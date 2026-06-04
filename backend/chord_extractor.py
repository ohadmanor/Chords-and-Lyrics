import numpy as np
import librosa

# Optional madmom DeepChroma front-end (Path C hybrid). madmom's deep-learned
# chroma is harmony-focused and far cleaner than template chroma; when it is
# available we use it as the treble chroma source and keep the rest of the
# pipeline (beat-sync, Viterbi, 7th/slash decoding, key biasing) unchanged.
try:
    from madmom.audio.chroma import DeepChromaProcessor as _DeepChromaProcessor
    _MADMOM_AVAILABLE = True
except Exception:  # pragma: no cover - madmom optional
    _DeepChromaProcessor = None
    _MADMOM_AVAILABLE = False

# Lazily-initialised processor (loads a NN model on first use).
_DCP = None
_MADMOM_FPS = 10  # DeepChromaProcessor default frame rate

PITCH_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
# State layout (length 49):
#   0..11   major triads          (C, C#, ... B)
#   12..23  minor triads          (Cm, C#m, ... Bm)
#   24..35  major 7 chords        (Cmaj7, ...)
#   36..47  minor 7 chords        (Cm7, ...)
#   48      no-chord / silence (N)
CHORD_NAMES = (
    [p for p in PITCH_NAMES]
    + [p + "m" for p in PITCH_NAMES]
    + [p + "maj7" for p in PITCH_NAMES]
    + [p + "m7" for p in PITCH_NAMES]
)
NUM_STATES = 49
N_CHORD = 48  # index of the no-chord state


# ----------------------------------------------------------------------------
# Templates
# ----------------------------------------------------------------------------
def generate_templates():
    """
    12 major + 12 minor + 12 maj7 + 12 m7 chord templates over 12 pitch classes.
    Root is weighted higher than other tones to stabilise the root choice
    when overtones bleed into neighbouring pitch classes.
    """
    templates = np.zeros((N_CHORD, 12))
    root_w, third_w, fifth_w, seventh_w = 1.0, 0.8, 0.6, 0.5
    for r in range(12):
        # Major triad
        templates[r, r] = root_w
        templates[r, (r + 4) % 12] = third_w
        templates[r, (r + 7) % 12] = fifth_w
        # Minor triad
        templates[12 + r, r] = root_w
        templates[12 + r, (r + 3) % 12] = third_w
        templates[12 + r, (r + 7) % 12] = fifth_w
        # Maj7 (root, major 3, 5, major 7)
        templates[24 + r, r] = root_w
        templates[24 + r, (r + 4) % 12] = third_w
        templates[24 + r, (r + 7) % 12] = fifth_w
        templates[24 + r, (r + 11) % 12] = seventh_w
        # m7 (root, minor 3, 5, minor 7)
        templates[36 + r, r] = root_w
        templates[36 + r, (r + 3) % 12] = third_w
        templates[36 + r, (r + 7) % 12] = fifth_w
        templates[36 + r, (r + 10) % 12] = seventh_w
    templates /= np.linalg.norm(templates, axis=1, keepdims=True)
    return templates


# ----------------------------------------------------------------------------
# Circle-of-fifths weighted transition matrix
# ----------------------------------------------------------------------------
def _fifth_distance(p1, p2):
    pos1 = (p1 * 7) % 12
    pos2 = (p2 * 7) % 12
    diff = abs(pos1 - pos2)
    return min(diff, 12 - diff)


def build_transition_matrix(self_trans=0.85, alpha=0.6, n_prob=0.015):
    """
    NxN transition matrix:
      - self-loop dominates so chords persist for several beats,
      - off-diagonal weights decay with circle-of-fifths distance between roots,
      - relative major/minor pairs (e.g. C <-> Am) get a small boost,
      - small fixed probability of moving to/from N (no-chord).
      - extension chords (maj7, m7) sharing the same root are very cheap to move between.
    """
    T = np.zeros((NUM_STATES, NUM_STATES))
    for s1 in range(NUM_STATES):
        for s2 in range(NUM_STATES):
            if s1 == s2:
                T[s1, s2] = self_trans
            elif s1 == N_CHORD or s2 == N_CHORD:
                T[s1, s2] = n_prob
            else:
                p1, p2 = s1 % 12, s2 % 12
                d = _fifth_distance(p1, p2)
                w = np.exp(-alpha * d)
                fam1, fam2 = s1 // 12, s2 // 12
                # Same root, different chord family (e.g. C ↔ Cmaj7, Cm ↔ Cm7)
                if p1 == p2 and fam1 != fam2:
                    w *= 3.0
                # Relative major/minor boost (works across triads and 7ths)
                is_minor_fam = lambda f: f in (1, 3)
                if is_minor_fam(fam1) != is_minor_fam(fam2):
                    maj_root = p1 if not is_minor_fam(fam1) else p2
                    min_root = p2 if is_minor_fam(fam2) else p1
                    if (maj_root - min_root) % 12 == 3:
                        w *= 1.5
                T[s1, s2] = w
        off = [j for j in range(NUM_STATES) if j != s1]
        row_sum = T[s1, off].sum()
        if row_sum > 0:
            T[s1, off] *= (1.0 - self_trans) / row_sum
        T[s1, s1] = self_trans
    return T


# ----------------------------------------------------------------------------
# Chroma extraction helpers
# ----------------------------------------------------------------------------
def _cqt_to_chroma(cqt, bin_start, bin_end):
    """Fold CQT bins [bin_start:bin_end) into 12 pitch classes."""
    chroma = np.zeros((12, cqt.shape[1]))
    for b in range(bin_start, bin_end):
        chroma[b % 12] += cqt[b]
    return chroma


def _log_normalise(chroma, eps=1e-6):
    """Log-compress then L2-normalise each frame."""
    chroma = np.log1p(chroma)
    norms = np.linalg.norm(chroma, axis=0, keepdims=True)
    norms = np.where(norms < eps, 1.0, norms)
    return chroma / norms


def _centre_chroma(chroma):
    """
    Per-frame mean subtraction. Pitch classes that are present go positive,
    absent ones go negative. This makes templates of chords that share
    notes (e.g. C major vs A minor: both contain C+E) much more separable.
    """
    return chroma - chroma.mean(axis=0, keepdims=True)


def _madmom_treble_chroma(audio_path, sr, hop_length, n_frames):
    """
    Path C hybrid front-end. Run madmom's deep-learned chroma and resample it
    onto the librosa hop-frame grid so the rest of the pipeline (beat-sync,
    Viterbi, key biasing) consumes it unchanged.

    madmom DeepChroma is pitch-class ordered starting at C (matching
    PITCH_NAMES) at a fixed frame rate of _MADMOM_FPS. We map each librosa
    frame's centre time to the nearest madmom frame.

    Returns (12, n_frames) or None if madmom is unavailable / fails.
    """
    global _DCP
    if not _MADMOM_AVAILABLE:
        return None
    try:
        if _DCP is None:
            _DCP = _DeepChromaProcessor()
        mchroma = np.asarray(_DCP(audio_path), dtype=np.float64)  # (T, 12)
        if mchroma.ndim != 2 or mchroma.shape[1] != 12 or mchroma.shape[0] == 0:
            return None
        # librosa frame centre times -> madmom frame indices
        frame_times = librosa.frames_to_time(
            np.arange(n_frames), sr=sr, hop_length=hop_length)
        idx = np.clip((frame_times * _MADMOM_FPS).astype(int),
                      0, mchroma.shape[0] - 1)
        treble = mchroma[idx].T  # (12, n_frames)
        return treble
    except Exception as e:
        print(f"[chord_extractor] madmom front-end unavailable, "
              f"falling back to librosa chroma: {e}")
        return None


# Krumhansl-Schmuckler key profiles (major and minor)
_KS_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_KS_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def _estimate_key(chroma_mean):
    """Krumhansl-Schmuckler key estimation. Returns (tonic_pc, is_minor)."""
    if chroma_mean.sum() <= 0:
        return 0, False
    cm = chroma_mean - chroma_mean.mean()
    best_score = -np.inf
    best = (0, False)
    for tonic in range(12):
        for is_minor, profile in ((False, _KS_MAJOR), (True, _KS_MINOR)):
            p = np.roll(profile - profile.mean(), tonic)
            denom = np.linalg.norm(cm) * np.linalg.norm(p)
            if denom == 0:
                continue
            score = float(np.dot(cm, p) / denom)
            if score > best_score:
                best_score = score
                best = (tonic, is_minor)
    return best


def _key_aware_prior(tonic, is_minor, in_key_boost=1.15, off_key_seventh_penalty=0.6):
    """
    Build a length-NUM_STATES prior that gently favours diatonic chords of
    the estimated key. Out-of-key 7th chords get a small penalty so the
    decoder doesn't reach for an exotic Gm7 / Ebmaj7 / etc. on weak frames.
    """
    if is_minor:
        major_tonic = (tonic + 3) % 12
    else:
        major_tonic = tonic
    diatonic_major_roots = {(major_tonic + s) % 12 for s in (0, 5, 7)}
    diatonic_minor_roots = {(major_tonic + s) % 12 for s in (2, 4, 9)}
    prior = np.ones(NUM_STATES)
    for s in range(N_CHORD):
        root, fam = s % 12, s // 12
        is_minor_chord = fam in (1, 3)
        is_seventh = fam in (2, 3)
        in_key = (
            (not is_minor_chord and root in diatonic_major_roots)
            or (is_minor_chord and root in diatonic_minor_roots)
        )
        if in_key:
            prior[s] = in_key_boost
        elif is_seventh:
            prior[s] = off_key_seventh_penalty
    prior /= prior.sum()
    return prior


def _key_aware_weights(tonic, is_minor, diatonic_w=1.0, borrowed_w=0.35,
                       chromatic_w=0.12):
    """
    Per-state multiplicative weights enforcing the estimated key on *every*
    beat (unlike the prior, which only shapes the Viterbi initial state).

    Three tiers:
      * diatonic   – chord is exactly a chord built on a scale degree with the
                     expected quality (e.g. in C: C, Dm, Em, F, G, Am, Cmaj7,
                     Fmaj7, Dm7, Em7, Am7)            -> diatonic_w
      * borrowed   – root sits in the key's scale but the quality is off
                     (e.g. Gmaj7 / Cm in C major)     -> borrowed_w
      * chromatic  – root is not in the scale at all  -> chromatic_w

    Strong evidence can still overcome the weight, but weak/ambiguous beats
    no longer drift to out-of-scale chords.
    """
    if is_minor:
        major_tonic = (tonic + 3) % 12
    else:
        major_tonic = tonic
    # Major scale pitch classes relative to the major tonic.
    scale_pcs = {(major_tonic + s) % 12 for s in (0, 2, 4, 5, 7, 9, 11)}
    diatonic_major_roots = {(major_tonic + s) % 12 for s in (0, 5, 7)}
    diatonic_minor_roots = {(major_tonic + s) % 12 for s in (2, 4, 9)}
    diatonic_maj7_roots = {(major_tonic + s) % 12 for s in (0, 5)}
    diatonic_m7_roots = {(major_tonic + s) % 12 for s in (2, 4, 9)}

    weights = np.ones(N_CHORD)
    for s in range(N_CHORD):
        root, fam = s % 12, s // 12
        if fam == 0:        # major triad
            diatonic = root in diatonic_major_roots
        elif fam == 1:      # minor triad
            diatonic = root in diatonic_minor_roots
        elif fam == 2:      # maj7
            diatonic = root in diatonic_maj7_roots
        else:               # m7
            diatonic = root in diatonic_m7_roots
        if diatonic:
            weights[s] = diatonic_w
        elif root in scale_pcs:
            weights[s] = borrowed_w
        else:
            weights[s] = chromatic_w
    return weights


# ----------------------------------------------------------------------------
# Vocal onset estimation (unchanged behaviour)
# ----------------------------------------------------------------------------
def estimate_vocals_start_time(y, sr):
    try:
        hop_length = 512
        n_fft = 2048
        S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        band = np.where((freqs >= 500) & (freqs <= 3000))[0]
        if len(band) == 0:
            return 0.0
        S_v = S[band, :]
        sums = np.sum(S_v, axis=0)
        sums_safe = np.where(sums == 0, 1e-6, sums)
        S_vn = S_v / sums_safe
        flux = np.sum(np.diff(S_vn, axis=1) ** 2, axis=0)
        flux = np.concatenate(([0.0], flux))
        win_len = int(3.0 * sr / hop_length)
        if win_len % 2 == 0:
            win_len += 1
        if win_len > len(flux):
            win_len = max(3, len(flux) // 2)
            if win_len % 2 == 0:
                win_len += 1
        flux_smooth = np.convolve(flux, np.ones(win_len) / win_len, mode='same')
        times = librosa.frames_to_time(np.arange(len(flux)), sr=sr, hop_length=hop_length)
        intro = np.where(times < 5.0)[0]
        baseline = float(np.mean(flux_smooth[intro])) if len(intro) else 0.0015
        if baseline > 0.0035:
            return 0.0
        thr = max(0.0018, baseline * 1.5)
        for i in range(len(times)):
            if times[i] < 3.0:
                continue
            if flux_smooth[i] > thr:
                end = min(len(times), i + int(1.5 * sr / hop_length))
                if np.all(flux_smooth[i:end] > thr * 0.75):
                    return float(times[i])
        return 0.0
    except Exception as e:
        print(f"Error estimating vocals start: {e}")
        return 0.0


# ----------------------------------------------------------------------------
# Slash-chord resolution
# ----------------------------------------------------------------------------
def _decode_with_slash(path, bass_norm):
    """
    Walk the Viterbi path one chord-segment at a time. For each segment,
    pick the bass note dominant *across the whole segment*; only emit a
    slash chord when that bass note is a 3rd or 5th and clearly outweighs
    the root, so single-beat percussion bleed cannot create spurious slashes.
    """
    num_beats = len(path)
    decoded = [""] * num_beats
    i = 0
    while i < num_beats:
        state = int(path[i])
        j = i
        while j < num_beats and path[j] == state:
            j += 1
        if state == N_CHORD:
            i = j
            continue

        chord_label = CHORD_NAMES[state]
        root_idx = state % 12
        fam = state // 12
        is_minor_chord = fam in (1, 3)
        third = (root_idx + (3 if is_minor_chord else 4)) % 12
        fifth = (root_idx + 7) % 12

        seg_bass = bass_norm[:, i:j].mean(axis=1)
        bass_idx = int(np.argmax(seg_bass))
        chord_name = chord_label
        if (bass_idx != root_idx
                and bass_idx in (third, fifth)
                and seg_bass[bass_idx] > seg_bass[root_idx] * 1.25):
            chord_name = f"{chord_label}/{PITCH_NAMES[bass_idx]}"

        for k in range(i, j):
            decoded[k] = chord_name
        i = j
    return decoded


# ----------------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------------
def extract_chords_from_audio(audio_path, progress_callback=None):
    """
    Returns: { "chords": [...], "bpm": float, "bars": [...], "estimated_lyrics_start": float }
    """
    def _p(msg, frac):
        if progress_callback:
            progress_callback(msg, frac)

    _p("Loading audio file...", 0.05)
    y, sr = librosa.load(audio_path, sr=None, mono=True)

    _p("Separating harmonic component...", 0.20)
    y_harm = librosa.effects.harmonic(y, margin=4.0)

    _p("Estimating tuning...", 0.30)
    tuning = librosa.estimate_tuning(y=y_harm, sr=sr)

    _p("Tracking beats and tempo...", 0.40)
    tempo, beat_frames = librosa.beat.beat_track(
        y=y_harm, sr=sr, tightness=120, start_bpm=90.0,
    )
    bpm = float(tempo[0]) if isinstance(tempo, np.ndarray) else float(tempo)
    # Octave-correct doubled BPMs (most songs sit in 60-140 BPM).
    if bpm > 150.0:
        bpm /= 2.0
        print(f"[chord_extractor] BPM halved to {bpm:.1f} (octave correction)")

    _p("Computing Constant-Q transform...", 0.55)
    hop_length = 512
    n_octaves = 7
    n_bins = 12 * n_octaves
    cqt = np.abs(librosa.cqt(
        y=y_harm, sr=sr,
        hop_length=hop_length,
        fmin=librosa.note_to_hz('C1'),
        n_bins=n_bins, bins_per_octave=12,
        tuning=tuning,
    ))
    rms = librosa.feature.rms(y=y_harm, hop_length=hop_length)[0]

    # Bass: octaves 1-3 (bins 0..35, ~32-260 Hz) — wider so high-key bass notes survive
    # Treble: prefer madmom's deep-learned chroma (Path C hybrid) which is far
    #         cleaner/harmony-focused; fall back to librosa chroma_cqt when
    #         madmom is unavailable.
    bass_chroma_full = _cqt_to_chroma(cqt, 0, 36)
    treble_chroma_full = _madmom_treble_chroma(
        audio_path, sr, hop_length, cqt.shape[1])
    if treble_chroma_full is not None:
        print("[chord_extractor] Treble chroma: madmom DeepChroma (hybrid)")
    else:
        print("[chord_extractor] Treble chroma: librosa chroma_cqt (fallback)")
        treble_chroma_full = librosa.feature.chroma_cqt(
            y=y_harm, sr=sr, hop_length=hop_length,
            fmin=librosa.note_to_hz('C3'),
            n_octaves=4, bins_per_octave=36, tuning=tuning,
        )

    _p("Synchronising features to beats...", 0.70)
    if len(beat_frames) > 0:
        # Beat tracking often only locks on once a quiet/ambient intro ends, so
        # the first detected beat can land many seconds in. librosa.util.sync
        # would then collapse the entire intro into a single leading segment,
        # producing one giant "Bar 1" that spans everything up to the first
        # beat. Backfill a regular beat grid (using the median beat period)
        # from the first detected beat down to the start so the intro is
        # covered by normal-length bars.
        if len(beat_frames) >= 2:
            beat_period = int(round(np.median(np.diff(beat_frames))))
        else:
            beat_period = max(1, int(round((60.0 / max(bpm, 1.0)) * sr / hop_length)))
        if beat_period > 0 and beat_frames[0] > beat_period:
            prefix = np.arange(int(beat_frames[0]) - beat_period, 0, -beat_period, dtype=int)[::-1]
            prefix = prefix[prefix > 0]
            if len(prefix) > 0:
                beat_frames = np.concatenate([prefix, beat_frames])

        bass_sync = librosa.util.sync(bass_chroma_full, beat_frames, aggregate=np.median)
        treble_sync = librosa.util.sync(treble_chroma_full, beat_frames, aggregate=np.median)
        rms_sync = librosa.util.sync(rms[np.newaxis, :], beat_frames, aggregate=np.mean)[0]
        beat_times = [0.0] + list(librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length))
    else:
        duration = librosa.get_duration(y=y, sr=sr)
        ft = np.arange(0, duration, 0.5)
        ff = librosa.time_to_frames(ft, sr=sr, hop_length=hop_length)
        bass_sync = librosa.util.sync(bass_chroma_full, ff, aggregate=np.median)
        treble_sync = librosa.util.sync(treble_chroma_full, ff, aggregate=np.median)
        rms_sync = librosa.util.sync(rms[np.newaxis, :], ff, aggregate=np.mean)[0]
        beat_times = [0.0] + list(ft)

    num_beats = treble_sync.shape[1]
    # Align lengths defensively
    if len(beat_times) > num_beats:
        beat_times = beat_times[:num_beats]
    elif len(beat_times) < num_beats:
        last = beat_times[-1] if beat_times else 0.0
        beat_times = beat_times + [last] * (num_beats - len(beat_times))

    treble_norm = _log_normalise(treble_sync)
    bass_norm = _log_normalise(bass_sync)
    treble_centred = _centre_chroma(treble_norm)

    _p("Estimating song key...", 0.80)
    tonic, is_minor_key = _estimate_key(treble_norm.mean(axis=1))
    key_name = f"{PITCH_NAMES[tonic]}{'m' if is_minor_key else ''}"
    print(f"[chord_extractor] Estimated key: {key_name}")

    _p("Decoding chords with Viterbi...", 0.85)
    templates = generate_templates()                                  # (N_CHORD, 12)
    templates_c = templates - templates.mean(axis=1, keepdims=True)
    transition = build_transition_matrix(self_trans=0.5, alpha=0.6)
    prior = _key_aware_prior(tonic, is_minor_key)

    # Silence mask based on relative RMS energy
    rms_thresh = max(1e-4, 0.10 * float(np.median(rms_sync)))
    silence_mask = rms_sync < rms_thresh

    # Centred dot-product → contrast between similar chords (e.g. C vs Am) widens
    sims = templates_c @ treble_centred                               # (N_CHORD, num_beats)
    # Per-beat relative scaling: rank each beat's chords against that beat's own
    # best match so a sustained tonic can't dominate every frame. This lets
    # genuine chord changes surface instead of being swallowed.
    sims = sims - sims.max(axis=0, keepdims=True)
    sims_sharp = np.exp(25.0 * sims)

    # Bass-informed boost: scale each candidate chord's emission by how strongly
    # its root pitch class is present in the bass band. This is what makes the
    # decoder pick the correct chord when triads share notes (C vs Am, F vs Dm).
    root_pcs = np.array([s % 12 for s in range(N_CHORD)])
    bass_strength = bass_norm[root_pcs, :]
    bass_boost = 1.0 + 1.5 * bass_strength
    sims_sharp = sims_sharp * bass_boost

    # Per-beat diatonic biasing: enforce the estimated key on every frame so
    # weak/ambiguous beats stop drifting to out-of-scale chords. Applied as a
    # multiplicative weight (broadcast over all beats) that strong evidence can
    # still overcome.
    key_weights = _key_aware_weights(tonic, is_minor_key)
    sims_sharp = sims_sharp * key_weights[:, np.newaxis]

    emissions = np.zeros((NUM_STATES, num_beats))
    emissions[:N_CHORD, :] = sims_sharp
    fallback = sims_sharp.mean(axis=0) * 0.05
    emissions[N_CHORD, :] = np.where(silence_mask, 1.0, fallback)
    emissions[:N_CHORD, silence_mask] = 1e-6
    col_sums = emissions.sum(axis=0, keepdims=True)
    col_sums = np.where(col_sums == 0, 1.0, col_sums)
    emissions /= col_sums

    path = librosa.sequence.viterbi(emissions, transition, p_init=prior)

    _p("Resolving slash chords...", 0.92)
    chords_decoded = _decode_with_slash(path, bass_norm)

    # Compress consecutive duplicates → time-aligned list
    compressed = []
    current = None
    for col in range(num_beats):
        name = chords_decoded[col]
        if name != current:
            compressed.append({"time": float(beat_times[col]), "chord": name})
            current = name

    # Group into 4/4 bars
    bars = []
    bar_beats = 4
    for i in range(0, num_beats, bar_beats):
        seg = list(chords_decoded[i:i + bar_beats])
        if len(seg) < bar_beats:
            seg += [""] * (bar_beats - len(seg))
        bars.append({
            "bar_index": (i // bar_beats) + 1,
            "chords": seg,
            "time": float(beat_times[i]),
        })

    # Vocal onset → nearest beat
    raw_vocals_start = estimate_vocals_start_time(y, sr)
    if beat_times:
        idx = int(np.argmin(np.abs(np.array(beat_times) - raw_vocals_start)))
        estimated_lyrics_start = float(beat_times[idx])
    else:
        estimated_lyrics_start = float(raw_vocals_start)

    _p("Extraction complete!", 1.0)
    return {
        "chords": compressed,
        "bpm": bpm,
        "bars": bars,
        "estimated_lyrics_start": estimated_lyrics_start,
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
        print(f"Testing chord extraction on {test_file}...")
        results = extract_chords_from_audio(
            test_file, lambda msg, p: print(f"{p*100:5.1f}%  {msg}")
        )
        print(f"\nDetected BPM: {results['bpm']:.2f}")
        print("First 12 bars:")
        for bar in results["bars"][:12]:
            print(f"  Bar {bar['bar_index']:>3}  t={bar['time']:6.2f}s  {bar['chords']}")
    else:
        print("Please provide an audio file path to run test.")
