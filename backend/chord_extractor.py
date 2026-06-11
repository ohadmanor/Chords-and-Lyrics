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
SHARP_PITCH_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
FLAT_PITCH_NAMES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
# State layout (length 61):
#   0..11   major triads          (C, C#, ... B)
#   12..23  minor triads          (Cm, C#m, ... Bm)
#   24..35  major 7 chords        (Cmaj7, ...)
#   36..47  minor 7 chords        (Cm7, ...)
#   48..59  dominant 7 chords     (C7, ...)
#   60      no-chord / silence (N)
CHORD_NAMES = (
    [p for p in PITCH_NAMES]
    + [p + "m" for p in PITCH_NAMES]
    + [p + "maj7" for p in PITCH_NAMES]
    + [p + "m7" for p in PITCH_NAMES]
    + [p + "7" for p in PITCH_NAMES]
)
NUM_STATES = 61
N_CHORD = 60  # index of the no-chord state
NUM_ROOT_STATES = 13  # 12 pitch classes + no-chord
N_ROOT = 12
NUM_QUALITY_STATES = 6  # maj, min, maj7, m7, 7, no-chord
N_QUALITY = 5


# ----------------------------------------------------------------------------
# Templates
# ----------------------------------------------------------------------------
def generate_templates():
    """
    12 major + 12 minor + 12 maj7 + 12 m7 + 12 dominant 7 chord templates over 12 pitch classes.
    Root is weighted higher than other tones to stabilise the root choice
    when overtones bleed into neighbouring pitch classes.
    """
    templates = np.zeros((N_CHORD, 12))
    root_w, third_w, fifth_w, seventh_w = 1.0, 0.8, 0.6, 0.5
    # Penalise the opposite third so C vs Am-type ambiguities separate better.
    anti_third_w = -0.28
    for r in range(12):
        # Major triad
        templates[r, r] = root_w
        templates[r, (r + 4) % 12] = third_w
        templates[r, (r + 7) % 12] = fifth_w
        templates[r, (r + 3) % 12] = anti_third_w
        # Minor triad
        templates[12 + r, r] = root_w
        templates[12 + r, (r + 3) % 12] = third_w
        templates[12 + r, (r + 7) % 12] = fifth_w
        templates[12 + r, (r + 4) % 12] = anti_third_w
        # Maj7 (root, major 3, 5, major 7)
        templates[24 + r, r] = root_w
        templates[24 + r, (r + 4) % 12] = third_w
        templates[24 + r, (r + 7) % 12] = fifth_w
        templates[24 + r, (r + 11) % 12] = seventh_w
        templates[24 + r, (r + 3) % 12] = anti_third_w
        # m7 (root, minor 3, 5, minor 7)
        templates[36 + r, r] = root_w
        templates[36 + r, (r + 3) % 12] = third_w
        templates[36 + r, (r + 7) % 12] = fifth_w
        templates[36 + r, (r + 10) % 12] = seventh_w
        templates[36 + r, (r + 4) % 12] = anti_third_w
        # Dominant 7 (root, major 3, 5, minor 7)
        templates[48 + r, r] = root_w
        templates[48 + r, (r + 4) % 12] = third_w
        templates[48 + r, (r + 7) % 12] = fifth_w
        templates[48 + r, (r + 10) % 12] = seventh_w
        templates[48 + r, (r + 3) % 12] = anti_third_w
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


def _match_frame_count(chroma, n_frames):
    """Pad/trim chroma matrix to a target frame count."""
    if chroma.shape[1] == n_frames:
        return chroma
    if chroma.shape[1] > n_frames:
        return chroma[:, :n_frames]
    if chroma.shape[1] == 0:
        return np.zeros((12, n_frames), dtype=np.float64)
    pad = np.repeat(chroma[:, -1:], n_frames - chroma.shape[1], axis=1)
    return np.concatenate([chroma, pad], axis=1)


def _l1_normalise(chroma, eps=1e-8):
    sums = chroma.sum(axis=0, keepdims=True)
    sums = np.where(sums < eps, 1.0, sums)
    return chroma / sums


def _temporal_median_filter(chroma, width=5):
    """Median-smooth each pitch-class curve across time."""
    if width <= 1 or chroma.shape[1] <= 2:
        return chroma
    if width % 2 == 0:
        width += 1
    pad = width // 2
    padded = np.pad(chroma, ((0, 0), (pad, pad)), mode='edge')
    out = np.empty_like(chroma)
    for t in range(chroma.shape[1]):
        out[:, t] = np.median(padded[:, t:t + width], axis=1)
    return out


def _fallback_treble_chroma(y_harm, sr, hop_length, tuning):
    """
    Stronger fallback front-end when madmom is unavailable.

    Blend chroma_cqt + chroma_stft + chroma_cens and apply temporal median
    smoothing to reduce jitter from vocals and percussive transients.
    """
    cqt = librosa.feature.chroma_cqt(
        y=y_harm,
        sr=sr,
        hop_length=hop_length,
        fmin=librosa.note_to_hz('C3'),
        n_octaves=4,
        bins_per_octave=36,
        tuning=tuning,
    )
    stft = librosa.feature.chroma_stft(
        y=y_harm,
        sr=sr,
        hop_length=hop_length,
        tuning=tuning,
        n_fft=4096,
    )
    cens = librosa.feature.chroma_cens(
        y=y_harm,
        sr=sr,
        hop_length=hop_length,
        n_chroma=12,
    )

    n_frames = cqt.shape[1]
    stft = _match_frame_count(stft, n_frames)
    cens = _match_frame_count(cens, n_frames)

    cqt = _l1_normalise(np.maximum(0.0, cqt))
    stft = _l1_normalise(np.maximum(0.0, stft))
    cens = _l1_normalise(np.maximum(0.0, cens))

    blended = 0.58 * cqt + 0.27 * stft + 0.15 * cens
    blended = _temporal_median_filter(blended, width=5)
    return blended


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


def _key_aware_prior(tonic, is_minor, in_key_boost=1.15, off_key_seventh_penalty=0.52):
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
    diatonic_maj7_roots = {(major_tonic + s) % 12 for s in (0, 5)}
    diatonic_m7_roots = {(major_tonic + s) % 12 for s in (2, 4, 9)}
    diatonic_7_roots = {(tonic + 7) % 12} if is_minor else {(major_tonic + 7) % 12}

    prior = np.ones(NUM_STATES)
    for s in range(N_CHORD):
        root, fam = s % 12, s // 12
        is_seventh = fam in (2, 3, 4)
        
        if fam == 0:
            in_key = root in diatonic_major_roots
        elif fam == 1:
            in_key = root in diatonic_minor_roots
        elif fam == 2:
            in_key = root in diatonic_maj7_roots
        elif fam == 3:
            in_key = root in diatonic_m7_roots
        else:
            in_key = root in diatonic_7_roots

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
                     Fmaj7, Dm7, Em7, Am7, G7)        -> diatonic_w
      * borrowed   – root sits in the key's scale but the quality is off
                     (e.g. Gmaj7 / Cm in C major)     -> borrowed_w
      * chromatic  – root is not in the scale at all  -> chromatic_w

    Strong evidence can still overcome the weight, but weak/ambiguous beats
    no longer drift to out-of-scale chords.
    """
    tiers = _key_tiers_by_state(tonic, is_minor)
    weights = np.full(N_CHORD, chromatic_w, dtype=np.float64)
    weights[tiers == 1] = borrowed_w
    weights[tiers == 0] = diatonic_w
    return weights


def _key_tiers_by_state(tonic, is_minor):
    """
    Return per-state key tiers for chord states (size N_CHORD):
      0: diatonic quality+root
      1: scale root but borrowed quality
      2: chromatic root outside scale
    """
    if is_minor:
        major_tonic = (tonic + 3) % 12
    else:
        major_tonic = tonic

    scale_pcs = {(major_tonic + s) % 12 for s in (0, 2, 4, 5, 7, 9, 11)}
    diatonic_major_roots = {(major_tonic + s) % 12 for s in (0, 5, 7)}
    diatonic_minor_roots = {(major_tonic + s) % 12 for s in (2, 4, 9)}
    diatonic_maj7_roots = {(major_tonic + s) % 12 for s in (0, 5)}
    diatonic_m7_roots = {(major_tonic + s) % 12 for s in (2, 4, 9)}
    diatonic_7_roots = {(tonic + 7) % 12} if is_minor else {(major_tonic + 7) % 12}

    tiers = np.full(N_CHORD, 2, dtype=np.int8)
    for s in range(N_CHORD):
        root, fam = s % 12, s // 12
        if fam == 0:
            diatonic = root in diatonic_major_roots
        elif fam == 1:
            diatonic = root in diatonic_minor_roots
        elif fam == 2:
            diatonic = root in diatonic_maj7_roots
        elif fam == 3:
            diatonic = root in diatonic_m7_roots
        else:
            diatonic = root in diatonic_7_roots

        if diatonic:
            tiers[s] = 0
        elif root in scale_pcs:
            tiers[s] = 1
    return tiers


def _family_complexity_weights(triad_w=1.0, maj7_w=0.60, m7_w=0.72, dom7_w=0.70):
    """Down-weight 7th families so triads win unless 7th evidence is clear."""
    weights = np.full(N_CHORD, triad_w, dtype=np.float64)
    weights[24:36] = maj7_w
    weights[36:48] = m7_w
    weights[48:60] = dom7_w
    return weights


def _build_root_transition_matrix(self_trans=0.84, alpha=0.55, n_prob=0.03):
    """Transition matrix for root-only decoding (12 roots + no-chord)."""
    T = np.zeros((NUM_ROOT_STATES, NUM_ROOT_STATES), dtype=np.float64)
    for s1 in range(NUM_ROOT_STATES):
        for s2 in range(NUM_ROOT_STATES):
            if s1 == s2:
                T[s1, s2] = self_trans
            elif s1 == N_ROOT or s2 == N_ROOT:
                T[s1, s2] = n_prob
            else:
                T[s1, s2] = np.exp(-alpha * _fifth_distance(s1, s2))

        off = [j for j in range(NUM_ROOT_STATES) if j != s1]
        row_sum = T[s1, off].sum()
        if row_sum > 0:
            T[s1, off] *= (1.0 - self_trans) / row_sum
        T[s1, s1] = self_trans
    return T


def _root_prior(tonic, is_minor):
    """Key-aware initial prior for root decoding."""
    major_tonic = (tonic + 3) % 12 if is_minor else tonic
    scale_pcs = {(major_tonic + s) % 12 for s in (0, 2, 4, 5, 7, 9, 11)}
    dominant = (major_tonic + 7) % 12
    rel_minor = (major_tonic + 9) % 12

    p = np.full(NUM_ROOT_STATES, 0.75, dtype=np.float64)
    for r in range(12):
        if r == major_tonic:
            p[r] = 1.55
        elif r == dominant:
            p[r] = 1.35
        elif r == rel_minor:
            p[r] = 1.20
        elif r in scale_pcs:
            p[r] = 1.05
    p[N_ROOT] = 0.08
    p /= p.sum()
    return p


def _build_quality_transition_matrix(self_trans=0.80, n_prob=0.03):
    """Transition matrix for quality decoding (maj/min/7 families + no-chord)."""
    T = np.zeros((NUM_QUALITY_STATES, NUM_QUALITY_STATES), dtype=np.float64)

    for q1 in range(NUM_QUALITY_STATES):
        for q2 in range(NUM_QUALITY_STATES):
            if q1 == q2:
                T[q1, q2] = self_trans
            elif q1 == N_QUALITY or q2 == N_QUALITY:
                T[q1, q2] = n_prob
            else:
                w = 1.0
                if (q1, q2) in ((0, 2), (2, 0), (0, 4), (4, 0), (1, 3), (3, 1)):
                    w = 1.45
                elif (q1 in (0, 2, 4) and q2 in (0, 2, 4)) or (q1 in (1, 3) and q2 in (1, 3)):
                    w = 1.25
                T[q1, q2] = w

        off = [j for j in range(NUM_QUALITY_STATES) if j != q1]
        row_sum = T[q1, off].sum()
        if row_sum > 0:
            T[q1, off] *= (1.0 - self_trans) / row_sum
        T[q1, q1] = self_trans
    return T


def _decode_root_then_quality(emissions, sims_sharp, silence_mask, tonic, is_minor):
    """
    Two-stage decode:
      1) Viterbi on roots (12 + N)
      2) Viterbi on quality constrained by chosen root per beat
    """
    num_beats = sims_sharp.shape[1]

    root_scores = np.zeros((12, num_beats), dtype=np.float64)
    for root in range(12):
        states = np.array([root, 12 + root, 24 + root, 36 + root, 48 + root], dtype=np.int32)
        per_family = sims_sharp[states, :]
        root_scores[root, :] = np.max(per_family, axis=0) + 0.35 * np.mean(per_family, axis=0)

    root_emissions = np.zeros((NUM_ROOT_STATES, num_beats), dtype=np.float64)
    root_emissions[:12, :] = root_scores
    root_fallback = np.maximum(root_scores.mean(axis=0) * 0.06, 1e-8)
    root_emissions[N_ROOT, :] = np.where(silence_mask, 1.0, root_fallback)
    root_emissions[:12, silence_mask] = 1e-6
    rs = root_emissions.sum(axis=0, keepdims=True)
    rs = np.where(rs == 0.0, 1.0, rs)
    root_emissions /= rs

    root_path = librosa.sequence.viterbi(
        root_emissions,
        _build_root_transition_matrix(),
        p_init=_root_prior(tonic, is_minor),
    )

    quality_emissions = np.zeros((NUM_QUALITY_STATES, num_beats), dtype=np.float64)
    for beat in range(num_beats):
        root = int(root_path[beat])
        if root >= N_ROOT:
            quality_emissions[N_QUALITY, beat] = 1.0
            quality_emissions[:N_QUALITY, beat] = 1e-6
            continue

        states = np.array([root, 12 + root, 24 + root, 36 + root, 48 + root], dtype=np.int32)
        quality_emissions[:N_QUALITY, beat] = emissions[states, beat]
        quality_emissions[N_QUALITY, beat] = emissions[N_CHORD, beat] * 0.2

        # Guard against inflated 7th-family picks on ambiguous beats.
        triad_peak = max(float(emissions[root, beat]), float(emissions[12 + root, beat]))
        if triad_peak > 0.0:
            if float(emissions[24 + root, beat]) < triad_peak * 0.90:
                quality_emissions[2, beat] *= 0.72
            if float(emissions[36 + root, beat]) < triad_peak * 0.95:
                quality_emissions[3, beat] *= 0.82
            if float(emissions[48 + root, beat]) < triad_peak * 0.95:
                quality_emissions[4, beat] *= 0.82

    qs = quality_emissions.sum(axis=0, keepdims=True)
    qs = np.where(qs == 0.0, 1.0, qs)
    quality_emissions /= qs

    quality_prior = np.array([0.36, 0.31, 0.10, 0.09, 0.10, 0.04], dtype=np.float64)
    quality_prior /= quality_prior.sum()

    quality_path = librosa.sequence.viterbi(
        quality_emissions,
        _build_quality_transition_matrix(),
        p_init=quality_prior,
    )

    path = np.full(num_beats, N_CHORD, dtype=np.int32)
    for beat in range(num_beats):
        root = int(root_path[beat])
        quality = int(quality_path[beat])
        if root < N_ROOT and quality < N_QUALITY:
            path[beat] = quality * 12 + root

    return path, root_emissions


def _compute_beat_confidence(path, emissions, root_emissions):
    """Confidence score [0,1] per beat from chord prob, root prob and margin."""
    num_beats = len(path)
    conf = np.zeros(num_beats, dtype=np.float64)
    for beat in range(num_beats):
        state = int(path[beat])
        if state >= N_CHORD:
            conf[beat] = float(np.clip(emissions[N_CHORD, beat], 0.0, 1.0))
            continue

        chord_probs = emissions[:N_CHORD, beat]
        p_state = float(chord_probs[state])
        if chord_probs.size > 1:
            # second-largest probability for margin confidence
            second = float(np.partition(chord_probs, -2)[-2])
        else:
            second = 0.0
        margin = max(0.0, (p_state - second) / max(p_state, 1e-8))
        root_prob = float(root_emissions[state % 12, beat])
        conf[beat] = float(np.clip(0.58 * p_state + 0.30 * root_prob + 0.12 * margin, 0.0, 1.0))
    return conf


def _stabilise_with_confidence(path, emissions, beat_confidence,
                               weak_change_conf=0.38,
                               outlier_conf=0.30,
                               keep_ratio=0.82):
    """
    Hold previous chord on low-confidence flips and remove single-beat outliers.
    """
    fixed = np.array(path, dtype=np.int32, copy=True)
    conf = np.array(beat_confidence, dtype=np.float64, copy=True)
    num_beats = len(fixed)

    for beat in range(1, num_beats):
        cur = int(fixed[beat])
        prev = int(fixed[beat - 1])
        if cur >= N_CHORD or prev >= N_CHORD or cur == prev:
            continue
        cur_p = float(emissions[cur, beat])
        prev_p = float(emissions[prev, beat])
        if conf[beat] < weak_change_conf and prev_p >= cur_p * keep_ratio:
            fixed[beat] = prev
            conf[beat] = max(conf[beat], conf[beat - 1] * 0.90)

    for beat in range(1, num_beats - 1):
        if conf[beat] >= outlier_conf:
            continue
        left = int(fixed[beat - 1])
        right = int(fixed[beat + 1])
        cur = int(fixed[beat])
        if left == right and left < N_CHORD and cur != left:
            left_p = float(emissions[left, beat])
            cur_p = float(emissions[cur, beat]) if cur < N_CHORD else 0.0
            if left_p >= cur_p * 0.85:
                fixed[beat] = left
                conf[beat] = max(conf[beat], conf[beat - 1] * 0.92, conf[beat + 1] * 0.92)

    return fixed, conf


def _bar_confidence(beat_confidence, bar_beats=4):
    vals = []
    for i in range(0, len(beat_confidence), bar_beats):
        seg = beat_confidence[i:i + bar_beats]
        vals.append(float(np.mean(seg)) if len(seg) else 0.0)
    return vals


def _quality_evidence_weights(treble_norm, bass_norm, boost=0.52):
    """
    Per-beat quality weighting for major/minor disambiguation.

    For each chord state, boost frames where the expected 3rd is stronger than
    the opposite 3rd, and damp frames where the opposite 3rd dominates.
    """
    num_beats = treble_norm.shape[1]
    weights = np.ones((N_CHORD, num_beats), dtype=np.float64)

    for s in range(N_CHORD):
        root = s % 12
        fam = s // 12
        is_major_quality = fam in (0, 2, 4)
        good_third = (root + (4 if is_major_quality else 3)) % 12
        bad_third = (root + (3 if is_major_quality else 4)) % 12

        good_support = treble_norm[good_third, :] + 0.25 * bass_norm[good_third, :]
        bad_support = treble_norm[bad_third, :] + 0.25 * bass_norm[bad_third, :]
        delta = good_support - bad_support
        weights[s, :] = np.clip(1.0 + boost * delta, 0.55, 1.70)

        if fam in (2, 3, 4):
            seventh_idx = (root + (11 if fam == 2 else 10)) % 12
            seventh_support = treble_norm[seventh_idx, :] + 0.20 * bass_norm[seventh_idx, :]
            root_support = treble_norm[root, :] + 0.20 * bass_norm[root, :]
            seventh_ratio = seventh_support / np.maximum(root_support, 1e-6)
            seventh_factor = np.clip(0.65 + 0.90 * seventh_ratio, 0.55, 1.15)
            weights[s, :] *= seventh_factor

    return weights


def _repair_off_scale_states(
    path,
    emissions,
    tonic,
    is_minor,
    chromatic_ratio=0.70,
    borrowed_ratio=0.86,
    strong_evidence_ratio=0.94,
):
    """
    Correct low-confidence out-of-scale states after Viterbi.

    This keeps strongly-supported chromatic moments, but when a chosen chord is
    weak and an in-key alternative is close, it snaps to the in-key candidate.
    """
    fixed = np.array(path, dtype=np.int32, copy=True)
    tiers = _key_tiers_by_state(tonic, is_minor)
    diatonic_idx = np.where(tiers == 0)[0]
    in_scale_idx = np.where(tiers <= 1)[0]
    if len(diatonic_idx) == 0 or len(in_scale_idx) == 0:
        return fixed

    swaps = 0
    num_beats = fixed.shape[0]
    for beat in range(num_beats):
        state = int(fixed[beat])
        if state >= N_CHORD:
            continue

        tier = int(tiers[state])
        if tier == 0:
            continue

        beat_em = emissions[:N_CHORD, beat]
        peak = float(np.max(beat_em))
        if peak <= 0.0:
            continue
        cur_p = float(beat_em[state])

        # Keep genuine out-of-key moments when frame evidence is strong.
        if cur_p >= peak * strong_evidence_ratio:
            continue

        # Single-beat outliers between matching neighbours are usually noise.
        if 0 < beat < (num_beats - 1):
            left = int(fixed[beat - 1])
            right = int(fixed[beat + 1])
            if left == right and left < N_CHORD:
                neigh_tier = int(tiers[left])
                neigh_p = float(beat_em[left])
                if neigh_tier <= tier and neigh_p >= cur_p * 0.60:
                    fixed[beat] = left
                    swaps += 1
                    continue

        if tier >= 2:
            pool = in_scale_idx
            ratio = chromatic_ratio
        else:
            pool = diatonic_idx
            ratio = borrowed_ratio

        best_local = int(np.argmax(beat_em[pool]))
        best_state = int(pool[best_local])
        best_p = float(beat_em[best_state])

        if best_p >= max(peak * 0.30, cur_p * ratio):
            fixed[beat] = best_state
            swaps += 1

    if swaps:
        print(f"[chord_extractor] Key-repair adjusted {swaps} beat(s).")
    return fixed


def _repair_major_minor_quality(
    path,
    emissions,
    treble_norm,
    bass_norm,
    min_to_maj_ratio=1.06,
    maj_to_min_ratio=1.16,
    min_to_maj_emission_ratio=0.84,
    maj_to_min_emission_ratio=0.95,
):
    """
    Correct likely major/minor confusions for the same root after Viterbi.

    The decoder can occasionally prefer a minor triad when a major quality is
    more plausible. We compare local major-vs-minor third evidence plus family
    emission support and only switch when the alternate quality is clearly more
    consistent.
    """
    fixed = np.array(path, dtype=np.int32, copy=True)
    num_beats = fixed.shape[0]
    if num_beats == 0:
        return fixed

    swaps = 0
    major_families = {0, 2, 4}
    minor_families = {1, 3}

    for beat in range(num_beats):
        state = int(fixed[beat])
        if state >= N_CHORD:
            continue

        fam = state // 12
        if fam not in (0, 1):
            continue

        root = state % 12
        major_state = root
        minor_state = 12 + root

        # Single-beat quality spikes are often noise; trust matching neighbours
        # on the same root before looking at frame-level evidence.
        if 0 < beat < (num_beats - 1):
            left = int(fixed[beat - 1])
            right = int(fixed[beat + 1])
            if left == right and left < N_CHORD and (left % 12) == root:
                neigh_fam = left // 12
                if fam == 1 and neigh_fam in major_families:
                    fixed[beat] = major_state
                    swaps += 1
                    continue
                if fam == 0 and neigh_fam in minor_families:
                    fixed[beat] = minor_state
                    swaps += 1
                    continue

        win_start = max(0, beat - 1)
        win_end = min(num_beats, beat + 2)

        major_third_idx = (root + 4) % 12
        minor_third_idx = (root + 3) % 12

        major_third_support = float(np.mean(
            treble_norm[major_third_idx, win_start:win_end]
            + 0.30 * bass_norm[major_third_idx, win_start:win_end]
        ))
        minor_third_support = float(np.mean(
            treble_norm[minor_third_idx, win_start:win_end]
            + 0.30 * bass_norm[minor_third_idx, win_start:win_end]
        ))

        # Compare quality support across related families with same root.
        major_family_support = float(np.mean(
            emissions[[major_state, 24 + root, 48 + root], win_start:win_end]
        ))
        minor_family_support = float(np.mean(
            emissions[[minor_state, 36 + root], win_start:win_end]
        ))

        if fam == 1:
            # Minor -> major: allow easier switch to reduce false minor labels.
            if (
                major_third_support >= (minor_third_support * min_to_maj_ratio)
                and major_family_support >= (minor_family_support * min_to_maj_emission_ratio)
            ):
                fixed[beat] = major_state
                swaps += 1
        else:
            # Major -> minor: require stronger evidence to avoid over-correcting.
            if (
                minor_third_support >= (major_third_support * maj_to_min_ratio)
                and minor_family_support >= (major_family_support * maj_to_min_emission_ratio)
            ):
                fixed[beat] = minor_state
                swaps += 1

    if swaps:
        print(f"[chord_extractor] Quality-repair adjusted {swaps} beat(s).")
    return fixed


def _prefer_sharp_names(tonic, is_minor):
    """Choose accidental style from key center (sharp keys vs flat keys)."""
    major_tonic = (tonic + 3) % 12 if is_minor else tonic
    return major_tonic not in {5, 10, 3, 8}


def _state_to_chord_name(state, tonic, is_minor):
    if state < 0 or state >= N_CHORD:
        return ""
    names = SHARP_PITCH_NAMES if _prefer_sharp_names(tonic, is_minor) else FLAT_PITCH_NAMES
    root = names[state % 12]
    fam = state // 12
    if fam == 0:
        suffix = ""
    elif fam == 1:
        suffix = "m"
    elif fam == 2:
        suffix = "maj7"
    elif fam == 3:
        suffix = "m7"
    else:
        suffix = "7"
    return root + suffix


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
def _decode_with_slash(path, bass_norm, tonic, is_minor):
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

        chord_label = _state_to_chord_name(state, tonic, is_minor)
        root_idx = state % 12
        fam = state // 12
        is_minor_chord = fam in (1, 3)
        third = (root_idx + (3 if is_minor_chord else 4)) % 12
        fifth = (root_idx + 7) % 12

        seg_bass = bass_norm[:, i:j].mean(axis=1)
        bass_idx = int(np.argmax(seg_bass))
        names = SHARP_PITCH_NAMES if _prefer_sharp_names(tonic, is_minor) else FLAT_PITCH_NAMES
        chord_name = chord_label
        if (bass_idx != root_idx
                and bass_idx in (third, fifth)
                and seg_bass[bass_idx] > seg_bass[root_idx] * 1.25):
            chord_name = f"{chord_label}/{names[bass_idx]}"

        for k in range(i, j):
            decoded[k] = chord_name
        i = j
    return decoded


# ----------------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------------
def extract_chords_from_audio(audio_path, progress_callback=None):
    """
        Returns: {
            "chords": [...],
            "bpm": float,
            "bars": [...],
            "estimated_lyrics_start": float,
            "beat_confidence": [...],
            "bar_confidence": [...]
        }
    """
    def _p(msg, frac):
        if progress_callback:
            progress_callback(msg, frac)

    _p("Loading audio file...", 0.05)
    y, sr = librosa.load(audio_path, sr=22050, mono=True)

    _p("Separating harmonic component...", 0.20)
    y_harm = librosa.effects.harmonic(y, margin=3.0)

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
    # Treble: prefer madmom deep chroma, blended with a stronger librosa
    # fallback front-end for stability.
    bass_chroma_full = _cqt_to_chroma(cqt, 0, 36)
    fallback_treble = _fallback_treble_chroma(y_harm, sr, hop_length, tuning)
    treble_chroma_full = _madmom_treble_chroma(
        audio_path, sr, hop_length, cqt.shape[1])
    if treble_chroma_full is not None:
        treble_chroma_full = _match_frame_count(treble_chroma_full, fallback_treble.shape[1])
        treble_chroma_full = 0.78 * treble_chroma_full + 0.22 * fallback_treble
        treble_chroma_full = _temporal_median_filter(treble_chroma_full, width=3)
        print("[chord_extractor] Treble chroma: madmom + librosa blended front-end")
    else:
        print("[chord_extractor] Treble chroma: librosa hybrid fallback")
        treble_chroma_full = fallback_treble

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
    transition = build_transition_matrix(self_trans=0.72, alpha=0.6)
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

    # Prefer the quality whose third is actually present on this beat.
    quality_weights = _quality_evidence_weights(treble_norm, bass_norm)
    sims_sharp = sims_sharp * quality_weights

    # Per-beat diatonic biasing: enforce the estimated key on every frame so
    # weak/ambiguous beats stop drifting to out-of-scale chords. Applied as a
    # multiplicative weight (broadcast over all beats) that strong evidence can
    # still overcome.
    key_weights = _key_aware_weights(tonic, is_minor_key)
    sims_sharp = sims_sharp * key_weights[:, np.newaxis]

    # 7th chords are available, but triads should win unless evidence is clear.
    family_weights = _family_complexity_weights()
    sims_sharp = sims_sharp * family_weights[:, np.newaxis]

    emissions = np.zeros((NUM_STATES, num_beats))
    emissions[:N_CHORD, :] = sims_sharp
    fallback = sims_sharp.mean(axis=0) * 0.05
    emissions[N_CHORD, :] = np.where(silence_mask, 1.0, fallback)
    emissions[:N_CHORD, silence_mask] = 1e-6
    col_sums = emissions.sum(axis=0, keepdims=True)
    col_sums = np.where(col_sums == 0, 1.0, col_sums)
    emissions /= col_sums

    # Root-first then quality-second decoding (more stable major/minor handling).
    path, root_emissions = _decode_root_then_quality(
        emissions=emissions,
        sims_sharp=sims_sharp,
        silence_mask=silence_mask,
        tonic=tonic,
        is_minor=is_minor_key,
    )

    # Keep classic full-state decode around as a fallback for severe mismatch.
    legacy_path = librosa.sequence.viterbi(emissions, transition, p_init=prior)
    mismatch = float(np.mean(path != legacy_path)) if num_beats else 0.0
    if mismatch > 0.72:
        path = legacy_path
        root_emissions = np.zeros((NUM_ROOT_STATES, num_beats), dtype=np.float64)
        root_emissions[:12, :] = emissions[:12, :]
        root_emissions[N_ROOT, :] = emissions[N_CHORD, :]
        print("[chord_extractor] Root/quality decode diverged heavily; using legacy path fallback.")

    path = _repair_off_scale_states(path, emissions, tonic, is_minor_key)
    path = _repair_major_minor_quality(path, emissions, treble_norm, bass_norm)

    beat_confidence = _compute_beat_confidence(path, emissions, root_emissions)
    path, beat_confidence = _stabilise_with_confidence(path, emissions, beat_confidence)

    _p("Resolving slash chords...", 0.92)
    chords_decoded = _decode_with_slash(path, bass_norm, tonic, is_minor_key)

    # Compress consecutive duplicates → time-aligned list
    compressed = []
    col = 0
    while col < num_beats:
        name = chords_decoded[col]
        end = col + 1
        while end < num_beats and chords_decoded[end] == name:
            end += 1
        seg_conf = float(np.mean(beat_confidence[col:end])) if end > col else 0.0
        compressed.append({
            "time": float(beat_times[col]),
            "chord": name,
            "confidence": seg_conf,
        })
        col = end

    # Group into 4/4 bars
    bars = []
    bar_confidence = _bar_confidence(beat_confidence, bar_beats=4)
    bar_beats = 4
    for i in range(0, num_beats, bar_beats):
        seg = list(chords_decoded[i:i + bar_beats])
        if len(seg) < bar_beats:
            seg += [""] * (bar_beats - len(seg))
        bar_idx = i // bar_beats
        bars.append({
            "bar_index": bar_idx + 1,
            "chords": seg,
            "time": float(beat_times[i]),
            "confidence": bar_confidence[bar_idx] if bar_idx < len(bar_confidence) else 0.0,
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
        "estimated_key": key_name,
        "estimated_lyrics_start": estimated_lyrics_start,
        "beat_confidence": beat_confidence.tolist(),
        "bar_confidence": bar_confidence,
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
