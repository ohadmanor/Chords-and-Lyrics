"""
Compare our DSP chord extractor against madmom's trained chord recognizer
(DeepChromaChordRecognitionProcessor) on the same audio file.

Usage:
    python compare_extractors.py [path-to-audio]

Defaults to the cached "Imagine" test file. Prints a side-by-side timeline,
each engine's vocabulary, and frame-level agreement (root and root+quality)
sampled on a uniform grid.
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))

DEFAULT_AUDIO = os.path.join("backend", "downloads", "iOs9Osz3UFQ.mp3")

# ----------------------------------------------------------------------------
# Chord-label parsing -> (root_pitch_class, is_minor) or None for no-chord
# ----------------------------------------------------------------------------
_NOTE_TO_PC = {
    "C": 0, "B#": 0,
    "C#": 1, "Db": 1,
    "D": 2,
    "D#": 3, "Eb": 3,
    "E": 4, "Fb": 4,
    "F": 5, "E#": 5,
    "F#": 6, "Gb": 6,
    "G": 7,
    "G#": 8, "Ab": 8,
    "A": 9,
    "A#": 10, "Bb": 10,
    "B": 11, "Cb": 11,
}


def parse_chord(label):
    """Return (root_pc, is_minor) or None for N / unparseable."""
    if not label or label in ("N", "X", ""):
        return None
    # Drop slash bass (C/E -> C) and madmom colon quality marker handling below.
    main = label.split("/")[0]
    # madmom style "C:maj", "A:min"
    quality_minor = False
    if ":" in main:
        root_str, qual = main.split(":", 1)
        quality_minor = qual.startswith("min")
    else:
        # our style "C", "Cm", "Cmaj7", "Cm7"
        # extract the leading root token (note letter + optional #/b)
        i = 1
        if len(main) > 1 and main[1] in ("#", "b"):
            i = 2
        root_str = main[:i]
        suffix = main[i:]
        quality_minor = suffix.startswith("m") and not suffix.startswith("maj")
    pc = _NOTE_TO_PC.get(root_str)
    if pc is None:
        return None
    return (pc, quality_minor)


# ----------------------------------------------------------------------------
# Engines
# ----------------------------------------------------------------------------
def run_ours(audio_path):
    from chord_extractor import extract_chords_from_audio
    res = extract_chords_from_audio(audio_path)
    # -> list of (time, label)
    return [(c["time"], c["chord"]) for c in res["chords"]], res.get("bpm")


def run_madmom(audio_path):
    from madmom.audio.chroma import DeepChromaProcessor
    from madmom.features.chords import DeepChromaChordRecognitionProcessor
    dcp = DeepChromaProcessor()
    decode = DeepChromaChordRecognitionProcessor()
    chroma = dcp(audio_path)
    segments = decode(chroma)  # list of (start, end, label)
    # Convert to (time, label) change-points.
    out = []
    prev = None
    for start, end, label in segments:
        if label != prev:
            out.append((float(start), str(label)))
            prev = label
    return out


# ----------------------------------------------------------------------------
# Sampling + comparison helpers
# ----------------------------------------------------------------------------
def label_at(changes, t):
    """Label active at time t for a list of (time, label) change-points."""
    cur = None
    for time, label in changes:
        if time <= t:
            cur = label
        else:
            break
    return cur


def display_label(changes, t):
    lab = label_at(changes, t)
    return lab if lab else "N"


def compare(ours, madmom, duration, step=0.5):
    grid = np.arange(0.0, duration, step)
    root_match = 0
    full_match = 0
    counted = 0
    for t in grid:
        a = parse_chord(label_at(ours, t))
        b = parse_chord(label_at(madmom, t))
        if a is None and b is None:
            continue
        counted += 1
        if a is not None and b is not None:
            if a[0] == b[0]:
                root_match += 1
                if a[1] == b[1]:
                    full_match += 1
    return counted, root_match, full_match, grid


def compare_tolerant(ours, madmom, duration, tol, step=0.5):
    """
    Timing-tolerant agreement: at each grid frame, count a match if *either*
    engine's chord matches the other's chord anywhere within +/- tol seconds.
    This filters out transition-boundary offsets (where both engines pick the
    same two chords but switch a beat apart) and reveals chord-level agreement.
    """
    grid = np.arange(0.0, duration, step)
    root_match = 0
    full_match = 0
    counted = 0
    win = max(1, int(round(tol / step)))
    for gi, t in enumerate(grid):
        a = parse_chord(label_at(ours, t))
        b = parse_chord(label_at(madmom, t))
        if a is None and b is None:
            continue
        counted += 1
        if a is None or b is None:
            continue
        lo = max(0, gi - win)
        hi = min(len(grid), gi + win + 1)
        # Candidate chords from the other engine within the window.
        ours_win = [parse_chord(label_at(ours, grid[k])) for k in range(lo, hi)]
        madmom_win = [parse_chord(label_at(madmom, grid[k])) for k in range(lo, hi)]
        # Root match if a's root appears in madmom_win OR b's root in ours_win.
        root_ok = any(c is not None and c[0] == a[0] for c in madmom_win) \
            or any(c is not None and c[0] == b[0] for c in ours_win)
        full_ok = any(c is not None and c == a for c in madmom_win) \
            or any(c is not None and c == b for c in ours_win)
        if root_ok:
            root_match += 1
        if full_ok:
            full_match += 1
    return counted, root_match, full_match


def vocab(changes):
    from collections import Counter
    return Counter(lbl if lbl else "N" for _, lbl in changes)


# ----------------------------------------------------------------------------
def main():
    audio = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_AUDIO
    if not os.path.exists(audio):
        print(f"Audio file not found: {audio}")
        sys.exit(1)

    import librosa
    duration = librosa.get_duration(path=audio)

    print(f"Audio: {audio}  ({duration:.1f}s)")
    print("=" * 64)

    print("Running OUR extractor...")
    ours, bpm = run_ours(audio)
    print(f"  -> {len(ours)} change-points, bpm={bpm:.1f}")

    print("Running madmom DeepChroma chord recognizer...")
    madmom = run_madmom(audio)
    print(f"  -> {len(madmom)} change-points")

    # Side-by-side timeline on a uniform grid (only print where either changes).
    print("\n" + "=" * 64)
    print(f"{'time':>7} | {'OURS':<12} | {'madmom':<12}")
    print("-" * 64)
    step = 0.5
    grid = np.arange(0.0, duration, step)
    last_a = last_b = None
    for t in grid:
        a = display_label(ours, t)
        b = display_label(madmom, t)
        if a != last_a or b != last_b:
            flag = "" if parse_chord(a) == parse_chord(b) else "  <-- differ"
            print(f"{t:7.2f} | {a:<12} | {b:<12}{flag}")
            last_a, last_b = a, b

    # Agreement metrics.
    counted, root_m, full_m, _ = compare(ours, madmom, duration, step)
    print("\n" + "=" * 64)
    print("STRICT AGREEMENT (exact frame, no timing tolerance):")
    print(f"  sampled frames compared : {counted}")
    if counted:
        print(f"  root match              : {root_m}/{counted} "
              f"({100.0 * root_m / counted:.1f}%)")
        print(f"  root + major/minor match: {full_m}/{counted} "
              f"({100.0 * full_m / counted:.1f}%)")

    # Timing-tolerant: allow chords to agree within +/- 1 beat.
    beat = 60.0 / bpm if bpm else 0.8
    tol = beat  # +/- one beat
    tcounted, troot, tfull = compare_tolerant(ours, madmom, duration, tol, step)
    print("\n" + "-" * 64)
    print(f"TIMING-TOLERANT AGREEMENT (+/- 1 beat = {tol:.2f}s):")
    print(f"  sampled frames compared : {tcounted}")
    if tcounted:
        print(f"  root match              : {troot}/{tcounted} "
              f"({100.0 * troot / tcounted:.1f}%)")
        print(f"  root + major/minor match: {tfull}/{tcounted} "
              f"({100.0 * tfull / tcounted:.1f}%)")

    # Vocabularies.
    print("\n" + "=" * 64)
    print("OURS vocabulary  :", dict(vocab(ours)))
    print("madmom vocabulary:", dict(vocab(madmom)))


if __name__ == "__main__":
    main()
