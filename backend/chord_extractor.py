import os
import numpy as np
import librosa

CHORD_NAMES = [
    # Major chords (0-11)
    "C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B",
    # Minor chords (12-23)
    "Cm", "C#m", "Dm", "Ebm", "Em", "Fm", "F#m", "Gm", "Abm", "Am", "Bbm", "Bm"
]

PITCH_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]

def get_fifth_distance(p1, p2):
    """
    Get the distance between two pitch class roots p1 and p2 on the Circle of Fifths.
    Arrangement: C(0)-G(7)-D(2)-A(9)-E(4)-B(11)-F#(6)-C#(1)-Ab(8)-Eb(3)-Bb(10)-F(5)
    Returns an integer distance between 0 (same pitch) and 6 (tritone).
    """
    pos1 = (p1 * 7) % 12
    pos2 = (p2 * 7) % 12
    diff = abs(pos1 - pos2)
    return min(diff, 12 - diff)

def construct_transition_matrix(num_states=25, self_trans=0.92, alpha=0.5):
    """
    Construct transition probability matrix where chord-to-chord transitions
    are weighted based on their distance on the Circle of Fifths.
    """
    transition = np.zeros((num_states, num_states))
    
    for s1 in range(num_states):
        for s2 in range(num_states):
            if s1 == s2:
                transition[s1, s2] = self_trans
            elif s1 == 24 or s2 == 24:
                # Flat transition probability to/from silence
                transition[s1, s2] = (1.0 - self_trans) / (num_states - 1)
            else:
                # Chord-to-chord transition
                p1 = s1 % 12
                p2 = s2 % 12
                d = get_fifth_distance(p1, p2)
                
                # Exponential decay weight based on Circle of Fifths distance
                weight = np.exp(-alpha * d)
                
                # Boost relative major/minor key transitions (e.g. C major (0) <-> A minor (21))
                is_m1 = s1 >= 12
                is_m2 = s2 >= 12
                if is_m1 != is_m2:
                    maj_root = p1 if not is_m1 else p2
                    min_root = p2 if is_m2 else p1
                    if (maj_root - min_root) % 12 == 3:
                        weight *= 1.5
                        
                transition[s1, s2] = weight
                
        # Normalize non-self transition probabilities to sum to (1.0 - self_trans)
        non_self_indices = [idx for idx in range(num_states) if idx != s1]
        row_sum = np.sum(transition[s1, non_self_indices])
        if row_sum > 0:
            transition[s1, non_self_indices] = transition[s1, non_self_indices] / row_sum * (1.0 - self_trans)
            transition[s1, s1] = self_trans
            
    return transition

def smooth_vector(vec, window_size=3):
    """
    Applies a moving average filter to smooth chroma features across beats.
    """
    if len(vec) < window_size:
        return vec
    filt = np.ones(window_size) / window_size
    return np.convolve(vec, filt, mode='same')

def generate_templates():
    """
    Generate pitch templates for 12 Major and 12 Minor chords.
    Each template is a normalized 12-dimensional vector.
    """
    templates = []
    
    # 12 Major Chords
    for r in range(12):
        t = np.zeros(12)
        t[r] = 1.0               # Root
        t[(r + 4) % 12] = 1.0    # Major 3rd
        t[(r + 7) % 12] = 1.0    # Perfect 5th
        t /= np.linalg.norm(t)
        templates.append(t)
        
    # 12 Minor Chords
    for r in range(12):
        t = np.zeros(12)
        t[r] = 1.0               # Root
        t[(r + 3) % 12] = 1.0    # Minor 3rd
        t[(r + 7) % 12] = 1.0    # Perfect 5th
        t /= np.linalg.norm(t)
        templates.append(t)
        
    return np.array(templates)

def estimate_vocals_start_time(y, sr):
    """
    Estimates where the singing starts by analyzing the spectral flux in the vocal frequency band.
    """
    try:
        hop_length = 512
        n_fft = 2048
        
        # Compute spectrogram
        S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
        frequencies = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        
        # Define vocal band (500 Hz to 3000 Hz where formants are strongest)
        vocal_band_idx = np.where((frequencies >= 500) & (frequencies <= 3000))[0]
        if len(vocal_band_idx) == 0:
            return 0.0
            
        S_vocal = S[vocal_band_idx, :]
        
        # Compute normalized frame-to-frame spectral flux in vocal band
        sums = np.sum(S_vocal, axis=0)
        sums_safe = np.where(sums == 0, 1e-6, sums)
        S_vocal_norm = S_vocal / sums_safe
        
        flux = np.sum(np.diff(S_vocal_norm, axis=1)**2, axis=0)
        flux = np.concatenate(([0.0], flux))
        
        # Smooth flux with a 3.0-second moving average window
        win_len = int(3.0 * sr / hop_length)
        if win_len % 2 == 0:
            win_len += 1
        if win_len > len(flux):
            win_len = max(3, len(flux) // 2)
            if win_len % 2 == 0:
                win_len += 1
                
        flux_smooth = np.convolve(flux, np.ones(win_len)/win_len, mode='same')
        times = librosa.frames_to_time(np.arange(len(flux)), sr=sr, hop_length=hop_length)
        
        # Define baseline from first 5 seconds
        intro_frames = np.where(times < 5.0)[0]
        if len(intro_frames) > 0:
            baseline_flux = np.mean(flux_smooth[intro_frames])
        else:
            baseline_flux = 0.0015
            
        # If baseline is extremely high, song likely starts with singing immediately
        if baseline_flux > 0.0035:
            return 0.0
            
        threshold = baseline_flux * 1.5
        threshold = max(0.0018, threshold) # Enforce a noise floor
        
        estimated_start = 0.0
        for i in range(len(times)):
            if times[i] < 3.0: # Ignore onset transients in first 3 seconds
                continue
            if flux_smooth[i] > threshold:
                # Check if it remains high for 1.5 seconds
                check_dur = 1.5
                check_frames = int(check_dur * sr / hop_length)
                check_end = min(len(times), i + check_frames)
                if np.all(flux_smooth[i:check_end] > threshold * 0.75):
                    estimated_start = float(times[i])
                    break
        return estimated_start
    except Exception as e:
        print(f"Error estimating vocals start: {e}")
        return 0.0

def extract_chords_from_audio(audio_path, progress_callback=None):
    """
    Extracts chords (with slash chords) and groups them by 4/4 bars.
    
    Returns:
        Dict: {
            "chords": List[dict],  # Compressed chords [{"time": float, "chord": str}]
            "bpm": float,          # Detected BPM
            "bars": List[dict]     # Chords arranged by bars [{"bar_index": int, "chords": list, "time": float}]
        }
    """
    if progress_callback:
        progress_callback("Loading audio file...", 0.1)
        
    # Load audio
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    
    if progress_callback:
        progress_callback("Separating harmonic components...", 0.3)
    y_harmonic = librosa.effects.harmonic(y)
    
    if progress_callback:
        progress_callback("Estimating song tuning...", 0.4)
    tuning = librosa.estimate_tuning(y=y_harmonic, sr=sr)
    
    if progress_callback:
        progress_callback("Tracking beats and tempo...", 0.5)
    tempo, beat_frames = librosa.beat.beat_track(y=y_harmonic, sr=sr)
    bpm = float(tempo[0]) if isinstance(tempo, np.ndarray) else float(tempo)
    
    if progress_callback:
        progress_callback("Computing Constant-Q features...", 0.6)
    # Use 7 octaves (84 bins) starting at C1
    cqt = np.abs(librosa.cqt(
        y=y_harmonic, sr=sr, 
        fmin=librosa.note_to_hz('C1'), 
        n_bins=84, bins_per_octave=12, 
        tuning=tuning
    ))
    rms = librosa.feature.rms(y=y_harmonic)
    
    if progress_callback:
        progress_callback("Synchronizing features to beats...", 0.7)
        
    # Synchronize CQT to beat intervals
    if len(beat_frames) > 0:
        cqt_sync = librosa.util.sync(cqt, beat_frames, aggregate=np.mean)
        rms_sync = librosa.util.sync(rms, beat_frames, aggregate=np.mean)[0]
        beat_times = [0.0] + list(librosa.frames_to_time(beat_frames, sr=sr))
    else:
        duration = librosa.get_duration(y=y, sr=sr)
        fallback_times = np.arange(0, duration, 0.5)
        fallback_frames = librosa.time_to_frames(fallback_times, sr=sr)
        cqt_sync = librosa.util.sync(cqt, fallback_frames, aggregate=np.mean)
        rms_sync = librosa.util.sync(rms, fallback_frames, aggregate=np.mean)[0]
        beat_times = [0.0] + list(fallback_times)

    num_beats = cqt_sync.shape[1]
    
    # Split CQT:
    # Bass chroma = octaves 1-3 (bins 0 to 35) representing low-end bass note energy
    # Treble chroma = octaves 4-6 (bins 36 to 71) representing mid-range triad chord notes
    bass_chroma_sync = np.zeros((12, num_beats))
    treble_chroma_sync = np.zeros((12, num_beats))
    
    for note in range(12):
        bass_chroma_sync[note, :] = np.sum(cqt_sync[[note, note+12, note+24], :], axis=0)
        treble_chroma_sync[note, :] = np.sum(cqt_sync[[note+36, note+48, note+60], :], axis=0)

    if progress_callback:
        progress_callback("Running Viterbi decoding...", 0.8)
        
    templates = generate_templates()
    num_states = 25 # 24 chords + 1 silence (index 24)
    
    # Transition probability matrix (prefer staying in the same chord)
    self_trans = 0.85
    transition = np.ones((num_states, num_states)) * ((1.0 - self_trans) / (num_states - 1))
    np.fill_diagonal(transition, self_trans)
    
    prior = np.ones(num_states) / num_states
    
    # Compute emissions for Treble triads
    emissions = np.zeros((num_states, num_beats))
    for col in range(num_beats):
        chroma_vec = treble_chroma_sync[:, col]
        norm = np.linalg.norm(chroma_vec)
        is_silence = rms_sync[col] < 0.005 or norm == 0
        
        for state in range(25):
            if state == 24:
                emissions[state, col] = 1.0 if is_silence else 1e-4
            else:
                if is_silence:
                    emissions[state, col] = 1e-4
                else:
                    chroma_vec_norm = chroma_vec / norm
                    sim = np.dot(templates[state], chroma_vec_norm)
                    emissions[state, col] = np.exp(sim * 10.0)
                    
        col_sum = np.sum(emissions[:, col])
        if col_sum > 0:
            emissions[:, col] /= col_sum
            
    path = librosa.sequence.viterbi(emissions, transition, p_init=prior)
    
    # Decode final chords (merging treble triad + bass note if different)
    chords_decoded = []
    for col in range(num_beats):
        state = path[col]
        if state == 24:
            chords_decoded.append("")
            continue
            
        triad = CHORD_NAMES[state]
        root_idx = state % 12
        
        # Bass note detection: pick note index with the highest energy in bass_chroma
        bass_vec = bass_chroma_sync[:, col]
        bass_idx = np.argmax(bass_vec) if np.max(bass_vec) > 0.001 else root_idx
        
        if bass_idx != root_idx:
            chord_name = f"{triad}/{PITCH_NAMES[bass_idx]}"
        else:
            chord_name = triad
        chords_decoded.append(chord_name)
        
    # Compress consecutive duplicates for time-aligned list
    compressed_chords = []
    current_chord = None
    for col in range(num_beats):
        chord_name = chords_decoded[col]
        t_time = beat_times[col]
        if chord_name != current_chord:
            compressed_chords.append({
                "time": float(t_time),
                "chord": chord_name
            })
            current_chord = chord_name
            
    # Group into 4/4 bars
    bars = []
    bar_beats = 4
    for i in range(0, num_beats, bar_beats):
        bar_chords = chords_decoded[i : i + bar_beats]
        if len(bar_chords) < bar_beats:
            bar_chords += [""] * (bar_beats - len(bar_chords))
        bars.append({
            "bar_index": (i // bar_beats) + 1,
            "chords": bar_chords,
            "time": float(beat_times[i])
        })
        
    # Estimate lyrics start time from audio spectral flux in vocal band
    raw_vocals_start = estimate_vocals_start_time(y, sr)
    
    # Map raw vocal start time to the closest beat time
    estimated_lyrics_start = 0.0
    if len(beat_times) > 0:
        closest_beat_idx = np.argmin(np.abs(np.array(beat_times) - raw_vocals_start))
        estimated_lyrics_start = float(beat_times[closest_beat_idx])
    else:
        estimated_lyrics_start = float(raw_vocals_start)
            
    if progress_callback:
        progress_callback("Extraction complete!", 1.0)
        
    return {
        "chords": compressed_chords,
        "bpm": bpm,
        "bars": bars,
        "estimated_lyrics_start": estimated_lyrics_start
    }

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
        print(f"Testing improved chord extraction on {test_file}...")
        results = extract_chords_from_audio(test_file, lambda msg, progress: print(f"{progress*100:.0f}%: {msg}"))
        print(f"Detected BPM: {results['bpm']}")
        print(f"First 10 Bars:")
        for bar in results['bars'][:10]:
            print(f"Bar {bar['bar_index']}: {bar['chords']}")
    else:
        print("Please provide an audio file path to run test.")
