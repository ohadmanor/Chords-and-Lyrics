import sys
sys.path.insert(0, 'backend')
from chord_extractor import extract_chords_from_audio

results = extract_chords_from_audio(
    'backend/downloads/NFk1WoXmsM8.mp3',
    lambda msg, p: print(f"{p*100:.0f}%: {msg}")
)

print(f"\nBPM: {results['bpm']:.1f}")
print(f"Total bars: {len(results['bars'])}")
print(f"Total compressed chords: {len(results['chords'])}")

print("\nFirst 30 bars:")
for bar in results['bars'][:30]:
    print(f"  Bar {bar['bar_index']:3d}: {bar['chords']}")

# Count unique chords and chord changes
all_beat_chords = []
for bar in results['bars']:
    all_beat_chords.extend(bar['chords'])

changes = sum(1 for i in range(1, len(all_beat_chords)) if all_beat_chords[i] != all_beat_chords[i-1])
unique = set(c for c in all_beat_chords if c)
print(f"\nTotal beat-level chord changes: {changes}")
print(f"Unique chords detected: {len(unique)}")
print(f"Unique chord names: {sorted(unique)}")

# Count how many bars have 3+ different chords (noisy)
noisy_bars = 0
for bar in results['bars']:
    non_empty = [c for c in bar['chords'] if c]
    if len(set(non_empty)) >= 3:
        noisy_bars += 1
print(f"Bars with 3+ different chords (noisy): {noisy_bars} / {len(results['bars'])}")
