import sys
sys.path.insert(0, 'backend')
import numpy as np

print("Testing madmom DeepChroma chord recognition...")
print("=" * 50)

from madmom.audio.chroma import DeepChromaProcessor
from madmom.features.chords import DeepChromaChordRecognitionProcessor

dcp = DeepChromaProcessor()
decode = DeepChromaChordRecognitionProcessor()

audio_file = 'backend/downloads/NFk1WoXmsM8.mp3'
print(f"Processing: {audio_file}")

chroma = dcp(audio_file)
print(f"Chroma shape: {chroma.shape}")

chords = decode(chroma)
print(f"\nDetected {len(chords)} chord segments:")
print("-" * 40)

unique_chords = set()
for start, end, label in chords[:40]:
    print(f"  {start:6.2f}s - {end:6.2f}s : {label}")
    unique_chords.add(label)

if len(chords) > 40:
    print(f"  ... ({len(chords) - 40} more segments)")
    for start, end, label in chords[40:]:
        unique_chords.add(label)

print(f"\nTotal chord segments: {len(chords)}")
print(f"Unique chords: {len(unique_chords)}")
print(f"Chord names: {sorted(unique_chords)}")
