# Harmonix (v1.0.0) - Chord & Lyrics Alignment Studio

Harmonix is a modern web application designed to automatically extract chords from audio tracks (local uploads or YouTube streams), retrieve/sync lyrics, group musical intervals into clean 4/4 bars, and provide an interactive workspace to verify, edit, and play along.

---

## Technical Features

### 1. Chord Extraction & Music Theory Analysis
- **CQT Frequency Splitting**: To isolate bass notes and treble triads:
  - **Bass Chroma** (bins 0–35, <130Hz): Summed to extract dominant bass notes.
  - **Treble Chroma** (bins 36–71, >130Hz): Decoded via Hidden Markov Models (Viterbi path search) to identify the core triad (Major/Minor chords).
- **Slash Chords Support**: Harmonix automatically compiles slash chords (e.g., `Am/G`, `C/E`) when the detected bass note differs from the chord root.
- **BPM & Bar Alignment**: Employs harmonic onset beat tracking (`librosa.beat.beat_track`) to detect BPM and group chords into 4/4 measures (bars).

### 2. Vocal Activity Detection (VAD)
- **Vocal Band Spectral Flux**: Recognizes where the singing voice starts by isolating the core speech formant frequencies (500 Hz to 3000 Hz) and monitoring sustained spikes in spectral dynamics.
- **Beat Snapping**: Maps the estimated raw vocal onset timestamp to the nearest musical beat interval.

### 3. Verify & Edit Review Editor
- **Compact Grid Layout**: Display bars in a clean, responsive layout fitting dozens of bars on-screen without layout stretching.
- **Redundancy Suppression**: Only prints chord changes. Repeating chords are displayed as faint gray parenthesized placeholders (e.g. `(C)`), keeping the screen clean.
- **Lyric Start Bar Selection**: Clickable bar badges allow the user to select the exact measure where the lyrics begin. The Lyric Start bar is highlighted in a distinct purple color.
- **Playback Tracking**: The active playing bar card highlights in glowing indigo and auto-scrolls into view during playback.
- **Seek Scrubbing**: Clicking any bar card seeks the audio player directly to that bar's start time for quick audio reference.

### 4. Dual Playback Viewer
- **Lyrics Sheet Mode**: Standard, space-aligned monospace plain text sheet with synchronized scroll highlighting.
- **Bar Grid Mode**: Grid cards showing chords beat-by-beat with active highlighting following the audio scrubber.

### 5. Multi-Mode Lyric Sync
- **Auto-Sync (YouTube/LRC Databases)**: Automatically offsets lyric timestamps relative to the selected start bar.
- **Manual Sync**: Tapping Space/Enter to sync lyrics. Timestamps are automatically offset-shifted to align the first lyric line exactly with the user's selected starting bar.

---

## Project Structure

```
Chords_and Lyrics/
├── backend/
│   ├── app.py                # FastAPI server (Uvicorn, project routes)
│   ├── chord_extractor.py    # librosa DSP engine (CQT, Viterbi HMM, VAD)
│   └── downloads/            # Local cache for YouTube streams
├── frontend/                 # Angular (standalone components, signals)
│   ├── src/
│   │   ├── app/
│   │   │   ├── components/
│   │   │   │   ├── review-editor/        # Compact bars review & edit panel
│   │   │   │   ├── lyrics-syncer/        # Manual tapper syncer (offset auto-adjust)
│   │   │   │   ├── chord-sheet-editor/   # Sheet/Bar grid player editor
│   │   │   │   └── waveform/             # Waveform progress tracker
│   │   │   ├── services/                 # api.service.ts, audio.service.ts
│   │   │   └── app.component.ts          # Root component / central controller
│   │   └── index.html
│   └── angular.json
├── run.py                    # Server startup runner script (dev/prod dispatcher)
└── README.md                 # Project documentation
```

---

## Installation & Setup

### Prerequisites
- **Python 3.8+**
- **Node.js (v18+)**
- **FFmpeg** (installed and added to system `%PATH%` for audio conversion)

### 1. Run via Startup Wrapper
Simply run the root helper script to spin up the FastAPI backend (port `8000`) and the Angular dev frontend (port `4200`) concurrently:
```bash
python run.py
```

### 2. Manual Start

**Backend (FastAPI):**
```bash
cd backend
pip install -r requirements.txt
python app.py
```

**Frontend (Angular):**
```bash
cd frontend
npm install
npm start
```

---

## Project Save Format (`.chordproj`)
Harmonix projects can be saved and loaded as JSON-serialized files preserving chord sheets, timestamps, BPM, and bar grids:
```json
{
    "audioPath": "C:/Develop/Github/Chords_and Lyrics/backend/downloads/video_id.mp3",
    "chordsheetText": "C             G\nWhen I find myself in times of trouble...",
    "timestamps": [10.0, 14.2],
    "bpm": 114.8,
    "bars": [
        {"bar_index": 1, "chords": ["C", "C", "C", "G"], "time": 0.0}
    ]
}
```
