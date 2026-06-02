import os
import re
import json
import sys
import tkinter as tk
from tkinter import filedialog
import concurrent.futures
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import urllib.parse
import librosa
import syncedlyrics
import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi

from chord_extractor import extract_chords_from_audio

# Configure UTF-8 encoding for standard output and error to prevent crashes in Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')


app = FastAPI(title="Chord & Lyrics Extractor API", version="1.0.0")

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

extraction_tasks = {}

class LyricsLine(BaseModel):
    text: str
    time: float
    duration: Optional[float] = None

class GenerateSheetRequest(BaseModel):
    chords: List[dict]
    lyrics: List[LyricsLine]
    duration: float

class SaveProjectRequest(BaseModel):
    filePath: str
    audioPath: str
    chordsheetText: str
    timestamps: List[float]
    bpm: Optional[float] = None
    bars: Optional[List[dict]] = None

class YoutubeRequest(BaseModel):
    url: str

# Helper to extract YouTube video ID from various URL shapes
def extract_video_id(url: str) -> Optional[str]:
    pattern = r'(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|\S*?[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})'
    match = re.search(pattern, url)
    return match.group(1) if match else None

# Helper to clean up filenames into queryable song titles
def get_song_query(audio_path):
    filename = os.path.basename(audio_path)
    name, _ = os.path.splitext(filename)
    
    # Strip bracket/parenthetical annotations
    name = re.sub(r'[\(\[\{].*?[\)\]\}]', '', name)
    name = re.sub(r'[-_\.]', ' ', name)
    name = " ".join(name.split())
    return name

# Search and parse LRC text from online databases
def search_and_parse_lyrics(query):
    try:
        print(f"Searching online synced lyrics for: '{query}'...")
        lrc_text = syncedlyrics.search(query, synced_only=True)
        if not lrc_text:
            print("No synced lyrics found online.")
            return None
        
        lines = []
        pattern = re.compile(r'\[(\d+):(\d+)(?:\.(\d+))?\](.*)')
        
        for line in lrc_text.split('\n'):
            match = pattern.match(line.strip())
            if match:
                minutes = int(match.group(1))
                seconds = int(match.group(2))
                ms_match = match.group(3)
                
                milliseconds = 0
                if ms_match:
                    if len(ms_match) == 1:
                        milliseconds = int(ms_match) * 100
                    elif len(ms_match) == 2:
                        milliseconds = int(ms_match) * 10
                    else:
                        milliseconds = int(ms_match[:3])
                        
                timestamp = minutes * 60 + seconds + milliseconds / 1000.0
                lyric_text = match.group(4).strip()
                lyric_text = lyric_text.replace('♪', '').strip()
                
                # Exclude metadata tags
                if lyric_text and not lyric_text.startswith('[') and not any(tag in lyric_text.lower() for tag in ['[by:', '[ar:', '[ti:', '[al:', '[length:']):
                    lines.append({"text": lyric_text, "time": timestamp})
                    
        lines = sorted(lines, key=lambda x: x["time"])
        return lines
    except Exception as e:
        print(f"Lyrics search failed: {e}")
        return None

def search_unsynced_lyrics(query):
    try:
        print(f"Searching web for unsynced lyrics: {repr(query)}...")
        txt = syncedlyrics.search(query, synced_only=False)
        if txt:
            # If the result contains LRC tags, strip them to get plain lyrics
            has_timestamps = any(re.match(r'\[\d+:\d+', line) for line in txt.split('\n')[:5])
            if has_timestamps:
                lines = []
                for line in txt.split('\n'):
                    cleaned = re.sub(r'\[\d+:\d+(?:\.\d+)?\]', '', line).strip()
                    cleaned = cleaned.replace('♪', '').strip()
                    if cleaned:
                        lines.append(cleaned)
                txt = "\n".join(lines)
            else:
                lines = []
                for line in txt.split('\n'):
                    cleaned = line.replace('♪', '').strip()
                    if cleaned:
                        lines.append(cleaned)
                txt = "\n".join(lines)
            return txt
    except Exception as e:
        print(f"Unsynced lyrics search failed: {e}")
    return None


# Fetch pre-synced transcript/captions directly from YouTube
def fetch_youtube_transcript(video_id: str) -> Optional[List[dict]]:
    try:
        print(f"Fetching YouTube captions for video: {video_id}...")
        transcript_list = YouTubeTranscriptApi().fetch(video_id, languages=["he", "iw", "en"])
        
        lyrics = []
        for item in transcript_list:
            text = item.text.strip()
            # Clean up YouTube caption noises like [Music] or (laughter)
            text = re.sub(r'[\(\[\{].*?[\)\]\}]', '', text).strip()
            # Remove music note symbols
            text = text.replace('♪', '').strip()
            # Skip empty entries
            if text:
                lyrics.append({
                    "text": text,
                    "time": float(item.start),
                    "duration": float(item.duration) if hasattr(item, "duration") else 0.0
                })
        print(f"Retrieved {len(lyrics)} synced caption lines from YouTube.")
        return lyrics
    except Exception as e:
        print(f"YouTube captions not found: {e}")
        return None

# Download audio using yt-dlp and convert to MP3 via FFmpeg
def download_youtube_audio(video_id: str, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    output_mp3 = os.path.join(output_dir, f"{video_id}.mp3")
    
    # Return cache if already downloaded
    if os.path.exists(output_mp3):
        print(f"Audio cache found for {video_id}.")
        return output_mp3
        
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': os.path.join(output_dir, f"{video_id}.%(ext)s"),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'quiet': True,
        'no_warnings': True,
    }
    
    url = f"https://www.youtube.com/watch?v={video_id}"
    print(f"Downloading YouTube audio for: {video_id}...")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
        
    return output_mp3

# Core chord alignment algorithm
def generate_aligned_sheet_internal(chords: list, lyrics: list, duration: float):
    if not lyrics:
        return {"chordsheet": "", "timestamps": []}
        
    output_lines = []
    timestamps = []
    
    sorted_lyrics = sorted(lyrics, key=lambda l: l["time"] if isinstance(l, dict) else l.time)
    
    # 1. Check for Intro Gap before first lyric line starts
    first_line_start = sorted_lyrics[0]["time"] if isinstance(sorted_lyrics[0], dict) else sorted_lyrics[0].time
    if first_line_start > 3.0:
        chords_in_intro = []
        for c in chords:
            if 0.0 <= c["time"] < first_line_start:
                chords_in_intro.append(c["chord"])
        
        clean_intro = []
        for chord in chords_in_intro:
            if chord and (not clean_intro or clean_intro[-1] != chord):
                clean_intro.append(chord)
                
        if clean_intro:
            instr_line = "// " + " / ".join(clean_intro) + " //"
            output_lines.append(instr_line)
            output_lines.append("")
            timestamps.append(0.0)
            
    # 2. Process all lines & Intermediate gaps
    for i, line in enumerate(sorted_lyrics):
        line_text = line["text"] if isinstance(line, dict) else line.text
        line_start = line["time"] if isinstance(line, dict) else line.time
        
        # Determine singing duration
        line_dur_val = line.get("duration") if isinstance(line, dict) else getattr(line, "duration", None)
        
        next_lyric = sorted_lyrics[i+1] if i+1 < len(sorted_lyrics) else None
        next_start = next_lyric["time"] if isinstance(next_lyric, dict) else next_lyric.time if next_lyric else duration
        
        if line_dur_val is not None and line_dur_val > 0:
            singing_duration = line_dur_val
        else:
            # Estimate: 4 chars per second, leaving at least 1 second gap
            singing_duration = len(line_text) * 0.25
            singing_duration = min(singing_duration, max(0.0, next_start - line_start - 1.0))
            
        singing_end = line_start + singing_duration
        
        # Chords during active singing
        chords_in_singing = []
        for c in chords:
            if line_start <= c["time"] < singing_end:
                chords_in_singing.append((c["time"], c["chord"]))
                
        # Carry over last active chord if singing starts chordless
        if not chords_in_singing:
            prev_chords = [c for c in chords if c["time"] <= line_start]
            if prev_chords:
                last_chord = prev_chords[-1]
                if last_chord["chord"]:
                    chords_in_singing.append((line_start, last_chord["chord"]))
                    
        # Write singing block
        line_len = len(line_text)
        if line_len > 0:
            chord_chars = [" "] * (line_len + 40)
            for t_chord, chord_name in chords_in_singing:
                if not chord_name:
                    continue
                ratio = (t_chord - line_start) / singing_duration if singing_duration > 0 else 0.0
                ratio = max(0.0, min(1.0, ratio))
                char_idx = int(round(ratio * line_len))
                for k, char in enumerate(chord_name):
                    target_idx = char_idx + k
                    if target_idx < len(chord_chars):
                        chord_chars[target_idx] = char
            chord_line = "".join(chord_chars).rstrip()
            
            output_lines.append(chord_line)
            output_lines.append(line_text)
            output_lines.append("")
            timestamps.append(line_start)
            
        # Check for gap before next line starts (Instrumental sections)
        gap_duration = next_start - singing_end
        if gap_duration > 3.0:
            chords_in_gap = []
            for c in chords:
                if singing_end <= c["time"] < next_start:
                    chords_in_gap.append(c["chord"])
                    
            clean_gap = []
            for chord in chords_in_gap:
                if chord and (not clean_gap or clean_gap[-1] != chord):
                    clean_gap.append(chord)
                    
            if clean_gap:
                instr_line = "// " + " / ".join(clean_gap) + " //"
                output_lines.append(instr_line)
                output_lines.append("")
                timestamps.append(singing_end)
                
    # 3. Check for Outro Gap at the end of the song
    last_line = sorted_lyrics[-1]
    last_text = last_line["text"] if isinstance(last_line, dict) else last_line.text
    last_start = last_line["time"] if isinstance(last_line, dict) else last_line.time
    last_dur = last_line.get("duration") if isinstance(last_line, dict) else getattr(last_line, "duration", None)
    
    if last_dur is not None and last_dur > 0:
        last_singing_end = last_start + last_dur
    else:
        last_singing_end = last_start + len(last_text) * 0.25
        last_singing_end = min(last_singing_end, duration)
        
    if duration - last_singing_end > 3.0:
        chords_in_outro = []
        for c in chords:
            if last_singing_end <= c["time"] < duration:
                chords_in_outro.append(c["chord"])
                
        clean_outro = []
        for chord in chords_in_outro:
            if chord and (not clean_outro or clean_outro[-1] != chord):
                clean_outro.append(chord)
                
        if clean_outro:
            instr_line = "// " + " / ".join(clean_outro) + " //"
            output_lines.append(instr_line)
            output_lines.append("")
            timestamps.append(last_singing_end)
            
    chordsheet_text = "\n".join(output_lines)
    return {
        "chordsheet": chordsheet_text,
        "timestamps": timestamps
    }

# File dialogue thread run executors
def run_file_dialog():
    root = tk.Tk()
    root.withdraw()
    root.focus_force()
    root.attributes("-topmost", True)
    file_path = filedialog.askopenfilename(
        title="Select Audio File",
        filetypes=[
            ("Audio Files", "*.mp3 *.wav *.ogg *.m4a *.flac"),
            ("All Files", "*.*")
        ]
    )
    root.destroy()
    return file_path

def run_save_dialog():
    root = tk.Tk()
    root.withdraw()
    root.focus_force()
    root.attributes("-topmost", True)
    file_path = filedialog.asksaveasfilename(
        title="Save Chord Project",
        defaultextension=".chordproj",
        filetypes=[("Chord Project Files", "*.chordproj"), ("All Files", "*.*")]
    )
    root.destroy()
    return file_path

def run_open_project_dialog():
    root = tk.Tk()
    root.withdraw()
    root.focus_force()
    root.attributes("-topmost", True)
    file_path = filedialog.askopenfilename(
        title="Open Chord Project",
        filetypes=[("Chord Project Files", "*.chordproj"), ("All Files", "*.*")]
    )
    root.destroy()
    return file_path

@app.get("/api/select-file")
async def select_file():
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(run_file_dialog)
        file_path = future.result()
    if not file_path:
        return {"status": "cancelled", "path": ""}
    return {"status": "selected", "path": file_path, "filename": os.path.basename(file_path)}

@app.get("/api/stream-audio")
def stream_audio(path: str):
    decoded_path = urllib.parse.unquote(path)
    if not os.path.exists(decoded_path):
        raise HTTPException(status_code=404, detail=f"Audio file not found at: {decoded_path}")
    return FileResponse(decoded_path)

def run_extraction_background(task_id: str, audio_path: str):
    def progress_cb(message, val):
        extraction_tasks[task_id] = {
            "status": "processing",
            "progress": val,
            "message": message,
            "result": None,
            "error": None
        }
        
    try:
        # 1. Chord Extraction
        chords_data = extract_chords_from_audio(audio_path, progress_cb)
        chords = chords_data["chords"]
        bpm = chords_data["bpm"]
        bars = chords_data["bars"]
        
        # 2. Lyrics search
        progress_cb("Searching online synced lyrics...", 0.92)
        query = get_song_query(audio_path)
        lyrics = search_and_parse_lyrics(query)
        
        chordsheet = ""
        timestamps = []
        auto_synced = False
        unsynced_lyrics = ""
        
        if lyrics:
            progress_cb("Aligning chords with synced lyrics...", 0.96)
            duration = float(librosa.get_duration(path=audio_path))
            aligned = generate_aligned_sheet_internal(chords, lyrics, duration)
            chordsheet = aligned["chordsheet"]
            timestamps = aligned["timestamps"]
            auto_synced = True
        else:
            progress_cb("Searching web for unsynced lyrics...", 0.94)
            unsynced_lyrics = search_unsynced_lyrics(query) or ""
            
        extraction_tasks[task_id] = {
            "status": "completed",
            "progress": 1.0,
            "message": "Extraction complete!",
            "result": {
                "chords": chords,
                "bpm": bpm,
                "bars": bars,
                "lyrics": lyrics,
                "chordsheet": chordsheet,
                "timestamps": timestamps,
                "auto_synced": auto_synced,
                "unsyncedLyrics": unsynced_lyrics,
                "estimatedLyricsStart": float(lyrics[0]["time"]) if lyrics else chords_data.get("estimated_lyrics_start", 0.0),
                "audioPath": audio_path,
                "filename": os.path.basename(audio_path)
            },
            "error": None
        }
    except Exception as e:
        extraction_tasks[task_id] = {
            "status": "failed",
            "progress": 1.0,
            "message": f"Error: {str(e)}",
            "result": None,
            "error": str(e)
        }

@app.post("/api/extract-chords")
async def start_extraction(request: Request, background_tasks: BackgroundTasks):
    body = await request.json()
    audio_path = body.get("path")
    if not audio_path or not os.path.exists(audio_path):
        raise HTTPException(status_code=400, detail="Invalid audio file path")
        
    task_id = str(hash(audio_path))
    extraction_tasks[task_id] = {
        "status": "starting",
        "progress": 0.0,
        "message": "Initializing...",
        "result": None,
        "error": None
    }
    
    background_tasks.add_task(run_extraction_background, task_id, audio_path)
    return {"task_id": task_id}

def run_youtube_extraction_background(task_id: str, video_id: str):
    def progress_cb(message, val):
        extraction_tasks[task_id] = {
            "status": "processing",
            "progress": val,
            "message": message,
            "result": None,
            "error": None
        }
        
    try:
        downloads_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "downloads")
        
        # 1. Fetch transcript captions from YouTube first (fail-safe or check availability)
        progress_cb("Retrieving YouTube captions...", 0.10)
        lyrics = fetch_youtube_transcript(video_id)
        
        # 2. Download and convert YouTube audio to MP3
        progress_cb("Downloading YouTube audio stream...", 0.20)
        mp3_path = download_youtube_audio(video_id, downloads_dir)
        
        # Resolve friendly metadata title
        progress_cb("Reading video metadata...", 0.40)
        friendly_title = f"YouTube Song ({video_id})"
        try:
            with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
                info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)
                friendly_title = info.get('title', friendly_title)
        except Exception:
            pass
            
        # 3. Extract chords from the downloaded track
        progress_cb("Extracting chords...", 0.50)
        # Scale progress callback between 50% and 90%
        chords_data = extract_chords_from_audio(
            mp3_path, 
            lambda msg, progress: progress_cb(msg, 0.50 + progress * 0.40)
        )
        chords = chords_data["chords"]
        bpm = chords_data["bpm"]
        bars = chords_data["bars"]
        
        # 4. Fallback: If YouTube captions were not available, search online databases
        auto_synced = False
        chordsheet = ""
        timestamps = []
        unsynced_lyrics = ""
        
        if not lyrics:
            progress_cb("Captions unavailable. Querying synced lyrics databases...", 0.92)
            query = friendly_title
            query = re.sub(r'[\(\[\{].*?[\)\]\}]', '', query)
            query = re.sub(r'[-_\.]', ' ', query)
            query = " ".join(query.split())
            lyrics = search_and_parse_lyrics(query)
            
        # 5. Build aligned chordsheet
        if lyrics:
            progress_cb("Auto-aligning chords with synced lyrics...", 0.96)
            duration = float(librosa.get_duration(path=mp3_path))
            aligned = generate_aligned_sheet_internal(chords, lyrics, duration)
            chordsheet = aligned["chordsheet"]
            timestamps = aligned["timestamps"]
            auto_synced = True
        else:
            progress_cb("Searching web for unsynced lyrics...", 0.94)
            unsynced_lyrics = search_unsynced_lyrics(query) or ""
            
        extraction_tasks[task_id] = {
            "status": "completed",
            "progress": 1.0,
            "message": "YouTube extraction completed successfully!",
            "result": {
                "chords": chords,
                "bpm": bpm,
                "bars": bars,
                "lyrics": lyrics,
                "chordsheet": chordsheet,
                "timestamps": timestamps,
                "auto_synced": auto_synced,
                "unsyncedLyrics": unsynced_lyrics,
                "estimatedLyricsStart": float(lyrics[0]["time"]) if lyrics else chords_data.get("estimated_lyrics_start", 0.0),
                "audioPath": mp3_path,
                "filename": friendly_title + ".mp3"
            },
            "error": None
        }
        
    except Exception as e:
        extraction_tasks[task_id] = {
            "status": "failed",
            "progress": 1.0,
            "message": f"Error: {str(e)}",
            "result": None,
            "error": str(e)
        }

@app.post("/api/extract-youtube")
async def extract_youtube(request: YoutubeRequest, background_tasks: BackgroundTasks):
    url = request.url
    video_id = extract_video_id(url)
    if not video_id:
        raise HTTPException(status_code=400, detail="Invalid YouTube URL path")
        
    task_id = str(hash(video_id))
    extraction_tasks[task_id] = {
        "status": "starting",
        "progress": 0.0,
        "message": "Initializing YouTube task...",
        "result": None,
        "error": None
    }
    
    background_tasks.add_task(run_youtube_extraction_background, task_id, video_id)
    return {"task_id": task_id}

@app.get("/api/extract-chords/status/{task_id}")
def get_extraction_status(task_id: str):
    task = extraction_tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Extraction task not found")
    return task

@app.post("/api/generate-chordsheet")
def generate_chordsheet(data: GenerateSheetRequest):
    lyrics_list = [{"text": l.text, "time": l.time, "duration": l.duration} for l in data.lyrics]
    return generate_aligned_sheet_internal(data.chords, lyrics_list, data.duration)

@app.post("/api/save-project")
async def save_project(data: SaveProjectRequest):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(run_save_dialog)
        save_path = future.result()
        
    if not save_path:
        return {"status": "cancelled", "path": ""}
        
    project_data = {
        "audioPath": data.audioPath,
        "chordsheetText": data.chordsheetText,
        "timestamps": data.timestamps,
        "bpm": data.bpm,
        "bars": data.bars
    }
    
    try:
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(project_data, f, indent=4)
        return {"status": "saved", "path": save_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save project: {str(e)}")

@app.get("/api/load-project")
async def load_project():
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(run_open_project_dialog)
        open_path = future.result()
        
    if not open_path:
        return {"status": "cancelled"}
        
    try:
        with open(open_path, "r", encoding="utf-8") as f:
            project_data = json.load(f)
        return {
            "status": "loaded",
            "path": open_path,
            "data": project_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load project: {str(e)}")

def get_static_dir():
    if getattr(sys, 'frozen', False):
        base_path = sys._MEIPASS
        return os.path.join(base_path, "static")
    else:
        return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend", "dist", "frontend", "browser"))

def open_browser():
    import webbrowser
    import time
    time.sleep(1.5)
    webbrowser.open("http://127.0.0.1:8000")

@app.on_event("startup")
def on_startup():
    import threading
    threading.Thread(target=open_browser, daemon=True).start()

@app.get("/{catchall:path}")
def serve_spa(request: Request, catchall: str):
    if catchall.startswith("api/"):
        raise HTTPException(status_code=404, detail="API endpoint not found")
        
    static_dir = get_static_dir()
    clean_path = os.path.normpath(catchall).lstrip(os.path.sep)
    if clean_path == "." or clean_path == "":
        file_path = os.path.join(static_dir, "index.html")
    else:
        file_path = os.path.join(static_dir, clean_path)
        
    if os.path.exists(file_path) and os.path.isfile(file_path):
        return FileResponse(file_path)
        
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
        
    raise HTTPException(status_code=404, detail="Not Found")

if __name__ == "__main__":
    import uvicorn
    is_frozen = getattr(sys, 'frozen', False)
    if is_frozen:
        uvicorn.run(app, host="127.0.0.1", port=8000)
    else:
        uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
