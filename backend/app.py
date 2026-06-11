import os
import re
import sys
import time
import multiprocessing as mp
from difflib import SequenceMatcher
import tkinter as tk
from tkinter import filedialog
import concurrent.futures
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Callable, List, Optional
import urllib.parse
import librosa
import syncedlyrics
import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi

from chord_extractor import extract_chords_from_audio

# When packaged as a windowed (--noconsole) exe there is no console, so
# sys.stdout/sys.stderr are None. Redirect them to a null sink so the app's
# print() calls and uvicorn's logging handlers don't crash.
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w", encoding="utf-8")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w", encoding="utf-8")

# Configure UTF-8 encoding for standard output and error to prevent crashes in Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')


app = FastAPI(title="Chord & Lyrics Extractor API", version="1.0.4")

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

extraction_tasks = {}
HEBREW_CHAR_PATTERN = re.compile(r"[\u0590-\u05FF]")
HEBREW_NIQQUD_PATTERN = re.compile(r"[\u0591-\u05C7]")
TEXT_NORMALIZE_PATTERN = re.compile(r"[^\w\u0590-\u05FF\s]")


def _make_progress_cb(task_id: str):
    """Build a progress callback that records processing status for a task."""
    def progress_cb(message, val):
        extraction_tasks[task_id] = {
            "status": "processing",
            "progress": val,
            "message": message,
            "result": None,
            "error": None,
        }
    return progress_cb


def _complete_task(task_id: str, message: str, result: dict):
    """Mark an extraction task as successfully completed."""
    extraction_tasks[task_id] = {
        "status": "completed",
        "progress": 1.0,
        "message": message,
        "result": result,
        "error": None,
    }


def _fail_task(task_id: str, error: Exception):
    """Mark an extraction task as failed."""
    extraction_tasks[task_id] = {
        "status": "failed",
        "progress": 1.0,
        "message": f"Error: {str(error)}",
        "result": None,
        "error": str(error),
    }


class LyricsLine(BaseModel):
    text: str
    time: float
    duration: Optional[float] = None

class GenerateSheetRequest(BaseModel):
    chords: List[dict]
    lyrics: List[LyricsLine]
    duration: float
    bars: Optional[List[dict]] = None

class YoutubeRequest(BaseModel):
    url: str


class AlignLyricsRequest(BaseModel):
    lyricsText: str
    referenceLyrics: List[LyricsLine]
    selectedStartTime: float = 0.0

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


def contains_hebrew(text: str) -> bool:
    return bool(HEBREW_CHAR_PATTERN.search(text or ""))


def resolve_transcription_language(query_hint: str = "") -> Optional[str]:
    """Resolve Whisper language preference, allowing env override.

    - WHISPER_LANGUAGE=auto|none -> automatic language detection
    - WHISPER_LANGUAGE=<code>    -> force a language (e.g. he, en)
    - default                    -> infer Hebrew when the query hint is Hebrew
    """
    language_override = os.getenv("WHISPER_LANGUAGE", "").strip()
    if language_override:
        if language_override.lower() in {"auto", "none"}:
            return None
        return language_override
    return "he" if contains_hebrew(query_hint) else None


def clean_transcribed_line(text: str) -> str:
    cleaned = (text or "").strip()
    cleaned = re.sub(r'[\(\[\{].*?[\)\]\}]', '', cleaned).strip()
    cleaned = cleaned.replace('♪', '').strip()
    cleaned = re.sub(r'\s+', ' ', cleaned)
    # Keep only lines that still contain meaningful characters.
    if not re.search(r'[\w\u0590-\u05FF]', cleaned):
        return ""
    return cleaned


def read_int_env(name: str, default: int, min_value: int = 1) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
        return value if value >= min_value else min_value
    except ValueError:
        print(f"Invalid integer for {name}: {raw!r}. Using default {default}.")
        return default


def read_float_env(name: str, default: float, min_value: float = 0.0) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
        return value if value >= min_value else min_value
    except ValueError:
        print(f"Invalid float for {name}: {raw!r}. Using default {default}.")
        return default


def read_bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    print(f"Invalid boolean for {name}: {raw!r}. Using default {default}.")
    return default


def has_hebrew_content(lines: List[dict]) -> bool:
    for item in lines:
        if contains_hebrew(item.get("text", "")):
            return True
    return False


def resolve_estimated_lyrics_start(chords_data: dict, lyrics: Optional[List[dict]]) -> float:
    """
    Prefer vocal-onset estimation from chord extraction when available.
    This avoids late ASR/transcript starts from shifting the default lyric-start bar.
    """
    onset = float(chords_data.get("estimated_lyrics_start", 0.0) or 0.0)
    if onset > 0.0:
        return onset
    if lyrics:
        first = lyrics[0] or {}
        return float(first.get("time", 0.0) or 0.0)
    return 0.0


def normalize_lyric_text_for_match(text: str) -> str:
    normalized = (text or "").strip().lower()
    normalized = HEBREW_NIQQUD_PATTERN.sub("", normalized)
    normalized = TEXT_NORMALIZE_PATTERN.sub(" ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def lyric_similarity_score(a: str, b: str) -> float:
    norm_a = normalize_lyric_text_for_match(a)
    norm_b = normalize_lyric_text_for_match(b)
    if not norm_a or not norm_b:
        return 0.0

    tokens_a = set(norm_a.split())
    tokens_b = set(norm_b.split())
    token_score = 0.0
    if tokens_a and tokens_b:
        token_score = (2.0 * len(tokens_a & tokens_b)) / (len(tokens_a) + len(tokens_b))

    char_score = SequenceMatcher(None, norm_a, norm_b).ratio()
    return 0.65 * token_score + 0.35 * char_score


def split_plain_lyrics_lines(lyrics_text: str, hebrew_context: bool = False) -> List[str]:
    lines: List[str] = []
    banned_markers = (
        "contributors",
        "contributor",
        "lyrics",
        "translation",
        "embed",
        "you might also like",
        "produced by",
        "written by",
        "release date",
        "see ",
    )
    hebrew_banned_markers = (
        "המשורר",
        "הביצוע המקורי",
        "מזכיר לשומעים",
        "ביקורת והמחאות",
        "להקת",
    )

    for raw_line in (lyrics_text or "").splitlines():
        cleaned = clean_transcribed_line(raw_line)
        if not cleaned:
            continue
        lowered = cleaned.lower()
        if lowered.startswith(("lyrics", "www.", "http://", "https://")):
            continue
        if any(marker in lowered for marker in banned_markers):
            continue
        if re.fullmatch(r"\d+\s+contributors?", lowered):
            continue

        if hebrew_context:
            heb_count = len(re.findall(r"[\u0590-\u05FF]", cleaned))
            lat_count = len(re.findall(r"[A-Za-z]", cleaned))
            if heb_count < 2:
                continue
            if lat_count > max(2, heb_count // 2):
                continue
            if len(cleaned) > 70:
                continue
            if any(marker in cleaned for marker in hebrew_banned_markers):
                continue
            if re.search(r"\b(19|20)\d{2}\b", cleaned):
                continue

        lines.append(cleaned)
    return lines


def align_unsynced_lyrics_to_reference_timing(
    plain_lyrics_text: str,
    reference_timed_lines: List[dict],
    min_similarity: float = 0.24,
    hebrew_context: bool = False,
) -> Optional[List[dict]]:
    candidate_lines = split_plain_lyrics_lines(plain_lyrics_text, hebrew_context=hebrew_context)
    references = [
        line for line in reference_timed_lines
        if clean_transcribed_line(line.get("text", ""))
    ]

    if len(candidate_lines) < 2 or len(references) < 2:
        return None

    ref_texts = [line.get("text", "") for line in references]
    n = len(candidate_lines)
    m = len(ref_texts)

    matches = {}
    matched_scores: List[float] = []
    last_ref_idx = -1

    for i, candidate in enumerate(candidate_lines):
        expected = int(round((i / max(1, n - 1)) * (m - 1)))
        best_ref_idx = None
        best_rank = float("-inf")
        best_score = 0.0

        for j in range(last_ref_idx + 1, m):
            sim = lyric_similarity_score(candidate, ref_texts[j])
            rank = sim - (abs(j - expected) / max(8.0, float(m)))
            if rank > best_rank:
                best_rank = rank
                best_score = sim
                best_ref_idx = j

        if best_ref_idx is not None and best_score >= min_similarity:
            matches[i] = best_ref_idx
            matched_scores.append(best_score)
            last_ref_idx = best_ref_idx

    min_required_matches = max(2, n // 5)
    if len(matches) < min_required_matches:
        return None

    avg_similarity = sum(matched_scores) / max(1, len(matched_scores))
    if avg_similarity < min_similarity:
        return None

    prev_match = [None] * n
    next_match = [None] * n

    prev_idx = None
    for i in range(n):
        if i in matches:
            prev_idx = i
        prev_match[i] = prev_idx

    next_idx = None
    for i in range(n - 1, -1, -1):
        if i in matches:
            next_idx = i
        next_match[i] = next_idx

    refined_lines = []
    last_time = 0.0

    for i, text in enumerate(candidate_lines):
        duration = 0.0

        if i in matches:
            ref_line = references[matches[i]]
            time_val = float(ref_line.get("time", 0.0))
            duration = float(ref_line.get("duration") or 0.0)
        else:
            prev_i = prev_match[i]
            next_i = next_match[i]

            if prev_i is not None and next_i is not None and prev_i != next_i:
                prev_ref = references[matches[prev_i]]
                next_ref = references[matches[next_i]]
                prev_t = float(prev_ref.get("time", 0.0))
                next_t = float(next_ref.get("time", prev_t))
                frac = (i - prev_i) / float(next_i - prev_i)
                time_val = prev_t + max(0.0, next_t - prev_t) * frac
                duration = max(0.0, (max(0.0, next_t - prev_t) / max(1, next_i - prev_i)) * 0.85)
            elif prev_i is not None:
                prev_ref = references[matches[prev_i]]
                step = max(1.6, float(prev_ref.get("duration") or 0.0))
                time_val = float(prev_ref.get("time", 0.0)) + step * (i - prev_i)
                duration = step
            elif next_i is not None:
                next_ref = references[matches[next_i]]
                step = max(1.6, float(next_ref.get("duration") or 0.0))
                time_val = max(0.0, float(next_ref.get("time", 0.0)) - step * (next_i - i))
                duration = step
            else:
                time_val = float(i) * 2.5
                duration = 2.0

        if refined_lines and time_val <= last_time:
            time_val = last_time + 0.12
        last_time = time_val

        refined_lines.append({
            "text": text,
            "time": time_val,
            "duration": duration,
        })

    print(
        "Aligned web lyrics to ASR timing "
        f"(lines={len(refined_lines)}, matches={len(matches)}, avg_similarity={avg_similarity:.3f})."
    )
    return refined_lines


def project_plain_lyrics_to_reference_timing(
    plain_lyrics_text: str,
    reference_timed_lines: List[dict],
    hebrew_context: bool = False,
) -> Optional[List[dict]]:
    """Project plain lyric text onto the ASR timing curve when text matching fails."""
    candidates = split_plain_lyrics_lines(plain_lyrics_text, hebrew_context=hebrew_context)
    references = [
        line for line in reference_timed_lines
        if clean_transcribed_line(line.get("text", ""))
    ]

    if len(candidates) < 2 or len(references) < 2:
        return None

    ref_times = [float(line.get("time", 0.0)) for line in references]
    ref_times = sorted(ref_times)
    n = len(candidates)
    m = len(ref_times)

    projected = []
    last_time = max(0.0, ref_times[0])
    avg_ref_step = (ref_times[-1] - ref_times[0]) / max(1, m - 1)
    min_step = max(0.12, avg_ref_step * 0.18)

    for i, text in enumerate(candidates):
        if n == 1:
            pos = 0.0
        else:
            pos = (i / float(n - 1)) * (m - 1)

        left = int(pos)
        right = min(m - 1, left + 1)
        alpha = pos - left
        projected_time = ref_times[left] * (1.0 - alpha) + ref_times[right] * alpha
        projected_time = max(0.0, projected_time)

        if projected and projected_time <= last_time:
            projected_time = last_time + min_step

        projected.append({
            "text": text,
            "time": projected_time,
            "duration": 0.0,
        })
        last_time = projected_time

    for i in range(len(projected)):
        if i + 1 < len(projected):
            step = max(min_step, projected[i + 1]["time"] - projected[i]["time"])
        else:
            step = max(min_step, avg_ref_step)
        projected[i]["duration"] = max(0.8, step * 0.88)

    print(
        "Projected web lyric text onto ASR timing "
        f"(web_lines={len(projected)}, ref_lines={len(references)})."
    )
    return projected


def maybe_refine_asr_lyrics_with_web(query: str, timed_lyrics: Optional[List[dict]]) -> tuple[Optional[List[dict]], str]:
    if not timed_lyrics:
        return timed_lyrics, ""

    unsynced_text = search_unsynced_lyrics(query) or ""
    if not unsynced_text:
        return timed_lyrics, ""

    hebrew_context = contains_hebrew(query) or has_hebrew_content(timed_lyrics)
    min_similarity = 0.20 if hebrew_context else 0.26
    refined = align_unsynced_lyrics_to_reference_timing(
        plain_lyrics_text=unsynced_text,
        reference_timed_lines=timed_lyrics,
        min_similarity=min_similarity,
        hebrew_context=hebrew_context,
    )
    if refined:
        return refined, unsynced_text

    use_projection_fallback = read_bool_env(
        "WHISPER_USE_WEB_TEXT_FALLBACK",
        contains_hebrew(query),
    )
    if use_projection_fallback:
        projected = project_plain_lyrics_to_reference_timing(
            plain_lyrics_text=unsynced_text,
            reference_timed_lines=timed_lyrics,
            hebrew_context=hebrew_context,
        )
        if projected:
            return projected, unsynced_text

    return timed_lyrics, unsynced_text


def split_whisper_segment_to_lines(
    seg,
    max_line_chars: int,
    max_line_seconds: float,
) -> List[dict]:
    words = getattr(seg, "words", None) or []
    if not words:
        return []

    lines: List[dict] = []
    current_words: List[str] = []
    current_start: Optional[float] = None
    current_len = 0
    last_end = max(0.0, float(getattr(seg, "start", 0.0) or 0.0))

    def flush(end_time: float) -> None:
        nonlocal current_words, current_start, current_len
        if not current_words or current_start is None:
            current_words = []
            current_start = None
            current_len = 0
            return

        text = clean_transcribed_line(" ".join(current_words))
        if text:
            lines.append({
                "text": text,
                "time": max(0.0, current_start),
                "duration": max(0.0, float(end_time) - float(current_start)),
            })

        current_words = []
        current_start = None
        current_len = 0

    for word in words:
        token = (getattr(word, "word", "") or "").strip()
        if not token:
            continue

        w_start = getattr(word, "start", None)
        w_end = getattr(word, "end", None)
        if w_start is None:
            w_start = last_end
        if w_end is None:
            w_end = max(w_start, last_end)
        w_start = float(w_start)
        w_end = float(w_end)

        if current_start is None:
            current_start = w_start
            current_words = [token]
            current_len = len(token)
            last_end = w_end
            if re.search(r"[\.!?;:\u05C3]$", token):
                flush(last_end)
            continue

        predicted_len = current_len + 1 + len(token)
        elapsed = max(0.0, w_end - current_start)
        gap = max(0.0, w_start - last_end)

        if gap >= 0.75 or predicted_len > max_line_chars or elapsed > max_line_seconds:
            flush(last_end)
            current_start = w_start
            current_words = [token]
            current_len = len(token)
        else:
            current_words.append(token)
            current_len = predicted_len

        last_end = w_end
        if re.search(r"[\.!?;:\u05C3]$", token):
            flush(last_end)

    flush(last_end)
    return lines


def whisper_transcribe_worker(
    audio_path: str,
    model_size: str,
    device: str,
    compute_type: str,
    language: Optional[str],
    beam_size: int,
    best_of: int,
    word_timestamps_enabled: bool,
    max_line_chars: int,
    max_line_seconds: float,
    vad_filter: bool,
    condition_on_previous_text: bool,
    initial_prompt: Optional[str],
    result_queue,
) -> None:
    """Run Whisper transcription in a child process so it can be timed out safely."""
    try:
        from faster_whisper import WhisperModel

        model = WhisperModel(model_size, device=device, compute_type=compute_type)
        segments, info = model.transcribe(
            audio_path,
            task="transcribe",
            language=language,
            beam_size=beam_size,
            best_of=best_of,
            temperature=0.0,
            vad_filter=vad_filter,
            condition_on_previous_text=condition_on_previous_text,
            word_timestamps=word_timestamps_enabled,
            initial_prompt=initial_prompt,
        )

        transcript_lines = []
        for seg in segments:
            if word_timestamps_enabled:
                split_lines = split_whisper_segment_to_lines(
                    seg=seg,
                    max_line_chars=max_line_chars,
                    max_line_seconds=max_line_seconds,
                )
                if split_lines:
                    transcript_lines.extend(split_lines)
                    continue

            line = clean_transcribed_line(seg.text)
            if not line:
                continue

            start = max(0.0, float(seg.start))
            end = max(start, float(seg.end))
            transcript_lines.append({
                "text": line,
                "time": start,
                "duration": max(0.0, end - start),
            })

        result_queue.put({
            "ok": True,
            "lines": transcript_lines,
            "detected_lang": getattr(info, "language", None),
        })
    except Exception as e:
        result_queue.put({"ok": False, "error": str(e)})


def run_whisper_attempt(
    audio_path: str,
    model_size: str,
    device: str,
    compute_type: str,
    language: Optional[str],
    beam_size: int,
    best_of: int,
    word_timestamps_enabled: bool,
    max_line_chars: int,
    max_line_seconds: float,
    vad_filter: bool,
    condition_on_previous_text: bool,
    initial_prompt: Optional[str],
    timeout_seconds: int,
    progress_cb: Optional[Callable[[str, float], None]],
    progress_start: float,
    progress_end: float,
    progress_message: str,
) -> dict:
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue(maxsize=1)
    worker = ctx.Process(
        target=whisper_transcribe_worker,
        args=(
            audio_path,
            model_size,
            device,
            compute_type,
            language,
            beam_size,
            best_of,
            word_timestamps_enabled,
            max_line_chars,
            max_line_seconds,
            vad_filter,
            condition_on_previous_text,
            initial_prompt,
            result_queue,
        ),
        daemon=True,
    )
    worker.start()

    started_at = time.time()
    while worker.is_alive():
        elapsed = time.time() - started_at
        if elapsed >= timeout_seconds:
            break

        if progress_cb:
            frac = min(0.92, elapsed / float(timeout_seconds))
            scaled = progress_start + (progress_end - progress_start) * frac
            progress_cb(progress_message, scaled)

        worker.join(timeout=1.0)

    if worker.is_alive():
        worker.terminate()
        worker.join(timeout=2.0)
        try:
            result_queue.close()
        except Exception:
            pass
        return {
            "ok": False,
            "timed_out": True,
            "error": f"timeout after {timeout_seconds}s",
            "lines": [],
            "detected_lang": None,
        }

    payload = None
    try:
        payload = result_queue.get(timeout=1.0)
    except Exception:
        payload = None
    finally:
        try:
            result_queue.close()
        except Exception:
            pass

    if not payload:
        return {
            "ok": False,
            "timed_out": False,
            "error": "worker finished without result",
            "lines": [],
            "detected_lang": None,
        }

    if not payload.get("ok"):
        return {
            "ok": False,
            "timed_out": False,
            "error": payload.get("error", "unknown error"),
            "lines": [],
            "detected_lang": None,
        }

    return {
        "ok": True,
        "timed_out": False,
        "error": None,
        "lines": payload.get("lines") or [],
        "detected_lang": payload.get("detected_lang"),
    }


def transcribe_audio_with_whisper(
    audio_path: str,
    query_hint: str = "",
    progress_cb: Optional[Callable[[str, float], None]] = None,
    progress_start: float = 0.93,
    progress_end: float = 0.97,
) -> Optional[List[dict]]:
    """Transcribe audio into timed lyric lines using faster-whisper.

    This is a true ASR fallback used when synced captions/lyrics are unavailable.
    """
    try:
        from faster_whisper import WhisperModel  # noqa: F401
    except Exception as e:
        print(f"Whisper fallback unavailable (faster-whisper import failed): {e}")
        return None

    language = resolve_transcription_language(query_hint)
    is_hebrew_target = language == "he" or contains_hebrew(query_hint)

    preset = os.getenv("WHISPER_PRESET", "high").strip().lower()
    if preset not in {"high", "balanced", "fast"}:
        print(f"Invalid WHISPER_PRESET: {preset!r}. Using 'high'.")
        preset = "high"

    if preset == "high":
        heb_model_default = "medium"
        heb_beam_default = 5
        heb_best_of_default = 5
        heb_timeout_default = 420
        heb_retry_timeout_default = 420
    elif preset == "balanced":
        heb_model_default = "small"
        heb_beam_default = 3
        heb_best_of_default = 3
        heb_timeout_default = 240
        heb_retry_timeout_default = 240
    else:
        heb_model_default = "tiny"
        heb_beam_default = 1
        heb_best_of_default = 1
        heb_timeout_default = 120
        heb_retry_timeout_default = 120

    model_override = os.getenv("WHISPER_MODEL", "").strip()
    model_size = model_override or (heb_model_default if is_hebrew_target else "tiny")
    retry_model = os.getenv("WHISPER_RETRY_MODEL", heb_model_default if is_hebrew_target else model_size).strip() or model_size

    compute_type = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
    device = os.getenv("WHISPER_DEVICE", "cpu")
    beam_default = heb_beam_default if is_hebrew_target else 1
    best_of_default = heb_best_of_default if is_hebrew_target else 1
    beam_size = read_int_env("WHISPER_BEAM_SIZE", beam_default)
    best_of = read_int_env("WHISPER_BEST_OF", best_of_default)
    timeout_seconds = read_int_env("WHISPER_TIMEOUT_SECONDS", heb_timeout_default if is_hebrew_target else 180, min_value=30)
    retry_timeout_seconds = read_int_env(
        "WHISPER_RETRY_TIMEOUT_SECONDS",
        heb_retry_timeout_default if is_hebrew_target else 180,
        min_value=30,
    )
    enable_retry = read_bool_env("WHISPER_ENABLE_RETRY", True)
    retry_without_vad = read_bool_env("WHISPER_RETRY_DISABLE_VAD", True)
    word_timestamps_enabled = read_bool_env("WHISPER_WORD_SPLIT", is_hebrew_target)
    max_line_chars = read_int_env("WHISPER_MAX_LINE_CHARS", 28 if is_hebrew_target else 36, min_value=10)
    max_line_seconds = read_float_env("WHISPER_MAX_LINE_SECONDS", 4.2 if is_hebrew_target else 6.0, min_value=1.5)

    initial_prompt_override = os.getenv("WHISPER_INITIAL_PROMPT", "").strip()
    initial_prompt = initial_prompt_override or (
        "These are Hebrew song lyrics. Prioritize accurate Hebrew words and natural lyric phrasing."
        if is_hebrew_target else None
    )

    if progress_cb:
        progress_cb("Loading Whisper model (first run can take a few minutes)...", progress_start)

    print(
        "Running Whisper transcription "
        f"(preset={preset}, model={model_size}, language={language or 'auto'}, device={device}, "
        f"compute={compute_type}, beam={beam_size}, best_of={best_of}, timeout={timeout_seconds}s, "
        f"word_split={word_timestamps_enabled}, max_chars={max_line_chars}, max_seconds={max_line_seconds})"
    )

    try:
        primary = run_whisper_attempt(
            audio_path=audio_path,
            model_size=model_size,
            device=device,
            compute_type=compute_type,
            language=language,
            beam_size=beam_size,
            best_of=best_of,
            word_timestamps_enabled=word_timestamps_enabled,
            max_line_chars=max_line_chars,
            max_line_seconds=max_line_seconds,
            vad_filter=True,
            condition_on_previous_text=False,
            initial_prompt=initial_prompt,
            timeout_seconds=timeout_seconds,
            progress_cb=progress_cb,
            progress_start=progress_start,
            progress_end=progress_end,
            progress_message="Transcribing lyrics from audio (Whisper)...",
        )

        transcript_lines = primary["lines"]
        detected_lang = primary["detected_lang"]
        retry_reason = None

        if not primary["ok"]:
            retry_reason = f"primary attempt failed ({primary['error']})"
        elif not transcript_lines:
            retry_reason = "primary attempt produced no lines"
        elif is_hebrew_target and not has_hebrew_content(transcript_lines):
            retry_reason = "primary attempt did not produce Hebrew text"

        should_retry = enable_retry and is_hebrew_target and retry_reason is not None
        if should_retry:
            print(f"Retrying Whisper for Hebrew quality: {retry_reason}.")
            retry_beam = max(3, beam_size)
            retry_best_of = max(3, best_of)
            retry_device = device
            primary_error_text = (primary.get("error") or "").lower()
            if (
                retry_device != "cpu"
                and any(token in primary_error_text for token in ["cublas", "cudnn", "cuda", "libcudart"])
            ):
                retry_device = "cpu"
                print("CUDA runtime not available. Retrying Whisper on CPU.")

            retry = run_whisper_attempt(
                audio_path=audio_path,
                model_size=retry_model,
                device=retry_device,
                compute_type=compute_type,
                language="he",
                beam_size=retry_beam,
                best_of=retry_best_of,
                word_timestamps_enabled=True,
                max_line_chars=max_line_chars,
                max_line_seconds=max_line_seconds,
                vad_filter=not retry_without_vad,
                condition_on_previous_text=True,
                initial_prompt=initial_prompt,
                timeout_seconds=retry_timeout_seconds,
                progress_cb=progress_cb,
                progress_start=progress_start,
                progress_end=progress_end,
                progress_message="Retrying Hebrew transcription with enhanced settings...",
            )
            if retry["ok"] and retry["lines"]:
                transcript_lines = retry["lines"]
                detected_lang = retry["detected_lang"]
                print(
                    "Hebrew retry succeeded "
                    f"(model={retry_model}, beam={retry_beam}, best_of={retry_best_of}, "
                    f"device={retry_device}, vad_filter={not retry_without_vad})."
                )
            else:
                print(f"Hebrew retry failed: {retry['error']}")

        if progress_cb:
            progress_cb("Transcription pass complete.", progress_end)

        if not transcript_lines:
            print("Whisper transcription returned no non-empty lyric lines.")
            if progress_cb:
                progress_cb("Whisper could not produce lyrics; falling back to web lyrics...", progress_end)
            return None

        print(
            f"Whisper produced {len(transcript_lines)} timed lyric lines. "
            f"Detected language: {detected_lang or 'unknown'}."
        )
        return transcript_lines
    except Exception as e:
        print(f"Whisper transcription failed: {e}")
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

def get_ffmpeg_location() -> Optional[str]:
    """Return the directory holding ffmpeg/ffprobe, or None to rely on PATH.

    yt-dlp's audio post-processing needs FFmpeg. To keep the standalone exe
    self-contained, the binaries are bundled and shipped with the app:
      * Frozen exe: PyInstaller unpacks them under <_MEIPASS>/ffmpeg.
      * Dev runs:   a project-local ./ffmpeg folder (created by build_exe.py).
    If neither exists we return None so yt-dlp falls back to a system FFmpeg.
    """
    candidates = []
    if getattr(sys, 'frozen', False):
        candidates.append(os.path.join(sys._MEIPASS, "ffmpeg"))
    candidates.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ffmpeg")))
    for path in candidates:
        if os.path.isfile(os.path.join(path, "ffmpeg.exe")) or os.path.isfile(os.path.join(path, "ffmpeg")):
            return path
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

    ffmpeg_dir = get_ffmpeg_location()
    if ffmpeg_dir:
        ydl_opts['ffmpeg_location'] = ffmpeg_dir
    
    url = f"https://www.youtube.com/watch?v={video_id}"
    print(f"Downloading YouTube audio for: {video_id}...")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
        
    return output_mp3

# Core chord alignment algorithm
# If the first chord change inside a line happens within this fraction of the
# line's sung duration, the singer is treated as entering on that chord, so the
# previously-sounding (leftover instrumental) chord is NOT prepended.
SEED_HOLD_FRACTION = 0.4

# Helper to snap character index to word boundaries for cleaner chord sheets
def snap_to_word_boundaries(text: str, char_idx: int) -> int:
    if char_idx <= 0 or char_idx >= len(text):
        return char_idx
    # If the character itself or the preceding character is whitespace,
    # it is already at a word boundary (start or end).
    if text[char_idx].isspace() or text[char_idx - 1].isspace():
        return char_idx

    # Find the start of the current word
    start_word = char_idx
    while start_word > 0 and not text[start_word - 1].isspace():
        start_word -= 1

    # Find the end of the current word
    end_word = char_idx
    while end_word < len(text) and not text[end_word].isspace():
        end_word += 1

    # Snap to the start of the word if we are closer to the start,
    # or if we land within the first 2 characters of the word.
    if (char_idx - start_word) < (end_word - char_idx) or (char_idx - start_word) <= 2:
        return start_word
    return char_idx


def _word_alignment_anchors(text: str) -> tuple[List[int], List[float]]:
    """
    Return per-word start columns plus normalized start ratios across the line.
    Ratios are weighted by token core length so chord times map to words, not
    raw character columns.
    """
    starts: List[int] = []
    weights: List[float] = []

    for match in re.finditer(r"\S+", text or ""):
        token = match.group(0)
        starts.append(match.start())

        # Ignore surrounding punctuation when estimating sung token weight.
        core = re.sub(r"^[^\w\u0590-\u05FF]+|[^\w\u0590-\u05FF]+$", "", token)
        core_chars = len(re.findall(r"[\w\u0590-\u05FF]", core or token))
        weights.append(float(max(1, core_chars)))

    if not starts:
        return [], []

    total_weight = sum(weights)
    if total_weight <= 0:
        ratios = [0.0 for _ in starts]
        return starts, ratios

    ratios: List[float] = []
    acc = 0.0
    for w in weights:
        ratios.append(acc / total_weight)
        acc += w
    return starts, ratios


def _map_ratio_to_word_start(ratio: float, word_starts: List[int], word_ratios: List[float]) -> Optional[int]:
    if not word_starts:
        return None
    if len(word_starts) == 1:
        return word_starts[0]

    best_idx = 0
    best_dist = abs(ratio - word_ratios[0])
    for idx in range(1, len(word_starts)):
        dist = abs(ratio - word_ratios[idx])
        if dist < best_dist:
            best_idx = idx
            best_dist = dist
    return word_starts[best_idx]


def _compact_chord_sequence(chords: List[str]) -> List[str]:
    compact: List[str] = []
    for chord in chords or []:
        token = (chord or "").strip()
        if not token:
            continue
        if not compact or compact[-1] != token:
            compact.append(token)
    return compact


def _bar_label_for_intro(bar: dict) -> str:
    if isinstance(bar, dict):
        raw_chords = bar.get("chords") or []
    else:
        raw_chords = getattr(bar, "chords", []) or []

    compact = _compact_chord_sequence(raw_chords)
    if not compact:
        return "-"
    if len(compact) == 1:
        return compact[0]
    return " ".join(compact)

def generate_aligned_sheet_internal(chords: list, lyrics: list, duration: float, bars: Optional[List[dict]] = None):
    if not lyrics:
        return {"chordsheet": "", "timestamps": []}
        
    output_lines = []
    timestamps = []
    
    sorted_lyrics = sorted(lyrics, key=lambda l: l["time"] if isinstance(l, dict) else l.time)
    
    # 1. Check for Intro Gap before first lyric line starts
    first_line_start = sorted_lyrics[0]["time"] if isinstance(sorted_lyrics[0], dict) else sorted_lyrics[0].time
    if first_line_start > 3.0:
        intro_rendered = False

        # Prefer bar-based rendering when bar grid exists so intro count matches
        # the reviewed bar timeline (one token per bar).
        if bars:
            intro_bar_labels: List[str] = []
            for bar in bars:
                if isinstance(bar, dict):
                    bar_time = float(bar.get("time", 0.0) or 0.0)
                else:
                    bar_time = float(getattr(bar, "time", 0.0) or 0.0)
                if 0.0 <= bar_time < first_line_start:
                    intro_bar_labels.append(_bar_label_for_intro(bar))

            if intro_bar_labels:
                instr_line = "// " + " | ".join(intro_bar_labels) + " //"
                output_lines.append(instr_line)
                output_lines.append("")
                timestamps.append(0.0)
                intro_rendered = True

        if not intro_rendered:
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

        # The line owns every chord change until the next line begins, *unless*
        # there is a long instrumental gap (handled separately further down).
        # Using the full span prevents a late chord change that lands a beat
        # before the next line (e.g. the bar-10 chord while still singing the
        # first line) from being dropped by the rough singing-duration estimate.
        gap_to_next = next_start - singing_end
        chord_window_end = singing_end if gap_to_next > 3.0 else next_start

        # Chords whose change-point falls within the line's span.
        chords_in_singing = []
        for c in chords:
            if line_start <= c["time"] < chord_window_end and c["chord"]:
                chords_in_singing.append((c["time"], c["chord"]))

        # Seed the line with the chord sounding at line_start ONLY when it
        # actually dominates the start of the line. If the first real chord
        # change lands early in the line, the singer effectively enters on that
        # chord and the previously-sounding chord is just a brief leftover from
        # before the vocal entry (e.g. the instrumental tail), so we drop it to
        # avoid a stale leading chord. If the first change comes late (or there
        # is none), the held chord is real and must be shown.
        first_change_t = chords_in_singing[0][0] if chords_in_singing else None
        starts_at_line_head = first_change_t is not None and first_change_t <= line_start
        if not starts_at_line_head:
            seed_needed = True
            if first_change_t is not None and singing_duration > 0:
                early_frac = (first_change_t - line_start) / singing_duration
                if early_frac <= SEED_HOLD_FRACTION:
                    seed_needed = False
            if seed_needed:
                prev_chords = [c for c in chords if c["time"] <= line_start and c["chord"]]
                if prev_chords:
                    chords_in_singing.insert(0, (line_start, prev_chords[-1]["chord"]))

        # Collapse consecutive duplicate chords so the same chord is not repeated.
        deduped = []
        for t_chord, chord_name in chords_in_singing:
            if not deduped or deduped[-1][1] != chord_name:
                deduped.append((t_chord, chord_name))
        chords_in_singing = deduped

        # Write singing block
        line_len = len(line_text)
        if line_len > 0:
            chord_chars = [" "] * (line_len + 40)
            cursor = 0  # left-most column the next chord label may occupy
            word_starts, word_ratios = _word_alignment_anchors(line_text)
            last_word_anchor = -1

            # If the line-level duration underestimates where chord changes keep
            # happening, stretch the placement span so late changes don't bunch
            # at the final word.
            layout_duration = max(0.25, singing_duration)
            if chords_in_singing:
                max_line_span = max(0.25, next_start - line_start)
                last_change_offset = max(0.0, chords_in_singing[-1][0] - line_start)
                layout_duration = max(layout_duration, min(max_line_span, last_change_offset + 0.25))

            for t_chord, chord_name in chords_in_singing:
                if not chord_name:
                    continue
                ratio = (t_chord - line_start) / layout_duration if layout_duration > 0 else 0.0
                ratio = max(0.0, min(1.0, ratio))
                
                # Snapping chords close to the line start
                time_diff = t_chord - line_start
                if ratio < 0.08 or (0 < time_diff < 0.35):
                    ratio = 0.0

                mapped_word_start = _map_ratio_to_word_start(ratio, word_starts, word_ratios)
                if mapped_word_start is not None:
                    if mapped_word_start == last_word_anchor:
                        continue
                    char_idx = mapped_word_start
                    last_word_anchor = mapped_word_start
                else:
                    char_idx = int(round(ratio * line_len))
                    char_idx = snap_to_word_boundaries(line_text, char_idx)
                
                # Never overwrite a previously placed chord label; keep a gap.
                if char_idx < cursor:
                    next_word_start = None
                    for ws in word_starts:
                        if ws >= cursor:
                            next_word_start = ws
                            break
                    char_idx = next_word_start if next_word_start is not None else cursor
                for k, char in enumerate(chord_name):
                    target_idx = char_idx + k
                    if target_idx < len(chord_chars):
                        chord_chars[target_idx] = char
                cursor = char_idx + len(chord_name) + 1
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
def _run_dialog(dialog_fn, **kwargs):
    """Run a tkinter file dialog on a topmost, hidden root window."""
    root = tk.Tk()
    root.withdraw()
    root.focus_force()
    root.attributes("-topmost", True)
    file_path = dialog_fn(**kwargs)
    root.destroy()
    return file_path

def run_file_dialog():
    return _run_dialog(
        filedialog.askopenfilename,
        title="Select Audio File",
        filetypes=[
            ("Audio Files", "*.mp3 *.wav *.ogg *.m4a *.flac"),
            ("All Files", "*.*")
        ]
    )

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
    progress_cb = _make_progress_cb(task_id)

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
        lyric_source = "synced" if lyrics else "none"
        
        chordsheet = ""
        timestamps = []
        auto_synced = False
        unsynced_lyrics = ""
        
        if not lyrics:
            lyrics = transcribe_audio_with_whisper(
                audio_path=audio_path,
                query_hint=query,
                progress_cb=progress_cb,
                progress_start=0.93,
                progress_end=0.97,
            )
            if lyrics:
                lyric_source = "asr"

        if lyrics:
            if lyric_source == "asr":
                progress_cb("Refining transcribed lyric text with web source...", 0.975)
                lyrics, refined_unsynced = maybe_refine_asr_lyrics_with_web(query, lyrics)
                if refined_unsynced:
                    unsynced_lyrics = refined_unsynced

            progress_cb("Aligning chords with timed lyrics...", 0.98)
            duration = float(librosa.get_duration(path=audio_path))
            aligned = generate_aligned_sheet_internal(chords, lyrics, duration, bars=bars)
            chordsheet = aligned["chordsheet"]
            timestamps = aligned["timestamps"]
            auto_synced = True
        else:
            if not unsynced_lyrics:
                progress_cb("Searching web for unsynced lyrics...", 0.96)
                unsynced_lyrics = search_unsynced_lyrics(query) or ""

        estimated_lyrics_start = resolve_estimated_lyrics_start(chords_data, lyrics)
            
        _complete_task(task_id, "Extraction complete!", {
            "chords": chords,
            "bpm": bpm,
            "bars": bars,
            "estimatedKey": chords_data.get("estimated_key"),
            "lyrics": lyrics,
            "chordsheet": chordsheet,
            "timestamps": timestamps,
            "auto_synced": auto_synced,
            "unsyncedLyrics": unsynced_lyrics,
            "estimatedLyricsStart": estimated_lyrics_start,
            "audioPath": audio_path,
            "filename": os.path.basename(audio_path)
        })
    except Exception as e:
        _fail_task(task_id, e)

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
    progress_cb = _make_progress_cb(task_id)

    try:
        downloads_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "downloads")
        
        # 1. Fetch transcript captions from YouTube first (fail-safe or check availability)
        progress_cb("Retrieving YouTube captions...", 0.10)
        lyrics = fetch_youtube_transcript(video_id)
        lyric_source = "captions" if lyrics else "none"
        
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

        query = re.sub(r'[\(\[\{].*?[\)\]\}]', '', friendly_title)
        query = re.sub(r'[-_\.]', ' ', query)
        query = " ".join(query.split())
            
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
            lyrics = search_and_parse_lyrics(query)
            if lyrics:
                lyric_source = "synced"

        if not lyrics:
            lyrics = transcribe_audio_with_whisper(
                audio_path=mp3_path,
                query_hint=query,
                progress_cb=progress_cb,
                progress_start=0.94,
                progress_end=0.97,
            )
            if lyrics:
                lyric_source = "asr"
            
        # 5. Build aligned chordsheet
        if lyrics:
            if lyric_source == "asr":
                progress_cb("Refining transcribed lyric text with web source...", 0.975)
                lyrics, refined_unsynced = maybe_refine_asr_lyrics_with_web(query, lyrics)
                if refined_unsynced:
                    unsynced_lyrics = refined_unsynced

            progress_cb("Auto-aligning chords with timed lyrics...", 0.98)
            duration = float(librosa.get_duration(path=mp3_path))
            aligned = generate_aligned_sheet_internal(chords, lyrics, duration, bars=bars)
            chordsheet = aligned["chordsheet"]
            timestamps = aligned["timestamps"]
            auto_synced = True
        else:
            if not unsynced_lyrics:
                progress_cb("Searching web for unsynced lyrics...", 0.96)
                unsynced_lyrics = search_unsynced_lyrics(query) or ""

        estimated_lyrics_start = resolve_estimated_lyrics_start(chords_data, lyrics)
            
        _complete_task(task_id, "YouTube extraction completed successfully!", {
            "chords": chords,
            "bpm": bpm,
            "bars": bars,
            "estimatedKey": chords_data.get("estimated_key"),
            "lyrics": lyrics,
            "chordsheet": chordsheet,
            "timestamps": timestamps,
            "auto_synced": auto_synced,
            "unsyncedLyrics": unsynced_lyrics,
            "estimatedLyricsStart": estimated_lyrics_start,
            "audioPath": mp3_path,
            "filename": friendly_title + ".mp3"
        })
        
    except Exception as e:
        _fail_task(task_id, e)

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


@app.post("/api/align-lyrics")
def align_lyrics_for_sync(data: AlignLyricsRequest):
    """
    Repair/align edited plain lyrics against existing timed reference lyrics.
    Used by the Approve & Sync flow when the original transcription text is noisy.
    """
    plain_text = (data.lyricsText or "").strip()
    if not plain_text:
        return {
            "lyrics": [],
            "method": "failed",
            "projected": False,
            "message": "No lyrics text was provided for alignment.",
        }

    reference_timed: List[dict] = []
    for line in data.referenceLyrics:
        cleaned = clean_transcribed_line(line.text)
        if not cleaned:
            continue
        reference_timed.append({
            "text": cleaned,
            "time": float(line.time),
            "duration": float(line.duration or 0.0),
        })

    if len(reference_timed) < 2:
        return {
            "lyrics": [],
            "method": "failed",
            "projected": False,
            "message": "Not enough timed lyric references to repair alignment.",
        }

    hebrew_context = contains_hebrew(plain_text) or has_hebrew_content(reference_timed)
    min_similarity = 0.20 if hebrew_context else 0.26

    aligned = align_unsynced_lyrics_to_reference_timing(
        plain_lyrics_text=plain_text,
        reference_timed_lines=reference_timed,
        min_similarity=min_similarity,
        hebrew_context=hebrew_context,
    )
    method = "match"

    if not aligned:
        aligned = project_plain_lyrics_to_reference_timing(
            plain_lyrics_text=plain_text,
            reference_timed_lines=reference_timed,
            hebrew_context=hebrew_context,
        )
        method = "projection"

    if not aligned:
        return {
            "lyrics": [],
            "method": "failed",
            "projected": False,
            "message": "Could not repair lyric timing from the current transcription.",
        }

    target_start = max(0.0, float(data.selectedStartTime or 0.0))
    offset = 0.0
    if target_start > 0.0 and aligned:
        offset = target_start - float(aligned[0].get("time", 0.0))

    shifted: List[dict] = []
    last_time = 0.0
    for item in aligned:
        cleaned = clean_transcribed_line(item.get("text", ""))
        if not cleaned:
            continue

        t = max(0.0, float(item.get("time", 0.0)) + offset)
        if shifted and t <= last_time:
            t = last_time + 0.12
        last_time = t

        shifted.append({
            "text": cleaned,
            "time": t,
            "duration": max(0.0, float(item.get("duration") or 0.0)),
        })

    return {
        "lyrics": shifted,
        "method": method,
        "projected": method == "projection",
        "message": "ok",
    }

@app.post("/api/generate-chordsheet")
def generate_chordsheet(data: GenerateSheetRequest):
    lyrics_list = [{"text": l.text, "time": l.time, "duration": l.duration} for l in data.lyrics]
    return generate_aligned_sheet_internal(data.chords, lyrics_list, data.duration, bars=data.bars)

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
    # Only auto-open a browser when running as the standalone frozen exe, where
    # the backend itself serves the full app on :8000. In dev/prod the run.py
    # scripts open the correct frontend URL (:4200 / :4300), so opening here too
    # would spawn a duplicate tab.
    if getattr(sys, 'frozen', False):
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
