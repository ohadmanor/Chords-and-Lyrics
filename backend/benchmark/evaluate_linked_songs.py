#!/usr/bin/env python3
"""Run a benchmark pass on linked songs and score chord extraction quality."""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yt_dlp

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import chord_extractor as ce

ROOT_EQUIV = {
    "CB": "B",
    "DB": "C#",
    "EB": "D#",
    "FB": "E",
    "GB": "F#",
    "AB": "G#",
    "BB": "A#",
    "E#": "F",
    "B#": "C",
}

CHORD_PARSE_RE = re.compile(r"^([A-Ga-g])([#b]?)(.*)$")


def canonical_root(root: str, accidental: str) -> str:
    key = (root.upper() + accidental).upper()
    return ROOT_EQUIV.get(key, root.upper() + accidental)


def normalize_chord_token(token: str) -> Optional[str]:
    if not token:
        return None

    t = (
        str(token)
        .replace("\u266f", "#")
        .replace("\u266d", "b")
        .strip()
        .strip("[](){}:;,.!?\"'`")
    )
    if not t:
        return None

    if t in {"/", "//", "x", "X", "N", "-", "--"}:
        return None

    t = t.split("/")[0].strip()
    m = CHORD_PARSE_RE.match(t)
    if not m:
        return None

    root = canonical_root(m.group(1), m.group(2))
    rest = (m.group(3) or "").lower()

    if rest.startswith("maj7"):
        quality = "maj7"
    elif rest.startswith("m7"):
        quality = "m7"
    elif rest.startswith("m") and not rest.startswith("maj"):
        quality = "m"
    elif rest.startswith("7"):
        quality = "7"
    else:
        quality = ""

    return f"{root}{quality}"


def extract_reference_chord_sequence(raw_text: str) -> List[str]:
    if not raw_text:
        return []

    seq: List[str] = []
    for line in raw_text.splitlines():
        s = line.strip()
        if not s:
            continue

        parts: List[str] = []
        for chunk in s.split("|"):
            parts.extend(re.split(r"\s+", chunk))

        line_tokens: List[str] = []
        for part in parts:
            cleaned = part.strip()
            if not cleaned:
                continue

            if "/" in cleaned and not cleaned.startswith("http"):
                maybe = [x for x in cleaned.split("/") if x.strip()]
            else:
                maybe = [cleaned]

            for item in maybe:
                norm = normalize_chord_token(item)
                if norm:
                    line_tokens.append(norm)

        if len(line_tokens) >= 2:
            seq.extend(line_tokens)
        elif len(line_tokens) == 1 and ("//" in s or re.search(r"\s{2,}", line)):
            seq.extend(line_tokens)

    return seq


def extract_predicted_chord_sequence(result: Dict[str, Any]) -> List[str]:
    seq: List[str] = []

    for bar in result.get("bars") or []:
        for chord in bar.get("chords") or []:
            norm = normalize_chord_token(chord)
            if norm:
                seq.append(norm)

    if seq:
        return seq

    for item in result.get("chords") or []:
        norm = normalize_chord_token(item.get("chord"))
        if norm:
            seq.append(norm)
    return seq


def overlap_metrics(pred_seq: List[str], ref_seq: List[str]) -> Dict[str, Any]:
    pred_set = set(pred_seq)
    ref_set = set(ref_seq)
    overlap = pred_set & ref_set

    set_precision = len(overlap) / len(pred_set) if pred_set else 0.0
    set_recall = len(overlap) / len(ref_set) if ref_set else 0.0
    set_f1 = 0.0
    if set_precision + set_recall > 0.0:
        set_f1 = 2.0 * set_precision * set_recall / (set_precision + set_recall)

    pred_counter = Counter(pred_seq)
    ref_counter = Counter(ref_seq)
    token_tp = sum(min(pred_counter[k], ref_counter[k]) for k in set(pred_counter) | set(ref_counter))
    token_precision = token_tp / len(pred_seq) if pred_seq else 0.0
    token_recall = token_tp / len(ref_seq) if ref_seq else 0.0
    token_f1 = 0.0
    if token_precision + token_recall > 0.0:
        token_f1 = 2.0 * token_precision * token_recall / (token_precision + token_recall)

    return {
        "predUnique": len(pred_set),
        "refUnique": len(ref_set),
        "uniqueOverlap": len(overlap),
        "setPrecision": set_precision,
        "setRecall": set_recall,
        "setF1": set_f1,
        "tokenPrecision": token_precision,
        "tokenRecall": token_recall,
        "tokenF1": token_f1,
        "missingReference": sorted(ref_set - pred_set),
        "unexpectedPredicted": sorted(pred_set - ref_set),
    }


def get_ffmpeg_location() -> Optional[str]:
    candidate = BACKEND_DIR.parent / "ffmpeg"
    if (candidate / "ffmpeg.exe").exists() or (candidate / "ffmpeg").exists():
        return str(candidate)
    return None


def download_youtube_audio(video_id: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_mp3 = output_dir / f"{video_id}.mp3"
    if output_mp3.exists():
        return output_mp3

    ydl_opts: Dict[str, Any] = {
        "format": "bestaudio/best",
        "noplaylist": True,
        "outtmpl": str(output_dir / f"{video_id}.%(ext)s"),
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
        "quiet": True,
        "no_warnings": True,
    }

    ffmpeg_dir = get_ffmpeg_location()
    if ffmpeg_dir:
        ydl_opts["ffmpeg_location"] = ffmpeg_dir

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([f"https://www.youtube.com/watch?v={video_id}"])

    if not output_mp3.exists():
        raise RuntimeError(f"Expected audio output was not created: {output_mp3}")
    return output_mp3


def evaluate_entry(entry: Dict[str, Any], downloads_dir: Path) -> Dict[str, Any]:
    video_id = str(entry.get("youtubeVideoId") or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
        raise ValueError("Invalid youtubeVideoId")

    audio_path = download_youtube_audio(video_id, downloads_dir)
    result = ce.extract_chords_from_audio(str(audio_path))

    pred_seq = extract_predicted_chord_sequence(result)
    ref_seq = extract_reference_chord_sequence(str(entry.get("rawText") or ""))
    metrics = overlap_metrics(pred_seq, ref_seq)

    beat_conf = [float(x) for x in (result.get("beat_confidence") or [])]
    bar_conf = [float(x) for x in (result.get("bar_confidence") or [])]

    low_conf_ratio = 0.0
    if beat_conf:
        low_conf_ratio = sum(1 for x in beat_conf if x < 0.40) / float(len(beat_conf))

    row = {
        "id": entry.get("id"),
        "title": entry.get("title"),
        "artist": entry.get("artist"),
        "youtubeUrl": entry.get("youtubeUrl"),
        "youtubeVideoId": video_id,
        "estimatedKey": result.get("estimated_key"),
        "predictedChordEvents": len(result.get("chords") or []),
        "predictedBars": len(result.get("bars") or []),
        "avgBeatConfidence": float(statistics.mean(beat_conf)) if beat_conf else 0.0,
        "avgBarConfidence": float(statistics.mean(bar_conf)) if bar_conf else 0.0,
        "lowConfidenceBeatRatio": low_conf_ratio,
        "metrics": {
            **metrics,
            "missingReference": metrics["missingReference"][:20],
            "unexpectedPredicted": metrics["unexpectedPredicted"][:20],
        },
    }
    return row


def summarise(rows: List[Dict[str, Any]], failures: List[Dict[str, Any]]) -> Dict[str, Any]:
    set_f1_vals = [r["metrics"]["setF1"] for r in rows]
    token_f1_vals = [r["metrics"]["tokenF1"] for r in rows]
    bar_conf_vals = [r["avgBarConfidence"] for r in rows]

    worst = sorted(rows, key=lambda x: x["metrics"]["setF1"])[:10]
    return {
        "evaluated": len(rows),
        "failed": len(failures),
        "meanSetF1": float(statistics.mean(set_f1_vals)) if set_f1_vals else 0.0,
        "medianSetF1": float(statistics.median(set_f1_vals)) if set_f1_vals else 0.0,
        "meanTokenF1": float(statistics.mean(token_f1_vals)) if token_f1_vals else 0.0,
        "meanBarConfidence": float(statistics.mean(bar_conf_vals)) if bar_conf_vals else 0.0,
        "worstBySetF1": [
            {
                "id": r["id"],
                "title": r["title"],
                "artist": r["artist"],
                "setF1": r["metrics"]["setF1"],
                "tokenF1": r["metrics"]["tokenF1"],
                "avgBarConfidence": r["avgBarConfidence"],
            }
            for r in worst
        ],
    }


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate linked songs benchmark")
    parser.add_argument(
        "--manifest",
        default=r"backend\benchmark\linked_songs_manifest.json",
        help="Linked songs manifest file",
    )
    parser.add_argument(
        "--downloads-dir",
        default=r"backend\downloads",
        help="Audio cache directory",
    )
    parser.add_argument(
        "--out-dir",
        default=r"backend\benchmark\results",
        help="Output folder for benchmark reports",
    )
    parser.add_argument("--offset", type=int, default=0, help="Start index in linked manifest")
    parser.add_argument("--limit", type=int, default=10, help="Number of songs to evaluate")
    parser.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep between songs")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    linked = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(linked, list):
        raise ValueError("Manifest is not a list")

    start = max(0, int(args.offset))
    end = start + max(0, int(args.limit))
    batch = linked[start:end]

    downloads_dir = Path(args.downloads_dir)
    rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    print(f"Evaluating {len(batch)} song(s) from linked manifest (offset={start}, limit={args.limit})")

    for i, entry in enumerate(batch, start=1):
        title = str(entry.get("title") or "")
        artist = str(entry.get("artist") or "")
        print(f"[{i}/{len(batch)}] {artist} - {title}")
        try:
            row = evaluate_entry(entry, downloads_dir)
            rows.append(row)
            print(
                "    setF1={:.3f} tokenF1={:.3f} avgBarConf={:.3f}".format(
                    row["metrics"]["setF1"],
                    row["metrics"]["tokenF1"],
                    row["avgBarConfidence"],
                )
            )
        except Exception as e:
            failures.append(
                {
                    "id": entry.get("id"),
                    "title": title,
                    "artist": artist,
                    "youtubeUrl": entry.get("youtubeUrl"),
                    "error": str(e),
                }
            )
            print(f"    failed: {e}")

        if args.sleep > 0 and i < len(batch):
            time.sleep(args.sleep)

    summary = summarise(rows, failures)
    now_utc = datetime.now(timezone.utc)
    payload = {
        "createdAt": now_utc.isoformat().replace("+00:00", "Z"),
        "offset": start,
        "limit": int(args.limit),
        "summary": summary,
        "results": rows,
        "failures": failures,
    }

    out_dir = Path(args.out_dir)
    stamp = now_utc.strftime("%Y%m%d_%H%M%S")
    run_path = out_dir / f"benchmark_run_{stamp}.json"
    latest_path = out_dir / "benchmark_latest.json"

    write_json(run_path, payload)
    write_json(latest_path, payload)

    print(json.dumps(summary, ensure_ascii=True, indent=2))
    print(f"Wrote: {run_path}")
    print(f"Wrote: {latest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
