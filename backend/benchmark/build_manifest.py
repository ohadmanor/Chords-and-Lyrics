#!/usr/bin/env python3
"""Build a benchmark manifest from a JavaScript songs dataset file."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

YOUTUBE_ID_RE = re.compile(r"([A-Za-z0-9_-]{11})")
HEBREW_RE = re.compile(r"[\u0590-\u05FF]")


def extract_js_array_literal(text: str, marker: str) -> str:
    marker_idx = text.find(marker)
    if marker_idx < 0:
        raise ValueError(f"Could not find marker: {marker}")

    start_idx = text.find("[", marker_idx)
    if start_idx < 0:
        raise ValueError(f"Could not find opening array bracket after: {marker}")

    depth = 0
    in_string = False
    string_quote = ""
    escaped = False

    for i in range(start_idx, len(text)):
        ch = text[i]

        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == string_quote:
                in_string = False
            continue

        if ch in ('"', "'"):
            in_string = True
            string_quote = ch
            continue

        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return text[start_idx : i + 1]

    raise ValueError("Could not find matching closing array bracket")


def remove_trailing_commas(js_like_text: str) -> str:
    """Remove trailing commas outside strings so json.loads can parse."""
    out: List[str] = []
    in_string = False
    string_quote = ""
    escaped = False
    i = 0
    n = len(js_like_text)

    while i < n:
        ch = js_like_text[i]

        if in_string:
            out.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == string_quote:
                in_string = False
            i += 1
            continue

        if ch in ('"', "'"):
            in_string = True
            string_quote = ch
            out.append(ch)
            i += 1
            continue

        if ch == ",":
            j = i + 1
            while j < n and js_like_text[j] in " \t\r\n":
                j += 1
            if j < n and js_like_text[j] in "]}":
                i += 1
                continue

        out.append(ch)
        i += 1

    return "".join(out)


def parse_songs_dataset(js_text: str) -> List[Dict[str, Any]]:
    array_literal = extract_js_array_literal(js_text, "window.defaultSongs")

    try:
        songs = json.loads(array_literal)
    except json.JSONDecodeError:
        cleaned = remove_trailing_commas(array_literal)
        songs = json.loads(cleaned)

    if not isinstance(songs, list):
        raise ValueError("window.defaultSongs is not a list")
    return songs


def extract_youtube_video_id(url: str) -> Optional[str]:
    if not url:
        return None

    raw = url.strip()
    if not raw:
        return None

    if raw.startswith("www."):
        raw = "https://" + raw
    if not raw.startswith("http://") and not raw.startswith("https://"):
        raw = "https://" + raw

    parsed = urlparse(raw)
    host = parsed.netloc.lower()
    path = parsed.path or ""

    video_id: Optional[str] = None

    if "youtu.be" in host:
        token = path.strip("/").split("/")[0]
        if token:
            video_id = token
    elif "youtube.com" in host:
        q = parse_qs(parsed.query)
        if q.get("v"):
            video_id = q["v"][0]
        elif path.startswith("/shorts/"):
            parts = path.split("/")
            if len(parts) >= 3:
                video_id = parts[2]
        elif path.startswith("/embed/"):
            parts = path.split("/")
            if len(parts) >= 3:
                video_id = parts[2]

    if not video_id:
        match = YOUTUBE_ID_RE.search(raw)
        if match:
            video_id = match.group(1)

    if not video_id:
        return None

    candidate = video_id.strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]{11}", candidate):
        return None
    return candidate


def canonical_youtube_url(url: str) -> Tuple[Optional[str], Optional[str]]:
    video_id = extract_youtube_video_id(url)
    if not video_id:
        return None, None
    return f"https://www.youtube.com/watch?v={video_id}", video_id


def song_has_hebrew(song: Dict[str, Any]) -> bool:
    text = " ".join(
        [
            str(song.get("title") or ""),
            str(song.get("artist") or ""),
            str(song.get("rawText") or "")[:600],
        ]
    )
    return bool(HEBREW_RE.search(text))


def get_link_value(song: Dict[str, Any]) -> str:
    for key in ("Youtube_Link", "youtube_link", "youtubeUrl", "youtube_url", "youtube"):
        value = song.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def build_manifest_entries(songs: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    linked: List[Dict[str, Any]] = []
    missing: List[Dict[str, Any]] = []

    for idx, song in enumerate(songs):
        if not isinstance(song, dict):
            continue

        raw_link = get_link_value(song)
        canonical, video_id = canonical_youtube_url(raw_link)

        base = {
            "index": idx,
            "id": str(song.get("id") or f"song_{idx}"),
            "title": str(song.get("title") or ""),
            "artist": str(song.get("artist") or ""),
            "key": str(song.get("key") or ""),
            "isRTL": bool(song.get("isRTL", False)),
            "hasHebrew": song_has_hebrew(song),
            "rawText": str(song.get("rawText") or ""),
        }

        if canonical and video_id:
            row = dict(base)
            row.update(
                {
                    "youtubeUrl": canonical,
                    "youtubeVideoId": video_id,
                    "sourceYoutubeField": raw_link,
                }
            )
            linked.append(row)
        else:
            row = dict(base)
            row.update({"sourceYoutubeField": raw_link})
            missing.append(row)

    return linked, missing


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build benchmark manifest from songs JS data")
    parser.add_argument(
        "--songs-file",
        default=r"C:\dev\songs-data.js",
        help="Path to songs-data.js",
    )
    parser.add_argument(
        "--out-dir",
        default=r"backend\benchmark",
        help="Directory for generated manifest files",
    )
    args = parser.parse_args()

    songs_path = Path(args.songs_file)
    if not songs_path.exists():
        raise FileNotFoundError(f"Songs file was not found: {songs_path}")

    js_text = songs_path.read_text(encoding="utf-8")
    songs = parse_songs_dataset(js_text)
    linked, missing = build_manifest_entries(songs)

    out_dir = Path(args.out_dir)
    write_json(out_dir / "linked_songs_manifest.json", linked)
    write_json(out_dir / "missing_youtube_links.json", missing)

    missing_compact = [
        {
            "index": row.get("index"),
            "id": row.get("id"),
            "title": row.get("title"),
            "artist": row.get("artist"),
            "isRTL": row.get("isRTL"),
            "hasHebrew": row.get("hasHebrew"),
        }
        for row in missing
    ]
    write_json(out_dir / "missing_youtube_links_compact.json", missing_compact)

    hebrew_linked = sum(1 for row in linked if row.get("hasHebrew"))
    english_linked = len(linked) - hebrew_linked

    summary = {
        "totalSongs": len(songs),
        "songsWithYouTube": len(linked),
        "songsWithoutYouTube": len(missing),
        "linkedHebrewOrRTL": hebrew_linked,
        "linkedNonHebrew": english_linked,
    }
    write_json(out_dir / "manifest_summary.json", summary)

    print(json.dumps(summary, ensure_ascii=True, indent=2))
    print(f"Wrote: {out_dir / 'linked_songs_manifest.json'}")
    print(f"Wrote: {out_dir / 'missing_youtube_links.json'}")
    print(f"Wrote: {out_dir / 'missing_youtube_links_compact.json'}")
    print(f"Wrote: {out_dir / 'manifest_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
