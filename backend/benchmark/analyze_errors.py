#!/usr/bin/env python3
"""Analyze benchmark output and report common chord-error patterns."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

CHORD_RE = re.compile(r"^([A-G](?:#|b)?)(.*)$")


def classify_quality(suffix: str) -> str:
    s = (suffix or "").lower()
    if s.startswith("maj7"):
        return "maj7"
    if s.startswith("m7"):
        return "m7"
    if s.startswith("m") and not s.startswith("maj"):
        return "min"
    if s.startswith("7"):
        return "dom7"
    if s == "":
        return "maj"
    return "other"


def parse_chord(chord: str) -> Tuple[Optional[str], str]:
    m = CHORD_RE.match(chord or "")
    if not m:
        return None, "other"
    return m.group(1), classify_quality(m.group(2))


def compute_flip_roots(
    missing_by_root: Dict[str, Set[str]],
    unexpected_by_root: Dict[str, Set[str]],
) -> List[str]:
    roots: List[str] = []
    for root in sorted(set(missing_by_root) & set(unexpected_by_root)):
        missing_q = missing_by_root[root]
        unexpected_q = unexpected_by_root[root]
        major_missing_minor_unexpected = "maj" in missing_q and ("min" in unexpected_q or "m7" in unexpected_q)
        minor_missing_major_unexpected = "min" in missing_q and ("maj" in unexpected_q or "maj7" in unexpected_q)
        if major_missing_minor_unexpected or minor_missing_major_unexpected:
            roots.append(root)
    return roots


def analyze(results: List[Dict[str, Any]], worst_n: int) -> Dict[str, Any]:
    worst = sorted(results, key=lambda r: r.get("metrics", {}).get("setF1", 0.0))[:worst_n]

    missing_root_counter: Counter[str] = Counter()
    unexpected_root_counter: Counter[str] = Counter()
    missing_quality_counter: Counter[str] = Counter()
    unexpected_quality_counter: Counter[str] = Counter()
    flip_root_counter: Counter[str] = Counter()

    worst_songs: List[Dict[str, Any]] = []

    for row in worst:
        metrics = row.get("metrics") or {}
        missing = metrics.get("missingReference") or []
        unexpected = metrics.get("unexpectedPredicted") or []

        missing_by_root: Dict[str, Set[str]] = defaultdict(set)
        unexpected_by_root: Dict[str, Set[str]] = defaultdict(set)

        for chord in missing:
            root, quality = parse_chord(chord)
            missing_quality_counter[quality] += 1
            if root:
                missing_root_counter[root] += 1
                missing_by_root[root].add(quality)

        for chord in unexpected:
            root, quality = parse_chord(chord)
            unexpected_quality_counter[quality] += 1
            if root:
                unexpected_root_counter[root] += 1
                unexpected_by_root[root].add(quality)

        flip_roots = compute_flip_roots(missing_by_root, unexpected_by_root)
        for root in flip_roots:
            flip_root_counter[root] += 1

        worst_songs.append(
            {
                "id": row.get("id"),
                "title": row.get("title"),
                "artist": row.get("artist"),
                "setF1": metrics.get("setF1", 0.0),
                "tokenF1": metrics.get("tokenF1", 0.0),
                "avgBarConfidence": row.get("avgBarConfidence", 0.0),
                "lowConfidenceBeatRatio": row.get("lowConfidenceBeatRatio", 0.0),
                "missingReference": missing,
                "unexpectedPredicted": unexpected,
                "suspectedMajorMinorFlipRoots": flip_roots,
            }
        )

    return {
        "analyzedWorstN": worst_n,
        "topMissingRootsWorstN": missing_root_counter.most_common(10),
        "topUnexpectedRootsWorstN": unexpected_root_counter.most_common(10),
        "topMissingQualitiesWorstN": missing_quality_counter.most_common(),
        "topUnexpectedQualitiesWorstN": unexpected_quality_counter.most_common(),
        "suspectedMajorMinorFlipRootsWorstN": flip_root_counter.most_common(),
        "worstSongs": worst_songs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze benchmark chord error patterns")
    parser.add_argument(
        "--input",
        default=r"backend\benchmark\results\benchmark_latest.json",
        help="Path to benchmark result JSON",
    )
    parser.add_argument(
        "--output",
        default=r"backend\benchmark\results\error_analysis_latest.json",
        help="Path to write analysis JSON",
    )
    parser.add_argument(
        "--worst-n",
        type=int,
        default=5,
        help="Analyze the worst N songs by setF1",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Benchmark input not found: {input_path}")

    payload = json.loads(input_path.read_text(encoding="utf-8"))
    rows = payload.get("results") or []
    if not isinstance(rows, list) or not rows:
        raise ValueError("Benchmark input has no results rows")

    analysis = analyze(rows, max(1, int(args.worst_n)))
    output_payload = {
        "source": str(input_path),
        "evaluated": (payload.get("summary") or {}).get("evaluated"),
        "failed": (payload.get("summary") or {}).get("failed"),
        "meanSetF1": (payload.get("summary") or {}).get("meanSetF1"),
        "meanTokenF1": (payload.get("summary") or {}).get("meanTokenF1"),
        "meanBarConfidence": (payload.get("summary") or {}).get("meanBarConfidence"),
        **analysis,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output_payload, indent=2, ensure_ascii=True), encoding="utf-8")

    print(f"Wrote: {output_path}")
    print("Top missing roots:", output_payload["topMissingRootsWorstN"][:6])
    print("Top unexpected roots:", output_payload["topUnexpectedRootsWorstN"][:6])
    print("Top missing qualities:", output_payload["topMissingQualitiesWorstN"])
    print("Top unexpected qualities:", output_payload["topUnexpectedQualitiesWorstN"])
    print("Suspected major/minor flip roots:", output_payload["suspectedMajorMinorFlipRootsWorstN"])
    print("Worst songs analyzed:")
    for song in output_payload["worstSongs"]:
        print(
            "- {artist} - {title}: setF1={setf1:.3f}, tokenF1={tokenf1:.3f}, avgBarConfidence={barc:.3f}, lowConfRatio={lcr:.3f}, flipRoots={fr}".format(
                artist=str(song.get("artist") or ""),
                title=str(song.get("title") or ""),
                setf1=float(song.get("setF1") or 0.0),
                tokenf1=float(song.get("tokenF1") or 0.0),
                barc=float(song.get("avgBarConfidence") or 0.0),
                lcr=float(song.get("lowConfidenceBeatRatio") or 0.0),
                fr=song.get("suspectedMajorMinorFlipRoots") or [],
            )
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())