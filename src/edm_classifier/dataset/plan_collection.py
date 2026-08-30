"""Plan the V2 data-collection pass from the current embedded dataset.

This script does NOT call Last.fm yet. It measures the current usable dataset
and calculates how many NEW candidates we should request per leaf genre.

Why leaf genres?
- Parent/root targets are already strengthened automatically by child examples.
- Collection effort is most useful at the fine-grained labels that are sparse
  or artist-concentrated.

Inputs:
    config/taxonomy.yaml
    data/splits/samples.jsonl
    optional: data/runs/artist_mlp/test_per_class_hierarchical.csv

Outputs:
    data/v2/collection_plan/
      class_targets.csv
      collection_plan.json
      collection_report.json

Default planning goals:
- at least 100 usable tracks per leaf
- at least 50 unique artists per leaf
- add at least 20 new usable tracks to every leaf, even if already healthy
- assume 50% of newly collected candidates survive labeling/resolution/download
- cap planned candidate requests at 200 per leaf

These are planning defaults only. Run the script, inspect the report, and then
adjust them before starting network collection if the total is too large/small.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml


DEFAULT_TAXONOMY = Path("config/taxonomy.yaml")
DEFAULT_SAMPLES = Path("data/splits/samples.jsonl")
DEFAULT_PERFORMANCE = Path(
    "data/runs/artist_mlp/test_per_class_hierarchical.csv"
)
DEFAULT_OUTPUT_DIR = Path("data/v2/collection_plan")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}:{line_number}: invalid JSON: {exc}"
                ) from exc
            if not isinstance(value, dict):
                raise ValueError(
                    f"{path}:{line_number}: expected JSON object"
                )
            records.append(value)
    return records


def normalize_artist(value: str) -> str:
    text = unicodedata.normalize("NFKD", value)
    text = "".join(
        char for char in text
        if not unicodedata.combining(char)
    )
    text = text.casefold()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_taxonomy(
    path: Path,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, set[str]], str | None]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: taxonomy root must be an object")

    genres = raw.get("genres")
    if not isinstance(genres, list):
        raise ValueError(f"{path}: expected genres list")

    ordered: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}
    children: dict[str, set[str]] = defaultdict(set)

    for position, item in enumerate(genres):
        if not isinstance(item, dict):
            raise ValueError(
                f"{path}: genre #{position} is not an object"
            )

        genre_id = item.get("id")
        if not isinstance(genre_id, str) or not genre_id:
            raise ValueError(
                f"{path}: genre #{position} has invalid id"
            )

        if genre_id in by_id:
            raise ValueError(
                f"{path}: duplicate genre id {genre_id!r}"
            )

        copy = dict(item)
        copy["_position"] = position
        ordered.append(copy)
        by_id[genre_id] = copy

    for genre_id, item in by_id.items():
        parent = item.get("parent")
        if isinstance(parent, str) and parent:
            if parent not in by_id:
                raise ValueError(
                    f"{genre_id!r} references unknown parent {parent!r}"
                )
            children[parent].add(genre_id)

    meta = raw.get("taxonomy")
    version = (
        meta.get("version")
        if isinstance(meta, dict)
        else None
    )

    return ordered, by_id, children, version


def get_artist_keys(record: dict[str, Any]) -> set[str]:
    raw_keys = record.get("artist_keys")
    keys: set[str] = set()

    if isinstance(raw_keys, list):
        for item in raw_keys:
            if isinstance(item, str) and item.strip():
                keys.add(normalize_artist(item))

    if keys:
        return {key for key in keys if key}

    raw_artists = record.get("artists")
    if isinstance(raw_artists, list):
        for item in raw_artists:
            if isinstance(item, str) and item.strip():
                key = normalize_artist(item)
                if key:
                    keys.add(key)

    if keys:
        return keys

    artist = record.get("artist")
    if isinstance(artist, str) and artist.strip():
        key = normalize_artist(artist)
        if key:
            keys.add(key)

    return keys


def get_direct_labels(record: dict[str, Any]) -> set[str]:
    labels = record.get("labels")
    if not isinstance(labels, list):
        return set()

    return {
        label
        for label in labels
        if isinstance(label, str) and label
    }


def load_performance(path: Path) -> dict[str, dict[str, float | int]]:
    if not path.exists():
        return {}

    result: dict[str, dict[str, float | int]] = {}

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            genre_id = row.get("id")
            if not genre_id:
                continue

            try:
                f1 = float(row.get("f1", ""))
            except (TypeError, ValueError):
                f1 = float("nan")

            try:
                support = int(row.get("support", "0"))
            except (TypeError, ValueError):
                support = 0

            result[genre_id] = {
                "artist_test_f1": f1,
                "artist_test_support": support,
            }

    return result


def median_or_zero(values: list[int]) -> float:
    return float(statistics.median(values)) if values else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plan targeted V2 collection using track and artist deficits."
    )

    parser.add_argument(
        "--taxonomy",
        type=Path,
        default=DEFAULT_TAXONOMY,
    )
    parser.add_argument(
        "--samples",
        type=Path,
        default=DEFAULT_SAMPLES,
    )
    parser.add_argument(
        "--performance-csv",
        type=Path,
        default=DEFAULT_PERFORMANCE,
        help=(
            "Optional full-taxonomy artist-split per-class metrics. "
            "Used for reporting/prioritization, not eligibility."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument(
        "--target-tracks",
        type=int,
        default=100,
        help="Desired usable unique tracks per leaf after V2 (default: 100).",
    )
    parser.add_argument(
        "--target-artists",
        type=int,
        default=50,
        help="Desired unique artists per leaf after V2 (default: 50).",
    )
    parser.add_argument(
        "--min-new-usable",
        type=int,
        default=20,
        help=(
            "Minimum desired NEW usable tracks for every leaf, even if already "
            "above the track/artist targets (default: 20)."
        ),
    )
    parser.add_argument(
        "--estimated-yield",
        type=float,
        default=0.50,
        help=(
            "Estimated fraction of collected candidates that become usable "
            "embedded tracks (default: 0.50)."
        ),
    )
    parser.add_argument(
        "--max-candidates-per-leaf",
        type=int,
        default=200,
        help="Cap planned candidate requests per leaf (default: 200).",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.target_tracks < 1:
        raise SystemExit("--target-tracks must be >= 1")
    if args.target_artists < 1:
        raise SystemExit("--target-artists must be >= 1")
    if args.min_new_usable < 0:
        raise SystemExit("--min-new-usable must be >= 0")
    if not 0 < args.estimated_yield <= 1:
        raise SystemExit("--estimated-yield must be in (0, 1]")
    if args.max_candidates_per_leaf < 1:
        raise SystemExit("--max-candidates-per-leaf must be >= 1")

    ordered, taxonomy, children, version = load_taxonomy(args.taxonomy)
    samples = load_jsonl(args.samples)
    performance = load_performance(args.performance_csv)

    roots = [
        item["id"]
        for item in ordered
        if not item.get("parent")
    ]
    leaves = [
        item["id"]
        for item in ordered
        if not children.get(item["id"])
    ]

    direct_track_counts: Counter[str] = Counter()
    direct_artist_sets: dict[str, set[str]] = defaultdict(set)
    direct_artist_track_counts: dict[str, Counter[str]] = defaultdict(Counter)

    unknown_labels: Counter[str] = Counter()

    for sample in samples:
        labels = get_direct_labels(sample)
        artists = get_artist_keys(sample)

        for label in labels:
            if label not in taxonomy:
                unknown_labels[label] += 1
                continue

            direct_track_counts[label] += 1

            for artist in artists:
                direct_artist_sets[label].add(artist)
                direct_artist_track_counts[label][artist] += 1

    rows: list[dict[str, Any]] = []

    for item in ordered:
        genre_id = item["id"]
        if genre_id not in leaves:
            continue

        current_tracks = direct_track_counts[genre_id]
        current_artists = len(direct_artist_sets[genre_id])

        track_deficit = max(
            0,
            args.target_tracks - current_tracks,
        )
        artist_deficit = max(
            0,
            args.target_artists - current_artists,
        )

        # One genuinely new-artist usable track can improve both deficits.
        desired_new_usable = max(
            args.min_new_usable,
            track_deficit,
            artist_deficit,
        )

        planned_candidates = math.ceil(
            desired_new_usable / args.estimated_yield
        )
        planned_candidates = min(
            planned_candidates,
            args.max_candidates_per_leaf,
        )

        artist_counts = direct_artist_track_counts[genre_id]
        max_tracks_one_artist = max(
            artist_counts.values(),
            default=0,
        )
        top_artist_share = (
            max_tracks_one_artist / current_tracks
            if current_tracks
            else 0.0
        )

        perf = performance.get(genre_id, {})
        artist_test_f1 = perf.get("artist_test_f1")
        artist_test_support = perf.get("artist_test_support")

        # Priority is for ordering the collection queue, not changing quota.
        # Track scarcity and artist scarcity dominate; low artist-test F1 is a
        # smaller secondary signal.
        track_gap_fraction = (
            track_deficit / args.target_tracks
        )
        artist_gap_fraction = (
            artist_deficit / args.target_artists
        )

        f1_gap = 0.0
        if isinstance(artist_test_f1, float) and math.isfinite(artist_test_f1):
            f1_gap = max(0.0, 0.50 - artist_test_f1) / 0.50

        priority_score = (
            0.45 * track_gap_fraction
            + 0.45 * artist_gap_fraction
            + 0.10 * f1_gap
        )

        rows.append(
            {
                "id": genre_id,
                "label": item.get("label", genre_id),
                "parent": item.get("parent") or "",
                "current_tracks": current_tracks,
                "current_unique_artists": current_artists,
                "tracks_per_artist": (
                    round(current_tracks / current_artists, 3)
                    if current_artists
                    else None
                ),
                "top_artist_share": round(top_artist_share, 4),
                "track_deficit_to_target": track_deficit,
                "artist_deficit_to_target": artist_deficit,
                "desired_new_usable_tracks": desired_new_usable,
                "planned_candidate_requests": planned_candidates,
                "artist_test_f1": (
                    round(float(artist_test_f1), 6)
                    if isinstance(artist_test_f1, float)
                    and math.isfinite(artist_test_f1)
                    else None
                ),
                "artist_test_support": artist_test_support,
                "priority_score": round(priority_score, 6),
            }
        )

    rows.sort(
        key=lambda row: (
            -float(row["priority_score"]),
            -int(row["planned_candidate_requests"]),
            int(row["current_tracks"]),
            str(row["id"]),
        )
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = args.output_dir / "class_targets.csv"
    plan_path = args.output_dir / "collection_plan.json"
    report_path = args.output_dir / "collection_report.json"

    fieldnames = list(rows[0]) if rows else []
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    plan = {
        "version": "v2",
        "taxonomy_version": version,
        "policy": {
            "collection_unit": "leaf_genre",
            "target_tracks": args.target_tracks,
            "target_unique_artists": args.target_artists,
            "minimum_new_usable_tracks_per_leaf": args.min_new_usable,
            "estimated_candidate_to_usable_yield": args.estimated_yield,
            "maximum_candidate_requests_per_leaf": args.max_candidates_per_leaf,
            "parent_collection": (
                "indirect: parent support is supplied by child-label examples"
            ),
            "priority": (
                "45% track scarcity + 45% artist scarcity + "
                "10% poor artist-separated F1"
            ),
        },
        "leaves": [
            {
                "id": row["id"],
                "label": row["label"],
                "parent": row["parent"],
                "current_tracks": row["current_tracks"],
                "current_unique_artists": row["current_unique_artists"],
                "desired_new_usable_tracks": row["desired_new_usable_tracks"],
                "planned_candidate_requests": row["planned_candidate_requests"],
                "priority_score": row["priority_score"],
            }
            for row in rows
        ],
    }

    plan_path.write_text(
        json.dumps(plan, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    track_counts = [
        int(row["current_tracks"])
        for row in rows
    ]
    artist_counts = [
        int(row["current_unique_artists"])
        for row in rows
    ]
    candidate_counts = [
        int(row["planned_candidate_requests"])
        for row in rows
    ]
    usable_goals = [
        int(row["desired_new_usable_tracks"])
        for row in rows
    ]

    report = {
        "version": "v2",
        "taxonomy_version": version,
        "current_unique_audio_samples": len(samples),
        "taxonomy_classes": len(taxonomy),
        "root_classes": len(roots),
        "leaf_classes_targeted": len(leaves),
        "unknown_label_assignments": dict(unknown_labels),
        "current_leaf_support": {
            "tracks": {
                "minimum": min(track_counts) if track_counts else 0,
                "median": median_or_zero(track_counts),
                "maximum": max(track_counts) if track_counts else 0,
                "leaves_below_target": sum(
                    count < args.target_tracks
                    for count in track_counts
                ),
            },
            "unique_artists": {
                "minimum": min(artist_counts) if artist_counts else 0,
                "median": median_or_zero(artist_counts),
                "maximum": max(artist_counts) if artist_counts else 0,
                "leaves_below_target": sum(
                    count < args.target_artists
                    for count in artist_counts
                ),
            },
        },
        "planned_v2_collection": {
            "desired_new_usable_tracks_sum": sum(usable_goals),
            "planned_candidate_requests_sum": sum(candidate_counts),
            "minimum_candidates_for_one_leaf": (
                min(candidate_counts) if candidate_counts else 0
            ),
            "median_candidates_per_leaf": median_or_zero(candidate_counts),
            "maximum_candidates_for_one_leaf": (
                max(candidate_counts) if candidate_counts else 0
            ),
        },
        "top_priority_leaves": [
            {
                "id": row["id"],
                "label": row["label"],
                "current_tracks": row["current_tracks"],
                "current_unique_artists": row["current_unique_artists"],
                "planned_candidate_requests": row["planned_candidate_requests"],
                "artist_test_f1": row["artist_test_f1"],
                "priority_score": row["priority_score"],
            }
            for row in rows[:25]
        ],
        "outputs": {
            "class_targets": str(csv_path),
            "collection_plan": str(plan_path),
        },
    }

    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("V2 collection planning complete")
    print(f"  current unique samples:        {len(samples)}")
    print(f"  taxonomy classes:              {len(taxonomy)}")
    print(f"  root classes:                  {len(roots)}")
    print(f"  leaf classes targeted:         {len(leaves)}")
    print()
    print(
        f"  current leaf tracks:           "
        f"min={min(track_counts) if track_counts else 0}, "
        f"median={median_or_zero(track_counts):.1f}, "
        f"max={max(track_counts) if track_counts else 0}"
    )
    print(
        f"  current unique artists:        "
        f"min={min(artist_counts) if artist_counts else 0}, "
        f"median={median_or_zero(artist_counts):.1f}, "
        f"max={max(artist_counts) if artist_counts else 0}"
    )
    print()
    print(
        f"  desired new usable tracks:     {sum(usable_goals)}"
    )
    print(
        f"  planned candidate requests:    {sum(candidate_counts)}"
    )
    print()
    print("Top 15 collection priorities:")
    for row in rows[:15]:
        f1_text = (
            "n/a"
            if row["artist_test_f1"] is None
            else f"{row['artist_test_f1']:.3f}"
        )
        print(
            f"  {row['id']:<28} "
            f"tracks={row['current_tracks']:>3} "
            f"artists={row['current_unique_artists']:>3} "
            f"candidates={row['planned_candidate_requests']:>3} "
            f"artistF1={f1_text}"
        )
    print()
    print(f"Plan:   {plan_path}")
    print(f"CSV:    {csv_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
