"""Build the supervised training manifest from weak-label results.

Inputs:
- data/candidates/candidate_tracks.jsonl
- data/label_job/results/labeled_tracks.jsonl
- config/taxonomy.yaml
- data/validation/class_coverage.csv (optional but preferred)

Outputs:
- data/training/training_manifest.jsonl
- data/training/classes.json
- data/training/build_report.json

Only status=labeled records are included. Labels whose class coverage is below
--min-label-tracks are excluded. Tracks left with no active labels are dropped.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CANDIDATES = Path("data/candidates/candidate_tracks.jsonl")
DEFAULT_LABELS = Path("data/label_job/results/labeled_tracks.jsonl")
DEFAULT_TAXONOMY = Path("config/taxonomy.yaml")
DEFAULT_COVERAGE = Path("data/validation/class_coverage.csv")
DEFAULT_OUTPUT_DIR = Path("data/training")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: each JSONL line must be an object")
            records.append(record)
    return records


def load_taxonomy(path: Path) -> tuple[dict[str, dict[str, Any]], str | None]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("genres"), list):
        raise ValueError(f"Invalid taxonomy structure: {path}")

    genres: dict[str, dict[str, Any]] = {}
    for item in raw["genres"]:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str):
            raise ValueError(f"Invalid genre entry in {path}: {item!r}")
        genre_id = item["id"]
        if genre_id in genres:
            raise ValueError(f"Duplicate taxonomy ID: {genre_id}")
        genres[genre_id] = item

    version = (raw.get("taxonomy") or {}).get("version")
    return genres, version


def index_unique(records: list[dict[str, Any]], *, source: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for record in records:
        candidate_id = record.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id:
            raise ValueError(f"{source}: record missing non-empty candidate_id")
        if candidate_id in result:
            raise ValueError(f"{source}: duplicate candidate_id {candidate_id!r}")
        result[candidate_id] = record
    return result


def label_counts_from_results(
    label_records: list[dict[str, Any]],
    taxonomy: dict[str, dict[str, Any]],
) -> Counter[str]:
    counts: Counter[str] = Counter()
    for record in label_records:
        if record.get("status") != "labeled":
            continue
        seen: set[str] = set()
        labels = record.get("labels")
        if not isinstance(labels, list):
            continue
        for item in labels:
            if not isinstance(item, dict):
                continue
            label_id = item.get("id")
            if isinstance(label_id, str) and label_id in taxonomy and label_id not in seen:
                counts[label_id] += 1
                seen.add(label_id)
    return counts


def label_counts_from_coverage(path: Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"id", "tracks"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{path} must contain columns: id, tracks")
        for row in reader:
            label_id = (row.get("id") or "").strip()
            raw_tracks = (row.get("tracks") or "0").strip()
            if not label_id:
                continue
            try:
                tracks = int(raw_tracks)
            except ValueError as exc:
                raise ValueError(f"{path}: invalid tracks count {raw_tracks!r} for {label_id}") from exc
            counts[label_id] = tracks
    return counts


def taxonomy_order(item: tuple[str, dict[str, Any]]) -> tuple[int, str]:
    label_id, genre = item
    try:
        order = int(genre.get("order", 1_000_000))
    except (TypeError, ValueError):
        order = 1_000_000
    return order, label_id


def accepted_labels(record: dict[str, Any]) -> list[tuple[str, float]]:
    result: list[tuple[str, float]] = []
    labels = record.get("labels")
    if not isinstance(labels, list):
        return result

    for item in labels:
        if not isinstance(item, dict):
            continue
        label_id = item.get("id")
        confidence = item.get("confidence")
        if not isinstance(label_id, str):
            continue
        if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
            continue
        result.append((label_id, float(confidence)))
    return result


def slim_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    """Keep identity/provenance fields useful to the audio resolver and auditing."""
    output: dict[str, Any] = {
        "candidate_id": candidate["candidate_id"],
        "artist": candidate.get("artist", ""),
        "title": candidate.get("title", ""),
    }

    for key in ("mbid", "artist_mbid", "lastfm_url"):
        value = candidate.get(key)
        if value not in (None, "", []):
            output[key] = value

    # Preserve weak-supervision provenance. It is not used as the target itself,
    # but is useful for later audits and targeted re-labeling.
    if isinstance(candidate.get("discovered_for"), list):
        output["discovered_for"] = candidate["discovered_for"]
    if isinstance(candidate.get("top_tags"), list):
        output["top_tags"] = candidate["top_tags"]

    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the supervised training manifest.")
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--taxonomy", type=Path, default=DEFAULT_TAXONOMY)
    parser.add_argument("--coverage", type=Path, default=DEFAULT_COVERAGE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--min-label-tracks",
        type=int,
        default=1,
        help="Minimum labeled tracks required for a class to remain active (default: 1).",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.85,
        help="Minimum accepted-label confidence retained in the manifest (default: 0.85).",
    )
    parser.add_argument(
        "--ignore-coverage-file",
        action="store_true",
        help="Recompute class counts directly from labeled_tracks.jsonl.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.min_label_tracks < 1:
        raise SystemExit("--min-label-tracks must be >= 1")
    if not 0.0 <= args.min_confidence <= 1.0:
        raise SystemExit("--min-confidence must be between 0 and 1")

    taxonomy, taxonomy_version = load_taxonomy(args.taxonomy)
    candidate_records = load_jsonl(args.candidates)
    label_records = load_jsonl(args.labels)

    candidates = index_unique(candidate_records, source=str(args.candidates))
    labels_by_id = index_unique(label_records, source=str(args.labels))

    if not args.ignore_coverage_file and args.coverage.exists():
        coverage_counts = label_counts_from_coverage(args.coverage)
        coverage_source = str(args.coverage)
    else:
        coverage_counts = label_counts_from_results(label_records, taxonomy)
        coverage_source = "recomputed from labeled_tracks.jsonl"

    unknown_coverage_ids = sorted(set(coverage_counts) - set(taxonomy))
    if unknown_coverage_ids:
        raise SystemExit(
            "Coverage contains unknown taxonomy IDs: " + ", ".join(unknown_coverage_ids[:20])
        )

    active_ids = {
        label_id
        for label_id in taxonomy
        if coverage_counts.get(label_id, 0) >= args.min_label_tracks
    }

    ordered_active = [
        label_id
        for label_id, _genre in sorted(taxonomy.items(), key=taxonomy_order)
        if label_id in active_ids
    ]
    class_index = {label_id: index for index, label_id in enumerate(ordered_active)}

    output_records: list[dict[str, Any]] = []
    output_label_counts: Counter[str] = Counter()
    skipped_status: Counter[str] = Counter()
    skipped_no_active_labels = 0
    skipped_missing_candidate = 0
    removed_low_confidence = 0
    removed_inactive_labels: Counter[str] = Counter()

    for candidate_id, label_record in labels_by_id.items():
        status = label_record.get("status")
        if status != "labeled":
            skipped_status[str(status)] += 1
            continue

        candidate = candidates.get(candidate_id)
        if candidate is None:
            skipped_missing_candidate += 1
            continue

        retained: list[tuple[str, float]] = []
        for label_id, confidence in accepted_labels(label_record):
            if label_id not in taxonomy:
                raise SystemExit(f"Unknown taxonomy label {label_id!r} on {candidate_id}")
            if confidence < args.min_confidence:
                removed_low_confidence += 1
                continue
            if label_id not in active_ids:
                removed_inactive_labels[label_id] += 1
                continue
            retained.append((label_id, confidence))

        # Keep taxonomy order for deterministic target vectors.
        retained.sort(key=lambda pair: class_index[pair[0]])

        if not retained:
            skipped_no_active_labels += 1
            continue

        record = slim_candidate(candidate)
        record["labels"] = [label_id for label_id, _ in retained]
        record["label_confidence"] = {
            label_id: round(confidence, 4) for label_id, confidence in retained
        }
        record["label_indices"] = [class_index[label_id] for label_id, _ in retained]
        output_records.append(record)

        for label_id, _ in retained:
            output_label_counts[label_id] += 1

    output_records.sort(key=lambda r: r["candidate_id"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "training_manifest.jsonl"
    classes_path = args.output_dir / "classes.json"
    report_path = args.output_dir / "build_report.json"

    with manifest_path.open("w", encoding="utf-8") as handle:
        for record in output_records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")

    classes_payload = {
        "taxonomy_version": taxonomy_version,
        "class_count": len(ordered_active),
        "min_label_tracks": args.min_label_tracks,
        "classes": [
            {
                "index": class_index[label_id],
                "id": label_id,
                "label": taxonomy[label_id].get("label", label_id),
                "role": taxonomy[label_id].get("role"),
                "parent": taxonomy[label_id].get("parent"),
                "training_tracks": output_label_counts.get(label_id, 0),
            }
            for label_id in ordered_active
        ],
    }
    classes_path.write_text(
        json.dumps(classes_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    zero_coverage = [label_id for label_id in taxonomy if coverage_counts.get(label_id, 0) == 0]
    inactive = [label_id for label_id in taxonomy if label_id not in active_ids]

    report = {
        "taxonomy_version": taxonomy_version,
        "inputs": {
            "candidates": str(args.candidates),
            "labels": str(args.labels),
            "taxonomy": str(args.taxonomy),
            "coverage": coverage_source,
        },
        "settings": {
            "min_label_tracks": args.min_label_tracks,
            "min_confidence": args.min_confidence,
        },
        "candidate_records": len(candidate_records),
        "label_records": len(label_records),
        "training_tracks": len(output_records),
        "active_classes": len(ordered_active),
        "accepted_label_assignments": sum(output_label_counts.values()),
        "skipped_status": dict(sorted(skipped_status.items())),
        "skipped_missing_candidate": skipped_missing_candidate,
        "skipped_no_active_labels": skipped_no_active_labels,
        "removed_low_confidence_assignments": removed_low_confidence,
        "removed_inactive_label_assignments": dict(sorted(removed_inactive_labels.items())),
        "zero_coverage_labels": zero_coverage,
        "inactive_labels": inactive,
        "class_counts": {
            label_id: output_label_counts.get(label_id, 0) for label_id in ordered_active
        },
    }
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("Training manifest built")
    print(f"  candidate records:  {len(candidate_records)}")
    print(f"  label records:      {len(label_records)}")
    print(f"  training tracks:    {len(output_records)}")
    print(f"  active classes:     {len(ordered_active)}")
    print(f"  label assignments:  {sum(output_label_counts.values())}")
    print(f"  zero-coverage:      {len(zero_coverage)}")
    print()
    print(f"Manifest: {manifest_path}")
    print(f"Classes:  {classes_path}")
    print(f"Report:   {report_path}")


if __name__ == "__main__":
    main()
