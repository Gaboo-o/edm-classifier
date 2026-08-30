"""Validate, deduplicate, and select provisional V2 classes after LLM labeling.

Inputs:
    config/taxonomy.yaml
    data/v2/label_job/label_input_active.jsonl
    data/v2/label_job/labeled_tracks.jsonl
    data/v2/candidates/candidate_tracks.jsonl

V1 dedup/support inputs:
    data/splits/samples.jsonl
    data/candidates/candidate_tracks.jsonl

Outputs:
    data/v2/validation/
      accepted_candidates.jsonl
      non_labeled_candidates.jsonl
      duplicate_records.jsonl
      class_support.csv
      provisional_active_classes.json
      validation_report.json

Important:
- Only status=labeled records enter accepted_candidates.jsonl.
- Uncertain/reject records are preserved separately.
- Duplicate V2/V1 recordings are excluded from accepted_candidates.
- A provisionally dropped leaf does NOT cause its tracks to be discarded.
  The original leaf label is preserved so the track can still contribute to
  its active taxonomy ancestors during training.
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
DEFAULT_LABEL_INPUT = Path("data/v2/label_job/label_input_active.jsonl")
DEFAULT_LABELS = Path("data/v2/label_job/labeled_tracks.jsonl")
DEFAULT_CANDIDATES = Path("data/v2/candidates/candidate_tracks.jsonl")
DEFAULT_V1_SAMPLES = Path("data/splits/samples.jsonl")
DEFAULT_V1_CANDIDATES = Path("data/candidates/candidate_tracks.jsonl")
DEFAULT_OUTPUT_DIR = Path("data/v2/validation")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            line = raw.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            rows.append(value)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            )


def normalize_text(value: str) -> str:
    text = unicodedata.normalize("NFKD", value)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.casefold()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_artist(value: str) -> str:
    return normalize_text(value)


def load_taxonomy(
    path: Path,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, set[str]]]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    genres = raw.get("genres") if isinstance(raw, dict) else None
    if not isinstance(genres, list):
        raise ValueError(f"{path}: expected genres list")

    ordered: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}
    children: dict[str, set[str]] = defaultdict(set)

    for position, item in enumerate(genres):
        if not isinstance(item, dict) or not isinstance(item.get("id"), str):
            raise ValueError(f"{path}: invalid genre at position {position}")
        copy = dict(item)
        copy["_position"] = position
        ordered.append(copy)
        by_id[copy["id"]] = copy

    for genre_id, item in by_id.items():
        parent = item.get("parent")
        if isinstance(parent, str) and parent:
            if parent not in by_id:
                raise ValueError(f"{genre_id!r}: unknown parent {parent!r}")
            children[parent].add(genre_id)

    return ordered, by_id, children


def ancestor_ids(
    genre_id: str,
    taxonomy: dict[str, dict[str, Any]],
) -> set[str]:
    result: set[str] = set()
    current = genre_id
    seen: set[str] = set()

    while True:
        if current in seen:
            raise ValueError(f"taxonomy cycle involving {genre_id!r}")
        seen.add(current)

        parent = taxonomy[current].get("parent")
        if not isinstance(parent, str) or not parent:
            return result

        result.add(parent)
        current = parent


def extract_labels(row: dict[str, Any]) -> list[dict[str, Any]]:
    value = row.get("labels")
    return value if isinstance(value, list) else []


def direct_label_ids(row: dict[str, Any]) -> set[str]:
    result = set()
    for item in extract_labels(row):
        if isinstance(item, str):
            result.add(item)
        elif isinstance(item, dict) and isinstance(item.get("id"), str):
            result.add(item["id"])
    return result


def sample_label_ids(row: dict[str, Any]) -> set[str]:
    raw = row.get("labels")
    if not isinstance(raw, list):
        return set()

    result = set()
    for item in raw:
        if isinstance(item, str):
            result.add(item)
        elif isinstance(item, dict) and isinstance(item.get("id"), str):
            result.add(item["id"])
    return result


def artist_keys(row: dict[str, Any]) -> set[str]:
    values: list[str] = []

    for key in ("artist_keys", "artists"):
        raw = row.get(key)
        if isinstance(raw, list):
            values.extend(x for x in raw if isinstance(x, str))

    for key in ("artist", "original_artist"):
        value = row.get(key)
        if isinstance(value, str):
            values.append(value)

    return {
        normalize_artist(value)
        for value in values
        if value.strip() and normalize_artist(value)
    }


def identity_keys(row: dict[str, Any]) -> set[str]:
    keys: set[str] = set()

    mbid = row.get("mbid")
    if isinstance(mbid, str) and mbid.strip():
        keys.add("mbid:" + mbid.strip().casefold())

    artist = row.get("artist")
    if not isinstance(artist, str):
        artist = row.get("original_artist")

    title = row.get("title")

    if isinstance(artist, str) and isinstance(title, str):
        keys.add(
            "text:"
            + normalize_text(artist)
            + "\0"
            + normalize_text(title)
        )

    return keys


def max_confidence(row: dict[str, Any]) -> float:
    values = []
    for item in extract_labels(row):
        if isinstance(item, dict) and isinstance(item.get("confidence"), (int, float)):
            values.append(float(item["confidence"]))
    return max(values, default=0.0)


def mean_confidence_for_label(
    rows: list[dict[str, Any]],
) -> dict[str, float]:
    values: dict[str, list[float]] = defaultdict(list)

    for row in rows:
        for item in extract_labels(row):
            if (
                isinstance(item, dict)
                and isinstance(item.get("id"), str)
                and isinstance(item.get("confidence"), (int, float))
            ):
                values[item["id"]].append(float(item["confidence"]))

    return {
        genre_id: statistics.mean(scores)
        for genre_id, scores in values.items()
        if scores
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate/deduplicate V2 labels and compute class support."
    )
    parser.add_argument("--taxonomy", type=Path, default=DEFAULT_TAXONOMY)
    parser.add_argument("--label-input", type=Path, default=DEFAULT_LABEL_INPUT)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--v1-samples", type=Path, default=DEFAULT_V1_SAMPLES)
    parser.add_argument("--v1-candidates", type=Path, default=DEFAULT_V1_CANDIDATES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--min-leaf-tracks",
        type=int,
        default=50,
        help="Minimum combined direct labeled tracks for a provisional leaf.",
    )
    parser.add_argument(
        "--min-leaf-artists",
        type=int,
        default=30,
        help="Minimum combined unique artists for a provisional leaf.",
    )
    parser.add_argument(
        "--strong-leaf-tracks",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--strong-leaf-artists",
        type=int,
        default=50,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ordered_taxonomy, taxonomy, children = load_taxonomy(args.taxonomy)
    label_input = load_jsonl(args.label_input)
    outputs = load_jsonl(args.labels)
    candidates = load_jsonl(args.candidates)
    v1_samples = load_jsonl(args.v1_samples)
    v1_candidates = load_jsonl(args.v1_candidates)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    input_by_id = {
        row.get("candidate_id"): row
        for row in label_input
        if isinstance(row.get("candidate_id"), str)
    }
    candidate_by_id = {
        row.get("candidate_id"): row
        for row in candidates
        if isinstance(row.get("candidate_id"), str)
    }

    issues: list[dict[str, Any]] = []
    output_ids: list[str] = []
    seen_output_ids: set[str] = set()

    invalid_status = 0
    invalid_labels = 0
    invalid_confidence = 0
    too_many_labels = 0
    labeled_without_labels = 0
    redundant_pairs = 0
    missing_input_ids = 0
    missing_candidate_metadata = 0
    low_confidence_labeled = 0

    valid_statuses = {"labeled", "uncertain", "reject"}

    joined: list[dict[str, Any]] = []

    for position, output in enumerate(outputs):
        cid = output.get("candidate_id")
        if not isinstance(cid, str):
            issues.append({"position": position, "issue": "missing_candidate_id"})
            continue

        output_ids.append(cid)

        if cid in seen_output_ids:
            issues.append({"candidate_id": cid, "issue": "duplicate_output_candidate_id"})
        seen_output_ids.add(cid)

        input_row = input_by_id.get(cid)
        if input_row is None:
            missing_input_ids += 1
            issues.append({"candidate_id": cid, "issue": "candidate_id_not_in_label_input"})
            continue

        status = output.get("status")
        if status not in valid_statuses:
            invalid_status += 1
            issues.append({"candidate_id": cid, "issue": "invalid_status", "value": status})

        labels = extract_labels(output)
        if len(labels) > 2:
            too_many_labels += 1
            issues.append({"candidate_id": cid, "issue": "too_many_labels"})

        if status == "labeled" and not labels:
            labeled_without_labels += 1
            issues.append({"candidate_id": cid, "issue": "labeled_without_labels"})

        options = input_row.get("label_options")
        allowed = {
            item["id"]
            for item in options
            if isinstance(item, dict) and isinstance(item.get("id"), str)
        } if isinstance(options, list) else set()

        assigned_ids: list[str] = []

        for label in labels:
            if not isinstance(label, dict):
                invalid_labels += 1
                issues.append({"candidate_id": cid, "issue": "label_not_object"})
                continue

            genre_id = label.get("id")
            confidence = label.get("confidence")

            if not isinstance(genre_id, str) or genre_id not in allowed or genre_id not in taxonomy:
                invalid_labels += 1
                issues.append(
                    {
                        "candidate_id": cid,
                        "issue": "invalid_label_id",
                        "value": genre_id,
                    }
                )
            else:
                assigned_ids.append(genre_id)

            if (
                not isinstance(confidence, (int, float))
                or isinstance(confidence, bool)
                or not 0.0 <= float(confidence) <= 1.0
            ):
                invalid_confidence += 1
                issues.append(
                    {
                        "candidate_id": cid,
                        "issue": "invalid_confidence",
                        "value": confidence,
                    }
                )

        if status == "labeled" and labels and max_confidence(output) < 0.60:
            low_confidence_labeled += 1

        redundant = False
        assigned_set = set(assigned_ids)
        for genre_id in assigned_ids:
            if ancestor_ids(genre_id, taxonomy) & assigned_set:
                redundant = True
                break

        if redundant:
            redundant_pairs += 1
            issues.append({"candidate_id": cid, "issue": "redundant_parent_child"})

        metadata = candidate_by_id.get(cid)
        if metadata is None:
            # label_input contains enough metadata for a safe fallback.
            metadata = input_row
            missing_candidate_metadata += 1

        merged = dict(metadata)
        merged["candidate_id"] = cid
        merged["status"] = status
        merged["labels"] = labels
        merged["reason"] = output.get("reason", "")
        merged["label_input_position"] = position
        joined.append(merged)

    input_ids = [
        row.get("candidate_id")
        for row in label_input
        if isinstance(row.get("candidate_id"), str)
    ]

    line_count_matches = len(outputs) == len(label_input)
    id_set_matches = set(output_ids) == set(input_ids)
    order_matches = output_ids == input_ids

    # Build V1 identity index using both the usable sample set and the complete
    # raw candidate set, matching the Stage-1 exclusion policy.
    v1_identity: set[str] = set()
    for row in v1_samples + v1_candidates:
        v1_identity.update(identity_keys(row))

    duplicate_rows: list[dict[str, Any]] = []
    accepted: list[dict[str, Any]] = []
    non_labeled: list[dict[str, Any]] = []

    seen_v2_identity: dict[str, str] = {}

    cross_v1_duplicates = 0
    within_v2_duplicates = 0

    for row in joined:
        cid = row["candidate_id"]
        status = row.get("status")

        if status != "labeled":
            non_labeled.append(row)
            continue

        keys = identity_keys(row)

        matching_v1 = sorted(keys & v1_identity)
        if matching_v1:
            cross_v1_duplicates += 1
            duplicate_rows.append(
                {
                    "candidate_id": cid,
                    "duplicate_type": "v1_overlap",
                    "matched_identity_keys": matching_v1,
                    "artist": row.get("artist"),
                    "title": row.get("title"),
                }
            )
            continue

        prior_ids = {
            seen_v2_identity[key]
            for key in keys
            if key in seen_v2_identity
        }

        if prior_ids:
            within_v2_duplicates += 1
            duplicate_rows.append(
                {
                    "candidate_id": cid,
                    "duplicate_type": "within_v2",
                    "duplicate_of": sorted(prior_ids),
                    "matched_identity_keys": sorted(
                        key for key in keys if key in seen_v2_identity
                    ),
                    "artist": row.get("artist"),
                    "title": row.get("title"),
                }
            )
            continue

        for key in keys:
            seen_v2_identity[key] = cid

        accepted.append(row)

    # Direct class support. V1 samples are already unique usable audio samples.
    v1_tracks: Counter[str] = Counter()
    v2_tracks: Counter[str] = Counter()

    v1_artists: dict[str, set[str]] = defaultdict(set)
    v2_artists: dict[str, set[str]] = defaultdict(set)

    for row in v1_samples:
        row_artists = artist_keys(row)
        for genre_id in sample_label_ids(row):
            if genre_id in taxonomy:
                v1_tracks[genre_id] += 1
                v1_artists[genre_id].update(row_artists)

    for row in accepted:
        row_artists = artist_keys(row)
        for genre_id in direct_label_ids(row):
            if genre_id in taxonomy:
                v2_tracks[genre_id] += 1
                v2_artists[genre_id].update(row_artists)

    v2_mean_confidence = mean_confidence_for_label(accepted)

    class_rows: list[dict[str, Any]] = []
    provisional_active_ids: set[str] = set()

    for item in ordered_taxonomy:
        genre_id = item["id"]
        parent = item.get("parent")
        is_leaf = not children.get(genre_id)

        combined_artists = v1_artists[genre_id] | v2_artists[genre_id]
        combined_tracks = v1_tracks[genre_id] + v2_tracks[genre_id]

        if not parent:
            status = "parent_kept"
            provisional_active = True
        elif not is_leaf:
            status = "internal_parent_kept"
            provisional_active = True
        elif (
            combined_tracks >= args.strong_leaf_tracks
            and len(combined_artists) >= args.strong_leaf_artists
        ):
            status = "strong_leaf"
            provisional_active = True
        elif (
            combined_tracks >= args.min_leaf_tracks
            and len(combined_artists) >= args.min_leaf_artists
        ):
            status = "viable_leaf"
            provisional_active = True
        else:
            status = "drop_leaf"
            provisional_active = False

        if provisional_active:
            provisional_active_ids.add(genre_id)

        class_rows.append(
            {
                "id": genre_id,
                "label": item.get("label", genre_id),
                "parent": parent or "",
                "role": (
                    "leaf"
                    if is_leaf
                    else ("root" if not parent else "internal")
                ),
                "v1_direct_tracks": v1_tracks[genre_id],
                "v2_direct_tracks": v2_tracks[genre_id],
                "combined_direct_tracks": combined_tracks,
                "v1_unique_artists": len(v1_artists[genre_id]),
                "v2_unique_artists": len(v2_artists[genre_id]),
                "combined_unique_artists": len(combined_artists),
                "v2_mean_confidence": (
                    round(v2_mean_confidence[genre_id], 6)
                    if genre_id in v2_mean_confidence
                    else ""
                ),
                "provisional_status": status,
                "provisional_active": provisional_active,
            }
        )

    support_path = args.output_dir / "class_support.csv"
    fieldnames = list(class_rows[0]) if class_rows else []

    with support_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(class_rows)

    active_classes = [
        {
            "index": index,
            "id": item["id"],
            "label": item.get("label", item["id"]),
            "parent": item.get("parent"),
            "is_leaf": not children.get(item["id"]),
            "combined_direct_tracks": next(
                row["combined_direct_tracks"]
                for row in class_rows
                if row["id"] == item["id"]
            ),
            "combined_unique_artists": next(
                row["combined_unique_artists"]
                for row in class_rows
                if row["id"] == item["id"]
            ),
        }
        for index, item in enumerate(
            x for x in ordered_taxonomy if x["id"] in provisional_active_ids
        )
    ]

    active_path = args.output_dir / "provisional_active_classes.json"
    active_path.write_text(
        json.dumps(
            {
                "selection_stage": "post_label_pre_audio",
                "policy": {
                    "parents": "always retained",
                    "minimum_leaf_tracks": args.min_leaf_tracks,
                    "minimum_leaf_unique_artists": args.min_leaf_artists,
                    "strong_leaf_tracks": args.strong_leaf_tracks,
                    "strong_leaf_unique_artists": args.strong_leaf_artists,
                    "final_selection_repeated_after_audio_embeddings": True,
                },
                "class_count": len(active_classes),
                "classes": active_classes,
            },
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )

    accepted_path = args.output_dir / "accepted_candidates.jsonl"
    non_labeled_path = args.output_dir / "non_labeled_candidates.jsonl"
    duplicates_path = args.output_dir / "duplicate_records.jsonl"
    issues_path = args.output_dir / "validation_issues.jsonl"

    write_jsonl(accepted_path, accepted)
    write_jsonl(non_labeled_path, non_labeled)
    write_jsonl(duplicates_path, duplicate_rows)
    write_jsonl(issues_path, issues)

    leaf_rows = [row for row in class_rows if row["role"] == "leaf"]
    status_counts = Counter(row["provisional_status"] for row in leaf_rows)

    report = {
        "input": {
            "label_job_records": len(label_input),
            "label_output_records": len(outputs),
            "candidate_metadata_records": len(candidates),
        },
        "llm_status_counts": dict(Counter(row.get("status") for row in joined)),
        "validation": {
            "line_count_matches": line_count_matches,
            "candidate_id_set_matches": id_set_matches,
            "candidate_order_matches": order_matches,
            "duplicate_output_candidate_ids": len(output_ids) - len(set(output_ids)),
            "missing_input_candidate_ids": missing_input_ids,
            "missing_candidate_metadata": missing_candidate_metadata,
            "invalid_status_records": invalid_status,
            "invalid_label_records": invalid_labels,
            "invalid_confidence_records": invalid_confidence,
            "too_many_labels_records": too_many_labels,
            "labeled_without_labels_records": labeled_without_labels,
            "redundant_parent_child_records": redundant_pairs,
            "labeled_records_below_0_60_max_confidence": low_confidence_labeled,
            "issue_count": len(issues),
        },
        "deduplication": {
            "status_labeled_before_dedup": sum(
                row.get("status") == "labeled" for row in joined
            ),
            "within_v2_duplicates_removed": within_v2_duplicates,
            "v1_overlap_duplicates_removed": cross_v1_duplicates,
            "accepted_unique_v2_labeled_tracks": len(accepted),
            "non_labeled_preserved": len(non_labeled),
        },
        "post_label_class_selection": {
            "taxonomy_class_count": len(ordered_taxonomy),
            "provisional_active_class_count": len(active_classes),
            "root_or_parent_classes_retained": sum(
                row["role"] != "leaf" and row["provisional_active"]
                for row in class_rows
            ),
            "leaf_status_counts": dict(status_counts),
            "minimum_leaf_policy": {
                "tracks": args.min_leaf_tracks,
                "unique_artists": args.min_leaf_artists,
            },
            "strong_leaf_policy": {
                "tracks": args.strong_leaf_tracks,
                "unique_artists": args.strong_leaf_artists,
            },
            "note": (
                "This is provisional. Repeat class selection after YTM "
                "resolution/download/embedding survival."
            ),
        },
        "outputs": {
            "accepted_candidates": str(accepted_path),
            "non_labeled_candidates": str(non_labeled_path),
            "duplicates": str(duplicates_path),
            "class_support": str(support_path),
            "provisional_active_classes": str(active_path),
            "validation_issues": str(issues_path),
        },
    }

    report_path = args.output_dir / "validation_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("V2 validation + deduplication complete")
    print()
    print("LLM output:")
    for status in ("labeled", "uncertain", "reject"):
        print(f"  {status:10s} {report['llm_status_counts'].get(status, 0)}")
    print()
    print("Deduplication:")
    print(f"  labeled before dedup: {report['deduplication']['status_labeled_before_dedup']}")
    print(f"  within-V2 removed:    {within_v2_duplicates}")
    print(f"  V1 overlaps removed:  {cross_v1_duplicates}")
    print(f"  accepted unique V2:   {len(accepted)}")
    print()
    print("Provisional leaf classes:")
    for status in ("strong_leaf", "viable_leaf", "drop_leaf"):
        print(f"  {status:12s} {status_counts[status]}")
    print()
    print(f"Active classes: {len(active_classes)}")
    print(f"Report:         {report_path}")
    print(f"Class support:  {support_path}")
    print(f"Accepted:       {accepted_path}")


if __name__ == "__main__":
    main()
