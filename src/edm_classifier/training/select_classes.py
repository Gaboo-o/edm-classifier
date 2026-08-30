"""Select the shared active class set for the model.

The same active classes are used for BOTH the regular and artist-separated
experiments so their metrics are directly comparable.

For every training sample, direct labels are expanded upward through the
taxonomy. Example:

    melodic_dubstep -> dubstep -> ...

A sparse leaf can therefore be excluded while its well-supported parent remains.

Selection rule:
    expanded training support >= --min-train-tracks
    in BOTH split strategies.

Inputs:
    config/taxonomy.yaml
    data/splits/regular/train.jsonl
    data/splits/artist/train.jsonl

Outputs:
    data/training/
      active_classes.json
      class_selection.csv
      class_selection_report.json
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml


DEFAULT_TAXONOMY = Path("config/taxonomy.yaml")
DEFAULT_SPLITS_DIR = Path("data/splits")
DEFAULT_OUTPUT_DIR = Path("data/training")


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


def load_taxonomy(path: Path) -> tuple[dict[str, dict[str, Any]], str | None]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))

    if not isinstance(raw, dict):
        raise ValueError(f"{path}: taxonomy root must be an object")

    genres = raw.get("genres")
    if not isinstance(genres, list):
        raise ValueError(f"{path}: expected a genres list")

    by_id: dict[str, dict[str, Any]] = {}

    for position, item in enumerate(genres):
        if not isinstance(item, dict):
            raise ValueError(f"{path}: genre #{position} is not an object")

        genre_id = item.get("id")
        if not isinstance(genre_id, str) or not genre_id:
            raise ValueError(f"{path}: genre #{position} has no valid id")

        if genre_id in by_id:
            raise ValueError(f"{path}: duplicate genre id {genre_id!r}")

        copy = dict(item)
        copy["_taxonomy_position"] = position
        by_id[genre_id] = copy

    for genre_id, item in by_id.items():
        parent = item.get("parent")
        if parent is not None and parent not in by_id:
            raise ValueError(
                f"{path}: {genre_id!r} references unknown parent {parent!r}"
            )

    taxonomy_meta = raw.get("taxonomy")
    version = (
        taxonomy_meta.get("version")
        if isinstance(taxonomy_meta, dict)
        else None
    )

    return by_id, version


def build_children(
    taxonomy: dict[str, dict[str, Any]],
) -> dict[str, set[str]]:
    children: dict[str, set[str]] = defaultdict(set)

    for genre_id, item in taxonomy.items():
        parent = item.get("parent")
        if isinstance(parent, str) and parent:
            children[parent].add(genre_id)

    return children


def ancestors(
    genre_id: str,
    taxonomy: dict[str, dict[str, Any]],
) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()

    current = genre_id

    while True:
        item = taxonomy.get(current)
        if item is None:
            break

        parent = item.get("parent")
        if not isinstance(parent, str) or not parent:
            break

        if parent in seen:
            raise ValueError(
                f"taxonomy cycle detected while expanding {genre_id!r}"
            )

        seen.add(parent)
        result.append(parent)
        current = parent

    return result


def direct_labels(record: dict[str, Any]) -> set[str]:
    raw = record.get("labels")
    if not isinstance(raw, list):
        return set()

    return {
        item
        for item in raw
        if isinstance(item, str) and item
    }


def expanded_labels(
    labels: set[str],
    taxonomy: dict[str, dict[str, Any]],
) -> set[str]:
    result = set(labels)

    for label in labels:
        if label not in taxonomy:
            raise ValueError(
                f"split contains label not present in taxonomy: {label!r}"
            )
        result.update(ancestors(label, taxonomy))

    return result


def count_support(
    records: list[dict[str, Any]],
    taxonomy: dict[str, dict[str, Any]],
) -> tuple[Counter[str], Counter[str]]:
    direct: Counter[str] = Counter()
    expanded: Counter[str] = Counter()

    for record in records:
        labels = direct_labels(record)
        direct.update(labels)
        expanded.update(expanded_labels(labels, taxonomy))

    return direct, expanded


def taxonomy_depth(
    genre_id: str,
    taxonomy: dict[str, dict[str, Any]],
) -> int:
    return len(ancestors(genre_id, taxonomy))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select classes with sufficient expanded training support in both "
            "regular and artist-separated splits."
        )
    )

    parser.add_argument(
        "--taxonomy",
        type=Path,
        default=DEFAULT_TAXONOMY,
    )
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=DEFAULT_SPLITS_DIR,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--min-train-tracks",
        type=int,
        default=15,
        help=(
            "Minimum expanded training examples required in BOTH strategies "
            "(default: 15)."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.min_train_tracks < 1:
        raise SystemExit("--min-train-tracks must be >= 1")

    taxonomy, taxonomy_version = load_taxonomy(args.taxonomy)
    children = build_children(taxonomy)

    regular_path = args.splits_dir / "regular" / "train.jsonl"
    artist_path = args.splits_dir / "artist" / "train.jsonl"

    regular_records = load_jsonl(regular_path)
    artist_records = load_jsonl(artist_path)

    regular_direct, regular_expanded = count_support(
        regular_records,
        taxonomy,
    )
    artist_direct, artist_expanded = count_support(
        artist_records,
        taxonomy,
    )

    active_ids = {
        genre_id
        for genre_id in taxonomy
        if min(
            regular_expanded[genre_id],
            artist_expanded[genre_id],
        ) >= args.min_train_tracks
    }

    ordered_ids = sorted(
        taxonomy,
        key=lambda genre_id: (
            taxonomy[genre_id].get(
                "order",
                taxonomy[genre_id]["_taxonomy_position"],
            ),
            taxonomy[genre_id]["_taxonomy_position"],
        ),
    )

    active_ordered = [
        genre_id
        for genre_id in ordered_ids
        if genre_id in active_ids
    ]

    classes: list[dict[str, Any]] = []

    for new_index, genre_id in enumerate(active_ordered):
        item = taxonomy[genre_id]

        classes.append(
            {
                "index": new_index,
                "id": genre_id,
                "label": item.get("label", genre_id),
                "parent": item.get("parent"),
                "depth": taxonomy_depth(genre_id, taxonomy),
                "is_leaf": not bool(children.get(genre_id)),
                "regular_train_direct": regular_direct[genre_id],
                "artist_train_direct": artist_direct[genre_id],
                "regular_train_expanded": regular_expanded[genre_id],
                "artist_train_expanded": artist_expanded[genre_id],
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    active_path = args.output_dir / "active_classes.json"
    csv_path = args.output_dir / "class_selection.csv"
    report_path = args.output_dir / "class_selection_report.json"

    active_payload = {
        "taxonomy_version": taxonomy_version,
        "selection_version": 1,
        "policy": {
            "min_train_tracks": args.min_train_tracks,
            "support_type": "direct_labels_plus_taxonomy_ancestors",
            "required_in": [
                "regular_train",
                "artist_train",
            ],
            "same_class_set_for_both_experiments": True,
        },
        "class_count": len(classes),
        "classes": classes,
    }

    active_path.write_text(
        json.dumps(active_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    rows: list[dict[str, Any]] = []

    for genre_id in ordered_ids:
        item = taxonomy[genre_id]
        active = genre_id in active_ids

        rows.append(
            {
                "id": genre_id,
                "label": item.get("label", genre_id),
                "parent": item.get("parent") or "",
                "depth": taxonomy_depth(genre_id, taxonomy),
                "is_leaf": not bool(children.get(genre_id)),
                "regular_train_direct": regular_direct[genre_id],
                "artist_train_direct": artist_direct[genre_id],
                "regular_train_expanded": regular_expanded[genre_id],
                "artist_train_expanded": artist_expanded[genre_id],
                "minimum_expanded_train": min(
                    regular_expanded[genre_id],
                    artist_expanded[genre_id],
                ),
                "active": active,
            }
        )

    fieldnames = list(rows[0]) if rows else []

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    active_leaf_ids = [
        genre_id
        for genre_id in active_ordered
        if not children.get(genre_id)
    ]
    active_parent_ids = [
        genre_id
        for genre_id in active_ordered
        if children.get(genre_id)
    ]

    dropped_ids = [
        genre_id
        for genre_id in ordered_ids
        if genre_id not in active_ids
    ]
    dropped_leaf_ids = [
        genre_id
        for genre_id in dropped_ids
        if not children.get(genre_id)
    ]

    report = {
        "taxonomy_version": taxonomy_version,
        "minimum_train_tracks": args.min_train_tracks,
        "regular_training_samples": len(regular_records),
        "artist_training_samples": len(artist_records),
        "taxonomy_classes": len(taxonomy),
        "active_classes": len(active_ordered),
        "active_leaf_classes": len(active_leaf_ids),
        "active_parent_classes": len(active_parent_ids),
        "dropped_classes": len(dropped_ids),
        "dropped_leaf_classes": len(dropped_leaf_ids),
        "active_class_ids": active_ordered,
        "dropped_class_ids": dropped_ids,
        "dropped_leaf_details": [
            {
                "id": genre_id,
                "label": taxonomy[genre_id].get("label", genre_id),
                "regular_train_direct": regular_direct[genre_id],
                "artist_train_direct": artist_direct[genre_id],
                "regular_train_expanded": regular_expanded[genre_id],
                "artist_train_expanded": artist_expanded[genre_id],
            }
            for genre_id in dropped_leaf_ids
        ],
        "outputs": {
            "active_classes": str(active_path),
            "class_selection_csv": str(csv_path),
        },
    }

    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("Class selection complete")
    print(f"  taxonomy classes:       {len(taxonomy)}")
    print(f"  active classes:      {len(active_ordered)}")
    print(f"    leaf classes:         {len(active_leaf_ids)}")
    print(f"    parent classes:       {len(active_parent_ids)}")
    print(f"  dropped classes:        {len(dropped_ids)}")
    print(f"  minimum train support:  {args.min_train_tracks}")
    print()
    print(f"Active classes: {active_path}")
    print(f"Distribution:   {csv_path}")
    print(f"Report:         {report_path}")


if __name__ == "__main__":
    main()
