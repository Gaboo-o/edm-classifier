"""Build the 19-root parent-only class set.

Inputs:
    config/taxonomy.yaml
    data/splits/regular/{train,validation,test}.jsonl
    data/splits/artist/{train,validation,test}.jsonl

Outputs:
    data/training_parent/
      active_classes.json
      class_distribution.csv
      parent_selection_report.json

Every sample's direct labels are propagated to taxonomy ancestors. Since this
taxonomy has 19 roots and all subgenres sit directly beneath one root, each
training sample becomes positive for the appropriate parent genre(s).
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import yaml


DEFAULT_TAXONOMY = Path("config/taxonomy.yaml")
DEFAULT_SPLITS_DIR = Path("data/splits")
DEFAULT_OUTPUT_DIR = Path("data/training_parent")


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
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            records.append(value)
    return records


def load_taxonomy(path: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], str | None]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: taxonomy root must be an object")

    genres = raw.get("genres")
    if not isinstance(genres, list):
        raise ValueError(f"{path}: expected genres list")

    ordered: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}

    for position, item in enumerate(genres):
        if not isinstance(item, dict) or not isinstance(item.get("id"), str):
            raise ValueError(f"{path}: invalid genre at position {position}")
        copy = dict(item)
        copy["_position"] = position
        ordered.append(copy)
        by_id[copy["id"]] = copy

    meta = raw.get("taxonomy")
    version = meta.get("version") if isinstance(meta, dict) else None
    return ordered, by_id, version


def root_for(label_id: str, taxonomy: dict[str, dict[str, Any]]) -> str:
    if label_id not in taxonomy:
        raise ValueError(f"unknown taxonomy label {label_id!r}")

    current = label_id
    seen: set[str] = set()

    while True:
        if current in seen:
            raise ValueError(f"taxonomy cycle detected at {label_id!r}")
        seen.add(current)

        parent = taxonomy[current].get("parent")
        if not isinstance(parent, str) or not parent:
            return current
        if parent not in taxonomy:
            raise ValueError(f"{current!r} references unknown parent {parent!r}")
        current = parent


def parent_targets(record: dict[str, Any], taxonomy: dict[str, dict[str, Any]]) -> set[str]:
    labels = record.get("labels")
    if not isinstance(labels, list):
        return set()

    return {
        root_for(label, taxonomy)
        for label in labels
        if isinstance(label, str) and label
    }


def support_for(path: Path, taxonomy: dict[str, dict[str, Any]]) -> tuple[int, Counter[str]]:
    records = load_jsonl(path)
    counts: Counter[str] = Counter()
    for record in records:
        counts.update(parent_targets(record, taxonomy))
    return len(records), counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create parent-only EDM class set.")
    parser.add_argument("--taxonomy", type=Path, default=DEFAULT_TAXONOMY)
    parser.add_argument("--splits-dir", type=Path, default=DEFAULT_SPLITS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ordered, taxonomy, version = load_taxonomy(args.taxonomy)

    roots = [item for item in ordered if not item.get("parent")]
    if not roots:
        raise ValueError("taxonomy has no root classes")

    root_ids = {item["id"] for item in roots}

    # Verify every taxonomy node resolves to one of these roots.
    for genre_id in taxonomy:
        if root_for(genre_id, taxonomy) not in root_ids:
            raise ValueError(f"{genre_id!r} does not resolve to a known root")

    supports: dict[str, Counter[str]] = {}
    sample_counts: dict[str, int] = {}

    for strategy in ("regular", "artist"):
        for split in ("train", "validation", "test"):
            key = f"{strategy}_{split}"
            count, support = support_for(
                args.splits_dir / strategy / f"{split}.jsonl",
                taxonomy,
            )
            sample_counts[key] = count
            supports[key] = support

    classes = []
    for index, item in enumerate(roots):
        genre_id = item["id"]
        classes.append(
            {
                "index": index,
                "id": genre_id,
                "label": item.get("label", genre_id),
                "parent": None,
                "depth": 0,
                "is_leaf": False,
                "regular_train_direct": supports["regular_train"][genre_id],
                "artist_train_direct": supports["artist_train"][genre_id],
                "regular_train_expanded": supports["regular_train"][genre_id],
                "artist_train_expanded": supports["artist_train"][genre_id],
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    active_path = args.output_dir / "active_classes.json"
    distribution_path = args.output_dir / "class_distribution.csv"
    report_path = args.output_dir / "parent_selection_report.json"

    payload = {
        "taxonomy_version": version,
        "selection_version": 1,
        "task": "parent_only_multi_label",
        "class_count": len(classes),
        "policy": {
            "targets": "taxonomy_roots",
            "child_labels_propagated_to_root": True,
            "same_class_set_for_regular_and_artist": True,
        },
        "classes": classes,
    }
    active_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    fieldnames = [
        "index", "id", "label",
        "regular_train", "regular_validation", "regular_test",
        "artist_train", "artist_validation", "artist_test",
    ]
    with distribution_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, item in enumerate(roots):
            genre_id = item["id"]
            writer.writerow(
                {
                    "index": index,
                    "id": genre_id,
                    "label": item.get("label", genre_id),
                    "regular_train": supports["regular_train"][genre_id],
                    "regular_validation": supports["regular_validation"][genre_id],
                    "regular_test": supports["regular_test"][genre_id],
                    "artist_train": supports["artist_train"][genre_id],
                    "artist_validation": supports["artist_validation"][genre_id],
                    "artist_test": supports["artist_test"][genre_id],
                }
            )

    zero_support = {}
    for key, support in supports.items():
        zero_support[key] = [
            genre_id for genre_id in root_ids if support[genre_id] == 0
        ]

    report = {
        "taxonomy_version": version,
        "parent_class_count": len(classes),
        "parent_class_ids": [item["id"] for item in classes],
        "sample_counts": sample_counts,
        "zero_support_parent_classes": zero_support,
        "outputs": {
            "active_classes": str(active_path),
            "class_distribution": str(distribution_path),
        },
    }
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("Parent-only class selection complete")
    print(f"  parent classes: {len(classes)}")
    print()
    for key in (
        "regular_train", "regular_validation", "regular_test",
        "artist_train", "artist_validation", "artist_test",
    ):
        print(
            f"  {key:20s} samples={sample_counts[key]:5d} "
            f"zero-support parents={len(zero_support[key])}"
        )
    print()
    print(f"Classes:      {active_path}")
    print(f"Distribution: {distribution_path}")
    print(f"Report:       {report_path}")


if __name__ == "__main__":
    main()
