"""Build a single-label parent-genre dataset from existing splits.

The builder maps every direct taxonomy label to its root parent, then produces
one target class per physical sample.

Selection is driven by a JSON file produced by inspect_parent_errors.py.

Default policies:
- exactly one retained parent -> that parent
- exactly one excluded parent -> other
- multiple parent roots -> other
- zero parent roots -> other
- excluded/multi-parent rows are NOT discarded unless explicitly requested

This preserves the existing regular/artist split membership and embedding paths.

Outputs:
    data/parent_single/
      classes.json
      build_report.json
      artist/{train,validation,test}.jsonl
      regular/{train,validation,test}.jsonl
      class_distribution.csv

Each output sample preserves the source record and adds:
    target_label
    target_index
    source_parent_targets
    single_label_reason

For compatibility/readability it also sets:
    labels = [target_label]

A future single-label trainer should use target_index with CrossEntropyLoss.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import yaml


DEFAULT_SPLITS_DIR = Path("data/splits")
DEFAULT_TAXONOMY = Path("config/taxonomy.yaml")
DEFAULT_SELECTION = Path(
    "data/runs/parent_analysis/retention_selection.json"
)
DEFAULT_OUTPUT_DIR = Path("data/parent_single")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []

    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            line = raw.strip()
            if not line:
                continue

            value = json.loads(line)

            if not isinstance(value, dict):
                raise ValueError(
                    f"{path}:{line_number}: expected JSON object"
                )

            rows.append(value)

    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


def load_taxonomy(path: Path) -> dict[str, dict[str, Any]]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    genres = raw.get("genres") if isinstance(raw, dict) else None

    if not isinstance(genres, list):
        raise ValueError(f"{path}: expected genres list")

    result = {}

    for item in genres:
        if not isinstance(item, dict):
            raise ValueError(f"{path}: invalid genre")

        genre_id = item.get("id")

        if not isinstance(genre_id, str) or not genre_id:
            raise ValueError(f"{path}: invalid genre ID")

        result[genre_id] = dict(item)

    return result


def root_for(
    label_id: str,
    taxonomy: dict[str, dict[str, Any]],
) -> str:
    if label_id not in taxonomy:
        raise ValueError(f"unknown taxonomy label {label_id!r}")

    current = label_id
    seen: set[str] = set()

    while True:
        if current in seen:
            raise ValueError(f"taxonomy cycle at {label_id!r}")
        seen.add(current)

        parent = taxonomy[current].get("parent")

        if not isinstance(parent, str) or not parent:
            return current

        if parent not in taxonomy:
            raise ValueError(f"unknown parent {parent!r}")

        current = parent


def parent_targets(
    row: dict[str, Any],
    taxonomy: dict[str, dict[str, Any]],
) -> tuple[str, ...]:
    labels = row.get("labels")

    if not isinstance(labels, list):
        return ()

    return tuple(
        sorted(
            {
                root_for(label, taxonomy)
                for label in labels
                if isinstance(label, str) and label
            }
        )
    )


def load_selection(
    path: Path,
) -> tuple[
    list[dict[str, Any]],
    set[str],
    str | None,
    str,
]:
    raw = json.loads(path.read_text(encoding="utf-8"))

    classes = raw.get("classes")
    if not isinstance(classes, list):
        raise ValueError(f"{path}: missing classes list")

    kept = []
    kept_ids = set()

    for item in classes:
        if not isinstance(item, dict):
            continue

        class_id = item.get("id")
        keep = item.get("keep")

        if not isinstance(class_id, str):
            continue

        if keep is True:
            kept.append(item)
            kept_ids.add(class_id)

    other = raw.get("other")
    other_id: str | None = None
    other_label = "Other"

    if isinstance(other, dict) and other.get("enabled") is True:
        candidate = other.get("id")
        if not isinstance(candidate, str) or not candidate:
            raise ValueError(f"{path}: invalid other.id")
        other_id = candidate

        label = other.get("label")
        if isinstance(label, str) and label:
            other_label = label

    if not kept:
        raise ValueError(f"{path}: no retained parent classes")

    return kept, kept_ids, other_id, other_label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build single-label parent dataset."
    )

    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=DEFAULT_SPLITS_DIR,
    )
    parser.add_argument(
        "--taxonomy",
        type=Path,
        default=DEFAULT_TAXONOMY,
    )
    parser.add_argument(
        "--selection",
        type=Path,
        default=DEFAULT_SELECTION,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--ambiguous-policy",
        choices=["other", "drop"],
        default="other",
        help="How to handle samples resolving to 2+ parent roots.",
    )
    parser.add_argument(
        "--excluded-policy",
        choices=["other", "drop"],
        default="other",
        help="How to handle samples belonging to a dropped parent.",
    )
    parser.add_argument(
        "--zero-parent-policy",
        choices=["other", "drop"],
        default="other",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    taxonomy = load_taxonomy(args.taxonomy)

    kept_items, kept_ids, other_id, other_label = load_selection(
        args.selection
    )

    if (
        "other" in {
            args.ambiguous_policy,
            args.excluded_policy,
            args.zero_parent_policy,
        }
        and other_id is None
    ):
        raise SystemExit(
            "An 'other' policy is enabled, but selection JSON has "
            "other.enabled=false."
        )

    # Preserve selection-file ordering.
    output_classes: list[dict[str, Any]] = []

    for item in kept_items:
        class_id = item["id"]
        taxonomy_item = taxonomy.get(class_id, {})

        output_classes.append(
            {
                "index": len(output_classes),
                "id": class_id,
                "label": item.get(
                    "label",
                    taxonomy_item.get("label", class_id),
                ),
                "source": "retained_parent",
            }
        )

    if other_id is not None:
        if other_id in kept_ids:
            raise ValueError(
                f"other ID {other_id!r} conflicts with retained class"
            )

        output_classes.append(
            {
                "index": len(output_classes),
                "id": other_id,
                "label": other_label,
                "source": "catch_all",
            }
        )

    index_by_id = {
        item["id"]: item["index"]
        for item in output_classes
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)

    class_payload = {
        "task": "single_label_parent_genre",
        "loss": "cross_entropy",
        "prediction": "softmax_argmax",
        "class_count": len(output_classes),
        "classes": output_classes,
        "selection_file": str(args.selection),
        "policies": {
            "ambiguous": args.ambiguous_policy,
            "excluded_parent": args.excluded_policy,
            "zero_parent": args.zero_parent_policy,
        },
    }

    classes_path = args.output_dir / "classes.json"
    classes_path.write_text(
        json.dumps(class_payload, indent=2, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )

    distribution_rows: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "task": "single_label_parent_genre",
        "classes": [item["id"] for item in output_classes],
        "retained_parent_classes": [
            item["id"] for item in output_classes
            if item["source"] == "retained_parent"
        ],
        "other_class": other_id,
        "policies": class_payload["policies"],
        "splits": {},
    }

    for strategy in ("regular", "artist"):
        report["splits"][strategy] = {}

        for split in ("train", "validation", "test"):
            source_path = (
                args.splits_dir
                / strategy
                / f"{split}.jsonl"
            )
            records = load_jsonl(source_path)

            output_rows = []
            reasons = Counter()
            class_counts = Counter()
            source_parent_counts = Counter()

            for record in records:
                targets = parent_targets(record, taxonomy)
                source_parent_counts.update(targets)

                target_id: str | None
                reason: str

                if len(targets) == 0:
                    if args.zero_parent_policy == "drop":
                        target_id = None
                        reason = "dropped_zero_parent"
                    else:
                        target_id = other_id
                        reason = "other_zero_parent"

                elif len(targets) >= 2:
                    if args.ambiguous_policy == "drop":
                        target_id = None
                        reason = "dropped_multi_parent"
                    else:
                        target_id = other_id
                        reason = "other_multi_parent"

                else:
                    parent_id = targets[0]

                    if parent_id in kept_ids:
                        target_id = parent_id
                        reason = "retained_parent"
                    elif args.excluded_policy == "drop":
                        target_id = None
                        reason = "dropped_excluded_parent"
                    else:
                        target_id = other_id
                        reason = "other_excluded_parent"

                reasons[reason] += 1

                if target_id is None:
                    continue

                if target_id not in index_by_id:
                    raise ValueError(
                        f"target {target_id!r} not in output classes"
                    )

                item = dict(record)

                # Preserve original fine labels for audit before replacing
                # the training-facing labels field.
                item["source_labels"] = record.get("labels", [])
                item["source_parent_targets"] = list(targets)
                item["target_label"] = target_id
                item["target_index"] = index_by_id[target_id]
                item["single_label_reason"] = reason
                item["labels"] = [target_id]

                output_rows.append(item)
                class_counts[target_id] += 1

            output_path = (
                args.output_dir
                / strategy
                / f"{split}.jsonl"
            )
            write_jsonl(output_path, output_rows)

            for class_item in output_classes:
                class_id = class_item["id"]

                distribution_rows.append(
                    {
                        "strategy": strategy,
                        "split": split,
                        "index": class_item["index"],
                        "id": class_id,
                        "label": class_item["label"],
                        "support": class_counts[class_id],
                        "total_samples": len(output_rows),
                        "fraction": (
                            class_counts[class_id] / len(output_rows)
                            if output_rows
                            else 0.0
                        ),
                    }
                )

            report["splits"][strategy][split] = {
                "input_samples": len(records),
                "output_samples": len(output_rows),
                "dropped_samples": len(records) - len(output_rows),
                "reason_counts": dict(sorted(reasons.items())),
                "class_support": {
                    class_item["id"]: class_counts[class_item["id"]]
                    for class_item in output_classes
                },
                "source_parent_support": dict(
                    sorted(source_parent_counts.items())
                ),
                "output": str(output_path),
            }

    distribution_path = (
        args.output_dir / "class_distribution.csv"
    )
    write_csv(distribution_path, distribution_rows)

    report["outputs"] = {
        "classes": str(classes_path),
        "class_distribution": str(distribution_path),
    }

    report_path = args.output_dir / "build_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("Single-label parent dataset built")
    print(
        f"  retained parents: "
        f"{len([x for x in output_classes if x['source'] == 'retained_parent'])}"
    )
    print(
        f"  other enabled:    {other_id is not None}"
    )
    print(
        f"  output classes:   {len(output_classes)}"
    )
    print()

    for strategy in ("regular", "artist"):
        print(strategy)
        for split in ("train", "validation", "test"):
            info = report["splits"][strategy][split]
            print(
                f"  {split:10s} "
                f"{info['output_samples']:5d}/"
                f"{info['input_samples']:5d} samples"
            )

    print()
    print(f"Classes:      {classes_path}")
    print(f"Distribution: {distribution_path}")
    print(f"Report:       {report_path}")


if __name__ == "__main__":
    main()
