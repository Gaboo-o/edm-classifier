"""Inspect the existing artist-split parent-only classifier.

This is a diagnostic for redesigning the parent task as single-label.

It uses the already-trained parent-only MLP and:
- reports the existing multilabel per-parent test F1
- evaluates an argmax confusion matrix on ONLY single-parent test samples
- identifies each parent's largest confusions
- writes a retention_selection.json template for the next builder

Important:
The existing model was trained with sigmoid/BCE. Argmax is used here only as a
diagnostic of which broad parent classes compete with one another. It is not
reported as the final single-label model result.

Defaults:
    split:          data/splits/artist/test.jsonl
    active classes: data/training_parent/active_classes.json
    run:            data/runs/parent/artist_mlp
    output:         data/runs/parent_analysis

Outputs:
    per_parent_artist_test.csv
    confusion_counts.csv
    confusion_row_normalized.csv
    top_confusions.csv
    confusion_heatmap.png
    retention_selection.json
    report.json
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from edm_classifier.training.evaluate import (
    build_model,
    choose_device,
    load_active_classes,
    load_taxonomy,
    load_xy,
    predict,
)


DEFAULT_SPLITS_DIR = Path("data/splits")
DEFAULT_ACTIVE_CLASSES = Path("data/training_parent/active_classes.json")
DEFAULT_TAXONOMY = Path("config/taxonomy.yaml")
DEFAULT_RUN_DIR = Path("data/runs/parent/artist_mlp")
DEFAULT_OUTPUT_DIR = Path("data/runs/parent_analysis")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            line = raw.strip()
            if not line:
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            rows.append(value)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


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


def selected_threshold(run_dir: Path) -> float | None:
    for name in ("evaluation_report_v2.json", "evaluation_report.json"):
        path = run_dir / name
        if not path.is_file():
            continue

        report = json.loads(path.read_text(encoding="utf-8"))

        results = report.get("results")
        if isinstance(results, dict):
            raw = results.get("raw")
            if isinstance(raw, dict) and isinstance(
                raw.get("threshold"), (int, float)
            ):
                return float(raw["threshold"])

        policy = report.get("threshold_policy")
        if isinstance(policy, dict) and isinstance(
            policy.get("selected_threshold"), (int, float)
        ):
            return float(policy["selected_threshold"])

    return None


def existing_per_class_path(run_dir: Path) -> Path | None:
    candidates = [
        run_dir / "test_per_class_raw.csv",
        run_dir / "test_per_class.csv",
    ]
    return next((path for path in candidates if path.is_file()), None)


def load_existing_per_class(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}

    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            class_id = row.get("id")
            if not class_id:
                continue
            rows[class_id] = row

    return rows


def binary_metrics(
    probs: np.ndarray,
    targets: np.ndarray,
    threshold: float,
    classes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    pred = probs >= threshold
    true = targets >= 0.5

    rows = []

    for i, item in enumerate(classes):
        tp = int(np.sum(pred[:, i] & true[:, i]))
        fp = int(np.sum(pred[:, i] & ~true[:, i]))
        fn = int(np.sum(~pred[:, i] & true[:, i]))
        tn = int(np.sum(~pred[:, i] & ~true[:, i]))

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )

        rows.append(
            {
                "index": i,
                "id": item["id"],
                "label": item.get("label", item["id"]),
                "support": int(np.sum(true[:, i])),
                "predicted_positive": int(np.sum(pred[:, i])),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
            }
        )

    return rows


def confusion_rows(
    matrix: np.ndarray,
    classes: list[dict[str, Any]],
    normalized: bool,
) -> list[dict[str, Any]]:
    rows = []

    for i, item in enumerate(classes):
        total = matrix[i].sum()
        row: dict[str, Any] = {
            "true_id": item["id"],
            "true_label": item.get("label", item["id"]),
        }

        for j, other in enumerate(classes):
            value: float | int
            if normalized:
                value = float(matrix[i, j] / total) if total else 0.0
            else:
                value = int(matrix[i, j])

            row[other["id"]] = value

        rows.append(row)

    return rows


def top_confusion_rows(
    matrix: np.ndarray,
    classes: list[dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    rows = []

    for i, item in enumerate(classes):
        support = int(matrix[i].sum())
        correct = int(matrix[i, i])
        ranked = sorted(
            (
                (j, int(matrix[i, j]))
                for j in range(len(classes))
                if j != i
            ),
            key=lambda pair: pair[1],
            reverse=True,
        )

        for rank, (j, count) in enumerate(ranked[:top_k], 1):
            other = classes[j]
            rows.append(
                {
                    "true_id": item["id"],
                    "true_label": item.get("label", item["id"]),
                    "true_support_single_parent": support,
                    "argmax_correct": correct,
                    "argmax_accuracy": (
                        correct / support if support else 0.0
                    ),
                    "rank": rank,
                    "confused_with_id": other["id"],
                    "confused_with_label": other.get(
                        "label", other["id"]
                    ),
                    "count": count,
                    "fraction_of_true_class": (
                        count / support if support else 0.0
                    ),
                }
            )

    return rows


def plot_heatmap(
    path: Path,
    matrix: np.ndarray,
    classes: list[dict[str, Any]],
) -> None:
    normalized = matrix.astype(np.float64)
    row_sums = normalized.sum(axis=1, keepdims=True)
    normalized = np.divide(
        normalized,
        row_sums,
        out=np.zeros_like(normalized),
        where=row_sums > 0,
    )

    labels = [
        item.get("label", item["id"])
        for item in classes
    ]

    fig, ax = plt.subplots(figsize=(13, 11))
    image = ax.imshow(normalized, aspect="auto", vmin=0.0, vmax=1.0)
    fig.colorbar(image, ax=ax, label="Fraction of true class")

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Argmax predicted parent")
    ax.set_ylabel("True parent")
    ax.set_title(
        "Existing parent MLP: artist-test single-parent confusion"
    )

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect artist-test parent F1 and confusion patterns."
    )
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=DEFAULT_SPLITS_DIR,
    )
    parser.add_argument(
        "--active-classes",
        type=Path,
        default=DEFAULT_ACTIVE_CLASSES,
    )
    parser.add_argument(
        "--taxonomy",
        type=Path,
        default=DEFAULT_TAXONOMY,
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=DEFAULT_RUN_DIR,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--top-k", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    classes = load_active_classes(args.active_classes)
    class_to_index = {
        item["id"]: item["index"]
        for item in classes
    }
    taxonomy = load_taxonomy(args.taxonomy)

    checkpoint_path = args.run_dir / "model.pt"
    normalization_path = args.run_dir / "normalization.npz"

    if not checkpoint_path.is_file():
        raise SystemExit(f"missing checkpoint: {checkpoint_path}")
    if not normalization_path.is_file():
        raise SystemExit(
            f"missing normalization: {normalization_path}"
        )

    normalization = np.load(normalization_path, allow_pickle=False)
    mean = np.asarray(normalization["mean"], dtype=np.float32)
    std = np.asarray(normalization["std"], dtype=np.float32)

    device = choose_device(args.device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    expected_class_ids = [item["id"] for item in classes]
    if checkpoint.get("class_ids") != expected_class_ids:
        raise ValueError(
            "checkpoint class ordering does not match active classes"
        )

    model = build_model(checkpoint)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)

    test_path = args.splits_dir / "artist" / "test.jsonl"

    x_test, y_test, test_records = load_xy(
        test_path,
        class_to_index=class_to_index,
        taxonomy=taxonomy,
        mean=mean,
        std=std,
    )

    probs = predict(
        model,
        x_test,
        batch_size=args.batch_size,
        device=device,
    )

    threshold = selected_threshold(args.run_dir)
    existing_path = existing_per_class_path(args.run_dir)

    if existing_path is not None:
        existing = load_existing_per_class(existing_path)
        per_parent = []

        for item in classes:
            row = existing.get(item["id"])
            if row is None:
                continue

            per_parent.append(
                {
                    "index": item["index"],
                    "id": item["id"],
                    "label": item.get("label", item["id"]),
                    "support": int(float(row.get("support", 0))),
                    "predicted_positive": int(
                        float(row.get("predicted_positive", 0))
                    ),
                    "precision": float(row.get("precision", 0.0)),
                    "recall": float(row.get("recall", 0.0)),
                    "f1": float(row.get("f1", 0.0)),
                    "tp": int(float(row.get("tp", 0))),
                    "fp": int(float(row.get("fp", 0))),
                    "fn": int(float(row.get("fn", 0))),
                    "tn": int(float(row.get("tn", 0))),
                }
            )
    else:
        if threshold is None:
            raise SystemExit(
                "No per-class CSV and no evaluation threshold found."
            )
        per_parent = binary_metrics(
            probs,
            y_test,
            threshold,
            classes,
        )

    per_parent.sort(key=lambda row: row["f1"])

    single_indices = []
    true_indices = []

    class_id_to_index = {
        item["id"]: item["index"]
        for item in classes
    }

    multi_parent_samples = 0
    zero_parent_samples = 0

    for i, record in enumerate(test_records):
        targets = parent_targets(record, taxonomy)

        if len(targets) == 1:
            target = targets[0]
            if target in class_id_to_index:
                single_indices.append(i)
                true_indices.append(class_id_to_index[target])
        elif len(targets) == 0:
            zero_parent_samples += 1
        else:
            multi_parent_samples += 1

    single_probs = probs[np.asarray(single_indices)]
    predicted_indices = np.argmax(single_probs, axis=1)
    true_indices_array = np.asarray(true_indices, dtype=np.int64)

    matrix = np.zeros(
        (len(classes), len(classes)),
        dtype=np.int64,
    )

    for true_index, pred_index in zip(
        true_indices_array,
        predicted_indices,
    ):
        matrix[true_index, pred_index] += 1

    top_confusions = top_confusion_rows(
        matrix,
        classes,
        args.top_k,
    )

    confusion_accuracy = float(
        np.mean(predicted_indices == true_indices_array)
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    write_csv(
        args.output_dir / "per_parent_artist_test.csv",
        per_parent,
    )
    write_csv(
        args.output_dir / "confusion_counts.csv",
        confusion_rows(matrix, classes, normalized=False),
    )
    write_csv(
        args.output_dir / "confusion_row_normalized.csv",
        confusion_rows(matrix, classes, normalized=True),
    )
    write_csv(
        args.output_dir / "top_confusions.csv",
        top_confusions,
    )

    plot_heatmap(
        args.output_dir / "confusion_heatmap.png",
        matrix,
        classes,
    )

    # Deliberately do not auto-drop classes based on F1 alone.
    retention = {
        "selection_version": 1,
        "purpose": "single_label_parent_class_selection",
        "instructions": (
            "Set keep=false for parent genres to map into 'other'. "
            "Do not decide from F1 alone; consider relevance, support, "
            "and confusion pattern."
        ),
        "other": {
            "enabled": True,
            "id": "other",
            "label": "Other",
        },
        "classes": [
            {
                "id": row["id"],
                "label": row["label"],
                "keep": True,
                "artist_test_support": row["support"],
                "artist_test_f1_existing_multilabel": row["f1"],
            }
            for row in sorted(
                per_parent,
                key=lambda row: row["index"],
            )
        ],
    }

    retention_path = (
        args.output_dir / "retention_selection.json"
    )
    retention_path.write_text(
        json.dumps(retention, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    report = {
        "source_model": str(args.run_dir),
        "source_task": "19-parent multilabel sigmoid/BCE",
        "test_samples": len(test_records),
        "single_parent_test_samples": len(single_indices),
        "multi_parent_test_samples_excluded_from_confusion": (
            multi_parent_samples
        ),
        "zero_parent_test_samples": zero_parent_samples,
        "single_parent_argmax_accuracy_diagnostic": (
            confusion_accuracy
        ),
        "selected_multilabel_threshold": threshold,
        "per_class_source": (
            str(existing_path)
            if existing_path is not None
            else "recomputed_from_checkpoint"
        ),
        "weakest_existing_multilabel_f1": per_parent[:5],
        "strongest_existing_multilabel_f1": list(
            reversed(per_parent[-5:])
        ),
        "outputs": {
            "per_parent": str(
                args.output_dir / "per_parent_artist_test.csv"
            ),
            "confusion_counts": str(
                args.output_dir / "confusion_counts.csv"
            ),
            "confusion_normalized": str(
                args.output_dir / "confusion_row_normalized.csv"
            ),
            "top_confusions": str(
                args.output_dir / "top_confusions.csv"
            ),
            "heatmap": str(
                args.output_dir / "confusion_heatmap.png"
            ),
            "retention_selection": str(retention_path),
        },
    }

    report_path = args.output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("Parent error analysis complete")
    print()
    print(
        f"Single-parent test samples: {len(single_indices)} / "
        f"{len(test_records)}"
    )
    print(
        f"Argmax accuracy diagnostic: {confusion_accuracy:.4f}"
    )
    print()
    print("Weakest existing parent F1:")
    for row in per_parent[:8]:
        print(
            f"  {row['id']:18s} "
            f"support={row['support']:4d} "
            f"F1={row['f1']:.4f}"
        )
    print()
    print("Strongest existing parent F1:")
    for row in reversed(per_parent[-8:]):
        print(
            f"  {row['id']:18s} "
            f"support={row['support']:4d} "
            f"F1={row['f1']:.4f}"
        )
    print()
    print(f"Per-parent: {args.output_dir / 'per_parent_artist_test.csv'}")
    print(f"Confusions: {args.output_dir / 'top_confusions.csv'}")
    print(f"Heatmap:    {args.output_dir / 'confusion_heatmap.png'}")
    print(f"Selection:  {retention_path}")
    print(f"Report:     {report_path}")


if __name__ == "__main__":
    main()
