"""Evaluate a single-label parent-genre classifier.

Reports accuracy, macro/weighted F1, per-class metrics, confusion matrices,
top confusions, and a confidence summary. Predictions are softmax argmax.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from edm_classifier.training.train_single_label import (
    DEFAULT_DATA_DIR,
    DEFAULT_RUNS_DIR,
    EmbeddingDataset,
    build_model,
    choose_device,
    load_classes,
    load_jsonl,
)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def predict(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    probs = []
    pred = []
    true = []

    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        p = torch.softmax(logits, dim=1)

        probs.append(p.cpu().numpy())
        pred.append(torch.argmax(p, dim=1).cpu().numpy())
        true.append(y.numpy())

    return (
        np.concatenate(probs),
        np.concatenate(pred),
        np.concatenate(true),
    )


def class_metrics(
    true: np.ndarray,
    pred: np.ndarray,
    classes: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], np.ndarray]:
    n = len(classes)
    matrix = np.zeros((n, n), dtype=np.int64)

    for t, p in zip(true, pred):
        matrix[int(t), int(p)] += 1

    rows = []
    for i, item in enumerate(classes):
        tp = int(matrix[i, i])
        fp = int(matrix[:, i].sum() - tp)
        fn = int(matrix[i, :].sum() - tp)
        support = int(matrix[i, :].sum())
        predicted = int(matrix[:, i].sum())

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
                "source": item.get("source"),
                "support": support,
                "predicted": predicted,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "correct": tp,
            }
        )

    return rows, matrix


def aggregate_metrics(
    rows: list[dict[str, Any]],
    matrix: np.ndarray,
) -> dict[str, Any]:
    support = np.asarray([row["support"] for row in rows], dtype=np.float64)
    f1 = np.asarray([row["f1"] for row in rows], dtype=np.float64)

    supported = support > 0
    retained = np.asarray(
        [row["id"] != "other" for row in rows],
        dtype=bool,
    ) & supported

    total = int(matrix.sum())
    correct = int(np.trace(matrix))

    return {
        "accuracy": correct / total if total else 0.0,
        "macro_f1": float(f1[supported].mean()) if supported.any() else 0.0,
        "macro_f1_excluding_other": (
            float(f1[retained].mean()) if retained.any() else 0.0
        ),
        "weighted_f1": (
            float(np.sum(f1 * support) / support.sum())
            if support.sum()
            else 0.0
        ),
        "samples": total,
        "correct": correct,
        "supported_classes": int(supported.sum()),
        "retained_supported_classes": int(retained.sum()),
    }


def confusion_rows(
    matrix: np.ndarray,
    classes: list[dict[str, Any]],
    *,
    normalized: bool,
) -> list[dict[str, Any]]:
    rows = []
    for i, item in enumerate(classes):
        total = int(matrix[i].sum())
        row: dict[str, Any] = {
            "true_id": item["id"],
            "true_label": item.get("label", item["id"]),
        }

        for j, other in enumerate(classes):
            if normalized:
                value: float | int = (
                    float(matrix[i, j] / total) if total else 0.0
                )
            else:
                value = int(matrix[i, j])
            row[other["id"]] = value

        rows.append(row)

    return rows


def top_confusions(
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
            key=lambda x: x[1],
            reverse=True,
        )

        for rank, (j, count) in enumerate(ranked[:top_k], 1):
            other = classes[j]
            rows.append(
                {
                    "true_id": item["id"],
                    "true_label": item.get("label", item["id"]),
                    "support": support,
                    "correct": correct,
                    "accuracy": correct / support if support else 0.0,
                    "rank": rank,
                    "confused_with_id": other["id"],
                    "confused_with_label": other.get("label", other["id"]),
                    "count": count,
                    "fraction_of_true_class": (
                        count / support if support else 0.0
                    ),
                }
            )

    return rows


def plot_confusion(
    path: Path,
    matrix: np.ndarray,
    classes: list[dict[str, Any]],
) -> None:
    normalized = matrix.astype(np.float64)
    row_sum = normalized.sum(axis=1, keepdims=True)
    normalized = np.divide(
        normalized,
        row_sum,
        out=np.zeros_like(normalized),
        where=row_sum > 0,
    )

    labels = [item.get("label", item["id"]) for item in classes]

    fig, ax = plt.subplots(figsize=(13, 11))
    image = ax.imshow(normalized, aspect="auto", vmin=0.0, vmax=1.0)
    fig.colorbar(image, ax=ax, label="Fraction of true class")

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("Single-label parent genre confusion matrix")

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate single-label parent classifier."
    )
    parser.add_argument(
        "--split",
        choices=["regular", "artist"],
        required=True,
    )
    parser.add_argument(
        "--model",
        choices=["linear", "mlp"],
        default="mlp",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=DEFAULT_RUNS_DIR,
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    classes = load_classes(args.data_dir / "classes.json")
    class_count = len(classes)

    run_dir = args.runs_dir / f"{args.split}_{args.model}"
    checkpoint_path = run_dir / "model.pt"
    normalization_path = run_dir / "normalization.npz"

    if not checkpoint_path.is_file():
        raise SystemExit(f"missing checkpoint: {checkpoint_path}")
    if not normalization_path.is_file():
        raise SystemExit(f"missing normalization: {normalization_path}")

    device = choose_device(args.device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    expected_ids = [item["id"] for item in classes]
    if checkpoint.get("class_ids") != expected_ids:
        raise ValueError(
            "checkpoint classes do not match current data/parent_single/classes.json"
        )

    normalization = np.load(normalization_path, allow_pickle=False)
    mean = np.asarray(normalization["mean"], dtype=np.float32)
    std = np.asarray(normalization["std"], dtype=np.float32)

    model = build_model(
        checkpoint["model_name"],
        input_dim=int(checkpoint["input_dim"]),
        output_dim=int(checkpoint["output_dim"]),
        hidden_dim=int(checkpoint.get("hidden_dim", 512)),
        dropout=float(checkpoint.get("dropout", 0.2)),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])

    test_rows = load_jsonl(
        args.data_dir / args.split / "test.jsonl"
    )
    dataset = EmbeddingDataset(
        test_rows,
        mean=mean,
        std=std,
        class_count=class_count,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    probs, pred, true = predict(model, loader, device)

    per_class, matrix = class_metrics(true, pred, classes)
    aggregate = aggregate_metrics(per_class, matrix)

    write_csv(run_dir / "test_per_class.csv", per_class)
    write_csv(
        run_dir / "confusion_counts.csv",
        confusion_rows(matrix, classes, normalized=False),
    )
    write_csv(
        run_dir / "confusion_row_normalized.csv",
        confusion_rows(matrix, classes, normalized=True),
    )
    write_csv(
        run_dir / "top_confusions.csv",
        top_confusions(matrix, classes, args.top_k),
    )
    plot_confusion(
        run_dir / "confusion_heatmap.png",
        matrix,
        classes,
    )

    max_prob = probs.max(axis=1)
    confidence_summary = {
        "mean_max_probability": float(max_prob.mean()),
        "median_max_probability": float(np.median(max_prob)),
        "correct_mean_max_probability": (
            float(max_prob[pred == true].mean())
            if np.any(pred == true)
            else 0.0
        ),
        "incorrect_mean_max_probability": (
            float(max_prob[pred != true].mean())
            if np.any(pred != true)
            else 0.0
        ),
    }

    report = {
        "run_name": f"{args.split}_{args.model}",
        "task": "single_label_parent_genre",
        "encoder": checkpoint.get("encoder"),
        "pooling": checkpoint.get("pooling"),
        "split": args.split,
        "model": args.model,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "class_weight_mode": checkpoint.get("class_weight_mode"),
        "class_count": class_count,
        "test": aggregate,
        "confidence": confidence_summary,
        "outputs": {
            "per_class": str(run_dir / "test_per_class.csv"),
            "confusion_counts": str(run_dir / "confusion_counts.csv"),
            "confusion_normalized": str(
                run_dir / "confusion_row_normalized.csv"
            ),
            "top_confusions": str(run_dir / "top_confusions.csv"),
            "heatmap": str(run_dir / "confusion_heatmap.png"),
        },
    }

    report_path = run_dir / "evaluation_report.json"
    report_path.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print("Single-label evaluation")
    print(f"  split:                  {args.split}")
    print(f"  model:                  {args.model}")
    print(f"  samples:                {aggregate['samples']}")
    print(f"  accuracy:               {aggregate['accuracy']:.4f}")
    print(f"  macro F1:               {aggregate['macro_f1']:.4f}")
    print(
        f"  macro F1 excluding O:   "
        f"{aggregate['macro_f1_excluding_other']:.4f}"
    )
    print(f"  weighted F1:            {aggregate['weighted_f1']:.4f}")
    print()

    weakest = sorted(per_class, key=lambda row: row["f1"])[:6]
    strongest = sorted(
        per_class,
        key=lambda row: row["f1"],
        reverse=True,
    )[:6]

    print("Weakest classes:")
    for row in weakest:
        print(
            f"  {row['id']:18s} "
            f"support={row['support']:4d} "
            f"F1={row['f1']:.4f}"
        )

    print()
    print("Strongest classes:")
    for row in strongest:
        print(
            f"  {row['id']:18s} "
            f"support={row['support']:4d} "
            f"F1={row['f1']:.4f}"
        )

    print()
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
