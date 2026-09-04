"""Evaluate single-label parent-genre classifiers across EDM ablations."""
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

from edm_classifier.training.train import (
    DEFAULT_DATA_DIR,
    DEFAULT_FEATURES_DIR,
    DEFAULT_RUNS_DIR,
    ENCODER_MODEL_NAMES,
    RHYTHM_CHOICES,
    VIEW_CHOICES,
    EmbeddingDataset,
    EmbeddingResolver,
    FeatureResolver,
    build_model,
    choose_device,
    collate_sequences,
    load_classes,
    load_jsonl,
    make_run_name,
    representation_plan,
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
def predict(model, loader, device, model_representation):
    model.eval()
    probs: list[np.ndarray] = []
    pred: list[np.ndarray] = []
    true: list[np.ndarray] = []
    for batch in loader:
        if model_representation == "sequence":
            x, mask, rhythm, y = batch
            logits = model(x.to(device), mask.to(device), rhythm.to(device))
        else:
            x, y = batch
            logits = model(x.to(device))
        probability = torch.softmax(logits, dim=1)
        probs.append(probability.cpu().numpy())
        pred.append(torch.argmax(probability, dim=1).cpu().numpy())
        true.append(y.numpy())
    return np.concatenate(probs), np.concatenate(pred), np.concatenate(true)


def class_metrics(true, pred, classes):
    n = len(classes)
    matrix = np.zeros((n, n), dtype=np.int64)
    for target, prediction in zip(true, pred):
        matrix[int(target), int(prediction)] += 1
    rows = []
    for i, item in enumerate(classes):
        tp = int(matrix[i, i])
        fp = int(matrix[:, i].sum() - tp)
        fn = int(matrix[i, :].sum() - tp)
        support = int(matrix[i, :].sum())
        predicted = int(matrix[:, i].sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append({
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
        })
    return rows, matrix


def aggregate_metrics(rows, matrix):
    support = np.asarray([row["support"] for row in rows], dtype=np.float64)
    f1 = np.asarray([row["f1"] for row in rows], dtype=np.float64)
    supported = support > 0
    retained = np.asarray([row["id"] != "other" for row in rows], dtype=bool) & supported
    total = int(matrix.sum())
    correct = int(np.trace(matrix))
    return {
        "accuracy": correct / total if total else 0.0,
        "macro_f1": float(f1[supported].mean()) if supported.any() else 0.0,
        "macro_f1_excluding_other": float(f1[retained].mean()) if retained.any() else 0.0,
        "weighted_f1": float(np.sum(f1 * support) / support.sum()) if support.sum() else 0.0,
        "samples": total,
        "correct": correct,
        "supported_classes": int(supported.sum()),
        "retained_supported_classes": int(retained.sum()),
    }


def confusion_rows(matrix, classes, normalized):
    out = []
    for i, item in enumerate(classes):
        total = int(matrix[i].sum())
        row = {"true_id": item["id"], "true_label": item.get("label", item["id"])}
        for j, other in enumerate(classes):
            row[other["id"]] = float(matrix[i, j] / total) if normalized and total else (0.0 if normalized else int(matrix[i, j]))
        out.append(row)
    return out


def top_confusions(matrix, classes, top_k):
    out = []
    for i, item in enumerate(classes):
        support = int(matrix[i].sum())
        correct = int(matrix[i, i])
        ranked = sorted(
            ((j, int(matrix[i, j])) for j in range(len(classes)) if j != i),
            key=lambda pair: pair[1],
            reverse=True,
        )
        for rank, (j, count) in enumerate(ranked[:top_k], 1):
            other = classes[j]
            out.append({
                "true_id": item["id"],
                "true_label": item.get("label", item["id"]),
                "support": support,
                "correct": correct,
                "accuracy": correct / support if support else 0.0,
                "rank": rank,
                "confused_with_id": other["id"],
                "confused_with_label": other.get("label", other["id"]),
                "count": count,
                "fraction_of_true_class": count / support if support else 0.0,
            })
    return out


def plot_confusion(path, matrix, classes):
    normalized = matrix.astype(np.float64)
    row_sum = normalized.sum(axis=1, keepdims=True)
    normalized = np.divide(normalized, row_sum, out=np.zeros_like(normalized), where=row_sum > 0)
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


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate single-label EDM ablation classifier.")
    p.add_argument("--split", choices=["regular", "artist"], required=True)
    p.add_argument("--model", choices=["linear", "mlp", "attention"], default="mlp")
    p.add_argument("--encoder", choices=sorted(ENCODER_MODEL_NAMES), default="mert95m")
    p.add_argument("--view", choices=VIEW_CHOICES, default="whole")
    p.add_argument("--rhythm", choices=RHYTHM_CHOICES, default="none")
    p.add_argument("--embeddings-dir", type=Path, default=None)
    p.add_argument("--features-dir", type=Path, default=DEFAULT_FEATURES_DIR)
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--top-k", type=int, default=3)
    return p.parse_args()


def main():
    args = parse_args()
    classes = load_classes(args.data_dir / "classes.json")
    class_count = len(classes)
    run_name = make_run_name(args.encoder, args.split, args.model, args.view, args.rhythm)
    run_dir = args.runs_dir / run_name
    checkpoint_path = run_dir / "model.pt"
    normalization_path = run_dir / "normalization.npz"
    if not checkpoint_path.is_file():
        raise SystemExit(f"missing checkpoint: {checkpoint_path}")
    if not normalization_path.is_file():
        raise SystemExit(f"missing normalization: {normalization_path}")

    device = choose_device(args.device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    for key, requested in (("encoder", args.encoder), ("view", args.view), ("rhythm", args.rhythm)):
        saved = checkpoint.get(key)
        if saved != requested:
            raise ValueError(f"checkpoint {key} is {saved!r}, but requested {requested!r}")
    expected = [item["id"] for item in classes]
    if checkpoint.get("class_ids") != expected:
        raise ValueError("checkpoint classes do not match current classes.json")

    source_representation, model_representation = representation_plan(args.encoder, args.model, args.view)
    if checkpoint.get("source_representation") != source_representation:
        raise ValueError("checkpoint source representation does not match requested configuration")
    if checkpoint.get("representation") != model_representation:
        raise ValueError("checkpoint model representation does not match requested configuration")

    normalization = np.load(normalization_path, allow_pickle=False)
    mean = np.asarray(normalization["mean"], dtype=np.float32)
    std = np.asarray(normalization["std"], dtype=np.float32)
    rhythm_mean = rhythm_std = None
    rhythm_dim = int(checkpoint.get("rhythm_dim", 0))
    if args.rhythm == "global":
        rhythm_mean = np.asarray(normalization["rhythm_mean"], dtype=np.float32)
        rhythm_std = np.asarray(normalization["rhythm_std"], dtype=np.float32)

    model = build_model(
        checkpoint["model_name"],
        input_dim=int(checkpoint["input_dim"]),
        output_dim=int(checkpoint["output_dim"]),
        hidden_dim=int(checkpoint.get("hidden_dim", 512)),
        dropout=float(checkpoint.get("dropout", 0.2)),
        rhythm_dim=rhythm_dim if model_representation == "sequence" else 0,
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])

    checkpoint_source = checkpoint.get("embedding_source")
    checkpoint_dir = None
    if isinstance(checkpoint_source, dict):
        value = checkpoint_source.get("directory") or checkpoint_source.get("pooled_dir")
        if isinstance(value, str) and value:
            checkpoint_dir = Path(value)
    feature_source = checkpoint.get("feature_source")
    checkpoint_features = None
    if isinstance(feature_source, dict):
        value = feature_source.get("directory")
        if isinstance(value, str) and value:
            checkpoint_features = Path(value)

    resolver = EmbeddingResolver(
        args.encoder,
        args.embeddings_dir or checkpoint_dir,
        representation=source_representation,
    )
    features = FeatureResolver(checkpoint_features or args.features_dir)
    rows = load_jsonl(args.data_dir / args.split / "test.jsonl")
    dataset = EmbeddingDataset(
        rows,
        mean=mean,
        std=std,
        class_count=class_count,
        resolver=resolver,
        features=features,
        view=args.view,
        model_representation=model_representation,
        rhythm_mode=args.rhythm,
        rhythm_mean=rhythm_mean,
        rhythm_std=rhythm_std,
    )
    batch_size = args.batch_size or (16 if model_representation == "sequence" else 512)
    collate = collate_sequences if model_representation == "sequence" else None
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=torch.cuda.is_available(),
    )

    probs, pred, true = predict(model, loader, device, model_representation)
    per_class, matrix = class_metrics(true, pred, classes)
    aggregate = aggregate_metrics(per_class, matrix)
    write_csv(run_dir / "test_per_class.csv", per_class)
    write_csv(run_dir / "confusion_counts.csv", confusion_rows(matrix, classes, False))
    write_csv(run_dir / "confusion_row_normalized.csv", confusion_rows(matrix, classes, True))
    write_csv(run_dir / "top_confusions.csv", top_confusions(matrix, classes, args.top_k))
    plot_confusion(run_dir / "confusion_heatmap.png", matrix, classes)

    max_prob = probs.max(axis=1)
    confidence = {
        "mean_max_probability": float(max_prob.mean()),
        "median_max_probability": float(np.median(max_prob)),
        "correct_mean_max_probability": float(max_prob[pred == true].mean()) if np.any(pred == true) else 0.0,
        "incorrect_mean_max_probability": float(max_prob[pred != true].mean()) if np.any(pred != true) else 0.0,
    }
    report = {
        "run_name": run_name,
        "task": "single_label_parent_genre",
        "encoder": checkpoint.get("encoder"),
        "encoder_model": checkpoint.get("encoder_model"),
        "embedding_source": resolver.describe(),
        "feature_source": features.describe(),
        "source_representation": source_representation,
        "representation": model_representation,
        "view": args.view,
        "rhythm": args.rhythm,
        "pooling": checkpoint.get("pooling"),
        "split": args.split,
        "model": args.model,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "class_weight_mode": checkpoint.get("class_weight_mode"),
        "class_count": class_count,
        "test": aggregate,
        "confidence": confidence,
        "outputs": {
            "per_class": str(run_dir / "test_per_class.csv"),
            "confusion_counts": str(run_dir / "confusion_counts.csv"),
            "confusion_normalized": str(run_dir / "confusion_row_normalized.csv"),
            "top_confusions": str(run_dir / "top_confusions.csv"),
            "heatmap": str(run_dir / "confusion_heatmap.png"),
        },
    }
    report_path = run_dir / "evaluation_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"Single-label evaluation\n"
        f"  split:                  {args.split}\n"
        f"  model:                  {args.model}\n"
        f"  encoder:                {args.encoder}\n"
        f"  view:                   {args.view}\n"
        f"  rhythm:                 {args.rhythm}\n"
        f"  samples:                {aggregate['samples']}\n"
        f"  accuracy:               {aggregate['accuracy']:.4f}\n"
        f"  macro F1:               {aggregate['macro_f1']:.4f}\n"
        f"  macro F1 excluding O:   {aggregate['macro_f1_excluding_other']:.4f}\n"
        f"  weighted F1:            {aggregate['weighted_f1']:.4f}\n\n"
        f"Report: {report_path}"
    )


if __name__ == "__main__":
    main()
