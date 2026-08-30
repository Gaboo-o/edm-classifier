"""Evaluate the learned-attention Discogs-EffNet patch classifier.

Validation chooses one global threshold. Test is evaluated once using that
validation-selected threshold. Both raw and hierarchy-projected inference are
reported in the same general schema as the pooled evaluator.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from edm_classifier.training.train_attention import (
    AttentionMLP,
    PatchDataset,
    choose_device,
    collate_patch_batch,
    load_active_classes,
    load_jsonl,
    load_taxonomy,
)


DEFAULT_SPLITS_DIR = Path("data/splits")
DEFAULT_ACTIVE_CLASSES = Path("data/training/active_classes.json")
DEFAULT_TAXONOMY = Path("config/taxonomy.yaml")
DEFAULT_PATCHES_DIR = Path("data/embeddings/patches")
DEFAULT_RUNS_DIR = Path("data/runs/attention")


def active_ancestor_pairs(
    classes: list[dict[str, Any]],
    taxonomy: dict[str, dict[str, Any]],
) -> list[tuple[int, int]]:
    class_to_index = {
        item["id"]: item["index"]
        for item in classes
    }

    pairs: list[tuple[int, int]] = []

    for child in classes:
        current = child["id"]
        seen: set[str] = set()

        while current in taxonomy:
            parent = taxonomy[current].get("parent")
            if not isinstance(parent, str) or not parent:
                break
            if parent in seen:
                raise ValueError(f"taxonomy cycle at {child['id']!r}")
            seen.add(parent)

            parent_index = class_to_index.get(parent)
            if parent_index is not None:
                pairs.append((child["index"], parent_index))

            current = parent

    return sorted(set(pairs))


def project_hierarchy(
    probs: np.ndarray,
    pairs: list[tuple[int, int]],
) -> np.ndarray:
    result = probs.copy()

    # Iterate until stable so this remains correct for deeper taxonomies.
    changed = True
    iterations = 0

    while changed:
        changed = False
        iterations += 1

        for child, parent in pairs:
            before = result[:, parent].copy()
            result[:, parent] = np.maximum(
                result[:, parent],
                result[:, child],
            )
            if not np.array_equal(before, result[:, parent]):
                changed = True

        if iterations > len(pairs) + 1:
            break

    return result


def hierarchy_diagnostics(
    probs: np.ndarray,
    pairs: list[tuple[int, int]],
) -> dict[str, float]:
    if not pairs:
        return {
            "pair_violation_rate": 0.0,
            "sample_violation_rate": 0.0,
            "mean_violation_magnitude": 0.0,
        }

    violations = []
    sample_any = np.zeros(len(probs), dtype=bool)

    for child, parent in pairs:
        magnitude = np.maximum(
            probs[:, child] - probs[:, parent],
            0.0,
        )
        violations.append(magnitude)
        sample_any |= magnitude > 0

    matrix = np.stack(violations, axis=1)

    return {
        "pair_violation_rate": float(np.mean(matrix > 0)),
        "sample_violation_rate": float(np.mean(sample_any)),
        "mean_violation_magnitude": float(
            matrix[matrix > 0].mean()
            if np.any(matrix > 0)
            else 0.0
        ),
    }


@torch.no_grad()
def predict_probs(
    model: AttentionMLP,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()

    probs: list[np.ndarray] = []
    targets: list[np.ndarray] = []

    for patches, mask, y in loader:
        patches = patches.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        logits = model(patches, mask)

        probs.append(
            torch.sigmoid(logits).cpu().numpy()
        )
        targets.append(y.numpy())

    return (
        np.concatenate(probs, axis=0),
        np.concatenate(targets, axis=0),
    )


def safe_prf(
    tp: np.ndarray,
    fp: np.ndarray,
    fn: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    precision = np.divide(
        tp,
        tp + fp,
        out=np.zeros_like(tp, dtype=np.float64),
        where=(tp + fp) > 0,
    )
    recall = np.divide(
        tp,
        tp + fn,
        out=np.zeros_like(tp, dtype=np.float64),
        where=(tp + fn) > 0,
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) > 0,
    )
    return precision, recall, f1


def metrics(
    probs: np.ndarray,
    targets: np.ndarray,
    threshold: float,
    classes: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    pred = probs >= threshold
    true = targets >= 0.5

    tp = np.sum(pred & true, axis=0)
    fp = np.sum(pred & ~true, axis=0)
    fn = np.sum(~pred & true, axis=0)
    tn = np.sum(~pred & ~true, axis=0)
    support = np.sum(true, axis=0)

    precision, recall, f1 = safe_prf(tp, fp, fn)

    supported = support > 0
    parent_mask = np.array(
        [not bool(item.get("is_leaf")) for item in classes],
        dtype=bool,
    )
    leaf_mask = ~parent_mask
    supported_leaf = supported & leaf_mask

    def macro(mask: np.ndarray) -> dict[str, Any]:
        count = int(mask.sum())
        if count == 0:
            return {
                "class_count": 0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
            }
        return {
            "class_count": count,
            "precision": float(precision[mask].mean()),
            "recall": float(recall[mask].mean()),
            "f1": float(f1[mask].mean()),
        }

    micro_tp = int(tp.sum())
    micro_fp = int(fp.sum())
    micro_fn = int(fn.sum())

    micro_precision = (
        micro_tp / (micro_tp + micro_fp)
        if micro_tp + micro_fp
        else 0.0
    )
    micro_recall = (
        micro_tp / (micro_tp + micro_fn)
        if micro_tp + micro_fn
        else 0.0
    )
    micro_f1 = (
        2 * micro_precision * micro_recall
        / (micro_precision + micro_recall)
        if micro_precision + micro_recall
        else 0.0
    )

    result = {
        "macro_all": macro(np.ones(len(classes), dtype=bool)),
        "macro_supported": macro(supported),
        "macro_parent": macro(parent_mask & supported),
        "macro_leaf": macro(leaf_mask),
        "macro_supported_leaf": macro(supported_leaf),
        "micro": {
            "precision": micro_precision,
            "recall": micro_recall,
            "f1": micro_f1,
        },
        "subset_accuracy": float(np.mean(np.all(pred == true, axis=1))),
        "hamming_loss": float(np.mean(pred != true)),
        "zero_support_classes": [
            classes[i]["id"]
            for i in range(len(classes))
            if support[i] == 0
        ],
    }

    per_class = []

    for i, item in enumerate(classes):
        per_class.append(
            {
                "index": i,
                "id": item["id"],
                "label": item.get("label", item["id"]),
                "is_leaf": bool(item.get("is_leaf")),
                "support": int(support[i]),
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "tp": int(tp[i]),
                "fp": int(fp[i]),
                "fn": int(fn[i]),
                "tn": int(tn[i]),
            }
        )

    return result, per_class


def threshold_search(
    probs: np.ndarray,
    targets: np.ndarray,
    classes: list[dict[str, Any]],
    *,
    threshold_min: float,
    threshold_max: float,
    threshold_step: float,
) -> tuple[float, list[dict[str, float]]]:
    thresholds = np.arange(
        threshold_min,
        threshold_max + threshold_step / 2,
        threshold_step,
    )

    rows: list[dict[str, float]] = []
    best_threshold = float(thresholds[0])
    best_f1 = -1.0

    for threshold in thresholds:
        result, _ = metrics(
            probs,
            targets,
            float(threshold),
            classes,
        )

        macro = result["macro_supported"]["f1"]
        micro = result["micro"]["f1"]

        rows.append(
            {
                "threshold": float(threshold),
                "validation_macro_supported_f1": float(macro),
                "validation_micro_f1": float(micro),
            }
        )

        if macro > best_f1 + 1e-12:
            best_f1 = macro
            best_threshold = float(threshold)

    return best_threshold, rows


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
        )
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate attention-pooled EDM classifier."
    )

    parser.add_argument(
        "--split",
        choices=["regular", "artist"],
        required=True,
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
        "--patches-dir",
        type=Path,
        default=DEFAULT_PATCHES_DIR,
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=DEFAULT_RUNS_DIR,
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument("--threshold-min", type=float, default=0.10)
    parser.add_argument("--threshold-max", type=float, default=0.90)
    parser.add_argument("--threshold-step", type=float, default=0.01)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    classes = load_active_classes(args.active_classes)
    class_to_index = {
        item["id"]: item["index"]
        for item in classes
    }
    taxonomy = load_taxonomy(args.taxonomy)

    run_name = f"{args.split}_attention_mlp"
    run_dir = args.runs_dir / run_name
    checkpoint_path = run_dir / "model.pt"

    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    checkpoint_ids = checkpoint.get("class_ids")
    expected_ids = [item["id"] for item in classes]
    if checkpoint_ids != expected_ids:
        raise ValueError(
            "checkpoint class IDs do not match active_classes.json"
        )

    max_patches = checkpoint.get("max_patches")

    split_dir = args.splits_dir / args.split

    val_dataset = PatchDataset(
        load_jsonl(split_dir / "validation.jsonl"),
        class_to_index=class_to_index,
        taxonomy=taxonomy,
        patches_dir=args.patches_dir,
        max_patches=max_patches,
    )
    test_dataset = PatchDataset(
        load_jsonl(split_dir / "test.jsonl"),
        class_to_index=class_to_index,
        taxonomy=taxonomy,
        patches_dir=args.patches_dir,
        max_patches=max_patches,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_patch_batch,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_patch_batch,
        pin_memory=torch.cuda.is_available(),
    )

    device = choose_device(args.device)

    model = AttentionMLP(
        output_dim=len(classes),
        attention_dim=int(checkpoint["attention_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
        dropout=float(checkpoint["dropout"]),
    ).to(device)

    model.load_state_dict(checkpoint["state_dict"])

    print(f"Run:            {run_name}")
    print(f"Device:         {device}")
    print(f"Checkpoint:     epoch {checkpoint['epoch']}")
    print(f"Validation:     {len(val_dataset)}")
    print(f"Test:           {len(test_dataset)}")
    print(f"Max patches:    {max_patches or 'unlimited'}")
    print()

    val_raw_probs, val_targets = predict_probs(
        model,
        val_loader,
        device,
    )
    test_raw_probs, test_targets = predict_probs(
        model,
        test_loader,
        device,
    )

    pairs = active_ancestor_pairs(classes, taxonomy)

    val_hier_probs = project_hierarchy(val_raw_probs, pairs)
    test_hier_probs = project_hierarchy(test_raw_probs, pairs)

    results: dict[str, Any] = {}

    for mode, val_probs, test_probs in (
        ("raw", val_raw_probs, test_raw_probs),
        ("hierarchical", val_hier_probs, test_hier_probs),
    ):
        threshold, search_rows = threshold_search(
            val_probs,
            val_targets,
            classes,
            threshold_min=args.threshold_min,
            threshold_max=args.threshold_max,
            threshold_step=args.threshold_step,
        )

        validation_metrics, _ = metrics(
            val_probs,
            val_targets,
            threshold,
            classes,
        )
        test_metrics, per_class = metrics(
            test_probs,
            test_targets,
            threshold,
            classes,
        )

        threshold_path = (
            run_dir / f"threshold_search_{mode}.csv"
        )
        per_class_path = (
            run_dir / f"test_per_class_{mode}.csv"
        )

        write_csv(threshold_path, search_rows)
        write_csv(per_class_path, per_class)

        results[mode] = {
            "threshold": threshold,
            "validation": validation_metrics,
            "test": test_metrics,
            "outputs": {
                "threshold_search": str(threshold_path),
                "test_per_class": str(per_class_path),
            },
        }

        print(
            f"{mode:12s} "
            f"threshold={threshold:.2f} "
            f"test_macro={test_metrics['macro_supported']['f1']:.4f} "
            f"leaf={test_metrics['macro_supported_leaf']['f1']:.4f} "
            f"micro={test_metrics['micro']['f1']:.4f}"
        )

    report = {
        "run_name": run_name,
        "split": args.split,
        "model": "attention_mlp",
        "device": str(device),
        "checkpoint_epoch": checkpoint["epoch"],
        "active_class_count": len(classes),
        "active_ancestor_pairs": len(pairs),
        "validation_samples": len(val_dataset),
        "test_samples": len(test_dataset),
        "max_patches": max_patches,
        "confidence_policy": checkpoint.get("confidence_policy"),
        "hierarchy_diagnostics_before_projection": {
            "validation": hierarchy_diagnostics(
                val_raw_probs,
                pairs,
            ),
            "test": hierarchy_diagnostics(
                test_raw_probs,
                pairs,
            ),
        },
        "results": results,
    }

    report_path = run_dir / "evaluation_report.json"
    report_path.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print()
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
