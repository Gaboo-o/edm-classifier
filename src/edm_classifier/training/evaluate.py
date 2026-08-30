"""Evaluate flat or hierarchy-aware EDM classifiers.

For every checkpoint, this evaluator reports two inference modes:
  raw          - model probabilities as produced
  hierarchical - probabilities projected so each active ancestor probability
                 is at least as large as its descendants

Each mode gets its own validation-tuned global threshold. Test metrics include
all, supported, parent, leaf, and supported-leaf macro scores, plus hierarchy
consistency diagnostics.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import yaml

DEFAULT_SPLITS_DIR = Path("data/splits")
DEFAULT_ACTIVE_CLASSES = Path("data/training/active_classes.json")
DEFAULT_TAXONOMY = Path("config/taxonomy.yaml")
DEFAULT_RUNS_DIR = Path("data/runs")


def load_jsonl(path):
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if line:
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path}: expected JSON objects")
                records.append(value)
    return records


def load_active_classes(path):
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    classes = [item for item in raw["classes"] if isinstance(item, dict)]
    classes.sort(key=lambda x: int(x["index"]))
    return classes


def load_taxonomy(path):
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return {item["id"]: item for item in raw["genres"]}


def ancestors(label_id, taxonomy):
    result = []
    seen = set()
    current = label_id
    while True:
        item = taxonomy.get(current)
        if item is None:
            break
        parent = item.get("parent")
        if not isinstance(parent, str) or not parent:
            break
        if parent in seen:
            raise ValueError(f"taxonomy cycle at {label_id}")
        seen.add(parent)
        result.append(parent)
        current = parent
    return result


def active_ancestor_pairs(classes, taxonomy):
    class_to_index = {item["id"]: item["index"] for item in classes}
    pairs = []
    for item in classes:
        for ancestor_id in ancestors(item["id"], taxonomy):
            if ancestor_id in class_to_index:
                pairs.append((item["index"], class_to_index[ancestor_id]))
    return pairs


def load_xy(path, class_to_index, taxonomy, mean, std):
    records = load_jsonl(path)
    xs, ys = [], []
    for record in records:
        vector = np.asarray(np.load(record["embedding_path"], allow_pickle=False), dtype=np.float32)
        labels = {label for label in record.get("labels", []) if isinstance(label, str)}
        expanded = set(labels)
        for label in labels:
            expanded.update(ancestors(label, taxonomy))
        target = np.zeros(len(class_to_index), dtype=np.float32)
        for label in expanded:
            idx = class_to_index.get(label)
            if idx is not None:
                target[idx] = 1.0
        xs.append((vector - mean) / std)
        ys.append(target)
    return np.stack(xs).astype(np.float32), np.stack(ys).astype(np.float32), records


class LinearHead(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.classifier = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.classifier(x)


class MLPHead(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, dropout):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.network(x)


class HierarchicalMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, dropout, parent_indices, leaf_indices):
        super().__init__()
        self.output_dim = output_dim
        self.parent_indices = list(parent_indices)
        self.leaf_indices = list(leaf_indices)
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.parent_head = nn.Linear(hidden_dim, len(self.parent_indices))
        self.leaf_head = nn.Linear(hidden_dim, len(self.leaf_indices))

    def forward(self, x):
        h = self.trunk(x)
        out = x.new_zeros((x.shape[0], self.output_dim))
        if self.parent_indices:
            out[:, self.parent_indices] = self.parent_head(h)
        if self.leaf_indices:
            out[:, self.leaf_indices] = self.leaf_head(h)
        return out


def build_model(checkpoint):
    name = checkpoint["model_name"]
    if name == "linear":
        return LinearHead(checkpoint["input_dim"], checkpoint["output_dim"])
    if name == "mlp":
        return MLPHead(checkpoint["input_dim"], checkpoint["output_dim"], checkpoint["hidden_dim"], checkpoint["dropout"])
    if name == "hierarchical_mlp":
        return HierarchicalMLP(
            checkpoint["input_dim"], checkpoint["output_dim"], checkpoint["hidden_dim"], checkpoint["dropout"],
            checkpoint["parent_indices"], checkpoint["leaf_indices"]
        )
    raise ValueError(f"unknown model {name}")


def choose_device(requested):
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def predict(model, x, batch_size, device):
    loader = DataLoader(TensorDataset(torch.from_numpy(x)), batch_size=batch_size, shuffle=False)
    outputs = []
    model.eval()
    for (xb,) in loader:
        outputs.append(torch.sigmoid(model(xb.to(device))).cpu().numpy())
    return np.concatenate(outputs, axis=0)


def project_hierarchy(probs, pairs):
    adjusted = probs.copy()
    # Repeating to convergence is cheap and guarantees transitive consistency.
    changed = True
    while changed:
        changed = False
        for child_idx, ancestor_idx in pairs:
            new_values = np.maximum(adjusted[:, ancestor_idx], adjusted[:, child_idx])
            if np.any(new_values > adjusted[:, ancestor_idx]):
                adjusted[:, ancestor_idx] = new_values
                changed = True
    return adjusted


def hierarchy_diagnostics(probs, pairs):
    if not pairs:
        return {"pair_violation_rate": 0.0, "sample_violation_rate": 0.0, "mean_violation_magnitude": 0.0}
    violations = []
    sample_has = np.zeros(probs.shape[0], dtype=bool)
    magnitudes = []
    for child_idx, ancestor_idx in pairs:
        diff = probs[:, child_idx] - probs[:, ancestor_idx]
        bad = diff > 0
        violations.append(bad)
        sample_has |= bad
        if np.any(bad):
            magnitudes.extend(diff[bad].tolist())
    all_bad = np.stack(violations, axis=1)
    return {
        "pair_violation_rate": float(all_bad.mean()),
        "sample_violation_rate": float(sample_has.mean()),
        "mean_violation_magnitude": float(np.mean(magnitudes)) if magnitudes else 0.0,
    }


def per_class_arrays(probs, targets, threshold):
    pred = probs >= threshold
    true = targets >= 0.5
    tp = np.logical_and(pred, true).sum(axis=0).astype(float)
    fp = np.logical_and(pred, ~true).sum(axis=0).astype(float)
    fn = np.logical_and(~pred, true).sum(axis=0).astype(float)
    tn = np.logical_and(~pred, ~true).sum(axis=0).astype(float)
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) != 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) != 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp), where=(precision + recall) != 0)
    support = true.sum(axis=0).astype(int)
    predicted_positive = pred.sum(axis=0).astype(int)
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "f1": f1,
        "support": support, "predicted_positive": predicted_positive,
        "pred": pred, "true": true,
    }


def aggregate_metrics(arrays, classes):
    tp, fp, fn = arrays["tp"], arrays["fp"], arrays["fn"]
    p, r, f1, support = arrays["precision"], arrays["recall"], arrays["f1"], arrays["support"]
    all_mask = np.ones(len(classes), dtype=bool)
    supported = support > 0
    parent = np.array([not bool(item.get("is_leaf")) for item in classes])
    leaf = ~parent

    def macro(mask):
        idx = np.where(mask)[0]
        if len(idx) == 0:
            return {"class_count": 0, "precision": None, "recall": None, "f1": None}
        return {
            "class_count": int(len(idx)),
            "precision": float(p[idx].mean()),
            "recall": float(r[idx].mean()),
            "f1": float(f1[idx].mean()),
        }

    micro_tp, micro_fp, micro_fn = tp.sum(), fp.sum(), fn.sum()
    micro_p = micro_tp / (micro_tp + micro_fp) if micro_tp + micro_fp else 0.0
    micro_r = micro_tp / (micro_tp + micro_fn) if micro_tp + micro_fn else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if micro_p + micro_r else 0.0

    return {
        "macro_all": macro(all_mask),
        "macro_supported": macro(supported),
        "macro_parent": macro(parent & supported),
        "macro_leaf": macro(leaf & supported),
        "macro_supported_leaf": macro(leaf & supported),
        "micro": {"precision": float(micro_p), "recall": float(micro_r), "f1": float(micro_f1)},
        "subset_accuracy": float(np.all(arrays["pred"] == arrays["true"], axis=1).mean()),
        "hamming_loss": float(np.not_equal(arrays["pred"], arrays["true"]).mean()),
        "zero_support_classes": [classes[i]["id"] for i in range(len(classes)) if support[i] == 0],
    }


def tune_threshold(probs, targets, classes, minimum, maximum, step):
    best_threshold = 0.5
    best_f1 = -1.0
    rows = []
    for threshold in np.arange(minimum, maximum + step / 2, step):
        arrays = per_class_arrays(probs, targets, float(threshold))
        metrics = aggregate_metrics(arrays, classes)
        score = metrics["macro_supported"]["f1"]
        rows.append({"threshold": round(float(threshold), 4), "supported_macro_f1": score, "micro_f1": metrics["micro"]["f1"]})
        if score > best_f1 + 1e-12 or (abs(score - best_f1) <= 1e-12 and abs(threshold - 0.5) < abs(best_threshold - 0.5)):
            best_f1 = score
            best_threshold = float(threshold)
    return best_threshold, rows


def write_per_class(path, classes, arrays):
    fields = ["index", "id", "label", "parent", "is_leaf", "support", "predicted_positive", "precision", "recall", "f1", "tp", "fp", "fn", "tn"]
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for i, item in enumerate(classes):
            writer.writerow({
                "index": i,
                "id": item["id"],
                "label": item.get("label", item["id"]),
                "parent": item.get("parent") or "",
                "is_leaf": item.get("is_leaf", ""),
                "support": int(arrays["support"][i]),
                "predicted_positive": int(arrays["predicted_positive"][i]),
                "precision": round(float(arrays["precision"][i]), 6),
                "recall": round(float(arrays["recall"][i]), 6),
                "f1": round(float(arrays["f1"][i]), 6),
                "tp": int(arrays["tp"][i]), "fp": int(arrays["fp"][i]), "fn": int(arrays["fn"][i]), "tn": int(arrays["tn"][i]),
            })


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["regular", "artist"], required=True)
    p.add_argument("--model", choices=["linear", "mlp", "hierarchical_mlp"], default="mlp")
    p.add_argument("--splits-dir", type=Path, default=DEFAULT_SPLITS_DIR)
    p.add_argument("--active-classes", type=Path, default=DEFAULT_ACTIVE_CLASSES)
    p.add_argument("--taxonomy", type=Path, default=DEFAULT_TAXONOMY)
    p.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--device", default="auto")
    p.add_argument("--threshold-min", type=float, default=0.10)
    p.add_argument("--threshold-max", type=float, default=0.90)
    p.add_argument("--threshold-step", type=float, default=0.01)
    return p.parse_args()


def main():
    args = parse_args()
    classes = load_active_classes(args.active_classes)
    class_to_index = {item["id"]: item["index"] for item in classes}
    taxonomy = load_taxonomy(args.taxonomy)
    pairs = active_ancestor_pairs(classes, taxonomy)

    run_name = f"{args.split}_{args.model}"
    run_dir = args.runs_dir / run_name
    checkpoint_path = run_dir / "model.pt"
    normalization_path = run_dir / "normalization.npz"
    if not checkpoint_path.exists():
        raise SystemExit(f"missing checkpoint: {checkpoint_path}")

    norm = np.load(normalization_path, allow_pickle=False)
    mean = np.asarray(norm["mean"], dtype=np.float32)
    std = np.asarray(norm["std"], dtype=np.float32)

    device = choose_device(args.device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    expected = [item["id"] for item in classes]
    if checkpoint["class_ids"] != expected:
        raise ValueError("checkpoint class ordering differs from active_classes.json")
    model = build_model(checkpoint)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)

    split_dir = args.splits_dir / args.split
    x_val, y_val, val_records = load_xy(split_dir / "validation.jsonl", class_to_index, taxonomy, mean, std)
    x_test, y_test, test_records = load_xy(split_dir / "test.jsonl", class_to_index, taxonomy, mean, std)
    val_raw = predict(model, x_val, args.batch_size, device)
    test_raw = predict(model, x_test, args.batch_size, device)
    val_hier = project_hierarchy(val_raw, pairs)
    test_hier = project_hierarchy(test_raw, pairs)

    results = {}
    for mode, val_probs, test_probs in [("raw", val_raw, test_raw), ("hierarchical", val_hier, test_hier)]:
        threshold, threshold_rows = tune_threshold(val_probs, y_val, classes, args.threshold_min, args.threshold_max, args.threshold_step)
        val_arrays = per_class_arrays(val_probs, y_val, threshold)
        test_arrays = per_class_arrays(test_probs, y_test, threshold)
        val_metrics = aggregate_metrics(val_arrays, classes)
        test_metrics = aggregate_metrics(test_arrays, classes)

        threshold_path = run_dir / f"threshold_search_{mode}.csv"
        with threshold_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(threshold_rows[0]))
            writer.writeheader(); writer.writerows(threshold_rows)
        per_class_path = run_dir / f"test_per_class_{mode}.csv"
        write_per_class(per_class_path, classes, test_arrays)

        results[mode] = {
            "threshold": round(threshold, 4),
            "validation": val_metrics,
            "test": test_metrics,
            "outputs": {"threshold_search": str(threshold_path), "test_per_class": str(per_class_path)},
        }

    report = {
        "run_name": run_name,
        "split": args.split,
        "model": args.model,
        "device": str(device),
        "checkpoint_epoch": checkpoint["epoch"],
        "active_class_count": len(classes),
        "active_ancestor_pairs": len(pairs),
        "validation_samples": len(val_records),
        "test_samples": len(test_records),
        "hierarchy_diagnostics_before_projection": {
            "validation": hierarchy_diagnostics(val_raw, pairs),
            "test": hierarchy_diagnostics(test_raw, pairs),
        },
        "results": results,
    }
    report_path = run_dir / "evaluation_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"Evaluation: {run_name}")
    for mode in ("raw", "hierarchical"):
        test = results[mode]["test"]
        print()
        print(f"{mode} inference, threshold={results[mode]['threshold']:.2f}")
        def fmt_metric(value):
            return "n/a" if value is None else f"{value:.4f}"

        print(f"  supported macro F1: {fmt_metric(test['macro_supported']['f1'])}")
        print(f"  parent macro F1:    {fmt_metric(test['macro_parent']['f1'])}")
        print(f"  leaf macro F1:      {fmt_metric(test['macro_leaf']['f1'])}")
        print(f"  micro F1:           {fmt_metric(test['micro']['f1'])}")
    print()
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
