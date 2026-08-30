"""Train flat or hierarchy-aware multi-label EDM genre classifiers.

Models:
  linear            standardized embedding -> linear outputs
  mlp               standardized embedding -> shared MLP -> flat outputs
  hierarchical_mlp  standardized embedding -> shared MLP -> separate
                    parent/leaf heads + hierarchy consistency penalty

Targets are expanded through taxonomy ancestors before filtering to the active
class set. All models therefore receive hierarchical supervision.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
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


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
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


def load_active_classes(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    classes = raw.get("classes")
    if not isinstance(classes, list) or not classes:
        raise ValueError(f"{path}: missing classes")
    result = [item for item in classes if isinstance(item, dict)]
    result.sort(key=lambda item: int(item["index"]))
    for expected, item in enumerate(result):
        if item.get("index") != expected:
            raise ValueError(f"{path}: class indices must be contiguous")
    return result


def load_taxonomy(path: Path) -> dict[str, dict[str, Any]]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    genres = raw.get("genres") if isinstance(raw, dict) else None
    if not isinstance(genres, list):
        raise ValueError(f"{path}: expected genres list")
    result = {}
    for item in genres:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str):
            raise ValueError(f"{path}: invalid genre entry")
        result[item["id"]] = item
    return result


def ancestors(label_id: str, taxonomy: dict[str, dict[str, Any]]) -> list[str]:
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
            raise ValueError(f"taxonomy cycle while expanding {label_id!r}")
        seen.add(parent)
        result.append(parent)
        current = parent
    return result


def expanded_labels(labels: list[str], taxonomy: dict[str, dict[str, Any]]) -> set[str]:
    result = {label for label in labels if isinstance(label, str)}
    for label in list(result):
        if label not in taxonomy:
            raise ValueError(f"unknown taxonomy label {label!r}")
        result.update(ancestors(label, taxonomy))
    return result


def load_xy(path: Path, class_to_index: dict[str, int], taxonomy: dict[str, dict[str, Any]]):
    records = load_jsonl(path)
    xs = []
    ys = []
    for record in records:
        embedding_path = record.get("embedding_path")
        if not isinstance(embedding_path, str) or not embedding_path:
            raise ValueError(f"{path}: missing embedding_path")
        vector = np.asarray(np.load(embedding_path, allow_pickle=False), dtype=np.float32)
        if vector.ndim != 1 or not np.isfinite(vector).all():
            raise ValueError(f"{embedding_path}: invalid embedding")
        labels = record.get("labels")
        if not isinstance(labels, list):
            labels = []
        target = np.zeros(len(class_to_index), dtype=np.float32)
        for label in expanded_labels(labels, taxonomy):
            idx = class_to_index.get(label)
            if idx is not None:
                target[idx] = 1.0
        xs.append(vector)
        ys.append(target)
    if not xs:
        raise ValueError(f"{path}: no samples")
    return np.stack(xs), np.stack(ys), records


def nearest_active_parent_edges(classes, taxonomy):
    class_to_index = {item["id"]: item["index"] for item in classes}
    edges = []
    for item in classes:
        child_id = item["id"]
        current = child_id
        while True:
            tax = taxonomy.get(current)
            if tax is None:
                break
            parent = tax.get("parent")
            if not isinstance(parent, str) or not parent:
                break
            if parent in class_to_index:
                edges.append((item["index"], class_to_index[parent]))
                break
            current = parent
    return edges


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


def build_model(name, input_dim, classes, hidden_dim, dropout):
    output_dim = len(classes)
    if name == "linear":
        return LinearHead(input_dim, output_dim)
    if name == "mlp":
        return MLPHead(input_dim, output_dim, hidden_dim, dropout)
    if name == "hierarchical_mlp":
        parent_indices = [item["index"] for item in classes if not bool(item.get("is_leaf"))]
        leaf_indices = [item["index"] for item in classes if bool(item.get("is_leaf"))]
        return HierarchicalMLP(
            input_dim, output_dim, hidden_dim, dropout, parent_indices, leaf_indices
        )
    raise ValueError(name)


def hierarchy_penalty(logits: torch.Tensor, edges: list[tuple[int, int]]) -> torch.Tensor:
    if not edges:
        return logits.new_tensor(0.0)
    probs = torch.sigmoid(logits)
    violations = []
    for child_idx, parent_idx in edges:
        violations.append(torch.relu(probs[:, child_idx] - probs[:, parent_idx]))
    return torch.stack(violations, dim=1).mean()


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
def predict_all(model, loader, device):
    model.eval()
    logits_out = []
    targets_out = []
    loss_fn = nn.BCEWithLogitsLoss(reduction="sum")
    total_loss = 0.0
    total_count = 0
    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        logits = model(x_batch)
        total_loss += float(loss_fn(logits, y_batch).item())
        total_count += int(y_batch.numel())
        logits_out.append(logits.cpu())
        targets_out.append(y_batch.cpu())
    return torch.cat(logits_out), torch.cat(targets_out), total_loss / max(1, total_count)


def macro_f1(logits, targets, threshold=0.5):
    pred = torch.sigmoid(logits) >= threshold
    true = targets >= 0.5
    tp = (pred & true).sum(dim=0).float()
    fp = (pred & ~true).sum(dim=0).float()
    fn = (~pred & true).sum(dim=0).float()
    precision = tp / (tp + fp).clamp_min(1.0)
    recall = tp / (tp + fn).clamp_min(1.0)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
    micro_tp, micro_fp, micro_fn = tp.sum(), fp.sum(), fn.sum()
    micro_precision = micro_tp / (micro_tp + micro_fp).clamp_min(1.0)
    micro_recall = micro_tp / (micro_tp + micro_fn).clamp_min(1.0)
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall).clamp_min(1e-12)
    return float(f1.mean()), float(micro_f1)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["regular", "artist"], required=True)
    p.add_argument("--model", choices=["linear", "mlp", "hierarchical_mlp"], default="hierarchical_mlp")
    p.add_argument("--splits-dir", type=Path, default=DEFAULT_SPLITS_DIR)
    p.add_argument("--active-classes", type=Path, default=DEFAULT_ACTIVE_CLASSES)
    p.add_argument("--taxonomy", type=Path, default=DEFAULT_TAXONOMY)
    p.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--max-pos-weight", type=float, default=20.0)
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--hierarchy-lambda", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--device", default="auto")
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    classes = load_active_classes(args.active_classes)
    class_to_index = {item["id"]: item["index"] for item in classes}
    taxonomy = load_taxonomy(args.taxonomy)
    hierarchy_edges = nearest_active_parent_edges(classes, taxonomy)

    split_dir = args.splits_dir / args.split
    x_train, y_train, train_records = load_xy(split_dir / "train.jsonl", class_to_index, taxonomy)
    x_val, y_val, val_records = load_xy(split_dir / "validation.jsonl", class_to_index, taxonomy)

    mean = x_train.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = x_train.std(axis=0, dtype=np.float64).astype(np.float32)
    std[std < 1e-6] = 1.0
    x_train = ((x_train - mean) / std).astype(np.float32)
    x_val = ((x_val - mean) / std).astype(np.float32)

    positive = y_train.sum(axis=0)
    if np.any(positive == 0):
        bad = [classes[i]["id"] for i, count in enumerate(positive) if count == 0]
        raise ValueError("zero-positive active classes: " + ", ".join(bad))
    negative = len(y_train) - positive
    raw_pos_weight = negative / positive
    pos_weight = np.clip(raw_pos_weight, 1.0, args.max_pos_weight).astype(np.float32)

    train_dataset = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    val_dataset = TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val))
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    device = choose_device(args.device)
    model = build_model(args.model, x_train.shape[1], classes, args.hidden_dim, args.dropout).to(device)
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.from_numpy(pos_weight).to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    run_name = f"{args.split}_{args.model}"
    run_dir = args.runs_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "model.pt"
    normalization_path = run_dir / "normalization.npz"
    history_path = run_dir / "history.csv"
    report_path = run_dir / "training_report.json"
    np.savez(normalization_path, mean=mean, std=std, pos_weight=pos_weight, raw_pos_weight=raw_pos_weight.astype(np.float32))

    history = []
    best_macro = -1.0
    best_epoch = 0
    stale = 0
    started = time.monotonic()

    print(f"Run: {run_name}")
    print(f"Device: {device}")
    print(f"Classes: {len(classes)}")
    print(f"Hierarchy edges: {len(hierarchy_edges)}")
    print(f"Train/val: {len(x_train)} / {len(x_val)}")
    print()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_bce = 0.0
        total_hier = 0.0
        total_examples = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            bce_loss = bce(logits, yb)
            hier_loss = hierarchy_penalty(logits, hierarchy_edges) if args.model == "hierarchical_mlp" else logits.new_tensor(0.0)
            loss = bce_loss + args.hierarchy_lambda * hier_loss
            loss.backward()
            optimizer.step()
            n = xb.shape[0]
            total_loss += float(loss.item()) * n
            total_bce += float(bce_loss.item()) * n
            total_hier += float(hier_loss.item()) * n
            total_examples += n

        val_logits, val_targets, val_loss = predict_all(model, val_loader, device)
        val_macro, val_micro = macro_f1(val_logits, val_targets, 0.5)
        row = {
            "epoch": epoch,
            "train_loss": total_loss / total_examples,
            "train_bce": total_bce / total_examples,
            "train_hierarchy_penalty": total_hier / total_examples,
            "validation_loss": val_loss,
            "validation_macro_f1_at_0_5": val_macro,
            "validation_micro_f1_at_0_5": val_micro,
        }
        history.append(row)

        improved = val_macro > best_macro + 1e-6
        if improved:
            best_macro = val_macro
            best_epoch = epoch
            stale = 0
            torch.save({
                "state_dict": model.state_dict(),
                "model_name": args.model,
                "input_dim": x_train.shape[1],
                "output_dim": len(classes),
                "hidden_dim": args.hidden_dim,
                "dropout": args.dropout,
                "class_ids": [item["id"] for item in classes],
                "parent_indices": [item["index"] for item in classes if not bool(item.get("is_leaf"))],
                "leaf_indices": [item["index"] for item in classes if bool(item.get("is_leaf"))],
                "hierarchy_edges": hierarchy_edges,
                "hierarchy_lambda": args.hierarchy_lambda,
                "split": args.split,
                "seed": args.seed,
                "epoch": epoch,
                "validation_macro_f1_at_0_5": best_macro,
            }, checkpoint_path)
        else:
            stale += 1

        print(
            f"epoch {epoch:03d} loss={row['train_loss']:.4f} "
            f"hier={row['train_hierarchy_penalty']:.4f} "
            f"val_macro={val_macro:.4f} val_micro={val_micro:.4f}" + (" *" if improved else "")
        )
        if stale >= args.patience:
            print(f"Early stopping after {args.patience} stale epochs.")
            break

    elapsed = time.monotonic() - started
    with history_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)

    report = {
        "run_name": run_name,
        "split": args.split,
        "model": args.model,
        "device": str(device),
        "seed": args.seed,
        "input_dimension": int(x_train.shape[1]),
        "active_class_count": len(classes),
        "parent_class_count": sum(not bool(item.get("is_leaf")) for item in classes),
        "leaf_class_count": sum(bool(item.get("is_leaf")) for item in classes),
        "hierarchy_edges": len(hierarchy_edges),
        "hierarchy_lambda": args.hierarchy_lambda if args.model == "hierarchical_mlp" else 0.0,
        "train_samples": len(train_records),
        "validation_samples": len(val_records),
        "best_epoch": best_epoch,
        "best_validation_macro_f1_at_0_5": best_macro,
        "epochs_completed": len(history),
        "elapsed_seconds": round(elapsed, 3),
        "outputs": {
            "checkpoint": str(checkpoint_path),
            "normalization": str(normalization_path),
            "history": str(history_path),
        },
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print()
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation macro F1: {best_macro:.4f}")
    print(f"Checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
