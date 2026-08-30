"""Train a basic learned-attention pooler over Discogs-EffNet patch embeddings.

This is intentionally a small, controlled extension of the pooled MLP:

    [T, 1280] Discogs-EffNet patches
      -> LayerNorm per patch
      -> attention MLP 1280 -> attention_dim -> 1
      -> masked softmax over time
      -> weighted sum [1280]
      -> MLP 1280 -> hidden_dim -> classes

Discogs-EffNet itself remains frozen because patch embeddings are precomputed.

Defaults use the selected confidence-ablation policy:
    data/ablations/confidence/min_075/splits

Outputs:
    data/runs/attention/<split>_attention_mlp/
      model.pt
      history.csv
      training_report.json
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
from torch.utils.data import DataLoader, Dataset
import yaml


EMBEDDING_DIM = 1280

DEFAULT_SPLITS_DIR = Path("data/splits")
DEFAULT_ACTIVE_CLASSES = Path("data/training/active_classes.json")
DEFAULT_TAXONOMY = Path("config/taxonomy.yaml")
DEFAULT_PATCHES_DIR = Path("data/embeddings/patches")
DEFAULT_RUNS_DIR = Path("data/runs/attention")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            line = raw.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}:{line_number}: invalid JSON: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected object")
            rows.append(row)
    return rows


def load_active_classes(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    classes = raw.get("classes") if isinstance(raw, dict) else None
    if not isinstance(classes, list) or not classes:
        raise ValueError(f"{path}: missing classes array")

    result = [x for x in classes if isinstance(x, dict)]
    result.sort(key=lambda x: int(x["index"]))

    for expected, item in enumerate(result):
        if item.get("index") != expected:
            raise ValueError(f"{path}: class indices must be contiguous")
        if not isinstance(item.get("id"), str):
            raise ValueError(f"{path}: invalid class id at index {expected}")

    return result


def load_taxonomy(path: Path) -> dict[str, dict[str, Any]]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    genres = raw.get("genres") if isinstance(raw, dict) else None
    if not isinstance(genres, list):
        raise ValueError(f"{path}: expected genres list")

    result: dict[str, dict[str, Any]] = {}
    for item in genres:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str):
            raise ValueError(f"{path}: invalid genre")
        result[item["id"]] = item
    return result


def ancestors(
    label_id: str,
    taxonomy: dict[str, dict[str, Any]],
) -> set[str]:
    result: set[str] = set()
    seen: set[str] = set()
    current = label_id

    while True:
        item = taxonomy.get(current)
        if item is None:
            break

        parent = item.get("parent")
        if not isinstance(parent, str) or not parent:
            break

        if parent in seen:
            raise ValueError(f"taxonomy cycle at {label_id!r}")

        seen.add(parent)
        result.add(parent)
        current = parent

    return result


def expanded_labels(
    direct: list[str],
    taxonomy: dict[str, dict[str, Any]],
) -> set[str]:
    result = {x for x in direct if isinstance(x, str) and x}
    for label in list(result):
        if label not in taxonomy:
            raise ValueError(f"unknown taxonomy label {label!r}")
        result.update(ancestors(label, taxonomy))
    return result


def record_video_id(record: dict[str, Any]) -> str:
    for key in ("video_id", "sample_id"):
        value = record.get(key)
        if isinstance(value, str) and value:
            return value

    embedding_path = record.get("embedding_path")
    if isinstance(embedding_path, str) and embedding_path:
        return Path(embedding_path).stem

    raise ValueError(f"sample has no usable video ID: {record}")


def deterministic_subsample(
    patches: np.ndarray,
    max_patches: int | None,
) -> np.ndarray:
    if max_patches is None or len(patches) <= max_patches:
        return patches

    indices = np.linspace(
        0,
        len(patches) - 1,
        num=max_patches,
        dtype=np.int64,
    )
    return patches[indices]


class PatchDataset(Dataset):
    def __init__(
        self,
        records: list[dict[str, Any]],
        *,
        class_to_index: dict[str, int],
        taxonomy: dict[str, dict[str, Any]],
        patches_dir: Path,
        max_patches: int | None,
    ):
        self.items: list[tuple[Path, np.ndarray, str]] = []
        self.max_patches = max_patches

        missing: list[str] = []

        for record in records:
            video_id = record_video_id(record)
            patch_path = patches_dir / f"{video_id}.npy"

            if not patch_path.is_file():
                missing.append(video_id)
                continue

            raw_labels = record.get("labels")
            if not isinstance(raw_labels, list):
                raw_labels = []

            expanded = expanded_labels(raw_labels, taxonomy)
            target = np.zeros(len(class_to_index), dtype=np.float32)

            for label in expanded:
                index = class_to_index.get(label)
                if index is not None:
                    target[index] = 1.0

            self.items.append((patch_path, target, video_id))

        if missing:
            preview = ", ".join(missing[:10])
            raise FileNotFoundError(
                f"{len(missing)} split samples have no patch embedding in "
                f"{patches_dir}. First missing: {preview}"
            )

        if not self.items:
            raise ValueError("dataset contains no samples")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[np.ndarray, np.ndarray, str]:
        path, target, video_id = self.items[index]

        matrix = np.load(path, mmap_mode="r", allow_pickle=False)
        matrix = np.asarray(matrix)

        if (
            matrix.ndim != 2
            or matrix.shape[0] < 1
            or matrix.shape[1] != EMBEDDING_DIM
        ):
            raise ValueError(f"{path}: invalid patch shape {matrix.shape}")

        matrix = deterministic_subsample(matrix, self.max_patches)
        matrix = np.asarray(matrix, dtype=np.float32)

        if not np.isfinite(matrix).all():
            raise ValueError(f"{path}: patch matrix contains NaN/inf")

        return matrix, target, video_id


def collate_patch_batch(
    batch: list[tuple[np.ndarray, np.ndarray, str]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    lengths = [len(item[0]) for item in batch]
    max_length = max(lengths)
    batch_size = len(batch)

    patches = torch.zeros(
        (batch_size, max_length, EMBEDDING_DIM),
        dtype=torch.float32,
    )
    mask = torch.zeros(
        (batch_size, max_length),
        dtype=torch.bool,
    )
    targets = torch.from_numpy(
        np.stack([item[1] for item in batch]).astype(np.float32)
    )

    for i, (matrix, _, _) in enumerate(batch):
        length = len(matrix)
        patches[i, :length] = torch.from_numpy(matrix)
        mask[i, :length] = True

    return patches, mask, targets


class AttentionMLP(nn.Module):
    def __init__(
        self,
        *,
        output_dim: int,
        attention_dim: int = 256,
        hidden_dim: int = 512,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.input_norm = nn.LayerNorm(EMBEDDING_DIM)

        self.attention = nn.Sequential(
            nn.Linear(EMBEDDING_DIM, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1),
        )

        self.classifier = nn.Sequential(
            nn.Linear(EMBEDDING_DIM, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self,
        patches: torch.Tensor,
        mask: torch.Tensor,
        *,
        return_attention: bool = False,
    ):
        h = self.input_norm(patches)

        scores = self.attention(h).squeeze(-1)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1)

        pooled = torch.sum(h * weights.unsqueeze(-1), dim=1)
        logits = self.classifier(pooled)

        if return_attention:
            return logits, weights

        return logits


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)

    if torch.cuda.is_available():
        return torch.device("cuda")

    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def compute_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> dict[str, float]:
    probs = torch.sigmoid(logits)
    pred = probs >= threshold
    true = targets >= 0.5

    tp = (pred & true).sum(dim=0).float()
    fp = (pred & ~true).sum(dim=0).float()
    fn = (~pred & true).sum(dim=0).float()

    precision = tp / (tp + fp).clamp_min(1.0)
    recall = tp / (tp + fn).clamp_min(1.0)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)

    micro_tp = tp.sum()
    micro_fp = fp.sum()
    micro_fn = fn.sum()

    micro_precision = micro_tp / (micro_tp + micro_fp).clamp_min(1.0)
    micro_recall = micro_tp / (micro_tp + micro_fn).clamp_min(1.0)
    micro_f1 = (
        2
        * micro_precision
        * micro_recall
        / (micro_precision + micro_recall).clamp_min(1e-12)
    )

    return {
        "macro_f1": float(f1.mean().item()),
        "micro_f1": float(micro_f1.item()),
    }


@torch.no_grad()
def predict_all(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    model.eval()

    logits_out: list[torch.Tensor] = []
    targets_out: list[torch.Tensor] = []

    loss_fn = nn.BCEWithLogitsLoss(reduction="sum")
    total_loss = 0.0
    total_values = 0

    for patches, mask, targets in loader:
        patches = patches.to(device)
        mask = mask.to(device)
        targets = targets.to(device)

        logits = model(patches, mask)

        total_loss += float(loss_fn(logits, targets).item())
        total_values += int(targets.numel())

        logits_out.append(logits.cpu())
        targets_out.append(targets.cpu())

    return (
        torch.cat(logits_out, dim=0),
        torch.cat(targets_out, dim=0),
        total_loss / max(1, total_values),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train attention pooling over Discogs-EffNet patches."
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

    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-pos-weight", type=float, default=20.0)
    parser.add_argument("--attention-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument(
        "--max-patches",
        type=int,
        default=256,
        help=(
            "Evenly subsample at most this many patches across the full track. "
            "Use 0 to disable subsampling."
        ),
    )
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.epochs < 1 or args.patience < 1 or args.batch_size < 1:
        raise SystemExit("epochs, patience, and batch-size must be >= 1")

    max_patches = None if args.max_patches == 0 else args.max_patches
    if max_patches is not None and max_patches < 1:
        raise SystemExit("--max-patches must be >= 1, or 0 for unlimited")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    classes = load_active_classes(args.active_classes)
    class_to_index = {
        item["id"]: item["index"]
        for item in classes
    }
    taxonomy = load_taxonomy(args.taxonomy)

    split_dir = args.splits_dir / args.split

    train_records = load_jsonl(split_dir / "train.jsonl")
    validation_records = load_jsonl(split_dir / "validation.jsonl")

    train_dataset = PatchDataset(
        train_records,
        class_to_index=class_to_index,
        taxonomy=taxonomy,
        patches_dir=args.patches_dir,
        max_patches=max_patches,
    )
    val_dataset = PatchDataset(
        validation_records,
        class_to_index=class_to_index,
        taxonomy=taxonomy,
        patches_dir=args.patches_dir,
        max_patches=max_patches,
    )

    y_train = np.stack(
        [item[1] for item in train_dataset.items]
    ).astype(np.float32)

    positive = y_train.sum(axis=0)
    negative = len(y_train) - positive

    if np.any(positive == 0):
        empty = [
            classes[i]["id"]
            for i, value in enumerate(positive)
            if value == 0
        ]
        raise ValueError(
            "active class has zero positives in training split: "
            + ", ".join(empty)
        )

    raw_pos_weight = negative / positive
    pos_weight = np.clip(
        raw_pos_weight,
        1.0,
        args.max_pos_weight,
    ).astype(np.float32)

    generator = torch.Generator()
    generator.manual_seed(args.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=args.num_workers,
        collate_fn=collate_patch_batch,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_patch_batch,
        pin_memory=torch.cuda.is_available(),
    )

    device = choose_device(args.device)

    model = AttentionMLP(
        output_dim=len(classes),
        attention_dim=args.attention_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)

    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.from_numpy(pos_weight).to(device)
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    run_name = f"{args.split}_attention_mlp"
    run_dir = args.runs_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = run_dir / "model.pt"
    history_path = run_dir / "history.csv"
    report_path = run_dir / "training_report.json"

    best_macro_f1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []

    print(f"Run:              {run_name}")
    print(f"Device:           {device}")
    print(f"Active classes:   {len(classes)}")
    print(f"Train samples:    {len(train_dataset)}")
    print(f"Validation:       {len(val_dataset)}")
    print(f"Max patches:      {max_patches or 'unlimited'}")
    print(f"Attention dim:    {args.attention_dim}")
    print()

    started = time.monotonic()

    for epoch in range(1, args.epochs + 1):
        model.train()

        train_loss_sum = 0.0
        train_examples = 0

        for patches, mask, targets in train_loader:
            patches = patches.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            logits = model(patches, mask)
            loss = loss_fn(logits, targets)

            loss.backward()
            optimizer.step()

            batch_size = targets.shape[0]
            train_loss_sum += float(loss.item()) * batch_size
            train_examples += batch_size

        train_loss = train_loss_sum / max(1, train_examples)

        val_logits, val_targets, val_loss = predict_all(
            model,
            val_loader,
            device,
        )
        metrics = compute_metrics(
            val_logits,
            val_targets,
            threshold=0.5,
        )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": val_loss,
            "validation_macro_f1_at_0_5": metrics["macro_f1"],
            "validation_micro_f1_at_0_5": metrics["micro_f1"],
        }
        history.append(row)

        improved = metrics["macro_f1"] > best_macro_f1 + 1e-6

        if improved:
            best_macro_f1 = metrics["macro_f1"]
            best_epoch = epoch
            epochs_without_improvement = 0

            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "model_name": "attention_mlp",
                    "input_dim": EMBEDDING_DIM,
                    "output_dim": len(classes),
                    "attention_dim": args.attention_dim,
                    "hidden_dim": args.hidden_dim,
                    "dropout": args.dropout,
                    "max_patches": max_patches,
                    "class_ids": [item["id"] for item in classes],
                    "split": args.split,
                    "seed": args.seed,
                    "epoch": epoch,
                    "validation_macro_f1_at_0_5": best_macro_f1,
                    "confidence_policy": "min_075",
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1

        print(
            f"epoch {epoch:03d} "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"macro_f1={metrics['macro_f1']:.4f} "
            f"micro_f1={metrics['micro_f1']:.4f}"
            + (" *" if improved else "")
        )

        if epochs_without_improvement >= args.patience:
            print(
                f"Early stopping after {args.patience} epochs "
                f"without macro-F1 improvement."
            )
            break

    elapsed = time.monotonic() - started

    with history_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(history[0]),
        )
        writer.writeheader()
        writer.writerows(history)

    report = {
        "run_name": run_name,
        "model": "attention_mlp",
        "split": args.split,
        "device": str(device),
        "active_class_count": len(classes),
        "train_samples": len(train_dataset),
        "validation_samples": len(val_dataset),
        "best_epoch": best_epoch,
        "best_validation_macro_f1_at_0_5": best_macro_f1,
        "elapsed_seconds": round(elapsed, 2),
        "architecture": {
            "embedding_dim": EMBEDDING_DIM,
            "attention_dim": args.attention_dim,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "max_patches": max_patches,
            "patch_normalization": "LayerNorm(1280)",
            "pooling": "learned_scalar_attention",
        },
        "optimization": {
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "max_pos_weight": args.max_pos_weight,
            "batch_size": args.batch_size,
            "patience": args.patience,
            "seed": args.seed,
        },
        "inputs": {
            "splits_dir": str(args.splits_dir),
            "active_classes": str(args.active_classes),
            "patches_dir": str(args.patches_dir),
        },
        "outputs": {
            "checkpoint": str(checkpoint_path),
            "history": str(history_path),
        },
    }

    report_path.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print()
    print(f"Best epoch:       {best_epoch}")
    print(f"Best val macro:   {best_macro_f1:.4f}")
    print(f"Checkpoint:       {checkpoint_path}")
    print(f"Report:           {report_path}")


if __name__ == "__main__":
    main()
