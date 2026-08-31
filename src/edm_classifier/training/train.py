"""Train a single-label parent-genre classifier on pooled embeddings.

The split files define sample membership and labels. Embedding selection is
independent: ``--encoder`` resolves each pooled vector from the row video_id,
so Discogs, MERT-95M, and MuQ can share the exact same splits.

Task:
    one target_index per sample
    CrossEntropyLoss
    softmax/argmax inference

Models:
    linear
    mlp: input -> 512 -> ReLU -> dropout -> classes

Validation macro F1 excluding the optional "other" class drives early stopping.
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


DEFAULT_DATA_DIR = Path("data/parent_single")
DEFAULT_RUNS_DIR = Path("data/runs/single_label")

ENCODER_MODEL_NAMES = {
    "discogs": "discogs-effnet-bs64-1",
    "mert95m": "m-a-p/MERT-v1-95M",
    "muq": "OpenMuQ/MuQ-large-msd-iter",
}
DEFAULT_EMBEDDING_DIRS = {
    "discogs": Path("data/embeddings/discogs/pooled"),
    "mert95m": Path("data/embeddings/mert95m/pooled"),
    "muq": Path("data/embeddings/muq/pooled"),
}


class EmbeddingResolver:
    """Resolve one pooled embedding without encoder-specific split copies.

    Resolution order when --embeddings-dir is not explicitly supplied:
      1. row["embedding_paths"][encoder], if present
      2. legacy row["embedding_path"] / row["embedding"] for Discogs
      3. default encoder directory + <video_id>.npy

    An explicit --embeddings-dir always wins and derives the filename from
    video_id. This makes moving embeddings to cloud/local storage trivial.
    """

    def __init__(
        self,
        encoder: str,
        embeddings_dir: Path | None = None,
    ) -> None:
        if encoder not in DEFAULT_EMBEDDING_DIRS:
            raise ValueError(f"unknown encoder {encoder!r}")
        self.encoder = encoder
        self.explicit_dir = embeddings_dir is not None
        self.root = embeddings_dir or DEFAULT_EMBEDDING_DIRS[encoder]

    @staticmethod
    def _video_id(row: dict[str, Any]) -> str:
        value = row.get("video_id") or row.get("sample_id")
        if not isinstance(value, str) or not value:
            raise ValueError("sample has no video_id/sample_id")
        return value

    def path(self, row: dict[str, Any]) -> Path:
        video_id = self._video_id(row)

        if self.explicit_dir:
            return self.root / f"{video_id}.npy"

        paths = row.get("embedding_paths")
        if isinstance(paths, dict):
            value = paths.get(self.encoder)
            if isinstance(value, str) and value:
                return Path(value)

        embedding = row.get("embedding")
        if isinstance(embedding, dict):
            pooled_paths = embedding.get("pooled_paths")
            if isinstance(pooled_paths, dict):
                value = pooled_paths.get(self.encoder)
                if isinstance(value, str) and value:
                    return Path(value)

        if self.encoder == "discogs":
            value = row.get("embedding_path")
            if isinstance(value, str) and value:
                return Path(value)
            if isinstance(embedding, dict):
                value = embedding.get("pooled_path") or embedding.get("embedding_path")
                if isinstance(value, str) and value:
                    return Path(value)

        return self.root / f"{video_id}.npy"

    def describe(self) -> dict[str, Any]:
        return {
            "encoder": self.encoder,
            "model": ENCODER_MODEL_NAMES[self.encoder],
            "pooled_dir": str(self.root),
            "explicit_dir": self.explicit_dir,
        }



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


def load_classes(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    classes = raw.get("classes") if isinstance(raw, dict) else None
    if not isinstance(classes, list) or not classes:
        raise ValueError(f"{path}: missing classes array")

    result = [dict(x) for x in classes if isinstance(x, dict)]
    result.sort(key=lambda x: int(x["index"]))

    for expected, item in enumerate(result):
        if item.get("index") != expected:
            raise ValueError(f"{path}: class indices must be contiguous")
        if not isinstance(item.get("id"), str):
            raise ValueError(f"{path}: invalid class at index {expected}")

    return result



def target_index(row: dict[str, Any]) -> int:
    value = row.get("target_index")
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    raise ValueError(
        f"sample {row.get('video_id') or row.get('sample_id')!r} "
        "has no integer target_index"
    )


def infer_embedding_dim(
    rows: list[dict[str, Any]],
    resolver: EmbeddingResolver,
) -> int:
    for row in rows:
        path = resolver.path(row)
        if not path.is_file():
            continue
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if array.ndim != 1:
            raise ValueError(f"{path}: expected 1D pooled vector, got {array.shape}")
        return int(array.shape[0])
    raise FileNotFoundError("no readable embeddings found")


def compute_normalization(
    rows: list[dict[str, Any]],
    embedding_dim: int,
    resolver: EmbeddingResolver,
) -> tuple[np.ndarray, np.ndarray]:
    total = np.zeros(embedding_dim, dtype=np.float64)
    total_sq = np.zeros(embedding_dim, dtype=np.float64)
    count = 0

    for row in rows:
        path = resolver.path(row)
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        vector = np.asarray(array, dtype=np.float64)

        if vector.shape != (embedding_dim,):
            raise ValueError(f"{path}: expected {(embedding_dim,)}, got {vector.shape}")
        if not np.isfinite(vector).all():
            raise ValueError(f"{path}: non-finite embedding")

        total += vector
        total_sq += vector * vector
        count += 1

    if count == 0:
        raise ValueError("cannot normalize empty training set")

    mean = total / count
    variance = np.maximum(total_sq / count - mean * mean, 1e-12)
    std = np.sqrt(variance)

    return mean.astype(np.float32), std.astype(np.float32)


class EmbeddingDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        *,
        mean: np.ndarray,
        std: np.ndarray,
        class_count: int,
        resolver: EmbeddingResolver,
    ):
        self.items: list[tuple[Path, int]] = []
        self.mean = mean
        self.std = std
        self.embedding_dim = int(mean.shape[0])

        missing = []

        for row in rows:
            path = resolver.path(row)
            target = target_index(row)

            if not 0 <= target < class_count:
                raise ValueError(f"target index out of range: {target}")

            if not path.is_file():
                missing.append(str(path))
                continue

            self.items.append((path, target))

        if missing:
            preview = ", ".join(missing[:5])
            raise FileNotFoundError(
                f"{len(missing)} embeddings missing. First: {preview}"
            )
        if not self.items:
            raise ValueError("dataset is empty")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, target = self.items[index]
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        vector = np.asarray(array, dtype=np.float32)

        if vector.shape != (self.embedding_dim,):
            raise ValueError(
                f"{path}: expected {(self.embedding_dim,)}, got {vector.shape}"
            )
        if not np.isfinite(vector).all():
            raise ValueError(f"{path}: non-finite embedding")

        vector = (vector - self.mean) / self.std
        return torch.from_numpy(vector.copy()), target


class LinearClassifier(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.classifier = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class MLPClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


def build_model(
    name: str,
    *,
    input_dim: int,
    output_dim: int,
    hidden_dim: int,
    dropout: float,
) -> nn.Module:
    if name == "linear":
        return LinearClassifier(input_dim, output_dim)
    if name == "mlp":
        return MLPClassifier(
            input_dim,
            output_dim,
            hidden_dim,
            dropout,
        )
    raise ValueError(name)


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def class_weights(
    targets: np.ndarray,
    class_count: int,
    mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    counts = np.bincount(targets, minlength=class_count).astype(np.float64)

    if np.any(counts == 0):
        empty = np.flatnonzero(counts == 0).tolist()
        raise ValueError(f"training classes with zero support: {empty}")

    if mode == "none":
        weights = np.ones(class_count, dtype=np.float64)
    elif mode == "balanced":
        weights = len(targets) / (class_count * counts)
    elif mode == "sqrt_balanced":
        balanced = len(targets) / (class_count * counts)
        weights = np.sqrt(balanced)
    else:
        raise ValueError(mode)

    weights /= weights.mean()
    return counts.astype(np.int64), weights.astype(np.float32)


def per_class_f1(
    true: np.ndarray,
    pred: np.ndarray,
    class_count: int,
) -> np.ndarray:
    result = np.zeros(class_count, dtype=np.float64)
    for i in range(class_count):
        tp = np.sum((true == i) & (pred == i))
        fp = np.sum((true != i) & (pred == i))
        fn = np.sum((true == i) & (pred != i))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        result[i] = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
    return result


def metrics_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    class_count: int,
    other_index: int | None,
) -> dict[str, float]:
    pred = torch.argmax(logits, dim=1).cpu().numpy()
    true = targets.cpu().numpy()

    f1 = per_class_f1(true, pred, class_count)
    support = np.bincount(true, minlength=class_count)

    mask = support > 0
    retained_mask = mask.copy()
    if other_index is not None:
        retained_mask[other_index] = False

    weighted = (
        float(np.sum(f1 * support) / support.sum())
        if support.sum()
        else 0.0
    )

    return {
        "accuracy": float(np.mean(pred == true)),
        "macro_f1": float(f1[mask].mean()) if mask.any() else 0.0,
        "macro_f1_excluding_other": (
            float(f1[retained_mask].mean())
            if retained_mask.any()
            else 0.0
        ),
        "weighted_f1": weighted,
    }


@torch.no_grad()
def predict_all(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    model.eval()

    logits_out = []
    targets_out = []
    total_loss = 0.0
    total_examples = 0

    loss_fn = nn.CrossEntropyLoss(reduction="sum")

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        total_loss += float(loss_fn(logits, y).item())
        total_examples += int(y.shape[0])

        logits_out.append(logits.cpu())
        targets_out.append(y.cpu())

    return (
        torch.cat(logits_out),
        torch.cat(targets_out),
        total_loss / max(total_examples, 1),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train single-label classifier on cached embeddings."
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
        "--encoder",
        choices=sorted(DEFAULT_EMBEDDING_DIRS),
        default="discogs",
        help="Embedding encoder; all encoders reuse the same split rows.",
    )
    parser.add_argument(
        "--embeddings-dir",
        type=Path,
        default=None,
        help=(
            "Override the pooled embedding directory. Files are resolved as "
            "<dir>/<video_id>.npy. Useful for cloud-mounted storage."
        ),
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
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument(
        "--class-weight",
        choices=["none", "balanced", "sqrt_balanced"],
        default="sqrt_balanced",
    )
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    classes = load_classes(args.data_dir / "classes.json")
    class_count = len(classes)
    other_index = next(
        (
            int(item["index"])
            for item in classes
            if item.get("id") == "other"
        ),
        None,
    )

    split_dir = args.data_dir / args.split
    train_rows = load_jsonl(split_dir / "train.jsonl")
    val_rows = load_jsonl(split_dir / "validation.jsonl")

    resolver = EmbeddingResolver(args.encoder, args.embeddings_dir)
    embedding_dim = infer_embedding_dim(train_rows, resolver)
    mean, std = compute_normalization(train_rows, embedding_dim, resolver)

    train_targets = np.asarray(
        [target_index(row) for row in train_rows],
        dtype=np.int64,
    )
    counts, weights = class_weights(
        train_targets,
        class_count,
        args.class_weight,
    )

    train_dataset = EmbeddingDataset(
        train_rows,
        mean=mean,
        std=std,
        class_count=class_count,
        resolver=resolver,
    )
    val_dataset = EmbeddingDataset(
        val_rows,
        mean=mean,
        std=std,
        class_count=class_count,
        resolver=resolver,
    )

    generator = torch.Generator()
    generator.manual_seed(args.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    device = choose_device(args.device)

    model = build_model(
        args.model,
        input_dim=embedding_dim,
        output_dim=class_count,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)

    loss_fn = nn.CrossEntropyLoss(
        weight=torch.from_numpy(weights).to(device)
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    run_name = f"{args.encoder}_{args.split}_{args.model}"
    run_dir = args.runs_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    np.savez(
        run_dir / "normalization.npz",
        mean=mean,
        std=std,
    )

    history = []
    best_score = -1.0
    best_epoch = 0
    stale = 0
    started = time.monotonic()

    print(f"Run:             {run_name}")
    print(f"Device:          {device}")
    print(f"Model:           {args.model}")
    print(f"Encoder:         {args.encoder}")
    print(f"Embeddings:      {resolver.root}")
    print(f"Classes:         {class_count}")
    print(f"Embedding dim:   {embedding_dim}")
    print(f"Train samples:   {len(train_dataset)}")
    print(f"Validation:      {len(val_dataset)}")
    print(f"Class weighting: {args.class_weight}")
    print()

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_examples = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

            n = int(y.shape[0])
            train_loss_sum += float(loss.item()) * n
            train_examples += n

        train_loss = train_loss_sum / max(train_examples, 1)

        val_logits, val_targets, val_loss = predict_all(
            model,
            val_loader,
            device,
        )
        val_metrics = metrics_from_logits(
            val_logits,
            val_targets,
            class_count=class_count,
            other_index=other_index,
        )

        score = val_metrics["macro_f1_excluding_other"]
        improved = score > best_score + 1e-6

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": val_loss,
                **{f"validation_{k}": v for k, v in val_metrics.items()},
            }
        )

        if improved:
            best_score = score
            best_epoch = epoch
            stale = 0

            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "model_name": args.model,
                    "input_dim": embedding_dim,
                    "output_dim": class_count,
                    "hidden_dim": args.hidden_dim,
                    "dropout": args.dropout,
                    "class_ids": [item["id"] for item in classes],
                    "class_weight_mode": args.class_weight,
                    "class_counts": counts.tolist(),
                    "class_weights": weights.tolist(),
                    "split": args.split,
                    "seed": args.seed,
                    "epoch": epoch,
                    "best_validation_macro_f1_excluding_other": best_score,
                    "encoder": args.encoder,
                    "encoder_model": ENCODER_MODEL_NAMES[args.encoder],
                    "embedding_source": resolver.describe(),
                    "pooling": "mean",
                },
                run_dir / "model.pt",
            )
        else:
            stale += 1

        print(
            f"epoch {epoch:03d} "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"acc={val_metrics['accuracy']:.4f} "
            f"macro={val_metrics['macro_f1']:.4f} "
            f"macro_no_other={score:.4f}"
            + (" *" if improved else "")
        )

        if stale >= args.patience:
            print(
                f"Early stopping after {args.patience} epochs without "
                "validation macro-F1 improvement."
            )
            break

    history_path = run_dir / "history.csv"
    with history_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)

    report = {
        "run_name": run_name,
        "task": "single_label_parent_genre",
        "encoder": args.encoder,
        "encoder_model": ENCODER_MODEL_NAMES[args.encoder],
        "embedding_source": resolver.describe(),
        "pooling": "mean",
        "split": args.split,
        "model": args.model,
        "device": str(device),
        "class_count": class_count,
        "other_index": other_index,
        "embedding_dim": embedding_dim,
        "train_samples": len(train_dataset),
        "validation_samples": len(val_dataset),
        "best_epoch": best_epoch,
        "best_validation_macro_f1_excluding_other": best_score,
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "optimization": {
            "loss": "CrossEntropyLoss",
            "class_weight": args.class_weight,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "hidden_dim": args.hidden_dim if args.model == "mlp" else None,
            "dropout": args.dropout if args.model == "mlp" else None,
            "seed": args.seed,
        },
        "outputs": {
            "checkpoint": str(run_dir / "model.pt"),
            "normalization": str(run_dir / "normalization.npz"),
            "history": str(history_path),
        },
    }

    (run_dir / "training_report.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print()
    print(f"Best epoch:       {best_epoch}")
    print(f"Best macro/no-O:  {best_score:.4f}")
    print(f"Checkpoint:       {run_dir / 'model.pt'}")


if __name__ == "__main__":
    main()
