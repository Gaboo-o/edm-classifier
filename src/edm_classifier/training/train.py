"""Train single-label parent-genre classifiers on cached audio embeddings.

The new EDM ablation path keeps the split rows fixed while varying:

* encoder: Discogs / MERT-95M / MuQ
* model: linear / MLP / learned temporal attention
* view: whole / center30 / energy20
* rhythm: none / global hand-engineered rhythm vector

For MERT/MuQ, pooled models derive their vector from the saved 5-second window
sequence so whole/center30/energy20 differ only in which windows are retained.
No new foundation-model inference is required.
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
DEFAULT_FEATURES_DIR = Path("data/features/edm")

ENCODER_MODEL_NAMES = {
    "discogs": "discogs-effnet-bs64-1",
    "mert95m": "m-a-p/MERT-v1-95M",
    "muq": "OpenMuQ/MuQ-large-msd-iter",
}
WINDOWED_ENCODERS = {"mert95m", "muq"}
VIEW_CHOICES = ("whole", "center30", "energy20")
RHYTHM_CHOICES = ("none", "global")

DEFAULT_POOLED_DIR_CANDIDATES = {
    "discogs": [Path("data/embeddings/discogs/pooled"), Path("data/embeddings/pooled")],
    "mert95m": [Path("data/embeddings/mert95m/pooled")],
    "muq": [Path("data/embeddings/muq/pooled")],
}
DEFAULT_SEQUENCE_DIR_CANDIDATES = {
    "discogs": [Path("data/embeddings/discogs/patches"), Path("data/embeddings/patches")],
    "mert95m": [Path("data/embeddings/mert95m/windows")],
    "muq": [Path("data/embeddings/muq/windows")],
}
DEFAULT_EMBEDDING_DIRS = {k: v[0] for k, v in DEFAULT_POOLED_DIR_CANDIDATES.items()}


def _first_existing(candidates: list[Path]) -> Path:
    for path in candidates:
        if path.is_dir():
            return path
    return candidates[0]


class EmbeddingResolver:
    """Resolve one canonical pooled or sequence embedding from a split row."""

    def __init__(self, encoder: str, embeddings_dir: Path | None = None, *, representation: str = "pooled") -> None:
        if encoder not in ENCODER_MODEL_NAMES:
            raise ValueError(f"unknown encoder {encoder!r}")
        if representation not in {"pooled", "sequence"}:
            raise ValueError(representation)
        self.encoder = encoder
        self.representation = representation
        self.explicit_dir = embeddings_dir is not None
        candidates = DEFAULT_POOLED_DIR_CANDIDATES[encoder] if representation == "pooled" else DEFAULT_SEQUENCE_DIR_CANDIDATES[encoder]
        self.root = embeddings_dir or _first_existing(candidates)

    @staticmethod
    def video_id(row: dict[str, Any]) -> str:
        value = row.get("video_id") or row.get("sample_id")
        if not isinstance(value, str) or not value:
            raise ValueError("sample has no video_id/sample_id")
        return value

    def path(self, row: dict[str, Any]) -> Path:
        video_id = self.video_id(row)
        if self.explicit_dir:
            return self.root / f"{video_id}.npy"

        paths = row.get("embedding_paths")
        if isinstance(paths, dict):
            key = self.encoder if self.representation == "pooled" else f"{self.encoder}_sequence"
            value = paths.get(key)
            if isinstance(value, str) and value:
                return Path(value)

        embedding = row.get("embedding")
        if isinstance(embedding, dict):
            map_key = "pooled_paths" if self.representation == "pooled" else "sequence_paths"
            mapped = embedding.get(map_key)
            if isinstance(mapped, dict):
                value = mapped.get(self.encoder)
                if isinstance(value, str) and value:
                    return Path(value)

        if self.encoder == "discogs" and self.representation == "pooled":
            value = row.get("embedding_path")
            if isinstance(value, str) and value and Path(value).is_file():
                return Path(value)
            if isinstance(embedding, dict):
                value = embedding.get("pooled_path") or embedding.get("embedding_path")
                if isinstance(value, str) and value and Path(value).is_file():
                    return Path(value)

        return self.root / f"{video_id}.npy"

    def describe(self) -> dict[str, Any]:
        return {
            "encoder": self.encoder,
            "model": ENCODER_MODEL_NAMES[self.encoder],
            "representation": self.representation,
            "directory": str(self.root),
            "explicit_dir": self.explicit_dir,
        }


class FeatureResolver:
    """Resolve rhythm vectors and MERT-aligned section metadata."""

    def __init__(self, root: Path = DEFAULT_FEATURES_DIR) -> None:
        self.root = root
        self.rhythm_dir = root / "rhythm"
        self.sections_dir = root / "sections"

    def rhythm_path(self, row: dict[str, Any]) -> Path:
        return self.rhythm_dir / f"{EmbeddingResolver.video_id(row)}.npy"

    def section_path(self, row: dict[str, Any]) -> Path:
        return self.sections_dir / f"{EmbeddingResolver.video_id(row)}.npz"

    def bounds(self, row: dict[str, Any], view: str, window_count: int) -> tuple[int, int]:
        if view == "whole":
            return 0, window_count
        if view not in {"center30", "energy20"}:
            raise ValueError(view)
        path = self.section_path(row)
        if not path.is_file():
            raise FileNotFoundError(f"missing section features: {path}")
        with np.load(path, allow_pickle=False) as section:
            expected = int(np.asarray(section["window_count"]).item())
            if expected != window_count:
                raise ValueError(
                    f"{path}: section window_count={expected} does not match embedding T={window_count}"
                )
            start = int(np.asarray(section[f"{view}_start"]).item())
            end = int(np.asarray(section[f"{view}_end"]).item())
        if not 0 <= start < end <= window_count:
            raise ValueError(f"{path}: invalid {view} bounds [{start}, {end}) for T={window_count}")
        return start, end

    def describe(self) -> dict[str, Any]:
        return {"directory": str(self.root)}


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
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected object")
            rows.append(row)
    return rows


def load_classes(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    classes = raw.get("classes") if isinstance(raw, dict) else None
    if not isinstance(classes, list) or not classes:
        raise ValueError(f"{path}: missing classes array")
    result = [dict(item) for item in classes if isinstance(item, dict)]
    result.sort(key=lambda item: int(item["index"]))
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
    raise ValueError(f"sample {row.get('video_id') or row.get('sample_id')!r} has no integer target_index")


def representation_plan(encoder: str, model: str, view: str) -> tuple[str, str]:
    """Return (source_representation, model_representation)."""
    model_representation = "sequence" if model == "attention" else "pooled"
    if view != "whole" and encoder not in WINDOWED_ENCODERS:
        raise ValueError(f"--view {view} requires a 5-second-window encoder (MERT/MuQ)")
    # For a controlled MERT/MuQ ablation, even pooled models derive their vector
    # from the saved windows. This keeps whole/center/energy pooling identical
    # except for the selected window indices.
    if encoder in WINDOWED_ENCODERS:
        source_representation = "sequence"
    else:
        source_representation = "sequence" if model_representation == "sequence" else "pooled"
    return source_representation, model_representation


def _load_selected_deep(
    row: dict[str, Any],
    *,
    resolver: EmbeddingResolver,
    features: FeatureResolver,
    view: str,
    model_representation: str,
) -> np.ndarray:
    path = resolver.path(row)
    if not path.is_file():
        raise FileNotFoundError(path)
    raw = np.load(path, mmap_mode="r", allow_pickle=False)
    if resolver.representation == "pooled":
        if view != "whole" or model_representation != "pooled":
            raise ValueError("pooled source embeddings only support whole pooled models")
        if raw.ndim != 1:
            raise ValueError(f"{path}: expected [D], got {raw.shape}")
        x = np.asarray(raw, dtype=np.float32)
    else:
        if raw.ndim != 2 or raw.shape[0] < 1:
            raise ValueError(f"{path}: expected [T,D], got {raw.shape}")
        start, end = features.bounds(row, view, int(raw.shape[0]))
        selected = raw[start:end]
        if model_representation == "pooled":
            x = np.asarray(selected, dtype=np.float32).mean(axis=0)
        else:
            x = np.asarray(selected, dtype=np.float32)
    if not np.isfinite(x).all():
        raise ValueError(f"{path}: non-finite embedding")
    return x


def infer_embedding_dim(rows: list[dict[str, Any]], resolver: EmbeddingResolver) -> int:
    for row in rows:
        path = resolver.path(row)
        if not path.is_file():
            continue
        a = np.load(path, mmap_mode="r", allow_pickle=False)
        expected_ndim = 1 if resolver.representation == "pooled" else 2
        if a.ndim != expected_ndim:
            raise ValueError(f"{path}: expected {expected_ndim}D embedding, got {a.shape}")
        return int(a.shape[-1])
    raise FileNotFoundError(f"no readable {resolver.representation} embeddings found under {resolver.root}")


def compute_embedding_normalization(
    rows: list[dict[str, Any]],
    embedding_dim: int,
    *,
    resolver: EmbeddingResolver,
    features: FeatureResolver,
    view: str,
    model_representation: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit track-balanced normalization using training rows only."""
    total = np.zeros(embedding_dim, dtype=np.float64)
    total_sq = np.zeros(embedding_dim, dtype=np.float64)
    count = 0
    for row in rows:
        x = _load_selected_deep(
            row,
            resolver=resolver,
            features=features,
            view=view,
            model_representation=model_representation,
        ).astype(np.float64)
        if model_representation == "pooled":
            if x.shape != (embedding_dim,):
                raise ValueError(f"expected {(embedding_dim,)}, got {x.shape}")
            first, second = x, x * x
        else:
            if x.ndim != 2 or x.shape[1] != embedding_dim or x.shape[0] < 1:
                raise ValueError(f"expected [T,{embedding_dim}], got {x.shape}")
            first = x.mean(axis=0)
            second = np.square(x).mean(axis=0)
        total += first
        total_sq += second
        count += 1
    if count == 0:
        raise ValueError("cannot normalize empty training set")
    mean = total / count
    variance = np.maximum(total_sq / count - mean * mean, 1e-12)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


def infer_rhythm_dim(rows: list[dict[str, Any]], features: FeatureResolver) -> int:
    for row in rows:
        path = features.rhythm_path(row)
        if path.is_file():
            vector = np.load(path, mmap_mode="r", allow_pickle=False)
            if vector.ndim != 1:
                raise ValueError(f"{path}: expected 1-D rhythm vector, got {vector.shape}")
            return int(vector.shape[0])
    raise FileNotFoundError(f"no rhythm vectors found under {features.rhythm_dir}")


def compute_rhythm_normalization(
    rows: list[dict[str, Any]],
    rhythm_dim: int,
    features: FeatureResolver,
) -> tuple[np.ndarray, np.ndarray]:
    total = np.zeros(rhythm_dim, dtype=np.float64)
    total_sq = np.zeros(rhythm_dim, dtype=np.float64)
    count = 0
    for row in rows:
        path = features.rhythm_path(row)
        vector = np.asarray(np.load(path, mmap_mode="r", allow_pickle=False), dtype=np.float64)
        if vector.shape != (rhythm_dim,):
            raise ValueError(f"{path}: expected {(rhythm_dim,)}, got {vector.shape}")
        if not np.isfinite(vector).all():
            raise ValueError(f"{path}: non-finite rhythm vector")
        total += vector
        total_sq += vector * vector
        count += 1
    mean = total / max(count, 1)
    variance = np.maximum(total_sq / max(count, 1) - mean * mean, 1e-12)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


class EmbeddingDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        *,
        mean: np.ndarray,
        std: np.ndarray,
        class_count: int,
        resolver: EmbeddingResolver,
        features: FeatureResolver,
        view: str,
        model_representation: str,
        rhythm_mode: str,
        rhythm_mean: np.ndarray | None = None,
        rhythm_std: np.ndarray | None = None,
    ) -> None:
        self.items: list[tuple[dict[str, Any], int]] = []
        self.mean = mean
        self.std = std
        self.embedding_dim = int(mean.shape[0])
        self.resolver = resolver
        self.features = features
        self.view = view
        self.model_representation = model_representation
        self.rhythm_mode = rhythm_mode
        self.rhythm_mean = rhythm_mean
        self.rhythm_std = rhythm_std
        self.rhythm_dim = 0 if rhythm_mean is None else int(rhythm_mean.shape[0])
        missing: list[str] = []

        for row in rows:
            target = target_index(row)
            if not 0 <= target < class_count:
                raise ValueError(f"target index out of range: {target}")
            embedding_path = resolver.path(row)
            if not embedding_path.is_file():
                missing.append(str(embedding_path))
                continue
            if resolver.representation == "sequence" and view != "whole":
                # Validate alignment once at dataset construction time.
                shape = np.load(embedding_path, mmap_mode="r", allow_pickle=False).shape
                if len(shape) != 2:
                    raise ValueError(f"{embedding_path}: expected [T,D], got {shape}")
                features.bounds(row, view, int(shape[0]))
            if rhythm_mode == "global" and not features.rhythm_path(row).is_file():
                missing.append(str(features.rhythm_path(row)))
                continue
            self.items.append((row, target))

        if missing:
            raise FileNotFoundError(f"{len(missing)} required files missing. First: {', '.join(missing[:5])}")
        if not self.items:
            raise ValueError("dataset is empty")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        row, target = self.items[index]
        x = _load_selected_deep(
            row,
            resolver=self.resolver,
            features=self.features,
            view=self.view,
            model_representation=self.model_representation,
        )
        if self.model_representation == "pooled":
            if x.shape != (self.embedding_dim,):
                raise ValueError(f"expected {(self.embedding_dim,)}, got {x.shape}")
        else:
            if x.ndim != 2 or x.shape[1] != self.embedding_dim or x.shape[0] < 1:
                raise ValueError(f"expected [T,{self.embedding_dim}], got {x.shape}")
        x = (x - self.mean) / self.std

        rhythm = np.empty((0,), dtype=np.float32)
        if self.rhythm_mode == "global":
            assert self.rhythm_mean is not None and self.rhythm_std is not None
            rhythm = np.asarray(
                np.load(self.features.rhythm_path(row), mmap_mode="r", allow_pickle=False),
                dtype=np.float32,
            )
            rhythm = (rhythm - self.rhythm_mean) / self.rhythm_std

        if self.model_representation == "pooled":
            if rhythm.size:
                x = np.concatenate([x, rhythm]).astype(np.float32, copy=False)
            return torch.from_numpy(np.asarray(x, dtype=np.float32).copy()), target

        return (
            torch.from_numpy(np.asarray(x, dtype=np.float32).copy()),
            torch.from_numpy(rhythm.copy()),
            target,
        )


def collate_sequences(batch):
    xs, rhythms, ys = zip(*batch)
    lengths = torch.tensor([x.shape[0] for x in xs], dtype=torch.long)
    embedding_dim = xs[0].shape[1]
    max_t = int(lengths.max())
    padded = torch.zeros((len(xs), max_t, embedding_dim), dtype=torch.float32)
    mask = torch.zeros((len(xs), max_t), dtype=torch.bool)
    for i, x in enumerate(xs):
        t = x.shape[0]
        padded[i, :t] = x
        mask[i, :t] = True
    rhythm = torch.stack(rhythms) if rhythms[0].numel() else torch.empty((len(xs), 0), dtype=torch.float32)
    return padded, mask, rhythm, torch.tensor(ys, dtype=torch.long)


class LinearClassifier(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.classifier = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class MLPClassifier(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class AttentionClassifier(nn.Module):
    """Single-head temporal attention with optional post-pooling rhythm fusion."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, dropout: float, rhythm_dim: int = 0):
        super().__init__()
        self.rhythm_dim = int(rhythm_dim)
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
        )
        self.score = nn.Linear(hidden_dim, 1, bias=False)
        self.classifier = nn.Linear(hidden_dim + self.rhythm_dim, output_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None, rhythm: torch.Tensor | None = None) -> torch.Tensor:
        h = self.projection(x)
        scores = self.score(h).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1)
        pooled = torch.sum(h * weights.unsqueeze(-1), dim=1)
        if self.rhythm_dim:
            if rhythm is None or rhythm.shape[-1] != self.rhythm_dim:
                raise ValueError(f"expected rhythm dimension {self.rhythm_dim}")
            pooled = torch.cat([pooled, rhythm], dim=1)
        return self.classifier(pooled)


def build_model(
    name: str,
    *,
    input_dim: int,
    output_dim: int,
    hidden_dim: int,
    dropout: float,
    rhythm_dim: int = 0,
) -> nn.Module:
    if name == "linear":
        return LinearClassifier(input_dim, output_dim)
    if name == "mlp":
        return MLPClassifier(input_dim, output_dim, hidden_dim, dropout)
    if name == "attention":
        return AttentionClassifier(input_dim, output_dim, hidden_dim, dropout, rhythm_dim=rhythm_dim)
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


def class_weights(targets: np.ndarray, class_count: int, mode: str) -> tuple[np.ndarray, np.ndarray]:
    counts = np.bincount(targets, minlength=class_count).astype(np.float64)
    if np.any(counts == 0):
        raise ValueError(f"training classes with zero support: {np.flatnonzero(counts == 0).tolist()}")
    if mode == "none":
        weights = np.ones(class_count)
    elif mode == "balanced":
        weights = len(targets) / (class_count * counts)
    elif mode == "sqrt_balanced":
        weights = np.sqrt(len(targets) / (class_count * counts))
    else:
        raise ValueError(mode)
    weights /= weights.mean()
    return counts.astype(np.int64), weights.astype(np.float32)


def per_class_f1(true: np.ndarray, pred: np.ndarray, class_count: int) -> np.ndarray:
    result = np.zeros(class_count, dtype=np.float64)
    for i in range(class_count):
        tp = np.sum((true == i) & (pred == i))
        fp = np.sum((true != i) & (pred == i))
        fn = np.sum((true == i) & (pred != i))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        result[i] = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return result


def metrics_from_logits(logits: torch.Tensor, targets: torch.Tensor, *, class_count: int, other_index: int | None) -> dict[str, float]:
    pred = torch.argmax(logits, dim=1).cpu().numpy()
    true = targets.cpu().numpy()
    f1 = per_class_f1(true, pred, class_count)
    support = np.bincount(true, minlength=class_count)
    mask = support > 0
    retained = mask.copy()
    if other_index is not None:
        retained[other_index] = False
    return {
        "accuracy": float(np.mean(pred == true)),
        "macro_f1": float(f1[mask].mean()) if mask.any() else 0.0,
        "macro_f1_excluding_other": float(f1[retained].mean()) if retained.any() else 0.0,
        "weighted_f1": float(np.sum(f1 * support) / support.sum()) if support.sum() else 0.0,
    }


def _forward_batch(model: nn.Module, batch, device: torch.device, model_representation: str):
    if model_representation == "sequence":
        x, mask, rhythm, y = batch
        return model(x.to(device), mask.to(device), rhythm.to(device)), y.to(device)
    x, y = batch
    return model(x.to(device)), y.to(device)


@torch.no_grad()
def predict_all(model: nn.Module, loader: DataLoader, device: torch.device, model_representation: str = "pooled"):
    model.eval()
    logits_out: list[torch.Tensor] = []
    targets_out: list[torch.Tensor] = []
    total_loss = 0.0
    total_examples = 0
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    for batch in loader:
        logits, y = _forward_batch(model, batch, device, model_representation)
        total_loss += float(loss_fn(logits, y).item())
        total_examples += int(y.shape[0])
        logits_out.append(logits.cpu())
        targets_out.append(y.cpu())
    return torch.cat(logits_out), torch.cat(targets_out), total_loss / max(total_examples, 1)


def make_run_name(encoder: str, split: str, model: str, view: str, rhythm: str) -> str:
    name = f"{encoder}_{split}_{model}_{view}"
    if rhythm == "global":
        name += "_rhythm"
    return name


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train single-label classifier with optional EDM section/rhythm features.")
    p.add_argument("--split", choices=["regular", "artist"], required=True)
    p.add_argument("--model", choices=["linear", "mlp", "attention"], default="mlp")
    p.add_argument("--encoder", choices=sorted(ENCODER_MODEL_NAMES), default="mert95m")
    p.add_argument("--view", choices=VIEW_CHOICES, default="whole")
    p.add_argument("--rhythm", choices=RHYTHM_CHOICES, default="none")
    p.add_argument("--embeddings-dir", type=Path, default=None, help="Override canonical embedding source directory.")
    p.add_argument("--features-dir", type=Path, default=DEFAULT_FEATURES_DIR)
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=None, help="Default: 128 pooled, 16 attention")
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--class-weight", choices=["none", "balanced", "sqrt_balanced"], default="sqrt_balanced")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--device", default="auto")
    p.add_argument("--num-workers", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    source_representation, model_representation = representation_plan(args.encoder, args.model, args.view)
    batch_size = args.batch_size or (16 if model_representation == "sequence" else 128)

    classes = load_classes(args.data_dir / "classes.json")
    class_count = len(classes)
    other_index = next((int(item["index"]) for item in classes if item.get("id") == "other"), None)
    split_dir = args.data_dir / args.split
    train_rows = load_jsonl(split_dir / "train.jsonl")
    val_rows = load_jsonl(split_dir / "validation.jsonl")

    resolver = EmbeddingResolver(args.encoder, args.embeddings_dir, representation=source_representation)
    features = FeatureResolver(args.features_dir)
    embedding_dim = infer_embedding_dim(train_rows, resolver)
    mean, std = compute_embedding_normalization(
        train_rows,
        embedding_dim,
        resolver=resolver,
        features=features,
        view=args.view,
        model_representation=model_representation,
    )

    rhythm_dim = 0
    rhythm_mean = rhythm_std = None
    if args.rhythm == "global":
        rhythm_dim = infer_rhythm_dim(train_rows, features)
        rhythm_mean, rhythm_std = compute_rhythm_normalization(train_rows, rhythm_dim, features)

    train_targets = np.asarray([target_index(row) for row in train_rows], dtype=np.int64)
    counts, weights = class_weights(train_targets, class_count, args.class_weight)
    dataset_kwargs = dict(
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
    train_ds = EmbeddingDataset(train_rows, **dataset_kwargs)
    val_ds = EmbeddingDataset(val_rows, **dataset_kwargs)

    generator = torch.Generator().manual_seed(args.seed)
    collate = collate_sequences if model_representation == "sequence" else None
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=torch.cuda.is_available(),
    )

    device = choose_device(args.device)
    model_input_dim = embedding_dim + rhythm_dim if model_representation == "pooled" else embedding_dim
    model = build_model(
        args.model,
        input_dim=model_input_dim,
        output_dim=class_count,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        rhythm_dim=rhythm_dim if model_representation == "sequence" else 0,
    ).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=torch.from_numpy(weights).to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    run_name = make_run_name(args.encoder, args.split, args.model, args.view, args.rhythm)
    run_dir = args.runs_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    norm_payload = {"mean": mean, "std": std}
    if rhythm_mean is not None and rhythm_std is not None:
        norm_payload.update({"rhythm_mean": rhythm_mean, "rhythm_std": rhythm_std})
    np.savez(run_dir / "normalization.npz", **norm_payload)

    history: list[dict[str, Any]] = []
    best_score = -1.0
    best_epoch = 0
    stale = 0
    started = time.monotonic()
    print(
        f"Run:             {run_name}\n"
        f"Device:          {device}\n"
        f"Model:           {args.model}\n"
        f"Encoder:         {args.encoder}\n"
        f"View:            {args.view}\n"
        f"Rhythm:          {args.rhythm}\n"
        f"Source repr:     {source_representation}\n"
        f"Model repr:      {model_representation}\n"
        f"Embeddings:      {resolver.root}\n"
        f"Features:        {features.root}\n"
        f"Classes:         {class_count}\n"
        f"Deep dim:        {embedding_dim}\n"
        f"Rhythm dim:      {rhythm_dim}\n"
        f"Model input dim: {model_input_dim}\n"
        f"Train samples:   {len(train_ds)}\n"
        f"Validation:      {len(val_ds)}\n"
        f"Batch size:      {batch_size}\n"
        f"Class weighting: {args.class_weight}\n"
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        n_examples = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits, y = _forward_batch(model, batch, device, model_representation)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            n = int(y.shape[0])
            loss_sum += float(loss.item()) * n
            n_examples += n
        train_loss = loss_sum / max(n_examples, 1)
        val_logits, val_targets, val_loss = predict_all(model, val_loader, device, model_representation)
        val_metrics = metrics_from_logits(val_logits, val_targets, class_count=class_count, other_index=other_index)
        score = val_metrics["macro_f1_excluding_other"]
        improved = score > best_score + 1e-6
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": val_loss,
            **{f"validation_{key}": value for key, value in val_metrics.items()},
        })

        if improved:
            best_score = score
            best_epoch = epoch
            stale = 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "model_name": args.model,
                    "input_dim": model_input_dim,
                    "deep_embedding_dim": embedding_dim,
                    "rhythm_dim": rhythm_dim,
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
                    "feature_source": features.describe(),
                    "source_representation": source_representation,
                    "representation": model_representation,
                    "view": args.view,
                    "rhythm": args.rhythm,
                    "pooling": "learned_attention" if model_representation == "sequence" else ("window_mean" if source_representation == "sequence" else "canonical_pooled"),
                },
                run_dir / "model.pt",
            )
        else:
            stale += 1

        print(
            f"epoch {epoch:03d} train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"acc={val_metrics['accuracy']:.4f} macro={val_metrics['macro_f1']:.4f} "
            f"macro_no_other={score:.4f}" + (" *" if improved else "")
        )
        if stale >= args.patience:
            print(f"Early stopping after {args.patience} epochs without validation macro-F1 improvement.")
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
        "feature_source": features.describe(),
        "source_representation": source_representation,
        "representation": model_representation,
        "view": args.view,
        "rhythm": args.rhythm,
        "pooling": "learned_attention" if model_representation == "sequence" else ("window_mean" if source_representation == "sequence" else "canonical_pooled"),
        "split": args.split,
        "model": args.model,
        "device": str(device),
        "class_count": class_count,
        "other_index": other_index,
        "deep_embedding_dim": embedding_dim,
        "rhythm_dim": rhythm_dim,
        "model_input_dim": model_input_dim,
        "train_samples": len(train_ds),
        "validation_samples": len(val_ds),
        "best_epoch": best_epoch,
        "best_validation_macro_f1_excluding_other": best_score,
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "optimization": {
            "loss": "CrossEntropyLoss",
            "class_weight": args.class_weight,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "batch_size": batch_size,
            "hidden_dim": args.hidden_dim if args.model in {"mlp", "attention"} else None,
            "dropout": args.dropout if args.model in {"mlp", "attention"} else None,
            "seed": args.seed,
        },
        "outputs": {
            "checkpoint": str(run_dir / "model.pt"),
            "normalization": str(run_dir / "normalization.npz"),
            "history": str(history_path),
        },
    }
    (run_dir / "training_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nBest epoch:       {best_epoch}\nBest macro/no-O:  {best_score:.4f}\nCheckpoint:       {run_dir / 'model.pt'}")


if __name__ == "__main__":
    main()
