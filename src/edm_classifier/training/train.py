"""Train single-label parent-genre classifiers on cached audio embeddings.

Models:
  linear / mlp  -> pooled per-track embeddings [D]
  attention     -> temporal embeddings [T,D] (Discogs patches; MERT/MuQ windows)

The split files remain encoder-independent. Embeddings are resolved from video_id.
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

# First entry is canonical. Additional entries preserve compatibility with the
# pre-namespaced Discogs extraction layout recorded by patch_report.json.
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
# Backward-compatible export used by evaluate.py / older callers.
DEFAULT_EMBEDDING_DIRS = {k: v[0] for k, v in DEFAULT_POOLED_DIR_CANDIDATES.items()}


def _first_existing(candidates: list[Path]) -> Path:
    for p in candidates:
        if p.is_dir():
            return p
    return candidates[0]


class EmbeddingResolver:
    """Resolve pooled or temporal embeddings for a split row."""
    def __init__(
        self,
        encoder: str,
        embeddings_dir: Path | None = None,
        *,
        representation: str = "pooled",
    ) -> None:
        if encoder not in ENCODER_MODEL_NAMES:
            raise ValueError(f"unknown encoder {encoder!r}")
        if representation not in {"pooled", "sequence"}:
            raise ValueError(representation)
        self.encoder = encoder
        self.representation = representation
        self.explicit_dir = embeddings_dir is not None
        candidates = (
            DEFAULT_POOLED_DIR_CANDIDATES[encoder]
            if representation == "pooled"
            else DEFAULT_SEQUENCE_DIR_CANDIDATES[encoder]
        )
        self.root = embeddings_dir or _first_existing(candidates)

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

        # Optional explicit multi-encoder paths in split rows.
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

        # Preserve legacy Discogs pooled path only for pooled models.
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


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
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
    raise ValueError(f"sample {row.get('video_id') or row.get('sample_id')!r} has no integer target_index")


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


def compute_normalization(rows: list[dict[str, Any]], embedding_dim: int, resolver: EmbeddingResolver) -> tuple[np.ndarray, np.ndarray]:
    """Track-balanced normalization.

    For pooled vectors this is the ordinary per-dimension mean/std over tracks.
    For sequences each track contributes its own first/second moment once, so a
    3-minute song does not outweigh a 30-second song merely because it has more
    windows/patches.
    """
    total = np.zeros(embedding_dim, dtype=np.float64)
    total_sq = np.zeros(embedding_dim, dtype=np.float64)
    count = 0
    for row in rows:
        path = resolver.path(row)
        a = np.load(path, mmap_mode="r", allow_pickle=False)
        x = np.asarray(a, dtype=np.float64)
        if resolver.representation == "pooled":
            if x.shape != (embedding_dim,):
                raise ValueError(f"{path}: expected {(embedding_dim,)}, got {x.shape}")
            first, second = x, x * x
        else:
            if x.ndim != 2 or x.shape[1] != embedding_dim or x.shape[0] < 1:
                raise ValueError(f"{path}: expected [T,{embedding_dim}], got {x.shape}")
            first = x.mean(axis=0)
            second = np.square(x).mean(axis=0)
        if not np.isfinite(first).all() or not np.isfinite(second).all():
            raise ValueError(f"{path}: non-finite embedding")
        total += first
        total_sq += second
        count += 1
    if count == 0:
        raise ValueError("cannot normalize empty training set")
    mean = total / count
    variance = np.maximum(total_sq / count - mean * mean, 1e-12)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


class EmbeddingDataset(Dataset):
    def __init__(self, rows, *, mean, std, class_count, resolver):
        self.items = []
        self.mean = mean
        self.std = std
        self.embedding_dim = int(mean.shape[0])
        self.representation = resolver.representation
        missing = []
        for row in rows:
            path = resolver.path(row)
            target = target_index(row)
            if not 0 <= target < class_count:
                raise ValueError(f"target index out of range: {target}")
            if not path.is_file():
                missing.append(str(path)); continue
            self.items.append((path, target))
        if missing:
            raise FileNotFoundError(f"{len(missing)} embeddings missing. First: {', '.join(missing[:5])}")
        if not self.items:
            raise ValueError("dataset is empty")

    def __len__(self): return len(self.items)

    def __getitem__(self, index):
        path, target = self.items[index]
        x = np.asarray(np.load(path, mmap_mode="r", allow_pickle=False), dtype=np.float32)
        if self.representation == "pooled":
            if x.shape != (self.embedding_dim,):
                raise ValueError(f"{path}: expected {(self.embedding_dim,)}, got {x.shape}")
        else:
            if x.ndim != 2 or x.shape[1] != self.embedding_dim or x.shape[0] < 1:
                raise ValueError(f"{path}: expected [T,{self.embedding_dim}], got {x.shape}")
        if not np.isfinite(x).all():
            raise ValueError(f"{path}: non-finite embedding")
        x = (x - self.mean) / self.std
        return torch.from_numpy(x.copy()), target


def collate_sequences(batch):
    xs, ys = zip(*batch)
    lengths = torch.tensor([x.shape[0] for x in xs], dtype=torch.long)
    d = xs[0].shape[1]
    max_t = int(lengths.max())
    padded = torch.zeros((len(xs), max_t, d), dtype=torch.float32)
    mask = torch.zeros((len(xs), max_t), dtype=torch.bool)
    for i, x in enumerate(xs):
        t = x.shape[0]
        padded[i, :t] = x
        mask[i, :t] = True
    return padded, mask, torch.tensor(ys, dtype=torch.long)


class LinearClassifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__(); self.classifier = nn.Linear(input_dim, output_dim)
    def forward(self, x): return self.classifier(x)


class MLPClassifier(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, dropout):
        super().__init__(); self.classifier = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, output_dim))
    def forward(self, x): return self.classifier(x)


class AttentionClassifier(nn.Module):
    """Single-head learned attention pooling over a variable-length sequence."""
    def __init__(self, input_dim, output_dim, hidden_dim, dropout):
        super().__init__()
        self.projection = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh(), nn.Dropout(dropout))
        self.score = nn.Linear(hidden_dim, 1, bias=False)
        self.classifier = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, mask=None):
        h = self.projection(x)                       # [B,T,H]
        scores = self.score(h).squeeze(-1)           # [B,T]
        if mask is not None:
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1)
        pooled = torch.sum(h * weights.unsqueeze(-1), dim=1)
        return self.classifier(pooled)


def build_model(name, *, input_dim, output_dim, hidden_dim, dropout):
    if name == "linear": return LinearClassifier(input_dim, output_dim)
    if name == "mlp": return MLPClassifier(input_dim, output_dim, hidden_dim, dropout)
    if name == "attention": return AttentionClassifier(input_dim, output_dim, hidden_dim, dropout)
    raise ValueError(name)


def choose_device(requested):
    if requested != "auto": return torch.device(requested)
    if torch.cuda.is_available(): return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def class_weights(targets, class_count, mode):
    counts = np.bincount(targets, minlength=class_count).astype(np.float64)
    if np.any(counts == 0): raise ValueError(f"training classes with zero support: {np.flatnonzero(counts == 0).tolist()}")
    if mode == "none": weights = np.ones(class_count)
    elif mode == "balanced": weights = len(targets) / (class_count * counts)
    elif mode == "sqrt_balanced": weights = np.sqrt(len(targets) / (class_count * counts))
    else: raise ValueError(mode)
    weights /= weights.mean()
    return counts.astype(np.int64), weights.astype(np.float32)


def per_class_f1(true, pred, class_count):
    result = np.zeros(class_count, dtype=np.float64)
    for i in range(class_count):
        tp = np.sum((true == i) & (pred == i)); fp = np.sum((true != i) & (pred == i)); fn = np.sum((true == i) & (pred != i))
        precision = tp / (tp + fp) if tp + fp else 0.0; recall = tp / (tp + fn) if tp + fn else 0.0
        result[i] = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return result


def metrics_from_logits(logits, targets, *, class_count, other_index):
    pred = torch.argmax(logits, dim=1).cpu().numpy(); true = targets.cpu().numpy()
    f1 = per_class_f1(true, pred, class_count); support = np.bincount(true, minlength=class_count)
    mask = support > 0; retained = mask.copy()
    if other_index is not None: retained[other_index] = False
    return {
        "accuracy": float(np.mean(pred == true)),
        "macro_f1": float(f1[mask].mean()) if mask.any() else 0.0,
        "macro_f1_excluding_other": float(f1[retained].mean()) if retained.any() else 0.0,
        "weighted_f1": float(np.sum(f1 * support) / support.sum()) if support.sum() else 0.0,
    }


def _forward_batch(model, batch, device, representation):
    if representation == "sequence":
        x, mask, y = batch
        return model(x.to(device), mask.to(device)), y.to(device)
    x, y = batch
    return model(x.to(device)), y.to(device)


@torch.no_grad()
def predict_all(model, loader, device, representation="pooled"):
    model.eval(); logits_out=[]; targets_out=[]; total_loss=0.0; total_examples=0
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    for batch in loader:
        logits, y = _forward_batch(model, batch, device, representation)
        total_loss += float(loss_fn(logits, y).item()); total_examples += int(y.shape[0])
        logits_out.append(logits.cpu()); targets_out.append(y.cpu())
    return torch.cat(logits_out), torch.cat(targets_out), total_loss / max(total_examples, 1)


def parse_args():
    p = argparse.ArgumentParser(description="Train single-label classifier on cached pooled or temporal embeddings.")
    p.add_argument("--split", choices=["regular", "artist"], required=True)
    p.add_argument("--model", choices=["linear", "mlp", "attention"], default="mlp")
    p.add_argument("--encoder", choices=sorted(ENCODER_MODEL_NAMES), default="discogs")
    p.add_argument("--embeddings-dir", type=Path, default=None, help="Override embedding directory. For attention this must contain [T,D] files.")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    p.add_argument("--epochs", type=int, default=100); p.add_argument("--patience", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=None, help="Default: 128 pooled, 16 attention")
    p.add_argument("--learning-rate", type=float, default=1e-3); p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hidden-dim", type=int, default=512); p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--class-weight", choices=["none", "balanced", "sqrt_balanced"], default="sqrt_balanced")
    p.add_argument("--seed", type=int, default=1337); p.add_argument("--device", default="auto"); p.add_argument("--num-workers", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args(); random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    representation = "sequence" if args.model == "attention" else "pooled"
    batch_size = args.batch_size or (16 if representation == "sequence" else 128)

    classes = load_classes(args.data_dir / "classes.json"); class_count = len(classes)
    other_index = next((int(x["index"]) for x in classes if x.get("id") == "other"), None)
    split_dir = args.data_dir / args.split; train_rows = load_jsonl(split_dir / "train.jsonl"); val_rows = load_jsonl(split_dir / "validation.jsonl")
    resolver = EmbeddingResolver(args.encoder, args.embeddings_dir, representation=representation)
    embedding_dim = infer_embedding_dim(train_rows, resolver)
    mean, std = compute_normalization(train_rows, embedding_dim, resolver)
    train_targets = np.asarray([target_index(r) for r in train_rows], dtype=np.int64)
    counts, weights = class_weights(train_targets, class_count, args.class_weight)
    train_ds = EmbeddingDataset(train_rows, mean=mean, std=std, class_count=class_count, resolver=resolver)
    val_ds = EmbeddingDataset(val_rows, mean=mean, std=std, class_count=class_count, resolver=resolver)
    generator = torch.Generator().manual_seed(args.seed)
    collate = collate_sequences if representation == "sequence" else None
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=generator, num_workers=args.num_workers, collate_fn=collate, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate, pin_memory=torch.cuda.is_available())
    device = choose_device(args.device)
    model = build_model(args.model, input_dim=embedding_dim, output_dim=class_count, hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=torch.from_numpy(weights).to(device)); optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    run_name = f"{args.encoder}_{args.split}_{args.model}"; run_dir = args.runs_dir / run_name; run_dir.mkdir(parents=True, exist_ok=True)
    np.savez(run_dir / "normalization.npz", mean=mean, std=std)
    history=[]; best_score=-1.0; best_epoch=0; stale=0; started=time.monotonic()
    print(f"Run:             {run_name}\nDevice:          {device}\nModel:           {args.model}\nEncoder:         {args.encoder}\nRepresentation:  {representation}\nEmbeddings:      {resolver.root}\nClasses:         {class_count}\nEmbedding dim:   {embedding_dim}\nTrain samples:   {len(train_ds)}\nValidation:      {len(val_ds)}\nBatch size:      {batch_size}\nClass weighting: {args.class_weight}\n")

    for epoch in range(1, args.epochs + 1):
        model.train(); loss_sum=0.0; n_examples=0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True); logits, y = _forward_batch(model, batch, device, representation); loss = loss_fn(logits, y); loss.backward(); optimizer.step()
            n=int(y.shape[0]); loss_sum += float(loss.item())*n; n_examples += n
        train_loss = loss_sum / max(n_examples, 1)
        val_logits, val_targets, val_loss = predict_all(model, val_loader, device, representation)
        val_metrics = metrics_from_logits(val_logits, val_targets, class_count=class_count, other_index=other_index)
        score = val_metrics["macro_f1_excluding_other"]; improved = score > best_score + 1e-6
        history.append({"epoch":epoch,"train_loss":train_loss,"validation_loss":val_loss,**{f"validation_{k}":v for k,v in val_metrics.items()}})
        if improved:
            best_score=score; best_epoch=epoch; stale=0
            torch.save({
                "state_dict": model.state_dict(), "model_name": args.model, "input_dim": embedding_dim, "output_dim": class_count,
                "hidden_dim": args.hidden_dim, "dropout": args.dropout, "class_ids":[x["id"] for x in classes], "class_weight_mode":args.class_weight,
                "class_counts":counts.tolist(), "class_weights":weights.tolist(), "split":args.split, "seed":args.seed, "epoch":epoch,
                "best_validation_macro_f1_excluding_other":best_score, "encoder":args.encoder, "encoder_model":ENCODER_MODEL_NAMES[args.encoder],
                "embedding_source":resolver.describe(), "representation":representation, "pooling":"learned_attention" if representation=="sequence" else "mean",
            }, run_dir / "model.pt")
        else: stale += 1
        print(f"epoch {epoch:03d} train_loss={train_loss:.4f} val_loss={val_loss:.4f} acc={val_metrics['accuracy']:.4f} macro={val_metrics['macro_f1']:.4f} macro_no_other={score:.4f}" + (" *" if improved else ""))
        if stale >= args.patience:
            print(f"Early stopping after {args.patience} epochs without validation macro-F1 improvement."); break

    history_path=run_dir/"history.csv"
    with history_path.open("w",encoding="utf-8",newline="") as h:
        w=csv.DictWriter(h,fieldnames=list(history[0])); w.writeheader(); w.writerows(history)
    report={
        "run_name":run_name,"task":"single_label_parent_genre","encoder":args.encoder,"encoder_model":ENCODER_MODEL_NAMES[args.encoder],
        "embedding_source":resolver.describe(),"representation":representation,"pooling":"learned_attention" if representation=="sequence" else "mean",
        "split":args.split,"model":args.model,"device":str(device),"class_count":class_count,"other_index":other_index,"embedding_dim":embedding_dim,
        "train_samples":len(train_ds),"validation_samples":len(val_ds),"best_epoch":best_epoch,"best_validation_macro_f1_excluding_other":best_score,
        "elapsed_seconds":round(time.monotonic()-started,2),"optimization":{"loss":"CrossEntropyLoss","class_weight":args.class_weight,"learning_rate":args.learning_rate,
        "weight_decay":args.weight_decay,"batch_size":batch_size,"hidden_dim":args.hidden_dim if args.model in {"mlp","attention"} else None,"dropout":args.dropout if args.model in {"mlp","attention"} else None,"seed":args.seed},
        "outputs":{"checkpoint":str(run_dir/"model.pt"),"normalization":str(run_dir/"normalization.npz"),"history":str(history_path)}}
    (run_dir/"training_report.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(f"\nBest epoch:       {best_epoch}\nBest macro/no-O:  {best_score:.4f}\nCheckpoint:       {run_dir/'model.pt'}")

if __name__ == "__main__": main()
