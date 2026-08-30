"""Artist-split parent-only learning curve using the existing pooled MLP.

Builds deterministic nested 25/50/75/100% train subsets, keeps validation/test
fixed, audits 1-parent vs multi-parent samples, trains multiple seeds, evaluates,
and writes CSV + PNG summaries.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import matplotlib.pyplot as plt
import yaml

DEFAULT_SPLITS = Path("data/splits")
DEFAULT_ACTIVE = Path("data/training_parent/active_classes.json")
DEFAULT_TAXONOMY = Path("config/taxonomy.yaml")
DEFAULT_OUTPUT = Path("data/runs/parent_learning_curve")
DEFAULT_FRACTIONS = (0.25, 0.50, 0.75, 1.00)
DEFAULT_SEEDS = (1337, 2027, 9001)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for i, raw in enumerate(f, 1):
            line = raw.strip()
            if not line:
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{i}: expected object")
            rows.append(value)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def load_taxonomy(path: Path) -> dict[str, dict[str, Any]]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    genres = raw.get("genres") if isinstance(raw, dict) else None
    if not isinstance(genres, list):
        raise ValueError(f"{path}: expected genres list")
    out = {}
    for item in genres:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str):
            raise ValueError(f"{path}: invalid genre")
        out[item["id"]] = item
    return out


def root_for(label: str, taxonomy: dict[str, dict[str, Any]]) -> str:
    if label not in taxonomy:
        raise ValueError(f"unknown taxonomy label {label!r}")
    current = label
    seen = set()
    while True:
        if current in seen:
            raise ValueError(f"taxonomy cycle at {label!r}")
        seen.add(current)
        parent = taxonomy[current].get("parent")
        if not isinstance(parent, str) or not parent:
            return current
        if parent not in taxonomy:
            raise ValueError(f"unknown parent {parent!r}")
        current = parent


def parent_targets(row: dict[str, Any], taxonomy: dict[str, dict[str, Any]]) -> tuple[str, ...]:
    labels = row.get("labels")
    if not isinstance(labels, list):
        return ()
    return tuple(sorted({root_for(x, taxonomy) for x in labels if isinstance(x, str) and x}))


def identity(row: dict[str, Any], index: int) -> str:
    for key in ("video_id", "sample_id", "candidate_id"):
        value = row.get(key)
        if isinstance(value, str) and value:
            return value
    p = row.get("embedding_path")
    if isinstance(p, str) and p:
        return Path(p).stem
    return f"row:{index}"


def digest(text: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}\0{text}".encode()).hexdigest()


def fraction_name(f: float) -> str:
    return f"fraction_{round(100*f):03d}"


def build_subsets(rows: list[dict[str, Any]], taxonomy: dict[str, dict[str, Any]], fractions: list[float], seed: int):
    strata = defaultdict(list)
    for i, row in enumerate(rows):
        strata[parent_targets(row, taxonomy)].append((digest(identity(row, i), seed), row))
    for values in strata.values():
        values.sort(key=lambda x: x[0])

    result = {}
    for f in sorted(fractions):
        selected = []
        for values in strata.values():
            n = len(values)
            take = n if f >= 1 else max(1, math.floor(n * f))
            selected.extend(row for _, row in values[:take])
        selected.sort(key=lambda row: digest(identity(row, 0), seed))
        result[f] = selected
    return result


def audit(records: list[dict[str, Any]], taxonomy: dict[str, dict[str, Any]]) -> dict[str, Any]:
    card = Counter()
    support = Counter()
    for row in records:
        targets = parent_targets(row, taxonomy)
        card[len(targets)] += 1
        support.update(targets)
    total = len(records)
    multi = sum(v for k, v in card.items() if k >= 2)
    return {
        "samples": total,
        "parent_target_cardinality": {str(k): v for k, v in sorted(card.items())},
        "single_parent_samples": card[1],
        "multi_parent_samples": multi,
        "zero_parent_samples": card[0],
        "single_parent_fraction": card[1] / total if total else 0,
        "multi_parent_fraction": multi / total if total else 0,
        "parent_support": dict(sorted(support.items())),
    }


def find_report(run_root: Path) -> Path | None:
    run_dir = run_root / "artist_mlp"
    for name in ("evaluation_report_v2.json", "evaluation_report.json"):
        path = run_dir / name
        if path.is_file():
            return path
    return None


def has_checkpoint(run_root: Path) -> bool:
    run_dir = run_root / "artist_mlp"
    return any((run_dir / x).is_file() for x in ("model.pt", "best_model.pt", "checkpoint.pt"))


def run_cmd(args: list[str]) -> None:
    print("\n$ " + " ".join(args))
    subprocess.run(args, check=True)


def train_eval(split_root: Path, active: Path, taxonomy: Path, run_root: Path, seed: int, device: str) -> Path:
    report = find_report(run_root)
    if report:
        print(f"Reusing {report}")
        return report

    if not has_checkpoint(run_root):
        run_cmd([
            sys.executable, "-m", "edm_classifier.training.train",
            "--split", "artist", "--model", "mlp",
            "--splits-dir", str(split_root),
            "--active-classes", str(active),
            "--taxonomy", str(taxonomy),
            "--runs-dir", str(run_root),
            "--seed", str(seed), "--device", device,
        ])

    run_cmd([
        sys.executable, "-m", "edm_classifier.training.evaluate",
        "--split", "artist", "--model", "mlp",
        "--splits-dir", str(split_root),
        "--active-classes", str(active),
        "--taxonomy", str(taxonomy),
        "--runs-dir", str(run_root),
        "--device", device,
    ])
    report = find_report(run_root)
    if not report:
        raise FileNotFoundError(f"evaluation report missing under {run_root}")
    return report


def metrics(path: Path) -> dict[str, float]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    test = raw["results"]["raw"]["test"]
    return {
        "threshold": float(raw["results"]["raw"]["threshold"]),
        "macro_f1": float(test["macro_supported"]["f1"]),
        "micro_f1": float(test["micro"]["f1"]),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups = defaultdict(list)
    for row in rows:
        groups[(row["fraction"], row["train_samples"])].append(row)
    out = []
    for (fraction, train_samples), values in sorted(groups.items()):
        macro = [x["macro_f1"] for x in values]
        micro = [x["micro_f1"] for x in values]
        out.append({
            "fraction": fraction,
            "train_samples": train_samples,
            "runs": len(values),
            "macro_f1_mean": mean(macro),
            "macro_f1_std": stdev(macro) if len(macro) > 1 else 0.0,
            "micro_f1_mean": mean(micro),
            "micro_f1_std": stdev(micro) if len(micro) > 1 else 0.0,
        })
    return out


def plot(path: Path, rows: list[dict[str, Any]]) -> None:
    x = [r["train_samples"] for r in rows]
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.errorbar(x, [r["macro_f1_mean"] for r in rows], yerr=[r["macro_f1_std"] for r in rows], marker="o", capsize=4, label="Macro F1")
    ax.errorbar(x, [r["micro_f1_mean"] for r in rows], yerr=[r["micro_f1_std"] for r in rows], marker="o", capsize=4, label="Micro F1")
    ax.set_xlabel("Artist-split training samples")
    ax.set_ylabel("Test F1")
    ax.set_ylim(0, 1)
    ax.set_title("Parent-only pooled MLP learning curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Parent-only artist learning curve")
    p.add_argument("--splits-dir", type=Path, default=DEFAULT_SPLITS)
    p.add_argument("--active-classes", type=Path, default=DEFAULT_ACTIVE)
    p.add_argument("--taxonomy", type=Path, default=DEFAULT_TAXONOMY)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--fractions", type=float, nargs="+", default=list(DEFAULT_FRACTIONS))
    p.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    p.add_argument("--sampling-seed", type=int, default=424242)
    p.add_argument("--device", default="auto")
    p.add_argument("--build-only", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    fractions = sorted(set(args.fractions))
    if any(f <= 0 or f > 1 for f in fractions):
        raise SystemExit("fractions must be in (0,1]")

    taxonomy = load_taxonomy(args.taxonomy)
    artist = args.splits_dir / "artist"
    train = load_jsonl(artist / "train.jsonl")
    val = load_jsonl(artist / "validation.jsonl")
    test = load_jsonl(artist / "test.jsonl")
    subsets = build_subsets(train, taxonomy, fractions, args.sampling_seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "task": "parent_only_multi_label_learning_curve",
        "validation_and_test_held_fixed": True,
        "sampling_method": "deterministic nested stratification by parent-target signature",
        "sampling_seed": args.sampling_seed,
        "target_audit": {"train": audit(train, taxonomy), "validation": audit(val, taxonomy), "test": audit(test, taxonomy)},
        "fractions": {},
    }

    audit_rows = []
    for split_name, records in (("train", train), ("validation", val), ("test", test)):
        info = audit(records, taxonomy)
        for parent_id, support in info["parent_support"].items():
            audit_rows.append({"split": split_name, "parent_id": parent_id, "support": support, "total_samples": len(records), "sample_fraction": support / len(records) if records else 0})
    write_csv(args.output_dir / "parent_target_audit.csv", audit_rows)

    subset_roots = {}
    for fraction, rows in subsets.items():
        name = fraction_name(fraction)
        root = args.output_dir / "subsets" / name
        out_artist = root / "artist"
        write_jsonl(out_artist / "train.jsonl", rows)
        shutil.copy2(artist / "validation.jsonl", out_artist / "validation.jsonl")
        shutil.copy2(artist / "test.jsonl", out_artist / "test.jsonl")
        supports = audit(rows, taxonomy)["parent_support"]
        report["fractions"][name] = {
            "fraction": fraction,
            "train_samples": len(rows),
            "actual_fraction": len(rows) / len(train),
            "minimum_parent_support": min(supports.values()) if supports else 0,
            "parent_support": supports,
            "splits_dir": str(root),
        }
        subset_roots[fraction] = root

    (args.output_dir / "subset_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    a = report["target_audit"]["train"]
    print("Parent learning-curve subsets built")
    for fraction in fractions:
        x = report["fractions"][fraction_name(fraction)]
        print(f"  {fraction:>4.0%}: {x['train_samples']:5d} samples; min parent support={x['minimum_parent_support']}")
    print("\nFull artist-train parent cardinality")
    print(f"  exactly 1 parent: {a['single_parent_samples']} ({a['single_parent_fraction']:.1%})")
    print(f"  2+ parents:       {a['multi_parent_samples']} ({a['multi_parent_fraction']:.1%})")
    print(f"  zero parents:     {a['zero_parent_samples']}")

    if args.build_only:
        return

    run_rows = []
    for fraction in fractions:
        name = fraction_name(fraction)
        for seed in args.seeds:
            run_root = args.output_dir / "runs" / name / f"seed_{seed}"
            rp = train_eval(subset_roots[fraction], args.active_classes, args.taxonomy, run_root, seed, args.device)
            m = metrics(rp)
            run_rows.append({
                "fraction": fraction,
                "train_samples": len(subsets[fraction]),
                "seed": seed,
                **m,
                "evaluation_report": str(rp),
            })

    summary = summarize(run_rows)
    write_csv(args.output_dir / "learning_curve.csv", run_rows)
    write_csv(args.output_dir / "learning_curve_summary.csv", summary)
    plot(args.output_dir / "learning_curve.png", summary)

    print("\nLearning curve complete")
    print(f"{'Train':>7s} {'Frac':>6s} {'Macro F1':>18s} {'Micro F1':>18s}")
    print("-" * 55)
    for row in summary:
        print(f"{row['train_samples']:7d} {row['fraction']:6.0%} {row['macro_f1_mean']:.4f} +/- {row['macro_f1_std']:.4f}   {row['micro_f1_mean']:.4f} +/- {row['micro_f1_std']:.4f}")
    print(f"\nSummary: {args.output_dir / 'learning_curve_summary.csv'}")
    print(f"Plot:    {args.output_dir / 'learning_curve.png'}")


if __name__ == "__main__":
    main()
