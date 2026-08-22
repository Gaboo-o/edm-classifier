"""Compare regular vs artist-separated evaluation results on common supported classes."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

DEFAULT_RUNS_DIR = Path("data/runs")


def read_per_class(path: Path):
    rows = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows[row["id"]] = row
    return rows


def mean(values):
    return sum(values) / len(values) if values else None


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["linear", "mlp", "hierarchical_mlp"], default="mlp")
    p.add_argument("--mode", choices=["raw", "hierarchical"], default="hierarchical")
    p.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    return p.parse_args()


def main():
    args = parse_args()
    regular_dir = args.runs_dir / f"regular_{args.model}"
    artist_dir = args.runs_dir / f"artist_{args.model}"
    regular_path = regular_dir / f"test_per_class_{args.mode}.csv"
    artist_path = artist_dir / f"test_per_class_{args.mode}.csv"

    regular = read_per_class(regular_path)
    artist = read_per_class(artist_path)
    common_ids = sorted(set(regular) & set(artist))
    common_supported = [
        cid for cid in common_ids
        if int(regular[cid]["support"]) > 0 and int(artist[cid]["support"]) > 0
    ]
    common_supported_leaf = [
        cid for cid in common_supported
        if regular[cid]["is_leaf"].strip().lower() == "true"
    ]
    common_supported_parent = [
        cid for cid in common_supported
        if regular[cid]["is_leaf"].strip().lower() != "true"
    ]

    def summarize(ids):
        return {
            "class_count": len(ids),
            "regular_macro_f1": mean([float(regular[cid]["f1"]) for cid in ids]),
            "artist_macro_f1": mean([float(artist[cid]["f1"]) for cid in ids]),
        }

    report = {
        "model": args.model,
        "inference_mode": args.mode,
        "common_supported": summarize(common_supported),
        "common_supported_parent": summarize(common_supported_parent),
        "common_supported_leaf": summarize(common_supported_leaf),
    }

    output = args.runs_dir / f"comparison_{args.model}_{args.mode}.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(report, indent=2))
    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
