"""Compare frozen V1 benchmarks against the unified dataset runs.

Expected layout:

data/
├── baselines/v1/runs/
│   ├── regular_linear/
│   ├── artist_linear/
│   ├── regular_mlp/
│   ├── artist_mlp/
│   ├── regular_hierarchical_mlp/
│   ├── artist_hierarchical_mlp/
│   └── parent/                       # optional nested parent runs
│       ├── regular_mlp/
│       └── artist_mlp/
└── runs/
    ├── full/
    │   ├── regular_linear/
    │   ├── artist_linear/
    │   ├── regular_mlp/
    │   ├── artist_mlp/
    │   ├── regular_hierarchical_mlp/
    │   └── artist_hierarchical_mlp/
    └── parent/
        ├── regular_mlp/
        └── artist_mlp/

Outputs:

data/runs/comparison/
├── unified_vs_v1.csv
├── unified_vs_v1.md
├── unified_vs_v1.json
├── full_macro_f1.png
├── full_micro_f1.png
└── parent_macro_f1.png
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


MODELS = ("linear", "mlp", "hierarchical_mlp")
SPLITS = ("regular", "artist")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare V1 baseline metrics with unified-run metrics."
    )
    p.add_argument(
        "--baseline-runs-dir",
        type=Path,
        default=Path("data/baselines/v1/runs"),
    )
    p.add_argument(
        "--unified-full-runs-dir",
        type=Path,
        default=Path("data/runs/full"),
    )
    p.add_argument(
        "--unified-parent-runs-dir",
        type=Path,
        default=Path("data/runs/parent"),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/runs/comparison"),
    )
    p.add_argument(
        "--mode",
        choices=["raw", "hierarchical"],
        default="raw",
        help=(
            "Inference mode used when evaluation_report_v2-style files "
            "contain both raw and hierarchy-projected metrics. "
            "Default raw matches the historical V1 headline metrics."
        ),
    )
    return p.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def candidate_report_paths(root: Path) -> list[Path]:
    if not root.exists():
        return []

    paths: list[Path] = []
    for pattern in (
        "**/evaluation_report_v2.json",
        "**/evaluation_report.json",
    ):
        paths.extend(root.glob(pattern))

    # Prefer newer/v2 report when both exist in the same directory.
    by_dir: dict[Path, Path] = {}
    for path in sorted(paths):
        current = by_dir.get(path.parent)
        if current is None:
            by_dir[path.parent] = path
        elif path.name == "evaluation_report_v2.json":
            by_dir[path.parent] = path

    return sorted(by_dir.values())


def extract_metrics(
    report: dict[str, Any],
    mode: str,
) -> dict[str, Any]:
    """Handle both old flat evaluator and newer raw/hierarchical evaluator."""

    run_name = report.get("run_name")
    split = report.get("split")
    model = report.get("model")
    active_class_count = report.get("active_class_count")
    checkpoint_epoch = report.get("checkpoint_epoch")

    results = report.get("results")
    if isinstance(results, dict):
        mode_result = results.get(mode)
        if not isinstance(mode_result, dict):
            raise ValueError(
                f"{run_name}: requested mode {mode!r} not found"
            )

        test = mode_result.get("test")
        if not isinstance(test, dict):
            raise ValueError(f"{run_name}: no test metrics")

        threshold = mode_result.get("threshold")

        macro_supported = test.get("macro_supported")
        macro_parent = test.get("macro_parent")
        macro_leaf = test.get("macro_supported_leaf")
        micro = test.get("micro")

        def metric_f1(value: Any) -> float | None:
            if not isinstance(value, dict):
                return None
            v = value.get("f1")
            return float(v) if isinstance(v, (int, float)) else None

        return {
            "run_name": run_name,
            "split": split,
            "model": model,
            "active_class_count": active_class_count,
            "checkpoint_epoch": checkpoint_epoch,
            "threshold": threshold,
            "macro_f1": metric_f1(macro_supported),
            "parent_macro_f1": metric_f1(macro_parent),
            "leaf_macro_f1": metric_f1(macro_leaf),
            "micro_f1": metric_f1(micro),
            "subset_accuracy": (
                float(test["subset_accuracy"])
                if isinstance(test.get("subset_accuracy"), (int, float))
                else None
            ),
            "hamming_loss": (
                float(test["hamming_loss"])
                if isinstance(test.get("hamming_loss"), (int, float))
                else None
            ),
            "zero_support_classes": len(test.get("zero_support_classes", [])),
        }

    # Legacy flat evaluator.
    test = report.get("test")
    if not isinstance(test, dict):
        raise ValueError(f"{run_name}: unsupported evaluation report schema")

    threshold_policy = report.get("threshold_policy")
    threshold = (
        threshold_policy.get("selected_threshold")
        if isinstance(threshold_policy, dict)
        else None
    )

    def number(key: str) -> float | None:
        value = test.get(key)
        return float(value) if isinstance(value, (int, float)) else None

    zero = report.get("test_classes_with_zero_support", [])

    return {
        "run_name": run_name,
        "split": split,
        "model": model,
        "active_class_count": active_class_count,
        "checkpoint_epoch": checkpoint_epoch,
        "threshold": threshold,
        "macro_f1": number("macro_f1"),
        "parent_macro_f1": None,
        "leaf_macro_f1": None,
        "micro_f1": number("micro_f1"),
        "subset_accuracy": number("subset_accuracy"),
        "hamming_loss": number("hamming_loss"),
        "zero_support_classes": len(zero) if isinstance(zero, list) else None,
    }


def load_reports(root: Path, mode: str) -> list[dict[str, Any]]:
    rows = []
    for path in candidate_report_paths(root):
        report = load_json(path)
        try:
            metrics = extract_metrics(report, mode)
        except ValueError:
            continue
        metrics["report_path"] = str(path)
        rows.append(metrics)
    return rows


def index_reports(
    rows: list[dict[str, Any]],
) -> dict[tuple[str, str, int | None], dict[str, Any]]:
    result = {}
    for row in rows:
        split = row.get("split")
        model = row.get("model")
        count = row.get("active_class_count")
        if isinstance(split, str) and isinstance(model, str):
            result[(split, model, count)] = row
    return result


def find_run(
    rows: list[dict[str, Any]],
    *,
    split: str,
    model: str,
    parent_only: bool,
) -> dict[str, Any] | None:
    matches = [
        row
        for row in rows
        if row.get("split") == split
        and row.get("model") == model
        and (
            row.get("active_class_count") == 19
            if parent_only
            else row.get("active_class_count") != 19
        )
    ]

    if not matches:
        return None

    # If duplicates remain, prefer a v2 evaluation report.
    matches.sort(
        key=lambda row: (
            "evaluation_report_v2.json" in row.get("report_path", ""),
            row.get("checkpoint_epoch") or 0,
        ),
        reverse=True,
    )
    return matches[0]


def delta(new: float | None, old: float | None) -> float | None:
    if new is None or old is None:
        return None
    return new - old


def pct_change(new: float | None, old: float | None) -> float | None:
    if new is None or old is None or old == 0:
        return None
    return 100.0 * (new - old) / old


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def comparison_rows(
    baseline_rows: list[dict[str, Any]],
    unified_full_rows: list[dict[str, Any]],
    unified_parent_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    output = []

    for split in SPLITS:
        for model in MODELS:
            old = find_run(
                baseline_rows,
                split=split,
                model=model,
                parent_only=False,
            )
            new = find_run(
                unified_full_rows,
                split=split,
                model=model,
                parent_only=False,
            )

            if old is None and new is None:
                continue

            row = {
                "task": "full",
                "split": split,
                "model": model,
                "v1_classes": old.get("active_class_count") if old else None,
                "unified_classes": new.get("active_class_count") if new else None,
                "v1_threshold": old.get("threshold") if old else None,
                "unified_threshold": new.get("threshold") if new else None,
            }

            for metric in (
                "macro_f1",
                "parent_macro_f1",
                "leaf_macro_f1",
                "micro_f1",
                "subset_accuracy",
                "hamming_loss",
            ):
                old_value = old.get(metric) if old else None
                new_value = new.get(metric) if new else None
                row[f"v1_{metric}"] = old_value
                row[f"unified_{metric}"] = new_value
                row[f"delta_{metric}"] = delta(new_value, old_value)
                if metric in ("macro_f1", "micro_f1"):
                    row[f"pct_{metric}"] = pct_change(new_value, old_value)

            output.append(row)

    # Parent-only MLP benchmark.
    for split in SPLITS:
        old = find_run(
            baseline_rows,
            split=split,
            model="mlp",
            parent_only=True,
        )
        new = find_run(
            unified_parent_rows,
            split=split,
            model="mlp",
            parent_only=True,
        )

        if old is None and new is None:
            continue

        row = {
            "task": "parent",
            "split": split,
            "model": "mlp",
            "v1_classes": old.get("active_class_count") if old else None,
            "unified_classes": new.get("active_class_count") if new else None,
            "v1_threshold": old.get("threshold") if old else None,
            "unified_threshold": new.get("threshold") if new else None,
        }

        for metric in (
            "macro_f1",
            "parent_macro_f1",
            "leaf_macro_f1",
            "micro_f1",
            "subset_accuracy",
            "hamming_loss",
        ):
            old_value = old.get(metric) if old else None
            new_value = new.get(metric) if new else None
            row[f"v1_{metric}"] = old_value
            row[f"unified_{metric}"] = new_value
            row[f"delta_{metric}"] = delta(new_value, old_value)
            if metric in ("macro_f1", "micro_f1"):
                row[f"pct_{metric}"] = pct_change(new_value, old_value)

        output.append(row)

    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Unified vs V1 benchmark",
        "",
        "All values are test-set metrics. "
        "Delta = Unified − V1. Positive F1 deltas are improvements.",
        "",
        "## Full taxonomy",
        "",
        "| Split | Model | V1 classes | Unified classes | V1 macro F1 | Unified macro F1 | Δ macro | V1 micro F1 | Unified micro F1 | Δ micro |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for row in rows:
        if row["task"] != "full":
            continue
        lines.append(
            "| {split} | {model} | {v1c} | {newc} | {v1m} | {newm} | {dm} | {v1micro} | {newmicro} | {dmicro} |".format(
                split=row["split"],
                model=row["model"],
                v1c=fmt(row["v1_classes"], 0),
                newc=fmt(row["unified_classes"], 0),
                v1m=fmt(row["v1_macro_f1"]),
                newm=fmt(row["unified_macro_f1"]),
                dm=fmt(row["delta_macro_f1"]),
                v1micro=fmt(row["v1_micro_f1"]),
                newmicro=fmt(row["unified_micro_f1"]),
                dmicro=fmt(row["delta_micro_f1"]),
            )
        )

    lines += [
        "",
        "## Parent-only",
        "",
        "| Split | Model | V1 macro F1 | Unified macro F1 | Δ macro | V1 micro F1 | Unified micro F1 | Δ micro |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]

    for row in rows:
        if row["task"] != "parent":
            continue
        lines.append(
            "| {split} | {model} | {v1m} | {newm} | {dm} | {v1micro} | {newmicro} | {dmicro} |".format(
                split=row["split"],
                model=row["model"],
                v1m=fmt(row["v1_macro_f1"]),
                newm=fmt(row["unified_macro_f1"]),
                dm=fmt(row["delta_macro_f1"]),
                v1micro=fmt(row["v1_micro_f1"]),
                newmicro=fmt(row["unified_micro_f1"]),
                dmicro=fmt(row["delta_micro_f1"]),
            )
        )

    lines += [
        "",
        "## Notes",
        "",
        "- Macro F1 uses supported classes when the evaluator exposes that distinction.",
        "- Raw inference is the default because it matches the historical V1 headline metrics.",
        "- The unified and V1 full-taxonomy runs may have different active-class counts. "
        "Therefore this table measures each model on its own selected taxonomy/test split; "
        "it is not a strict same-class benchmark.",
        "- The frozen V1 artist test should be used separately for a strict historical benchmark "
        "after training a leakage-quarantined unified model.",
        "",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")


def grouped_bar(
    path: Path,
    labels: list[str],
    old_values: list[float],
    new_values: list[float],
    title: str,
    ylabel: str,
) -> None:
    x = list(range(len(labels)))
    width = 0.38

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(
        [i - width / 2 for i in x],
        old_values,
        width=width,
        label="V1",
    )
    ax.bar(
        [i + width / 2 for i in x],
        new_values,
        width=width,
        label="Unified",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0.0, 1.0)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_full_plot(
    path: Path,
    rows: list[dict[str, Any]],
    metric: str,
    title: str,
) -> None:
    selected = [
        row for row in rows
        if row["task"] == "full"
        and row.get(f"v1_{metric}") is not None
        and row.get(f"unified_{metric}") is not None
    ]

    labels = [
        f"{row['split']} {row['model'].replace('_', ' ')}"
        for row in selected
    ]
    old_values = [float(row[f"v1_{metric}"]) for row in selected]
    new_values = [float(row[f"unified_{metric}"]) for row in selected]

    if selected:
        grouped_bar(
            path,
            labels,
            old_values,
            new_values,
            title,
            "F1",
        )


def make_parent_plot(path: Path, rows: list[dict[str, Any]]) -> None:
    selected = [
        row for row in rows
        if row["task"] == "parent"
        and row.get("v1_macro_f1") is not None
        and row.get("unified_macro_f1") is not None
    ]

    labels = [row["split"] for row in selected]
    old_values = [float(row["v1_macro_f1"]) for row in selected]
    new_values = [float(row["unified_macro_f1"]) for row in selected]

    if selected:
        grouped_bar(
            path,
            labels,
            old_values,
            new_values,
            "Parent-only MLP: V1 vs Unified",
            "Macro F1",
        )


def print_table(rows: list[dict[str, Any]]) -> None:
    print()
    print("FULL TAXONOMY")
    print(
        f"{'Split':8s} {'Model':18s} "
        f"{'V1 Macro':>9s} {'Unified':>9s} {'Delta':>9s} "
        f"{'V1 Micro':>9s} {'Unified':>9s} {'Delta':>9s}"
    )
    print("-" * 91)

    for row in rows:
        if row["task"] != "full":
            continue
        print(
            f"{row['split']:8s} {row['model']:18s} "
            f"{fmt(row['v1_macro_f1']):>9s} "
            f"{fmt(row['unified_macro_f1']):>9s} "
            f"{fmt(row['delta_macro_f1']):>9s} "
            f"{fmt(row['v1_micro_f1']):>9s} "
            f"{fmt(row['unified_micro_f1']):>9s} "
            f"{fmt(row['delta_micro_f1']):>9s}"
        )

    print()
    print("PARENT-ONLY MLP")
    print(
        f"{'Split':8s} "
        f"{'V1 Macro':>9s} {'Unified':>9s} {'Delta':>9s} "
        f"{'V1 Micro':>9s} {'Unified':>9s} {'Delta':>9s}"
    )
    print("-" * 70)

    for row in rows:
        if row["task"] != "parent":
            continue
        print(
            f"{row['split']:8s} "
            f"{fmt(row['v1_macro_f1']):>9s} "
            f"{fmt(row['unified_macro_f1']):>9s} "
            f"{fmt(row['delta_macro_f1']):>9s} "
            f"{fmt(row['v1_micro_f1']):>9s} "
            f"{fmt(row['unified_micro_f1']):>9s} "
            f"{fmt(row['delta_micro_f1']):>9s}"
        )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    baseline_rows = load_reports(args.baseline_runs_dir, args.mode)
    unified_full_rows = load_reports(args.unified_full_runs_dir, args.mode)
    unified_parent_rows = load_reports(args.unified_parent_runs_dir, args.mode)

    rows = comparison_rows(
        baseline_rows,
        unified_full_rows,
        unified_parent_rows,
    )

    if not rows:
        raise SystemExit(
            "No comparable evaluation reports found. "
            "Check the three runs-directory arguments."
        )

    csv_path = args.output_dir / "unified_vs_v1.csv"
    md_path = args.output_dir / "unified_vs_v1.md"
    json_path = args.output_dir / "unified_vs_v1.json"

    write_csv(csv_path, rows)
    write_markdown(md_path, rows)
    json_path.write_text(
        json.dumps(
            {
                "inference_mode": args.mode,
                "rows": rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    make_full_plot(
        args.output_dir / "full_macro_f1.png",
        rows,
        "macro_f1",
        "Full Taxonomy: V1 vs Unified — Macro F1",
    )
    make_full_plot(
        args.output_dir / "full_micro_f1.png",
        rows,
        "micro_f1",
        "Full Taxonomy: V1 vs Unified — Micro F1",
    )
    make_parent_plot(
        args.output_dir / "parent_macro_f1.png",
        rows,
    )

    print_table(rows)
    print()
    print(f"CSV:       {csv_path}")
    print(f"Markdown:  {md_path}")
    print(f"JSON:      {json_path}")
    print(
        f"Charts:    {args.output_dir / 'full_macro_f1.png'}"
    )
    print(
        f"           {args.output_dir / 'full_micro_f1.png'}"
    )
    print(
        f"           {args.output_dir / 'parent_macro_f1.png'}"
    )


if __name__ == "__main__":
    main()
