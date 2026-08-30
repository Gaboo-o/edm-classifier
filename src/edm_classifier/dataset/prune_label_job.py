"""Prune clearly inadequate V2 leaves before LLM labeling.

Inputs:
    data/v2/label_job/label_input.jsonl
    data/v2/label_job/prelabel_class_viability.csv
    config/taxonomy.yaml

Policy:
- drop all leaves marked `at_risk`
- keep `borderline` and `healthy`
- keep all taxonomy parents/root classes
- discard candidate records whose retrieval evidence refers only to dropped leaves
- for retained records, remove dropped-leaf evidence/options

Outputs:
    data/v2/label_job/active_leaf_classes.json
    data/v2/label_job/dropped_leaf_classes.json
    data/v2/label_job/label_input_active.jsonl
    data/v2/label_job/prune_report.json
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import yaml


DEFAULT_INPUT = Path("data/v2/label_job/label_input.jsonl")
DEFAULT_VIABILITY = Path("data/v2/label_job/prelabel_class_viability.csv")
DEFAULT_TAXONOMY = Path("config/taxonomy.yaml")
DEFAULT_OUTPUT_DIR = Path("data/v2/label_job")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            line = raw.strip()
            if not line:
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            rows.append(value)
    return rows


def load_taxonomy(path: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    genres = raw.get("genres") if isinstance(raw, dict) else None
    if not isinstance(genres, list):
        raise ValueError(f"{path}: expected genres list")

    ordered: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}

    for position, item in enumerate(genres):
        if not isinstance(item, dict) or not isinstance(item.get("id"), str):
            raise ValueError(f"{path}: invalid genre at position {position}")
        copy = dict(item)
        copy["_position"] = position
        ordered.append(copy)
        by_id[copy["id"]] = copy

    return ordered, by_id


def load_viability(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(description="Prune at-risk V2 leaf classes.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--viability", type=Path, default=DEFAULT_VIABILITY)
    parser.add_argument("--taxonomy", type=Path, default=DEFAULT_TAXONOMY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    records = load_jsonl(args.input)
    viability = load_viability(args.viability)
    ordered, taxonomy = load_taxonomy(args.taxonomy)

    dropped = {
        row["id"]
        for row in viability
        if row.get("prelabel_status") == "at_risk"
    }
    kept_leaves = {
        row["id"]
        for row in viability
        if row.get("prelabel_status") in {"healthy", "borderline"}
    }

    roots = {
        item["id"]
        for item in ordered
        if not item.get("parent")
    }

    # Internal parent classes are also always legal options. In the current
    # taxonomy this is primarily the 19 roots, but this remains generic.
    parent_ids = {
        item.get("parent")
        for item in ordered
        if isinstance(item.get("parent"), str) and item.get("parent")
    }
    parent_ids = {x for x in parent_ids if x in taxonomy}

    allowed_ids = kept_leaves | roots | parent_ids

    filtered: list[dict[str, Any]] = []
    dropped_records = 0
    evidence_removed = 0
    options_removed = 0
    retained_evidence_strength = Counter()

    for record in records:
        evidence = record.get("retrieval_evidence")
        if not isinstance(evidence, list):
            evidence = []

        kept_evidence = []
        for item in evidence:
            if not isinstance(item, dict):
                continue
            genre_id = item.get("genre_id")
            if genre_id in dropped:
                evidence_removed += 1
                continue
            kept_evidence.append(item)

        # A candidate sourced only for a leaf we have explicitly dropped is
        # no longer worth sending to the LLM.
        if not kept_evidence:
            dropped_records += 1
            continue

        options = record.get("label_options")
        if not isinstance(options, list):
            options = []

        kept_options = []
        for item in options:
            if not isinstance(item, dict):
                continue
            genre_id = item.get("id")
            if genre_id in allowed_ids:
                kept_options.append(item)
            else:
                options_removed += 1

        if not kept_options:
            dropped_records += 1
            continue

        updated = dict(record)
        updated["retrieval_evidence"] = kept_evidence
        updated["label_options"] = kept_options
        filtered.append(updated)

        strengths = {
            str(item.get("strength", "weak"))
            for item in kept_evidence
        }
        if "strong" in strengths:
            retained_evidence_strength["has_strong"] += 1
        elif "medium" in strengths:
            retained_evidence_strength["medium_only"] += 1
        else:
            retained_evidence_strength["weak_only"] += 1

    args.output_dir.mkdir(parents=True, exist_ok=True)

    input_path = args.output_dir / "label_input_active.jsonl"
    active_path = args.output_dir / "active_leaf_classes.json"
    dropped_path = args.output_dir / "dropped_leaf_classes.json"
    report_path = args.output_dir / "prune_report.json"

    with input_path.open("w", encoding="utf-8") as handle:
        for row in filtered:
            handle.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            )

    active_rows = [
        {
            "id": row["id"],
            "label": row.get("label", row["id"]),
            "parent": row.get("parent"),
            "prelabel_status": row.get("prelabel_status"),
            "v1_plus_strong_medium": int(row.get("v1_plus_strong_medium") or 0),
        }
        for row in viability
        if row["id"] in kept_leaves
    ]

    dropped_rows = [
        {
            "id": row["id"],
            "label": row.get("label", row["id"]),
            "parent": row.get("parent"),
            "prelabel_status": row.get("prelabel_status"),
            "v1_plus_strong_medium": int(row.get("v1_plus_strong_medium") or 0),
            "v2_weak_candidates": int(row.get("v2_weak_candidates") or 0),
        }
        for row in viability
        if row["id"] in dropped
    ]

    active_path.write_text(
        json.dumps(
            {
                "policy": "keep healthy + borderline leaves; drop at_risk leaves",
                "leaf_count": len(active_rows),
                "leaves": active_rows,
            },
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )

    dropped_path.write_text(
        json.dumps(
            {
                "policy": "pre-label evidence support below 50 strong/medium+V1",
                "leaf_count": len(dropped_rows),
                "leaves": dropped_rows,
            },
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )

    report = {
        "input_records": len(records),
        "output_records": len(filtered),
        "records_removed": dropped_records,
        "leaf_classes_before": len(viability),
        "leaf_classes_kept": len(kept_leaves),
        "leaf_classes_dropped": len(dropped),
        "parents_preserved": len(roots | parent_ids),
        "evidence_entries_removed": evidence_removed,
        "label_options_removed": options_removed,
        "output_evidence_profile": dict(retained_evidence_strength),
        "outputs": {
            "label_input_active": str(input_path),
            "active_leaf_classes": str(active_path),
            "dropped_leaf_classes": str(dropped_path),
        },
    }

    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("V2 pre-label pruning complete")
    print(f"  leaves before:   {len(viability)}")
    print(f"  leaves kept:     {len(kept_leaves)}")
    print(f"  leaves dropped:  {len(dropped)}")
    print(f"  records before:  {len(records)}")
    print(f"  records after:   {len(filtered)}")
    print(f"  records removed: {dropped_records}")
    print()
    print(f"Label input: {input_path}")
    print(f"Active:      {active_path}")
    print(f"Dropped:     {dropped_path}")
    print(f"Report:      {report_path}")


if __name__ == "__main__":
    main()
