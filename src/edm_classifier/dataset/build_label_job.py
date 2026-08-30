"""Build the V2 LLM labeling job.

This stage does not call a new retrieval source or alter the candidate set.
It converts V2 candidates into compact, evidence-aware labeling records.

Inputs:
    config/taxonomy.yaml
    data/v2/candidates/candidate_tracks.jsonl
    data/v2/candidates/coverage_after_rescue.json
    data/splits/samples.jsonl

Outputs:
    data/v2/label_job/
      label_input.jsonl
      prelabel_class_viability.csv
      label_job_report.json

The actual LLM runner can consume label_input.jsonl with
config/label_prompt_v2.md and validate outputs against
config/label_output_v2.schema.json.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml


DEFAULT_TAXONOMY = Path("config/taxonomy.yaml")
DEFAULT_CANDIDATES = Path("data/v2/candidates/candidate_tracks.jsonl")
DEFAULT_COVERAGE = Path("data/v2/candidates/coverage_after_rescue.json")
DEFAULT_V1_SAMPLES = Path("data/splits/samples.jsonl")
DEFAULT_OUTPUT = Path("data/v2/label_job")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            line = raw.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected object")
            rows.append(value)
    return rows


def load_taxonomy(path: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    genres = raw.get("genres") if isinstance(raw, dict) else None
    if not isinstance(genres, list):
        raise ValueError(f"{path}: expected genres list")

    ordered = []
    by_id = {}
    for position, row in enumerate(genres):
        if not isinstance(row, dict) or not isinstance(row.get("id"), str):
            raise ValueError(f"{path}: invalid genre at position {position}")
        item = dict(row)
        item["_position"] = position
        ordered.append(item)
        by_id[item["id"]] = item

    return ordered, by_id


def load_coverage(path: Path) -> dict[str, dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    genres = raw.get("genres") if isinstance(raw, dict) else None
    if not isinstance(genres, dict):
        raise ValueError(f"{path}: expected genres object")
    return {
        key: value
        for key, value in genres.items()
        if isinstance(key, str) and isinstance(value, dict)
    }


def direct_labels(row: dict[str, Any]) -> set[str]:
    raw = row.get("labels")
    if not isinstance(raw, list):
        return set()
    return {x for x in raw if isinstance(x, str) and x}


def compact_top_tags(value: Any, maximum: int = 12) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []

    out = []
    for item in value:
        if isinstance(item, str):
            out.append({"name": item})
        elif isinstance(item, dict) and isinstance(item.get("name"), str):
            entry = {"name": item["name"]}
            if isinstance(item.get("count"), (int, float)):
                entry["count"] = item["count"]
            out.append(entry)

        if len(out) >= maximum:
            break

    return out


def children_map(taxonomy: dict[str, dict[str, Any]]) -> dict[str, list[str]]:
    children: dict[str, list[str]] = defaultdict(list)
    for genre_id, row in taxonomy.items():
        parent = row.get("parent")
        if isinstance(parent, str) and parent:
            children[parent].append(genre_id)

    for parent in children:
        children[parent].sort(key=lambda gid: taxonomy[gid].get("_position", 0))

    return children


def evidence_for_candidate(
    candidate: dict[str, Any],
    taxonomy: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    discovered = candidate.get("discovered_for")
    if not isinstance(discovered, list):
        discovered = []

    raw_evidence = candidate.get("retrieval_evidence")
    by_genre: dict[str, list[dict[str, Any]]] = defaultdict(list)

    if isinstance(raw_evidence, list):
        for item in raw_evidence:
            if not isinstance(item, dict):
                continue
            genre_id = item.get("genre_id")
            if isinstance(genre_id, str):
                by_genre[genre_id].append(item)

    result = []

    for genre_id in discovered:
        if not isinstance(genre_id, str) or genre_id not in taxonomy:
            continue

        if by_genre.get(genre_id):
            for item in by_genre[genre_id]:
                result.append(
                    {
                        "genre_id": genre_id,
                        "genre_label": taxonomy[genre_id].get("label", genre_id),
                        "retrieval_type": item.get("type", "unknown"),
                        "strength": item.get("strength", "weak"),
                        "query": item.get("query"),
                    }
                )
        else:
            # Stage-1 candidates came directly from the leaf tag itself.
            result.append(
                {
                    "genre_id": genre_id,
                    "genre_label": taxonomy[genre_id].get("label", genre_id),
                    "retrieval_type": "exact_leaf_tag",
                    "strength": "strong",
                    "query": taxonomy[genre_id].get("label", genre_id),
                }
            )

    return result


def relevant_options(
    evidence: list[dict[str, Any]],
    taxonomy: dict[str, dict[str, Any]],
    children: dict[str, list[str]],
) -> list[dict[str, Any]]:
    option_ids: set[str] = set()

    for item in evidence:
        genre_id = item["genre_id"]
        genre = taxonomy[genre_id]
        parent = genre.get("parent")

        if isinstance(parent, str) and parent in taxonomy:
            option_ids.add(parent)
            option_ids.update(children.get(parent, []))
        else:
            option_ids.add(genre_id)
            option_ids.update(children.get(genre_id, []))

    ordered_ids = sorted(
        option_ids,
        key=lambda gid: taxonomy[gid].get("_position", 0),
    )

    result = []
    for genre_id in ordered_ids:
        row = taxonomy[genre_id]
        entry: dict[str, Any] = {
            "id": genre_id,
            "label": row.get("label", genre_id),
            "parent": row.get("parent"),
        }

        aliases = row.get("aliases")
        if isinstance(aliases, list):
            cleaned = [x for x in aliases if isinstance(x, str) and x.strip()]
            if cleaned:
                entry["aliases"] = cleaned

        characteristics = row.get("characteristics")
        if characteristics:
            entry["characteristics"] = characteristics

        result.append(entry)

    return result


def evidence_strength_counts(
    candidates: list[dict[str, Any]],
    taxonomy: dict[str, dict[str, Any]],
) -> tuple[Counter[str], Counter[str], Counter[str]]:
    strong: Counter[str] = Counter()
    medium: Counter[str] = Counter()
    weak: Counter[str] = Counter()

    for candidate in candidates:
        seen_strength: set[tuple[str, str]] = set()

        for evidence in evidence_for_candidate(candidate, taxonomy):
            genre_id = evidence["genre_id"]
            strength = str(evidence.get("strength", "weak")).lower()
            marker = (genre_id, strength)
            if marker in seen_strength:
                continue
            seen_strength.add(marker)

            if strength == "strong":
                strong[genre_id] += 1
            elif strength == "medium":
                medium[genre_id] += 1
            else:
                weak[genre_id] += 1

    return strong, medium, weak


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build V2 evidence-aware LLM label job.")
    parser.add_argument("--taxonomy", type=Path, default=DEFAULT_TAXONOMY)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--coverage", type=Path, default=DEFAULT_COVERAGE)
    parser.add_argument("--v1-samples", type=Path, default=DEFAULT_V1_SAMPLES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ordered, taxonomy = load_taxonomy(args.taxonomy)
    children = children_map(taxonomy)
    candidates = load_jsonl(args.candidates)
    v1_samples = load_jsonl(args.v1_samples)
    coverage = load_coverage(args.coverage)

    v1_support: Counter[str] = Counter()
    for sample in v1_samples:
        v1_support.update(direct_labels(sample))

    strong, medium, weak = evidence_strength_counts(candidates, taxonomy)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    label_input_path = args.output_dir / "label_input.jsonl"
    viability_path = args.output_dir / "prelabel_class_viability.csv"
    report_path = args.output_dir / "label_job_report.json"

    job_rows = []

    for candidate in candidates:
        candidate_id = candidate.get("candidate_id")
        artist = candidate.get("artist")
        title = candidate.get("title")

        if not all(isinstance(x, str) and x for x in (candidate_id, artist, title)):
            continue

        evidence = evidence_for_candidate(candidate, taxonomy)
        if not evidence:
            continue

        job_rows.append(
            {
                "candidate_id": candidate_id,
                "artist": artist,
                "title": title,
                "mbid": candidate.get("mbid"),
                "top_tags": compact_top_tags(candidate.get("top_tags")),
                "retrieval_evidence": evidence,
                "label_options": relevant_options(evidence, taxonomy, children),
            }
        )

    with label_input_path.open("w", encoding="utf-8") as handle:
        for row in job_rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    leaf_ids = [
        row["id"]
        for row in ordered
        if not children.get(row["id"])
    ]

    fieldnames = [
        "id",
        "label",
        "parent",
        "v1_usable_direct_tracks",
        "v2_strong_candidates",
        "v2_medium_candidates",
        "v2_weak_candidates",
        "v1_plus_strong_medium",
        "planned_assignments",
        "final_candidate_assignments",
        "remaining_collection_shortfall",
        "prelabel_status",
    ]

    viability_rows = []

    for genre_id in leaf_ids:
        row = taxonomy[genre_id]
        cov = coverage.get(genre_id, {})

        evidence_total = (
            v1_support[genre_id]
            + strong[genre_id]
            + medium[genre_id]
        )

        # This is a diagnostic only, not a final keep/drop decision.
        # Actual dropping occurs after LLM labels and downstream usable support
        # are known.
        if evidence_total >= 100:
            status = "healthy"
        elif evidence_total >= 50:
            status = "borderline"
        else:
            status = "at_risk"

        viability_rows.append(
            {
                "id": genre_id,
                "label": row.get("label", genre_id),
                "parent": row.get("parent") or "",
                "v1_usable_direct_tracks": v1_support[genre_id],
                "v2_strong_candidates": strong[genre_id],
                "v2_medium_candidates": medium[genre_id],
                "v2_weak_candidates": weak[genre_id],
                "v1_plus_strong_medium": evidence_total,
                "planned_assignments": cov.get("planned", 0),
                "final_candidate_assignments": cov.get("final_selected_assignments", 0),
                "remaining_collection_shortfall": cov.get("remaining_shortfall", 0),
                "prelabel_status": status,
            }
        )

    viability_rows.sort(
        key=lambda row: (
            {"at_risk": 0, "borderline": 1, "healthy": 2}[row["prelabel_status"]],
            int(row["v1_plus_strong_medium"]),
            str(row["id"]),
        )
    )

    with viability_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(viability_rows)

    evidence_hist = Counter()
    for row in job_rows:
        strengths = {
            str(item.get("strength", "weak"))
            for item in row["retrieval_evidence"]
        }
        if "strong" in strengths:
            evidence_hist["has_strong"] += 1
        elif "medium" in strengths:
            evidence_hist["medium_only"] += 1
        else:
            evidence_hist["weak_only"] += 1

    status_hist = Counter(row["prelabel_status"] for row in viability_rows)

    report = {
        "version": "v2",
        "input_candidates": len(candidates),
        "label_job_records": len(job_rows),
        "leaf_classes": len(leaf_ids),
        "candidate_evidence_profile": dict(evidence_hist),
        "prelabel_class_status": dict(status_hist),
        "labeling_policy": {
            "max_labels_per_track": 2,
            "exact_leaf_tag": "strong weak-supervision evidence",
            "leaf_tag_artist": (
                "medium evidence: the artist is associated with the leaf, "
                "but the individual track must still be judged"
            ),
            "parent_fallback": (
                "weak candidate sourcing only; it must never by itself justify "
                "assigning the requested leaf"
            ),
            "similar_tag": "weak evidence",
            "track_specific_model_knowledge": "allowed when confident",
            "generic_artist_style_inference": "not sufficient by itself",
        },
        "future_drop_policy": {
            "when": (
                "after LLM validation and again after audio/embedding survival; "
                "do not drop solely for missing the collection quota"
            ),
            "minimum_viable_leaf": {
                "usable_tracks": 50,
                "unique_artists": 30,
            },
            "preferred_strong_leaf": {
                "usable_tracks": 100,
                "unique_artists": 50,
            },
        },
        "outputs": {
            "label_input": str(label_input_path),
            "prelabel_class_viability": str(viability_path),
        },
    }

    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("V2 label job built")
    print(f"  candidate input:        {len(candidates)}")
    print(f"  label-job records:      {len(job_rows)}")
    print(f"  leaf classes:           {len(leaf_ids)}")
    print()
    print("Candidate evidence:")
    for key in ("has_strong", "medium_only", "weak_only"):
        print(f"  {key:18s} {evidence_hist[key]}")
    print()
    print("Pre-label leaf status:")
    for key in ("healthy", "borderline", "at_risk"):
        print(f"  {key:18s} {status_hist[key]}")
    print()
    print(f"Input:     {label_input_path}")
    print(f"Viability: {viability_path}")
    print(f"Report:    {report_path}")


if __name__ == "__main__":
    main()
