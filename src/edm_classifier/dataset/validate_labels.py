"""Validate weak labels and report taxonomy coverage.

Checks:
- expected candidate IDs are present exactly once
- no unexpected/duplicate candidate IDs
- statuses, reasons, labels, candidates, and confidence ranges are valid
- label IDs exist in taxonomy.yaml
- status semantics are consistent
- parent+child labels are not redundantly assigned
- class coverage and confidence distributions
- uncertain candidate-label counts
- optional artist concentration diagnostics using candidate_tracks.jsonl

Outputs:
- label_report.json
- class_coverage.csv
- issues.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml


DEFAULT_RESULTS = Path("data/label_job/results/labeled_tracks.jsonl")
DEFAULT_MANIFEST = Path("data/label_job/manifest.json")
DEFAULT_TAXONOMY = Path("config/taxonomy.yaml")
DEFAULT_CANDIDATES = Path("data/candidates/candidate_tracks.jsonl")
DEFAULT_OUTPUT_DIR = Path("data/validation")

VALID_STATUSES = {"labeled", "uncertain", "out_of_scope"}
VALID_REASONS = {
    "exact_tag_support",
    "cross_tag_support",
    "broad_family_only",
    "boundary_ambiguous",
    "conflicting_evidence",
    "insufficient_evidence",
    "out_of_scope",
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    parse_issues: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                parse_issues.append(
                    {
                        "type": "invalid_json",
                        "line": line_number,
                        "message": str(exc),
                    }
                )
                continue

            if not isinstance(value, dict):
                parse_issues.append(
                    {
                        "type": "not_object",
                        "line": line_number,
                        "message": "JSONL record must be an object",
                    }
                )
                continue

            value["_line_number"] = line_number
            records.append(value)

    return records, parse_issues


def load_taxonomy(path: Path) -> tuple[dict[str, dict[str, Any]], str | None]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("genres"), list):
        raise ValueError(f"Invalid taxonomy structure: {path}")

    genres: dict[str, dict[str, Any]] = {}
    for item in raw["genres"]:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str):
            raise ValueError(f"Invalid genre entry in {path}: {item!r}")
        genre_id = item["id"]
        if genre_id in genres:
            raise ValueError(f"Duplicate taxonomy ID: {genre_id}")
        genres[genre_id] = item

    version = (raw.get("taxonomy") or {}).get("version")
    return genres, version


def load_candidate_artists(path: Path | None) -> dict[str, str]:
    if path is None or not path.exists():
        return {}

    result: dict[str, str] = {}
    records, _ = load_jsonl(path)
    for record in records:
        cid = record.get("candidate_id")
        artist = record.get("artist")
        if isinstance(cid, str) and isinstance(artist, str):
            result[cid] = artist.strip()
    return result


def ancestors(label_id: str, taxonomy: dict[str, dict[str, Any]]) -> set[str]:
    result: set[str] = set()
    current = taxonomy.get(label_id)
    seen: set[str] = set()

    while current:
        parent = current.get("parent")
        if not isinstance(parent, str) or not parent or parent in seen:
            break
        seen.add(parent)
        result.add(parent)
        current = taxonomy.get(parent)

    return result


def issue(
    issues: list[dict[str, Any]],
    issue_type: str,
    *,
    candidate_id: str | None = None,
    line: int | None = None,
    message: str,
    severity: str = "error",
) -> None:
    item: dict[str, Any] = {
        "severity": severity,
        "type": issue_type,
        "message": message,
    }
    if candidate_id is not None:
        item["candidate_id"] = candidate_id
    if line is not None:
        item["line"] = line
    issues.append(item)


def validate_record(
    record: dict[str, Any],
    taxonomy: dict[str, dict[str, Any]],
    issues: list[dict[str, Any]],
) -> None:
    cid = record.get("candidate_id")
    line = record.get("_line_number")
    if not isinstance(cid, str) or not cid.strip():
        issue(
            issues,
            "invalid_candidate_id",
            line=line,
            message="candidate_id must be a non-empty string",
        )
        return

    status = record.get("status")
    labels = record.get("labels")
    candidates = record.get("candidates")
    reason = record.get("reason")

    if status not in VALID_STATUSES:
        issue(
            issues,
            "invalid_status",
            candidate_id=cid,
            line=line,
            message=f"Invalid status: {status!r}",
        )

    if reason not in VALID_REASONS:
        issue(
            issues,
            "invalid_reason",
            candidate_id=cid,
            line=line,
            message=f"Invalid reason: {reason!r}",
        )

    if not isinstance(labels, list):
        issue(
            issues,
            "invalid_labels",
            candidate_id=cid,
            line=line,
            message="labels must be an array",
        )
        labels = []

    if not isinstance(candidates, list):
        issue(
            issues,
            "invalid_candidates",
            candidate_id=cid,
            line=line,
            message="candidates must be an array",
        )
        candidates = []

    if len(labels) > 2:
        issue(
            issues,
            "too_many_labels",
            candidate_id=cid,
            line=line,
            message=f"labels has {len(labels)} entries; maximum is 2",
        )

    if len(candidates) > 3:
        issue(
            issues,
            "too_many_candidates",
            candidate_id=cid,
            line=line,
            message=f"candidates has {len(candidates)} entries; maximum is 3",
        )

    if status == "labeled" and not labels:
        issue(
            issues,
            "labeled_without_labels",
            candidate_id=cid,
            line=line,
            message="status=labeled requires at least one accepted label",
        )

    if status in {"uncertain", "out_of_scope"} and labels:
        issue(
            issues,
            "non_labeled_status_has_labels",
            candidate_id=cid,
            line=line,
            message=f"status={status} must have an empty labels array",
        )

    if status == "out_of_scope" and reason != "out_of_scope":
        issue(
            issues,
            "out_of_scope_reason_mismatch",
            candidate_id=cid,
            line=line,
            message="status=out_of_scope should use reason=out_of_scope",
        )

    accepted_ids: list[str] = []
    candidate_ids: list[str] = []

    for kind, items, low, high, sink in (
        ("label", labels, 0.85, 1.0, accepted_ids),
        ("candidate", candidates, 0.50, 0.84, candidate_ids),
    ):
        seen: set[str] = set()
        for index, item in enumerate(items):
            if not isinstance(item, dict):
                issue(
                    issues,
                    f"invalid_{kind}_entry",
                    candidate_id=cid,
                    line=line,
                    message=f"{kind}[{index}] must be an object",
                )
                continue

            label_id = item.get("id")
            confidence = item.get("confidence")

            if not isinstance(label_id, str) or not label_id:
                issue(
                    issues,
                    f"invalid_{kind}_id",
                    candidate_id=cid,
                    line=line,
                    message=f"{kind}[{index}].id must be a non-empty string",
                )
                continue

            sink.append(label_id)

            if label_id not in taxonomy:
                issue(
                    issues,
                    "unknown_taxonomy_label",
                    candidate_id=cid,
                    line=line,
                    message=f"Unknown taxonomy label: {label_id}",
                )

            if label_id in seen:
                issue(
                    issues,
                    f"duplicate_{kind}",
                    candidate_id=cid,
                    line=line,
                    message=f"Duplicate {kind} ID: {label_id}",
                )
            seen.add(label_id)

            if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
                issue(
                    issues,
                    f"invalid_{kind}_confidence",
                    candidate_id=cid,
                    line=line,
                    message=f"{kind} {label_id}: confidence must be numeric",
                )
            elif not (low <= float(confidence) <= high):
                issue(
                    issues,
                    f"{kind}_confidence_out_of_range",
                    candidate_id=cid,
                    line=line,
                    message=(
                        f"{kind} {label_id}: confidence {confidence} "
                        f"outside [{low:.2f}, {high:.2f}]"
                    ),
                )

    overlap = set(accepted_ids) & set(candidate_ids)
    for label_id in sorted(overlap):
        issue(
            issues,
            "accepted_candidate_overlap",
            candidate_id=cid,
            line=line,
            message=f"{label_id} appears in both labels and candidates",
        )

    # Parent + child should not be emitted together as accepted labels.
    accepted_set = set(accepted_ids)
    for label_id in accepted_ids:
        redundant_parents = ancestors(label_id, taxonomy) & accepted_set
        for parent in sorted(redundant_parents):
            issue(
                issues,
                "redundant_parent_child",
                candidate_id=cid,
                line=line,
                message=f"Accepted labels contain child {label_id} and parent {parent}",
            )


def confidence_bin(value: float) -> str:
    if value >= 0.95:
        return "0.95-1.00"
    if value >= 0.90:
        return "0.90-0.94"
    return "0.85-0.89"


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate full weak-label dataset.")
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--taxonomy", type=Path, default=DEFAULT_TAXONOMY)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    taxonomy, taxonomy_version = load_taxonomy(args.taxonomy)
    manifest = load_json(args.manifest)

    expected_ids_raw = manifest.get("candidate_ids")
    if not isinstance(expected_ids_raw, list) or not all(
        isinstance(item, str) for item in expected_ids_raw
    ):
        raise SystemExit(f"Invalid candidate_ids in manifest: {args.manifest}")

    expected_ids = expected_ids_raw
    expected_set = set(expected_ids)
    if len(expected_set) != len(expected_ids):
        raise SystemExit("Manifest itself contains duplicate candidate IDs")

    records, parse_issues = load_jsonl(args.results)
    issues: list[dict[str, Any]] = list(parse_issues)

    seen_counts: Counter[str] = Counter()
    valid_record_by_id: dict[str, dict[str, Any]] = {}

    for record in records:
        cid = record.get("candidate_id")
        if isinstance(cid, str):
            seen_counts[cid] += 1
            if seen_counts[cid] == 1:
                valid_record_by_id[cid] = record

        validate_record(record, taxonomy, issues)

    returned_ids = set(seen_counts)
    missing_ids = sorted(expected_set - returned_ids)
    unexpected_ids = sorted(returned_ids - expected_set)
    duplicate_ids = sorted(cid for cid, count in seen_counts.items() if count > 1)

    for cid in missing_ids:
        issue(
            issues,
            "missing_candidate",
            candidate_id=cid,
            message="Candidate is present in manifest but absent from results",
        )

    for cid in unexpected_ids:
        issue(
            issues,
            "unexpected_candidate",
            candidate_id=cid,
            message="Candidate appears in results but not in manifest",
        )

    for cid in duplicate_ids:
        issue(
            issues,
            "duplicate_candidate",
            candidate_id=cid,
            message=f"Candidate appears {seen_counts[cid]} times in results",
        )

    # Aggregate only expected IDs with exactly one result record.
    aggregate_records: list[dict[str, Any]] = []
    for cid in expected_ids:
        if seen_counts[cid] == 1:
            aggregate_records.append(valid_record_by_id[cid])

    status_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    label_track_counts: Counter[str] = Counter()
    label_assignment_counts: Counter[str] = Counter()
    uncertain_candidate_counts: Counter[str] = Counter()
    confidence_bins: Counter[str] = Counter()
    label_confidences: defaultdict[str, list[float]] = defaultdict(list)
    label_tracks: defaultdict[str, set[str]] = defaultdict(set)

    candidate_artists = load_candidate_artists(args.candidates)
    label_artist_counts: defaultdict[str, Counter[str]] = defaultdict(Counter)

    total_assignments = 0

    for record in aggregate_records:
        cid = record["candidate_id"]
        status = record.get("status")
        reason = record.get("reason")

        if isinstance(status, str):
            status_counts[status] += 1
        if isinstance(reason, str):
            reason_counts[reason] += 1

        labels = record.get("labels") if isinstance(record.get("labels"), list) else []
        seen_on_track: set[str] = set()

        for item in labels:
            if not isinstance(item, dict):
                continue
            label_id = item.get("id")
            confidence = item.get("confidence")
            if label_id not in taxonomy:
                continue

            label_assignment_counts[label_id] += 1
            total_assignments += 1

            if label_id not in seen_on_track:
                label_track_counts[label_id] += 1
                label_tracks[label_id].add(cid)
                seen_on_track.add(label_id)

            if isinstance(confidence, (int, float)) and not isinstance(confidence, bool):
                value = float(confidence)
                if 0.85 <= value <= 1.0:
                    confidence_bins[confidence_bin(value)] += 1
                    label_confidences[label_id].append(value)

            artist = candidate_artists.get(cid)
            if artist:
                label_artist_counts[label_id][artist] += 1

        candidates = (
            record.get("candidates")
            if isinstance(record.get("candidates"), list)
            else []
        )
        if status == "uncertain":
            for item in candidates:
                if isinstance(item, dict):
                    label_id = item.get("id")
                    if label_id in taxonomy:
                        uncertain_candidate_counts[label_id] += 1

    coverage_rows: list[dict[str, Any]] = []

    for genre in sorted(taxonomy.values(), key=lambda g: int(g.get("order", 999999))):
        label_id = genre["id"]
        count = label_track_counts[label_id]
        confidences = label_confidences[label_id]

        artist_counter = label_artist_counts[label_id]
        top_artist = ""
        top_artist_count = 0
        top_artist_share = 0.0
        unique_artists = len(artist_counter)

        if artist_counter:
            top_artist, top_artist_count = artist_counter.most_common(1)[0]
            if count:
                top_artist_share = top_artist_count / count

        coverage_rows.append(
            {
                "id": label_id,
                "label": genre.get("label", label_id),
                "role": genre.get("role", ""),
                "parent": genre.get("parent") or "",
                "tracks": count,
                "uncertain_mentions": uncertain_candidate_counts[label_id],
                "mean_confidence": (
                    round(sum(confidences) / len(confidences), 4)
                    if confidences
                    else ""
                ),
                "unique_artists": unique_artists,
                "top_artist": top_artist,
                "top_artist_tracks": top_artist_count,
                "top_artist_share": round(top_artist_share, 4) if artist_counter else "",
                "coverage_band": (
                    "0"
                    if count == 0
                    else "<10"
                    if count < 10
                    else "10-19"
                    if count < 20
                    else "20-29"
                    if count < 30
                    else "30+"
                ),
            }
        )

    coverage_band_counts = Counter(row["coverage_band"] for row in coverage_rows)
    zero_labels = [row["id"] for row in coverage_rows if row["tracks"] == 0]
    under_10 = [row["id"] for row in coverage_rows if row["tracks"] < 10]
    under_20 = [row["id"] for row in coverage_rows if row["tracks"] < 20]
    under_30 = [row["id"] for row in coverage_rows if row["tracks"] < 30]

    error_count = sum(1 for x in issues if x.get("severity") == "error")
    warning_count = sum(1 for x in issues if x.get("severity") == "warning")

    report = {
        "taxonomy_version": taxonomy_version,
        "expected_records": len(expected_ids),
        "parsed_result_records": len(records),
        "unique_returned_ids": len(returned_ids),
        "complete": not missing_ids and not unexpected_ids and not duplicate_ids,
        "missing_count": len(missing_ids),
        "unexpected_count": len(unexpected_ids),
        "duplicate_id_count": len(duplicate_ids),
        "status_counts": dict(sorted(status_counts.items())),
        "reason_counts": dict(sorted(reason_counts.items())),
        "accepted_label_assignments": total_assignments,
        "taxonomy_label_count": len(taxonomy),
        "labels_with_at_least_one_track": len(taxonomy) - len(zero_labels),
        "coverage_bands": {
            "0": coverage_band_counts["0"],
            "<10": coverage_band_counts["<10"],
            "10-19": coverage_band_counts["10-19"],
            "20-29": coverage_band_counts["20-29"],
            "30+": coverage_band_counts["30+"],
        },
        "labels_with_zero_tracks": zero_labels,
        "labels_under_10_tracks": under_10,
        "labels_under_20_tracks": under_20,
        "labels_under_30_tracks": under_30,
        "accepted_confidence_distribution": {
            "0.85-0.89": confidence_bins["0.85-0.89"],
            "0.90-0.94": confidence_bins["0.90-0.94"],
            "0.95-1.00": confidence_bins["0.95-1.00"],
        },
        "issue_counts": {
            "errors": error_count,
            "warnings": warning_count,
            "total": len(issues),
        },
        "missing_candidate_ids": missing_ids,
        "unexpected_candidate_ids": unexpected_ids,
        "duplicate_candidate_ids": duplicate_ids,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)

    report_path = args.output_dir / "label_report.json"
    coverage_path = args.output_dir / "class_coverage.csv"
    issues_path = args.output_dir / "issues.jsonl"

    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    fieldnames = [
        "id",
        "label",
        "role",
        "parent",
        "tracks",
        "uncertain_mentions",
        "mean_confidence",
        "unique_artists",
        "top_artist",
        "top_artist_tracks",
        "top_artist_share",
        "coverage_band",
    ]

    with coverage_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(coverage_rows)

    with issues_path.open("w", encoding="utf-8") as handle:
        for item in issues:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("Validation complete")
    print(f"  expected:       {len(expected_ids)}")
    print(f"  returned IDs:   {len(returned_ids)}")
    print(f"  labeled:        {status_counts['labeled']}")
    print(f"  uncertain:      {status_counts['uncertain']}")
    print(f"  out_of_scope:   {status_counts['out_of_scope']}")
    print(f"  assignments:    {total_assignments}")
    print(f"  missing:        {len(missing_ids)}")
    print(f"  unexpected:     {len(unexpected_ids)}")
    print(f"  duplicate IDs:  {len(duplicate_ids)}")
    print(f"  errors:         {error_count}")
    print()
    print("Class coverage")
    print(f"  30+ tracks:     {coverage_band_counts['30+']}")
    print(f"  20-29 tracks:   {coverage_band_counts['20-29']}")
    print(f"  10-19 tracks:   {coverage_band_counts['10-19']}")
    print(f"  <10 tracks:     {coverage_band_counts['<10']}")
    print(f"  zero tracks:    {coverage_band_counts['0']}")
    print()
    print(f"Report:   {report_path}")
    print(f"Coverage: {coverage_path}")
    print(f"Issues:   {issues_path}")

    if error_count:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
