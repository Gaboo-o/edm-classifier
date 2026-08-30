"""Rebuild unified candidate + label artifacts from the canonical merged manifest.

Input:
    data/embeddings/embedding_manifest.jsonl

Outputs:
    data/candidates/candidate_tracks.jsonl
    data/label_job/results/labeled_tracks.jsonl
    data/candidates/rebuild_report.json

This is a post-merge compatibility bridge for the existing
build_training_manifest.py pipeline.

Only successfully embedded unified candidates are represented here, which is
exactly the population intended for supervised training.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path("data/embeddings/embedding_manifest.jsonl")
DEFAULT_CANDIDATES = Path("data/candidates/candidate_tracks.jsonl")
DEFAULT_LABELS = Path("data/label_job/results/labeled_tracks.jsonl")
DEFAULT_REPORT = Path("data/candidates/rebuild_report.json")


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
                raise ValueError(
                    f"{path}:{line_number}: invalid JSON: {exc}"
                ) from exc

            if not isinstance(value, dict):
                raise ValueError(
                    f"{path}:{line_number}: expected JSON object"
                )

            rows.append(value)

    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )


def labels_and_confidence(
    row: dict[str, Any],
) -> tuple[list[str], dict[str, float]]:
    labels = []

    raw_labels = row.get("labels")
    if isinstance(raw_labels, list):
        for item in raw_labels:
            if isinstance(item, str):
                genre_id = item
            elif isinstance(item, dict):
                genre_id = item.get("id")
            else:
                continue

            if (
                isinstance(genre_id, str)
                and genre_id
                and genre_id not in labels
            ):
                labels.append(genre_id)

    confidence: dict[str, float] = {}

    for key in ("label_confidence", "label_confidences"):
        raw = row.get(key)
        if not isinstance(raw, dict):
            continue

        for genre_id, value in raw.items():
            if (
                isinstance(genre_id, str)
                and isinstance(value, (int, float))
                and not isinstance(value, bool)
            ):
                confidence[genre_id] = max(
                    float(value),
                    confidence.get(genre_id, 0.0),
                )

    return labels, confidence


def candidate_row(row: dict[str, Any]) -> dict[str, Any]:
    cid = row.get("candidate_id")
    if not isinstance(cid, str) or not cid:
        raise ValueError("merged row missing candidate_id")

    output = {
        "candidate_id": cid,
        "artist": row.get("artist"),
        "title": row.get("title"),
        "mbid": row.get("mbid"),
        "artist_mbid": row.get("artist_mbid"),
        "lastfm_url": row.get("lastfm_url"),
        "discovered_for": row.get("discovered_for", []),
        "top_tags": row.get("top_tags", []),
        "audio_source": row.get("audio_source"),
        "local_audio": row.get("local_audio"),
        "embedding": row.get("embedding"),
        "video_id": row.get("video_id"),
        "ytm_artists": row.get("ytm_artists", []),
        "dataset_sources": row.get(
            "dataset_sources",
            [row.get("dataset_source")]
            if row.get("dataset_source")
            else [],
        ),
    }

    return output


def label_row(row: dict[str, Any]) -> dict[str, Any]:
    cid = row.get("candidate_id")
    if not isinstance(cid, str) or not cid:
        raise ValueError("merged row missing candidate_id")

    labels, confidence = labels_and_confidence(row)

    label_objects = [
        {
            "id": genre_id,
            "confidence": confidence.get(genre_id, 1.0),
        }
        for genre_id in labels
    ]

    return {
        "candidate_id": cid,
        "status": "labeled" if label_objects else "uncertain",
        "labels": label_objects,
        "reason": "Unified post-merge training label",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild unified candidate and label artifacts."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
    )
    parser.add_argument(
        "--candidates-output",
        type=Path,
        default=DEFAULT_CANDIDATES,
    )
    parser.add_argument(
        "--labels-output",
        type=Path,
        default=DEFAULT_LABELS,
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT,
    )
    args = parser.parse_args()

    rows = load_jsonl(args.input)

    candidates = []
    labels = []

    seen_ids = set()
    duplicate_ids = 0
    no_labels = 0
    source_counts = Counter()

    for row in rows:
        cid = row.get("candidate_id")

        if not isinstance(cid, str) or not cid:
            raise ValueError("merged row missing candidate_id")

        if cid in seen_ids:
            duplicate_ids += 1
            raise ValueError(
                f"duplicate candidate_id in merged manifest: {cid}"
            )

        seen_ids.add(cid)

        c = candidate_row(row)
        l = label_row(row)

        candidates.append(c)
        labels.append(l)

        if l["status"] != "labeled":
            no_labels += 1

        for source in c.get("dataset_sources", []):
            if isinstance(source, str):
                source_counts[source] += 1

    write_jsonl(
        args.candidates_output,
        candidates,
    )
    write_jsonl(
        args.labels_output,
        labels,
    )

    report = {
        "input_manifest": str(args.input),
        "input_records": len(rows),
        "candidate_records": len(candidates),
        "label_records": len(labels),
        "unique_candidate_ids": len(seen_ids),
        "duplicate_candidate_ids": duplicate_ids,
        "records_without_labels": no_labels,
        "dataset_source_memberships": dict(source_counts),
        "outputs": {
            "candidates": str(args.candidates_output),
            "labels": str(args.labels_output),
        },
        "next_command": (
            "python -m edm_classifier.dataset.build_training_manifest "
            "--ignore-coverage-file"
        ),
    }

    args.report.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    args.report.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print("Unified candidate/label artifacts rebuilt")
    print(f"  input records:      {len(rows)}")
    print(f"  candidate records:  {len(candidates)}")
    print(f"  label records:      {len(labels)}")
    print(f"  records no labels:  {no_labels}")
    print()
    print(f"Candidates: {args.candidates_output}")
    print(f"Labels:     {args.labels_output}")
    print(f"Report:     {args.report}")


if __name__ == "__main__":
    main()
