"""Build one self-contained LLM labeling job from all candidate tracks.

The output Markdown file contains the weak-labeling rules, complete taxonomy,
output contract, and every compact candidate record. It is intended to be
uploaded to a capable long-context LLM in one conversation. The model should
write the results to a JSONL file rather than echoing thousands of lines into
chat.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CANDIDATES = Path("data/candidates/candidate_tracks.jsonl")
DEFAULT_TAXONOMY = Path("config/taxonomy.yaml")
DEFAULT_RULES = Path("config/label_prompt.md")
DEFAULT_OUTPUT = Path("data/label_job/labeling_job.md")
DEFAULT_MANIFEST = Path("data/label_job/manifest.json")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: expected object")
            for field in ("candidate_id", "artist", "title"):
                if not isinstance(record.get(field), str) or not record[field].strip():
                    raise ValueError(f"{path}:{line_number}: invalid {field}")
            cid = record["candidate_id"]
            if cid in seen_ids:
                raise ValueError(f"{path}:{line_number}: duplicate candidate_id {cid!r}")
            seen_ids.add(cid)
            records.append(record)
    if not records:
        raise ValueError(f"No records in {path}")
    return records


def deterministic_mix(records: list[dict[str, Any]], seed: str) -> list[dict[str, Any]]:
    def key(record: dict[str, Any]) -> str:
        return hashlib.sha256(f"{seed}\0{record['candidate_id']}".encode()).hexdigest()
    return sorted(records, key=key)


def simplify_discoveries(items: Any, max_items: int) -> list[dict[str, Any]]:
    if not isinstance(items, list):
        return []
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        label_id = str(item.get("label_id") or "").strip()
        if not label_id or label_id in seen:
            continue
        seen.add(label_id)
        compact: dict[str, Any] = {"label_id": label_id}
        if item.get("label"):
            compact["label"] = str(item["label"])
        if isinstance(item.get("rank"), int):
            compact["rank"] = item["rank"]
        if item.get("query"):
            compact["query"] = str(item["query"])
        out.append(compact)
        if len(out) >= max_items:
            break
    return out


def simplify_tags(items: Any, max_items: int) -> list[dict[str, Any]]:
    if not isinstance(items, list):
        return []
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if not name or name.casefold() in seen:
            continue
        seen.add(name.casefold())
        compact: dict[str, Any] = {"name": name}
        try:
            compact["count"] = int(item.get("count"))
        except (TypeError, ValueError):
            pass
        out.append(compact)
        if len(out) >= max_items:
            break
    return out


def compact_candidate(record: dict[str, Any], top_tags: int, discoveries: int) -> dict[str, Any]:
    out: dict[str, Any] = {
        "candidate_id": record["candidate_id"],
        "artist": record["artist"],
        "title": record["title"],
    }
    if isinstance(record.get("mbid"), str) and record["mbid"].strip():
        out["mbid"] = record["mbid"].strip()
    ds = simplify_discoveries(record.get("discovered_for"), discoveries)
    if ds:
        out["discovered_for"] = ds
    tags = simplify_tags(record.get("top_tags"), top_tags)
    if tags:
        out["top_tags"] = tags
    if record.get("top_tags_error"):
        out["top_tags_unavailable"] = True
    return out


def taxonomy_summary(taxonomy: dict[str, Any]) -> str:
    meta = taxonomy.get("taxonomy") or {}
    policy = taxonomy.get("labeling_policy") or {}
    boundaries = taxonomy.get("boundary_rules") or []
    genres: list[dict[str, Any]] = []
    for genre in taxonomy["genres"]:
        item: dict[str, Any] = {
            "id": genre["id"],
            "label": genre["label"],
            "role": genre.get("role"),
        }
        if genre.get("parent"):
            item["parent"] = genre["parent"]
        if genre.get("bpm"):
            item["bpm"] = genre["bpm"]
        if genre.get("characteristics"):
            item["characteristics"] = genre["characteristics"]
        if genre.get("aliases"):
            item["aliases"] = genre["aliases"]
        genres.append(item)
    return "\n".join([
        "## Taxonomy metadata",
        "",
        f"- ID: `{meta.get('id', '')}`",
        f"- Version: `{meta.get('version', '')}`",
        f"- Task: `{meta.get('task', '')}`",
        "",
        "## Taxonomy policy",
        "",
        "```yaml",
        yaml.safe_dump(policy, sort_keys=False, allow_unicode=True).rstrip(),
        "```",
        "",
        "## Boundary rules",
        "",
        "```yaml",
        yaml.safe_dump(boundaries, sort_keys=False, allow_unicode=True).rstrip(),
        "```",
        "",
        "## Allowed labels",
        "",
        "```yaml",
        yaml.safe_dump(genres, sort_keys=False, allow_unicode=True).rstrip(),
        "```",
    ])


def output_contract() -> str:
    return r'''## Required output

Process every candidate in this file.

Create a file named `labeled_tracks.jsonl`. Do **not** paste thousands of result lines into conversational prose if file creation is available.

The file must contain exactly one JSON object per input candidate, one object per line, with no Markdown fences and no surrounding JSON array.

Each line must have this shape:

{"candidate_id":"lastfm:...","status":"labeled","labels":[{"id":"future_bass","confidence":0.94}],"candidates":[],"reason":"exact_tag_support"}

Allowed status values:
- `labeled`
- `uncertain`
- `out_of_scope`

Rules:
- Copy `candidate_id` exactly.
- `labels`: 0-2 accepted taxonomy labels, confidence 0.85-1.00.
- `candidates`: 0-3 plausible labels, confidence 0.50-0.84.
- `reason`: one of `exact_tag_support`, `cross_tag_support`, `broad_family_only`, `boundary_ambiguous`, `conflicting_evidence`, `insufficient_evidence`, `out_of_scope`.
- For `uncertain` and `out_of_scope`, `labels` must be empty.
- Never emit a child plus its derived parent for the same classification fact.
- Every input candidate must appear exactly once in the output file.

After creating the result file, report only a short summary containing total records, labeled, uncertain, and out_of_scope counts.'''


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build one full weak-labeling job for an LLM.")
    p.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    p.add_argument("--taxonomy", type=Path, default=DEFAULT_TAXONOMY)
    p.add_argument("--rules", type=Path, default=DEFAULT_RULES)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--top-tags", type=int, default=10)
    p.add_argument("--discoveries", type=int, default=5)
    p.add_argument("--mix-seed", default="edm-classifier-v0.2")
    p.add_argument("--preserve-order", action="store_true")
    p.add_argument("--limit", type=int)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    taxonomy = yaml.safe_load(args.taxonomy.read_text(encoding="utf-8"))
    if not isinstance(taxonomy, dict) or not isinstance(taxonomy.get("genres"), list):
        raise SystemExit(f"Invalid taxonomy: {args.taxonomy}")
    records = load_jsonl(args.candidates)
    if not args.preserve_order:
        records = deterministic_mix(records, args.mix_seed)
    if args.limit is not None:
        if args.limit <= 0:
            raise SystemExit("--limit must be positive")
        records = records[:args.limit]

    compact = [compact_candidate(r, args.top_tags, args.discoveries) for r in records]
    rules = args.rules.read_text(encoding="utf-8").strip()

    lines = [
        "# EDM Full Candidate Weak-Labeling Job",
        "",
        "This is one complete labeling job. Read the instructions and taxonomy first, then classify every candidate in the final section.",
        "",
        rules,
        "",
        output_contract(),
        "",
        taxonomy_summary(taxonomy),
        "",
        "## Candidate tracks",
        "",
        f"Total candidates: **{len(compact)}**",
        "",
        "Each following line is one candidate JSON object:",
        "",
        "```jsonl",
    ]
    lines.extend(json.dumps(r, ensure_ascii=False, separators=(",", ":")) for r in compact)
    lines.extend(["```", ""])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")

    manifest = {
        "taxonomy_version": (taxonomy.get("taxonomy") or {}).get("version"),
        "candidate_source": str(args.candidates),
        "candidate_count": len(compact),
        "top_tags_per_track": args.top_tags,
        "discoveries_per_track": args.discoveries,
        "mixed": not args.preserve_order,
        "mix_seed": None if args.preserve_order else args.mix_seed,
        "candidate_ids": [r["candidate_id"] for r in compact],
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    size_mb = args.output.stat().st_size / (1024 * 1024)
    print(f"built single labeling job for {len(compact)} tracks")
    print(f"job:      {args.output}")
    print(f"manifest: {args.manifest}")
    print(f"size:     {size_mb:.2f} MiB")
    print("upload labeling_job.md to the LLM and ask it to follow the embedded instructions")


if __name__ == "__main__":
    main()
