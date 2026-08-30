"""Create regular and artist-separated train/validation/test splits.

Inputs:
  data/embeddings/embedding_manifest.jsonl
  data/training/classes.json

Outputs:
  data/splits/
    samples.jsonl
    class_distribution.csv
    split_report.json
    regular/{train,validation,test}.jsonl
    regular/class_distribution.csv
    artist/{train,validation,test}.jsonl
    artist/class_distribution.csv

The script first collapses duplicate candidate rows that share the same video_id
into one unique audio sample and unions their labels. The regular split assigns
unique tracks independently. The artist split builds connected components from
credited artists, so collaborations cannot leak an artist across splits.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

DEFAULT_INPUT = Path("data/embeddings/embedding_manifest.jsonl")
DEFAULT_CLASSES = Path("data/training/classes.json")
DEFAULT_OUTPUT_DIR = Path("data/splits")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, 1):
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object")
            rows.append(obj)
    return rows


def load_classes(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    classes = raw.get("classes")
    if not isinstance(classes, list):
        raise ValueError(f"{path}: missing classes array")
    return [x for x in classes if isinstance(x, dict)]


def normalize_artist(value: str) -> str:
    text = unicodedata.normalize("NFKD", value)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.casefold()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def get_video_id(record: dict[str, Any]) -> str:
    for container_name in ("embedding", "local_audio", "audio_source"):
        container = record.get(container_name)
        if isinstance(container, dict):
            value = container.get("video_id")
            if isinstance(value, str) and value:
                return value
    raise ValueError(f"record {record.get('candidate_id')!r} has no video_id")


def get_embedding_path(record: dict[str, Any]) -> str:
    embedding = record.get("embedding")
    if isinstance(embedding, dict):
        value = embedding.get("pooled_path")
        if isinstance(value, str) and value:
            return value
    raise ValueError(f"record {record.get('candidate_id')!r} has no pooled embedding path")


def get_labels(record: dict[str, Any]) -> list[str]:
    labels = record.get("labels")
    if not isinstance(labels, list):
        return []
    return sorted({x for x in labels if isinstance(x, str) and x})


def get_artist_names(record: dict[str, Any]) -> list[str]:
    names: list[str] = []

    source = record.get("audio_source")
    if isinstance(source, dict):
        artists = source.get("artists")
        if isinstance(artists, list):
            for value in artists:
                if isinstance(value, str) and value.strip():
                    names.append(value.strip())

    original = record.get("artist")
    if isinstance(original, str) and original.strip():
        names.append(original.strip())

    out: list[str] = []
    seen: set[str] = set()
    for name in names:
        key = normalize_artist(name)
        if key and key not in seen:
            seen.add(key)
            out.append(name)
    return out


def build_unique_samples(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[get_video_id(record)].append(record)

    samples: list[dict[str, Any]] = []
    for video_id, group in grouped.items():
        embedding_paths = {get_embedding_path(x) for x in group}
        if len(embedding_paths) != 1:
            raise ValueError(f"{video_id}: records disagree on pooled embedding path")

        labels: set[str] = set()
        candidate_ids: set[str] = set()
        artist_map: dict[str, str] = {}
        for record in group:
            labels.update(get_labels(record))
            cid = record.get("candidate_id")
            if isinstance(cid, str) and cid:
                candidate_ids.add(cid)
            for artist in get_artist_names(record):
                artist_map.setdefault(normalize_artist(artist), artist)

        if not labels:
            continue

        first = group[0]
        samples.append({
            "sample_id": video_id,
            "video_id": video_id,
            "candidate_ids": sorted(candidate_ids),
            "artist": first.get("artist"),
            "title": first.get("title"),
            "artists": sorted(artist_map.values(), key=str.casefold),
            "artist_keys": sorted(artist_map),
            "labels": sorted(labels),
            "embedding_path": next(iter(embedding_paths)),
        })

    samples.sort(key=lambda x: x["sample_id"])
    return samples


class UnionFind:
    def __init__(self, items: Iterable[str]):
        self.parent = {x: x for x in items}
        self.rank = {x: 0 for x in items}

    def find(self, item: str) -> str:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


@dataclass
class Unit:
    unit_id: str
    sample_ids: list[str]
    size: int
    label_counts: Counter[str]


def regular_units(samples: list[dict[str, Any]]) -> list[Unit]:
    return [
        Unit(
            unit_id=s["sample_id"],
            sample_ids=[s["sample_id"]],
            size=1,
            label_counts=Counter(s["labels"]),
        )
        for s in samples
    ]


def artist_units(samples: list[dict[str, Any]]) -> tuple[list[Unit], dict[str, str]]:
    ids = [s["sample_id"] for s in samples]
    uf = UnionFind(ids)
    first_for_artist: dict[str, str] = {}

    for sample in samples:
        sid = sample["sample_id"]
        for artist_key in sample.get("artist_keys") or []:
            previous = first_for_artist.get(artist_key)
            if previous is None:
                first_for_artist[artist_key] = sid
            else:
                uf.union(sid, previous)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        grouped[uf.find(sample["sample_id"])].append(sample)

    units: list[Unit] = []
    sample_to_component: dict[str, str] = {}
    for group in grouped.values():
        sample_ids = sorted(s["sample_id"] for s in group)
        component_id = hashlib.sha256("\n".join(sample_ids).encode()).hexdigest()[:16]
        counts: Counter[str] = Counter()
        for sample in group:
            counts.update(sample["labels"])
            sample_to_component[sample["sample_id"]] = component_id
        units.append(Unit(component_id, sample_ids, len(group), counts))

    return units, sample_to_component


def targets(samples: list[dict[str, Any]], ratios: dict[str, float]):
    total = len(samples)
    train_n = round(total * ratios["train"])
    val_n = round(total * ratios["validation"])
    target_sizes = {
        "train": train_n,
        "validation": val_n,
        "test": total - train_n - val_n,
    }

    overall: Counter[str] = Counter()
    for sample in samples:
        overall.update(sample["labels"])

    target_labels = {
        split: {label: count * ratios[split] for label, count in overall.items()}
        for split in ratios
    }
    return target_sizes, target_labels, overall


def greedy_split(
    units: list[Unit],
    samples_by_id: dict[str, dict[str, Any]],
    ratios: dict[str, float],
    seed: str,
) -> dict[str, list[str]]:
    samples = [samples_by_id[sid] for unit in units for sid in unit.sample_ids]
    target_sizes, target_labels, overall = targets(samples, ratios)
    split_names = ["train", "validation", "test"]
    assigned = {name: [] for name in split_names}
    current_sizes: Counter[str] = Counter()
    current_labels = {name: Counter() for name in split_names}

    def rarity(unit: Unit) -> float:
        return sum(count / overall[label] for label, count in unit.label_counts.items())

    def jitter(text: str) -> float:
        digest = hashlib.sha256(f"{seed}|{text}".encode()).digest()
        return int.from_bytes(digest[:8], "big") / 2**64

    ordered = sorted(units, key=lambda u: (-rarity(u), -u.size, jitter(u.unit_id)))
    total_samples = sum(u.size for u in units)

    for unit in ordered:
        best_split = None
        best_score = None
        for split in split_names:
            target_size = max(1, target_sizes[split])
            after_size = current_sizes[split] + unit.size
            size_error = ((after_size - target_size) / target_size) ** 2

            label_error = 0.0
            weight_sum = 0.0
            for label, add in unit.label_counts.items():
                target = max(1.0, target_labels[split][label])
                after = current_labels[split][label] + add
                weight = 1.0 / math.sqrt(max(1, overall[label]))
                label_error += weight * ((after - target) / target) ** 2
                weight_sum += weight
            if weight_sum:
                label_error /= weight_sum

            desired_fraction = target_sizes[split] / max(1, total_samples)
            actual_fraction = after_size / max(1, total_samples)
            occupancy = (actual_fraction - desired_fraction) ** 2

            score = 4.0 * size_error + 2.5 * label_error + occupancy
            score += jitter(f"{unit.unit_id}|{split}") * 1e-9
            if best_score is None or score < best_score:
                best_score = score
                best_split = split

        assert best_split is not None
        assigned[best_split].extend(unit.sample_ids)
        current_sizes[best_split] += unit.size
        current_labels[best_split].update(unit.label_counts)

    for split in split_names:
        assigned[split].sort()
    return assigned


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def distribution_rows(
    classes: list[dict[str, Any]],
    all_samples: list[dict[str, Any]],
    split_samples: dict[str, list[dict[str, Any]]] | None = None,
) -> list[dict[str, Any]]:
    overall: Counter[str] = Counter()
    for sample in all_samples:
        overall.update(sample["labels"])

    per_split: dict[str, Counter[str]] = {}
    if split_samples is not None:
        for split, samples in split_samples.items():
            counter: Counter[str] = Counter()
            for sample in samples:
                counter.update(sample["labels"])
            per_split[split] = counter

    rows: list[dict[str, Any]] = []
    known: set[str] = set()
    for item in classes:
        label_id = item.get("id")
        if not isinstance(label_id, str):
            continue
        known.add(label_id)
        row: dict[str, Any] = {
            "index": item.get("index", ""),
            "id": label_id,
            "label": item.get("label", label_id),
            "tracks": overall[label_id],
        }
        if split_samples is not None:
            row.update({
                "train": per_split["train"][label_id],
                "validation": per_split["validation"][label_id],
                "test": per_split["test"][label_id],
            })
        rows.append(row)

    for label_id in sorted(set(overall) - known):
        row = {"index": "", "id": label_id, "label": label_id, "tracks": overall[label_id]}
        if split_samples is not None:
            row.update({
                "train": per_split["train"][label_id],
                "validation": per_split["validation"][label_id],
                "test": per_split["test"][label_id],
            })
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def artist_overlap(split_samples: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    artist_sets: dict[str, set[str]] = {}
    for split, samples in split_samples.items():
        values: set[str] = set()
        for sample in samples:
            values.update(sample.get("artist_keys") or [])
        artist_sets[split] = values

    result = {}
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        common = artist_sets[a] & artist_sets[b]
        result[f"{a}_{b}"] = {"count": len(common), "artists": sorted(common)[:100]}
    return result


def split_summary(name: str, assigned: dict[str, list[str]], samples_by_id: dict[str, dict[str, Any]]):
    split_samples = {
        split: [samples_by_id[sid] for sid in ids]
        for split, ids in assigned.items()
    }
    return {
        "name": name,
        "sample_counts": {split: len(rows) for split, rows in split_samples.items()},
        "artist_overlap": artist_overlap(split_samples),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create regular and artist-separated splits.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--classes", type=Path, default=DEFAULT_CLASSES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train", type=float, default=0.80)
    parser.add_argument("--validation", type=float, default=0.10)
    parser.add_argument("--test", type=float, default=0.10)
    parser.add_argument("--seed", default="edm-classifier-split")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ratios = {"train": args.train, "validation": args.validation, "test": args.test}
    if not math.isclose(sum(ratios.values()), 1.0, abs_tol=1e-9):
        raise SystemExit(f"split ratios must sum to 1.0; got {sum(ratios.values())}")

    input_records = load_jsonl(args.input)
    classes = load_classes(args.classes)
    samples = build_unique_samples(input_records)
    samples_by_id = {s["sample_id"]: s for s in samples}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "samples.jsonl", samples)
    write_csv(
        args.output_dir / "class_distribution.csv",
        distribution_rows(classes, samples),
    )

    reg_units = regular_units(samples)
    art_units, sample_to_component = artist_units(samples)

    strategies = {
        "regular": greedy_split(reg_units, samples_by_id, ratios, args.seed + "|regular"),
        "artist": greedy_split(art_units, samples_by_id, ratios, args.seed + "|artist"),
    }

    reports: dict[str, Any] = {}
    for strategy, assigned in strategies.items():
        strategy_dir = args.output_dir / strategy
        split_samples: dict[str, list[dict[str, Any]]] = {}
        for split in ("train", "validation", "test"):
            rows = [samples_by_id[sid] for sid in assigned[split]]
            split_samples[split] = rows
            output = []
            for row in rows:
                item = dict(row)
                item["split"] = split
                item["split_strategy"] = strategy
                if strategy == "artist":
                    item["artist_component"] = sample_to_component[row["sample_id"]]
                output.append(item)
            write_jsonl(strategy_dir / f"{split}.jsonl", output)

        write_csv(
            strategy_dir / "class_distribution.csv",
            distribution_rows(classes, samples, split_samples),
        )
        reports[strategy] = split_summary(strategy, assigned, samples_by_id)

    overall: Counter[str] = Counter()
    for sample in samples:
        overall.update(sample["labels"])

    report = {
        "input": str(args.input),
        "input_candidate_records": len(input_records),
        "unique_embedded_audio_samples": len(samples),
        "duplicate_candidate_records_collapsed": len(input_records) - len(samples),
        "total_label_assignments": sum(overall.values()),
        "classes_with_at_least_one_track": sum(1 for x in overall.values() if x > 0),
        "artist_components": len(art_units),
        "largest_artist_component_tracks": max((u.size for u in art_units), default=0),
        "ratios": ratios,
        "seed": args.seed,
        "strategies": reports,
        "outputs": {
            "samples": str(args.output_dir / "samples.jsonl"),
            "overall_class_distribution": str(args.output_dir / "class_distribution.csv"),
            "regular_dir": str(args.output_dir / "regular"),
            "artist_dir": str(args.output_dir / "artist"),
        },
    }
    (args.output_dir / "split_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("Split generation complete")
    print(f"  candidate records in:      {len(input_records)}")
    print(f"  unique audio samples:      {len(samples)}")
    print(f"  duplicates collapsed:      {len(input_records) - len(samples)}")
    print(f"  label assignments:         {sum(overall.values())}")
    print(f"  artist components:         {len(art_units)}")
    print()
    for strategy in ("regular", "artist"):
        summary = reports[strategy]
        counts = summary["sample_counts"]
        overlap = summary["artist_overlap"]
        print(f"{strategy}:")
        print(f"  train/val/test: {counts['train']} / {counts['validation']} / {counts['test']}")
        print(f"  train-test artist overlap: {overlap['train_test']['count']}")
        print(f"  train-val artist overlap:  {overlap['train_validation']['count']}")
        print()
    print(f"Class distribution: {args.output_dir / 'class_distribution.csv'}")
    print(f"Report:             {args.output_dir / 'split_report.json'}")


if __name__ == "__main__":
    main()
