"""Merge legacy V1 + V2 acquisition into one canonical dataset.

Canonical active data after this migration:

    data/audio/files/
    data/audio/audio_manifest.jsonl

    data/embeddings/pooled/
    data/embeddings/audio_embeddings.jsonl
    data/embeddings/embedding_manifest.jsonl
    data/embeddings/merge_report.json

The canonical candidate manifest deliberately preserves the legacy V1-style
shape because existing split/training code was already proven against it:

    candidate_id
    artist
    title
    labels                  -> list[str]
    label_confidence        -> dict[label_id, confidence]
    audio_source.video_id
    audio_source.artists
    local_audio.video_id
    local_audio.path
    embedding.video_id
    embedding.pooled_path

It ALSO exposes top-level video_id/ytm_artists for newer code.

V2 files are hard-linked into the canonical audio/embedding directories when
possible, with copy fallback.

Different candidate IDs sharing one video ID remain separate in the candidate
manifest; downstream split.py can collapse by video_id and union labels.

Run only after V1 baseline splits/runs have been archived under
data/baselines/v1/.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


EXPECTED_DIM = 1280
SAMPLE_RATE = 16000
MODEL_NAME = "discogs-effnet-bs64-1"
YT_VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")

V1_MANIFEST = Path("data/embeddings/embedding_manifest.jsonl")
V2_MANIFEST = Path("data/v2/embeddings/embedding_manifest.jsonl")
V2_RESOLUTION = Path("data/v2/audio_resolution/resolution_manifest.jsonl")

CANONICAL_AUDIO_DIR = Path("data/audio/files")
V2_AUDIO_DIR = Path("data/v2/audio/files")

CANONICAL_POOLED_DIR = Path("data/embeddings/pooled")
V2_POOLED_DIR = Path("data/v2/embeddings/pooled")

CANONICAL_AUDIO_MANIFEST = Path("data/audio/audio_manifest.jsonl")
CANONICAL_AUDIO_EMBEDDINGS = Path("data/embeddings/audio_embeddings.jsonl")
MERGE_REPORT = Path("data/embeddings/merge_report.json")
OVERLAP_REPORT = Path("data/embeddings/cross_source_overlaps.jsonl")

V1_BASELINE_MANIFEST = Path(
    "data/baselines/v1/manifests/embedding_manifest.jsonl"
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

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
                    f"{path}:{line_number}: expected object"
                )

            rows.append(value)

    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")

    with temp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )

    temp.replace(path)


def normalize_text(value: str) -> str:
    text = unicodedata.normalize("NFKD", value)
    text = "".join(
        c for c in text
        if not unicodedata.combining(c)
    )
    text = text.casefold()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def track_key(artist: Any, title: Any) -> str | None:
    if not isinstance(artist, str) or not isinstance(title, str):
        return None

    artist_n = normalize_text(artist)
    title_n = normalize_text(title)

    if not artist_n or not title_n:
        return None

    return artist_n + "\0" + title_n


def nested_string(
    row: dict[str, Any],
    container_key: str,
    value_key: str,
) -> str | None:
    container = row.get(container_key)
    if not isinstance(container, dict):
        return None

    value = container.get(value_key)
    if isinstance(value, str) and value.strip():
        return value.strip()

    return None


def resolve_video_id(row: dict[str, Any]) -> str | None:
    """Resolve IDs from both legacy V1 and V2/new schemas."""

    # New/top-level forms.
    for key in (
        "video_id",
        "videoId",
        "youtube_video_id",
        "yt_video_id",
    ):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    # Exact legacy V1 schema shown by the current dataset.
    for container_key in (
        "audio_source",
        "local_audio",
        "embedding",
        "resolution",
        "youtube",
        "ytm",
    ):
        for value_key in (
            "video_id",
            "videoId",
            "youtube_video_id",
        ):
            value = nested_string(
                row,
                container_key,
                value_key,
            )
            if value:
                return value

    # Legacy embedding paths.
    embedding = row.get("embedding")
    if isinstance(embedding, dict):
        for key in ("pooled_path", "path"):
            value = embedding.get(key)
            if isinstance(value, str) and value.strip():
                stem = Path(value).stem
                if YT_VIDEO_ID_RE.fullmatch(stem):
                    return stem

    # Newer flat path forms.
    for key in ("embedding_path", "audio_path"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            stem = Path(value).stem
            if YT_VIDEO_ID_RE.fullmatch(stem):
                return stem

    # Legacy local audio path.
    local_audio = row.get("local_audio")
    if isinstance(local_audio, dict):
        value = local_audio.get("path")
        if isinstance(value, str) and value.strip():
            stem = Path(value).stem
            if YT_VIDEO_ID_RE.fullmatch(stem):
                return stem

    return None


def normalize_labels(
    row: dict[str, Any],
) -> tuple[list[str], dict[str, float]]:
    labels: list[str] = []
    confidence: dict[str, float] = {}

    raw_labels = row.get("labels")
    if isinstance(raw_labels, list):
        for item in raw_labels:
            if isinstance(item, str):
                genre_id = item
                score = None
            elif isinstance(item, dict):
                genre_id = item.get("id")
                score = item.get("confidence")
            else:
                continue

            if not isinstance(genre_id, str) or not genre_id:
                continue

            if genre_id not in labels:
                labels.append(genre_id)

            if (
                isinstance(score, (int, float))
                and not isinstance(score, bool)
            ):
                old = confidence.get(genre_id)
                if old is None or float(score) > old:
                    confidence[genre_id] = float(score)

    # Legacy V1 spelling.
    for confidence_key in (
        "label_confidence",
        "label_confidences",
    ):
        raw = row.get(confidence_key)
        if not isinstance(raw, dict):
            continue

        for genre_id, score in raw.items():
            if not isinstance(genre_id, str):
                continue
            if (
                not isinstance(score, (int, float))
                or isinstance(score, bool)
            ):
                continue

            if genre_id not in labels:
                labels.append(genre_id)

            old = confidence.get(genre_id)
            if old is None or float(score) > old:
                confidence[genre_id] = float(score)

    return labels, confidence


def last_resolution_by_candidate(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    result = {}

    for row in rows:
        cid = row.get("candidate_id")
        if isinstance(cid, str):
            result[cid] = row

    return result


def duration_string(seconds: Any) -> str | None:
    if not isinstance(seconds, int) or seconds < 0:
        return None

    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)

    if hours:
        return f"{hours}:{minutes:02d}:{sec:02d}"

    return f"{minutes}:{sec:02d}"


def album_name(album: Any) -> str | None:
    if isinstance(album, str):
        return album

    if isinstance(album, dict):
        name = album.get("name")
        if isinstance(name, str):
            return name

    return None


def canonical_audio_source(
    row: dict[str, Any],
    *,
    video_id: str,
    source: str,
    resolution: dict[str, Any] | None,
) -> dict[str, Any]:
    if source == "v1":
        old = row.get("audio_source")
        if isinstance(old, dict):
            result = dict(old)
        else:
            result = {}

        result["provider"] = result.get(
            "provider",
            "youtube_music",
        )
        result["status"] = result.get(
            "status",
            "resolved",
        )
        result["video_id"] = video_id

        artists = result.get("artists")
        if not isinstance(artists, list):
            artists = []
        result["artists"] = artists

        return result

    resolved = resolution or {}

    artists = resolved.get("ytm_artists")
    if not isinstance(artists, list):
        artists = []

    artist_ids = resolved.get("ytm_artist_ids")
    if not isinstance(artist_ids, list):
        artist_ids = []

    seconds = resolved.get("duration_seconds")

    return {
        "provider": "youtube_music",
        "status": "resolved",
        "query": resolved.get("matched_query"),
        "video_id": video_id,
        "title": resolved.get(
            "ytm_title",
            row.get("title"),
        ),
        "artists": artists,
        "artist_ids": artist_ids,
        "album": album_name(resolved.get("album")),
        "album_id": (
            resolved.get("album", {}).get("id")
            if isinstance(resolved.get("album"), dict)
            else None
        ),
        "duration": duration_string(seconds),
        "duration_seconds": seconds,
        "result_type": resolved.get("result_type"),
        "video_type": resolved.get("video_type"),
        "resolution_source": resolved.get(
            "resolution_source",
            row.get("resolution_source"),
        ),
        "score": resolved.get("score"),
        "title_score": resolved.get("title_score"),
        "artist_score": resolved.get("artist_score"),
    }


def canonical_local_audio(
    row: dict[str, Any],
    *,
    video_id: str,
    source: str,
) -> dict[str, Any]:
    if source == "v1":
        old = row.get("local_audio")
        if isinstance(old, dict):
            result = dict(old)
            result["video_id"] = video_id
            return result

    # V2: path will be canonical after materialization. Extension may be
    # corrected later when the actual file is found.
    return {
        "video_id": video_id,
        "path": None,
        "filename": None,
        "extension": None,
        "bytes": None,
        "info_json": None,
    }


def canonical_embedding(
    row: dict[str, Any],
    *,
    video_id: str,
    source: str,
) -> dict[str, Any]:
    if source == "v1":
        old = row.get("embedding")
        result = dict(old) if isinstance(old, dict) else {}

        result["model"] = result.get(
            "model",
            MODEL_NAME,
        )
        result["video_id"] = video_id
        result["pooled_path"] = str(
            CANONICAL_POOLED_DIR / f"{video_id}.npy"
        )
        result["patches_path"] = None
        result["dimensions"] = EXPECTED_DIM
        result["pooling"] = "mean"
        result["sample_rate"] = SAMPLE_RATE
        return result

    return {
        "model": MODEL_NAME,
        "video_id": video_id,
        "pooled_path": str(
            CANONICAL_POOLED_DIR / f"{video_id}.npy"
        ),
        "patches_path": None,
        "dimensions": EXPECTED_DIM,
        "pooling": "mean",
        "sample_rate": SAMPLE_RATE,
    }


def canonical_row(
    row: dict[str, Any],
    source: str,
    resolution: dict[str, Any] | None,
) -> dict[str, Any]:
    cid = row.get("candidate_id")
    if not isinstance(cid, str) or not cid:
        raise ValueError(
            f"{source}: row missing candidate_id"
        )

    video_id = resolve_video_id(row)
    if not isinstance(video_id, str) or not video_id:
        raise ValueError(
            f"{source}:{cid}: missing video_id"
        )

    labels, confidence = normalize_labels(row)

    audio_source = canonical_audio_source(
        row,
        video_id=video_id,
        source=source,
        resolution=resolution,
    )

    output = {
        "candidate_id": cid,
        "artist": row.get("artist"),
        "title": row.get("title"),
        "labels": labels,
        "label_confidence": confidence,
        "video_id": video_id,
        "ytm_artists": list(
            audio_source.get("artists", [])
        ),
        "audio_source": audio_source,
        "local_audio": canonical_local_audio(
            row,
            video_id=video_id,
            source=source,
        ),
        "embedding": canonical_embedding(
            row,
            video_id=video_id,
            source=source,
        ),
        "dataset_source": source,
        "dataset_sources": [source],
    }

    # Preserve useful weak-supervision/provenance fields where they exist.
    for key in (
        "mbid",
        "artist_mbid",
        "lastfm_url",
        "discovered_for",
        "top_tags",
    ):
        if key in row:
            output[key] = row[key]

    return output


def merge_same_candidate(
    left: dict[str, Any],
    right: dict[str, Any],
) -> dict[str, Any]:
    if left["video_id"] != right["video_id"]:
        raise ValueError(
            f"candidate_id {left['candidate_id']} maps to "
            f"{left['video_id']} and {right['video_id']}"
        )

    merged = dict(left)

    sources = list(merged.get("dataset_sources", []))
    for source in right.get("dataset_sources", []):
        if source not in sources:
            sources.append(source)
    merged["dataset_sources"] = sources
    merged["dataset_source"] = "+".join(sorted(sources))

    labels = list(merged.get("labels", []))
    for genre_id in right.get("labels", []):
        if genre_id not in labels:
            labels.append(genre_id)
    merged["labels"] = labels

    confidence = dict(
        merged.get("label_confidence", {})
    )
    for genre_id, score in right.get(
        "label_confidence",
        {},
    ).items():
        old = confidence.get(genre_id)
        if old is None or score > old:
            confidence[genre_id] = score
    merged["label_confidence"] = confidence

    left_artists = merged.get("ytm_artists", [])
    right_artists = right.get("ytm_artists", [])
    union = []

    for artist in (
        left_artists if isinstance(left_artists, list) else []
    ) + (
        right_artists if isinstance(right_artists, list) else []
    ):
        if artist not in union:
            union.append(artist)

    merged["ytm_artists"] = union
    merged["audio_source"]["artists"] = union

    return merged


def valid_embedding(path: Path) -> bool:
    if not path.is_file():
        return False

    try:
        arr = np.load(
            path,
            mmap_mode="r",
            allow_pickle=False,
        )
    except Exception:
        return False

    return (
        arr.shape == (EXPECTED_DIM,)
        and np.issubdtype(arr.dtype, np.floating)
        and np.all(np.isfinite(arr))
    )


def equivalent_embeddings(
    left: Path,
    right: Path,
) -> bool:
    try:
        a = np.load(left, allow_pickle=False)
        b = np.load(right, allow_pickle=False)
    except Exception:
        return False

    return (
        a.shape == b.shape == (EXPECTED_DIM,)
        and np.allclose(
            a,
            b,
            rtol=1e-4,
            atol=1e-5,
        )
    )


def materialize(
    source: Path,
    destination: Path,
) -> str:
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if destination.exists():
        return "existing"

    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def audio_candidates(
    directory: Path,
    video_id: str,
) -> list[Path]:
    if not directory.exists():
        return []

    paths = []

    for path in directory.glob(f"{video_id}.*"):
        if not path.is_file():
            continue

        lower = path.name.casefold()

        if lower.endswith(
            (
                ".part",
                ".ytdl",
                ".tmp",
                ".temp",
                ".json",
            )
        ):
            continue

        paths.append(path)

    return sorted(
        paths,
        key=lambda p: (
            -p.stat().st_size,
            p.name,
        ),
    )


def update_local_audio_from_file(
    row: dict[str, Any],
    path: Path | None,
) -> None:
    video_id = row["video_id"]

    if path is None:
        row["local_audio"] = {
            "video_id": video_id,
            "path": None,
            "filename": None,
            "extension": None,
            "bytes": None,
            "info_json": None,
        }
        return

    info_path = path.with_name(
        f"{video_id}.info.json"
    )

    row["local_audio"] = {
        "video_id": video_id,
        "path": str(path),
        "filename": path.name,
        "extension": path.suffix.lstrip("."),
        "bytes": path.stat().st_size,
        "info_json": (
            str(info_path)
            if info_path.is_file()
            else None
        ),
    }


def backup_v1_manifest(
    v1_rows: list[dict[str, Any]],
) -> None:
    if V1_BASELINE_MANIFEST.exists():
        return

    write_jsonl(
        V1_BASELINE_MANIFEST,
        v1_rows,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge V1 + V2 into the canonical dataset."
    )

    parser.add_argument(
        "--v1-manifest",
        type=Path,
        default=V1_MANIFEST,
    )
    parser.add_argument(
        "--v2-manifest",
        type=Path,
        default=V2_MANIFEST,
    )
    parser.add_argument(
        "--v2-resolution",
        type=Path,
        default=V2_RESOLUTION,
    )

    args = parser.parse_args()

    # Load all old data before rewriting canonical files.
    v1 = load_jsonl(args.v1_manifest)
    v2 = load_jsonl(args.v2_manifest)
    resolution = last_resolution_by_candidate(
        load_jsonl(args.v2_resolution)
    )

    if not v1:
        raise SystemExit(
            f"No V1 records found: {args.v1_manifest}"
        )
    if not v2:
        raise SystemExit(
            f"No V2 records found: {args.v2_manifest}"
        )

    print("Unified dataset merge")
    print(f"  V1 candidate rows: {len(v1)}")
    print(f"  V2 candidate rows: {len(v2)}")

    # Preserve exact pre-merge V1 manifest in the baseline folder.
    backup_v1_manifest(v1)

    normalized = []

    for row in v1:
        normalized.append(
            canonical_row(
                row,
                "v1",
                None,
            )
        )

    for row in v2:
        cid = row.get("candidate_id")
        normalized.append(
            canonical_row(
                row,
                "v2",
                resolution.get(cid)
                if isinstance(cid, str)
                else None,
            )
        )

    by_candidate: dict[str, dict[str, Any]] = {}
    candidate_collisions = 0

    for row in normalized:
        cid = row["candidate_id"]

        if cid in by_candidate:
            candidate_collisions += 1
            by_candidate[cid] = merge_same_candidate(
                by_candidate[cid],
                row,
            )
        else:
            by_candidate[cid] = row

    candidates = list(by_candidate.values())

    by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        by_video[row["video_id"]].append(row)

    materialization = Counter()
    missing_embeddings = []
    missing_audio = []
    conflicts = []

    for index, video_id in enumerate(sorted(by_video), 1):
        if (
            index == 1
            or index % 1000 == 0
            or index == len(by_video)
        ):
            print(
                f"  materializing {index}/{len(by_video)} "
                f"unique recordings"
            )

        canonical_embedding = (
            CANONICAL_POOLED_DIR
            / f"{video_id}.npy"
        )
        v2_embedding = (
            V2_POOLED_DIR
            / f"{video_id}.npy"
        )

        if canonical_embedding.exists():
            if not valid_embedding(
                canonical_embedding
            ):
                raise RuntimeError(
                    f"Invalid canonical embedding: "
                    f"{canonical_embedding}"
                )

            if (
                v2_embedding.exists()
                and not equivalent_embeddings(
                    canonical_embedding,
                    v2_embedding,
                )
            ):
                conflicts.append(
                    {
                        "video_id": video_id,
                        "canonical": str(
                            canonical_embedding
                        ),
                        "v2": str(v2_embedding),
                    }
                )
        elif v2_embedding.exists():
            mode = materialize(
                v2_embedding,
                canonical_embedding,
            )
            materialization[
                f"embedding_{mode}"
            ] += 1
        else:
            missing_embeddings.append(video_id)

        canonical_audio = audio_candidates(
            CANONICAL_AUDIO_DIR,
            video_id,
        )

        if not canonical_audio:
            v2_audio = audio_candidates(
                V2_AUDIO_DIR,
                video_id,
            )

            if v2_audio:
                src = v2_audio[0]
                dst = CANONICAL_AUDIO_DIR / src.name
                mode = materialize(src, dst)
                materialization[
                    f"audio_{mode}"
                ] += 1
                canonical_audio = [dst]
            else:
                missing_audio.append(video_id)

        audio_path = (
            canonical_audio[0]
            if canonical_audio
            else None
        )

        for row in by_video[video_id]:
            update_local_audio_from_file(
                row,
                audio_path,
            )
            row["embedding"]["pooled_path"] = str(
                canonical_embedding
            )

    if conflicts:
        conflict_path = Path(
            "data/embeddings/embedding_conflicts.jsonl"
        )
        write_jsonl(
            conflict_path,
            conflicts,
        )
        raise RuntimeError(
            f"{len(conflicts)} V1/V2 embedding conflicts; "
            f"inspect {conflict_path}"
        )

    if missing_embeddings:
        missing_path = Path(
            "data/embeddings/missing_embeddings.jsonl"
        )
        write_jsonl(
            missing_path,
            [
                {"video_id": x}
                for x in missing_embeddings
            ],
        )
        raise RuntimeError(
            f"{len(missing_embeddings)} recordings are missing "
            f"pooled embeddings; inspect {missing_path}"
        )

    # Deterministic candidate ordering.
    candidates.sort(
        key=lambda row: (
            row["video_id"],
            row["candidate_id"],
        )
    )

    # One physical-recording row per video ID.
    audio_rows = []

    for video_id in sorted(by_video):
        rows = by_video[video_id]
        audio_files = audio_candidates(
            CANONICAL_AUDIO_DIR,
            video_id,
        )
        audio_path = (
            audio_files[0]
            if audio_files
            else None
        )

        labels = []
        label_confidence = {}
        candidate_ids = []
        artists = []
        ytm_artists = []
        dataset_sources = []
        resolution_sources = []

        for row in rows:
            candidate_ids.append(
                row["candidate_id"]
            )

            for genre_id in row["labels"]:
                if genre_id not in labels:
                    labels.append(genre_id)

            for genre_id, score in row[
                "label_confidence"
            ].items():
                old = label_confidence.get(
                    genre_id
                )
                if old is None or score > old:
                    label_confidence[
                        genre_id
                    ] = score

            artist = row.get("artist")
            if (
                isinstance(artist, str)
                and artist not in artists
            ):
                artists.append(artist)

            for artist_name in row.get(
                "ytm_artists",
                [],
            ):
                if (
                    isinstance(artist_name, str)
                    and artist_name not in ytm_artists
                ):
                    ytm_artists.append(
                        artist_name
                    )

            for source in row.get(
                "dataset_sources",
                [],
            ):
                if source not in dataset_sources:
                    dataset_sources.append(source)

            resolution_source = row.get(
                "audio_source",
                {},
            ).get("resolution_source")

            if (
                isinstance(
                    resolution_source,
                    str,
                )
                and resolution_source
                not in resolution_sources
            ):
                resolution_sources.append(
                    resolution_source
                )

        audio_rows.append(
            {
                "video_id": video_id,
                "status": "embedded",
                "audio_path": (
                    str(audio_path)
                    if audio_path
                    else None
                ),
                "embedding_path": str(
                    CANONICAL_POOLED_DIR
                    / f"{video_id}.npy"
                ),
                "embedding_shape": [
                    EXPECTED_DIM
                ],
                "pooling": "mean",
                "candidate_ids": sorted(
                    candidate_ids
                ),
                "labels": sorted(labels),
                "label_confidence": (
                    label_confidence
                ),
                "artists": sorted(artists),
                "ytm_artists": sorted(
                    ytm_artists
                ),
                "dataset_sources": sorted(
                    dataset_sources
                ),
                "resolution_sources": sorted(
                    resolution_sources
                ),
            }
        )

    # Report normalized artist/title overlaps across V1/V2 that resolved to
    # different videos. Do not silently delete them.
    text_groups: dict[
        str,
        list[dict[str, Any]],
    ] = defaultdict(list)

    for row in candidates:
        key = track_key(
            row.get("artist"),
            row.get("title"),
        )
        if key:
            text_groups[key].append(row)

    overlaps = []

    for rows in text_groups.values():
        sources = {
            source
            for row in rows
            for source in row.get(
                "dataset_sources",
                [],
            )
        }
        video_ids = {
            row["video_id"]
            for row in rows
        }

        if (
            "v1" in sources
            and "v2" in sources
            and len(video_ids) > 1
        ):
            overlaps.append(
                {
                    "artist": rows[0].get(
                        "artist"
                    ),
                    "title": rows[0].get(
                        "title"
                    ),
                    "video_ids": sorted(
                        video_ids
                    ),
                    "candidate_ids": sorted(
                        {
                            row["candidate_id"]
                            for row in rows
                        }
                    ),
                }
            )

    # Now commit canonical manifests.
    write_jsonl(
        args.v1_manifest,
        candidates,
    )
    write_jsonl(
        CANONICAL_AUDIO_MANIFEST,
        audio_rows,
    )
    write_jsonl(
        CANONICAL_AUDIO_EMBEDDINGS,
        audio_rows,
    )
    write_jsonl(
        OVERLAP_REPORT,
        overlaps,
    )

    video_source_counts = Counter()

    for row in audio_rows:
        sources = set(
            row["dataset_sources"]
        )

        if sources == {"v1"}:
            video_source_counts[
                "v1_only"
            ] += 1
        elif sources == {"v2"}:
            video_source_counts[
                "v2_only"
            ] += 1
        else:
            video_source_counts[
                "shared_v1_v2"
            ] += 1

    report = {
        "inputs": {
            "v1_candidate_rows": len(v1),
            "v2_candidate_rows": len(v2),
        },
        "unified": {
            "candidate_rows": len(
                candidates
            ),
            "unique_video_ids": len(
                audio_rows
            ),
            "candidate_id_collisions_merged": (
                candidate_collisions
            ),
            "video_source_counts": dict(
                video_source_counts
            ),
            "cross_source_text_overlaps_different_video_ids": len(
                overlaps
            ),
        },
        "physical_materialization": dict(
            materialization
        ),
        "validation": {
            "missing_embeddings": len(
                missing_embeddings
            ),
            "missing_audio": len(
                missing_audio
            ),
            "embedding_conflicts": len(
                conflicts
            ),
            "embedding_dimension": (
                EXPECTED_DIM
            ),
            "pooling": "mean",
        },
        "baseline_backup": {
            "v1_embedding_manifest": str(
                V1_BASELINE_MANIFEST
            ),
        },
        "canonical_outputs": {
            "embedding_manifest": str(
                args.v1_manifest
            ),
            "pooled_dir": str(
                CANONICAL_POOLED_DIR
            ),
            "audio_dir": str(
                CANONICAL_AUDIO_DIR
            ),
            "audio_manifest": str(
                CANONICAL_AUDIO_MANIFEST
            ),
            "audio_embeddings": str(
                CANONICAL_AUDIO_EMBEDDINGS
            ),
            "overlap_report": str(
                OVERLAP_REPORT
            ),
        },
    }

    MERGE_REPORT.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print("Unified dataset merge complete")
    print(
        f"  candidate rows:   "
        f"{len(candidates)}"
    )
    print(
        f"  unique recordings:"
        f" {len(audio_rows)}"
    )
    print(
        f"  missing audio:    "
        f"{len(missing_audio)}"
    )
    print(
        f"  text overlaps:    "
        f"{len(overlaps)}"
    )
    print()
    print(f"Report: {MERGE_REPORT}")
    print(
        "Keep data/v2/ until merge_report.json and the new "
        "unified splits have been verified."
    )


if __name__ == "__main__":
    main()
