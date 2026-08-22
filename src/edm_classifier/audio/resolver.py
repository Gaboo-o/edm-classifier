"""Resolve weakly-labeled training tracks to YouTube Music song IDs.

Input:
    data/training/training_manifest.jsonl

Outputs:
    data/audio_resolution/audio_manifest.jsonl
    data/audio_resolution/review.jsonl
    data/audio_resolution/resolver_report.json
    data/audio_resolution/search_cache.jsonl

This stage does NOT download audio. It only maps each Artist + Title pair to a
public YouTube Music song result (videoId) with a confidence score.

Resolution policy:
- Search the public YouTube Music catalogue with filter="songs".
- Prefer MUSIC_VIDEO_TYPE_ATV when videoType is present.
- Score title similarity, artist similarity, search rank, and version-marker
  agreement (remix/live/acoustic/etc.).
- Auto-resolve only high-confidence matches with enough separation from the
  second-best result.
- Route ambiguous matches to review instead of guessing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

from rapidfuzz import fuzz
from ytmusicapi import YTMusic


DEFAULT_INPUT = Path("data/training/training_manifest.jsonl")
DEFAULT_OUTPUT_DIR = Path("data/audio_resolution")

VERSION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("remix", re.compile(r"\bremix\b", re.I)),
    ("live", re.compile(r"\blive\b", re.I)),
    ("acoustic", re.compile(r"\bacoustic\b", re.I)),
    ("instrumental", re.compile(r"\binstrumental\b", re.I)),
    ("vip", re.compile(r"\bvip\b", re.I)),
    ("extended", re.compile(r"\bextended(?:\s+mix)?\b", re.I)),
    ("radio_edit", re.compile(r"\bradio\s+edit\b", re.I)),
    ("edit", re.compile(r"\bedit\b", re.I)),
    ("remaster", re.compile(r"\bremaster(?:ed)?\b", re.I)),
    ("sped_up", re.compile(r"\bsped[\s-]*up\b", re.I)),
    ("slowed", re.compile(r"\bslowed\b", re.I)),
    ("reverb", re.compile(r"\breverb(?:ed)?\b", re.I)),
    ("nightcore", re.compile(r"\bnightcore\b", re.I)),
    ("cover", re.compile(r"\bcover\b", re.I)),
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
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
                raise ValueError(f"{path}:{line_number}: each JSONL line must be an object")
            records.append(record)
    return records


def normalize_text(value: str) -> str:
    """Normalize for fuzzy comparison without erasing meaningful words."""
    text = unicodedata.normalize("NFKD", value)
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = text.casefold()

    replacements = {
        "&": " and ",
        "×": " x ",
        "–": "-",
        "—": "-",
        "’": "'",
        "feat.": " feat ",
        "ft.": " feat ",
        "featuring": " feat ",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    text = re.sub(r"[_/|]+", " ", text)
    text = re.sub(r"[^\w\s'-]", " ", text)
    text = re.sub(r"[-']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def version_markers(value: str) -> set[str]:
    return {
        name
        for name, pattern in VERSION_PATTERNS
        if pattern.search(value)
    }


def title_similarity(source: str, candidate: str) -> float:
    a = normalize_text(source)
    b = normalize_text(candidate)
    if not a or not b:
        return 0.0

    # Ratio keeps version suffixes meaningful; WRatio adds tolerance for minor
    # formatting differences. Do not use token_set_ratio here because it can
    # make "Song" and "Song (Remix)" look artificially identical.
    ratio = fuzz.ratio(a, b)
    weighted = fuzz.WRatio(a, b)
    return 0.70 * ratio + 0.30 * weighted


def artist_similarity(source: str, candidate_artists: list[str]) -> float:
    source_norm = normalize_text(source)
    if not source_norm or not candidate_artists:
        return 0.0

    joined = normalize_text(" ".join(candidate_artists))
    joined_score = fuzz.token_set_ratio(source_norm, joined)

    # Also consider the strongest individual credited artist, useful when the
    # source credit contains "feat." or several collaborators.
    individual = max(
        (fuzz.token_set_ratio(source_norm, normalize_text(name)) for name in candidate_artists),
        default=0.0,
    )

    return 0.80 * joined_score + 0.20 * individual


def version_adjustment(source_title: str, candidate_title: str) -> tuple[float, dict[str, Any]]:
    source = version_markers(source_title)
    candidate = version_markers(candidate_title)

    if source == candidate:
        return 0.0, {
            "source_markers": sorted(source),
            "candidate_markers": sorted(candidate),
            "penalty": 0.0,
        }

    missing = source - candidate
    extra = candidate - source

    # Missing a version marker from the requested title is more serious than
    # the candidate having an extra remaster/edit marker.
    penalty = min(28.0, 16.0 * len(missing) + 10.0 * len(extra))

    return -penalty, {
        "source_markers": sorted(source),
        "candidate_markers": sorted(candidate),
        "missing_markers": sorted(missing),
        "extra_markers": sorted(extra),
        "penalty": penalty,
    }


def candidate_artist_names(result: dict[str, Any]) -> list[str]:
    artists = result.get("artists")
    if not isinstance(artists, list):
        return []

    names: list[str] = []
    for artist in artists:
        if isinstance(artist, dict):
            name = artist.get("name")
            if isinstance(name, str) and name.strip():
                names.append(name.strip())
    return names


def score_result(
    source_artist: str,
    source_title: str,
    result: dict[str, Any],
    rank: int,
) -> dict[str, Any] | None:
    video_id = result.get("videoId")
    title = result.get("title")
    result_type = result.get("resultType")
    video_type = result.get("videoType")

    if not isinstance(video_id, str) or not video_id:
        return None
    if not isinstance(title, str) or not title:
        return None
    if result_type not in (None, "song"):
        return None

    # "songs" search should normally return ATV/song results. If videoType is
    # explicitly present and identifies something else, do not use it.
    if isinstance(video_type, str) and video_type and video_type != "MUSIC_VIDEO_TYPE_ATV":
        return None

    artists = candidate_artist_names(result)

    t_score = title_similarity(source_title, title)
    a_score = artist_similarity(source_artist, artists)
    version_delta, version_info = version_adjustment(source_title, title)

    # Rank contributes at most 4 points, and decays quickly.
    rank_bonus = max(0.0, 4.0 - 0.5 * rank)

    raw = (
        0.66 * t_score
        + 0.30 * a_score
        + rank_bonus
        + version_delta
    )
    total = max(0.0, min(100.0, raw))

    album = result.get("album")
    album_name = None
    if isinstance(album, dict):
        raw_name = album.get("name")
        if isinstance(raw_name, str):
            album_name = raw_name

    return {
        "video_id": video_id,
        "title": title,
        "artists": artists,
        "album": album_name,
        "duration": result.get("duration"),
        "duration_seconds": result.get("duration_seconds"),
        "video_type": video_type,
        "rank": rank,
        "score": round(total, 2),
        "title_score": round(t_score, 2),
        "artist_score": round(a_score, 2),
        "rank_bonus": round(rank_bonus, 2),
        "version": version_info,
    }


class SearchCache:
    """Append-only JSONL cache keyed by normalized search query."""

    def __init__(self, path: Path):
        self.path = path
        self.records: dict[str, list[dict[str, Any]]] = {}

        if path.exists():
            with path.open("r", encoding="utf-8") as handle:
                for raw in handle:
                    line = raw.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    key = item.get("key")
                    results = item.get("results")
                    if isinstance(key, str) and isinstance(results, list):
                        self.records[key] = results

    @staticmethod
    def key(query: str, limit: int) -> str:
        payload = f"songs|{limit}|{normalize_text(query)}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(self, query: str, limit: int) -> list[dict[str, Any]] | None:
        return self.records.get(self.key(query, limit))

    def put(self, query: str, limit: int, results: list[dict[str, Any]]) -> None:
        key = self.key(query, limit)
        self.records[key] = results
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "key": key,
                        "query": query,
                        "limit": limit,
                        "results": results,
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )


def search_with_retry(
    yt: YTMusic,
    query: str,
    *,
    limit: int,
    attempts: int,
    delay: float,
) -> list[dict[str, Any]]:
    last_error: Exception | None = None

    for attempt in range(1, attempts + 1):
        try:
            results = yt.search(query, filter="songs", limit=limit)
            return [item for item in results if isinstance(item, dict)]
        except Exception as exc:  # ytmusicapi can raise several request/server types
            last_error = exc
            if attempt == attempts:
                break
            wait = delay * (2 ** (attempt - 1))
            print(
                f"warning: search failed for {query!r} "
                f"(attempt {attempt}/{attempts}); retrying in {wait:.1f}s: {exc}"
            )
            time.sleep(wait)

    assert last_error is not None
    raise last_error


def load_previous(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            cid = item.get("candidate_id")
            if isinstance(cid, str):
                ids.add(cid)
    return ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resolve training tracks to public YouTube Music song IDs."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--search-limit",
        type=int,
        default=10,
        help="YouTube Music song results requested per track (default: 10).",
    )
    parser.add_argument(
        "--accept-score",
        type=float,
        default=90.0,
        help="Minimum overall score for automatic resolution (default: 90).",
    )
    parser.add_argument(
        "--accept-title",
        type=float,
        default=88.0,
        help="Minimum title score for automatic resolution (default: 88).",
    )
    parser.add_argument(
        "--accept-artist",
        type=float,
        default=72.0,
        help="Minimum artist score for automatic resolution (default: 72).",
    )
    parser.add_argument(
        "--accept-gap",
        type=float,
        default=3.0,
        help="Minimum score gap over second-best candidate (default: 3).",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.15,
        help="Delay after uncached searches in seconds (default: 0.15).",
    )
    parser.add_argument(
        "--attempts",
        type=int,
        default=3,
        help="Search attempts after transient failures (default: 3).",
    )
    parser.add_argument(
        "--max-tracks",
        type=int,
        default=None,
        help="Process only the first N unresolved tracks, useful for smoke tests.",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore existing output files instead of resuming them.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.search_limit < 1:
        raise SystemExit("--search-limit must be >= 1")
    if args.attempts < 1:
        raise SystemExit("--attempts must be >= 1")

    records = load_jsonl(args.input)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    resolved_path = args.output_dir / "audio_manifest.jsonl"
    review_path = args.output_dir / "review.jsonl"
    report_path = args.output_dir / "resolver_report.json"
    cache_path = args.output_dir / "search_cache.jsonl"

    if args.fresh:
        for path in (resolved_path, review_path, report_path):
            if path.exists():
                path.unlink()

    completed = set()
    if not args.fresh:
        completed |= load_previous(resolved_path)
        completed |= load_previous(review_path)

    pending = [
        record
        for record in records
        if isinstance(record.get("candidate_id"), str)
        and record["candidate_id"] not in completed
    ]

    if args.max_tracks is not None:
        pending = pending[: args.max_tracks]

    yt = YTMusic()
    cache = SearchCache(cache_path)

    counters: Counter[str] = Counter()
    score_samples: list[float] = []

    resolved_handle = resolved_path.open("a", encoding="utf-8")
    review_handle = review_path.open("a", encoding="utf-8")

    try:
        for index, record in enumerate(pending, start=1):
            candidate_id = record.get("candidate_id")
            artist = str(record.get("artist") or "").strip()
            title = str(record.get("title") or "").strip()

            print(f"[{index}/{len(pending)}] {artist} - {title}")

            if not candidate_id or not artist or not title:
                counters["invalid_input"] += 1
                review_handle.write(
                    json.dumps(
                        {
                            "candidate_id": candidate_id,
                            "artist": artist,
                            "title": title,
                            "status": "invalid_input",
                            "reason": "missing candidate_id, artist, or title",
                            "source": record,
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                review_handle.flush()
                continue

            query = f"{artist} {title}"
            cached = cache.get(query, args.search_limit)

            try:
                if cached is None:
                    raw_results = search_with_retry(
                        yt,
                        query,
                        limit=args.search_limit,
                        attempts=args.attempts,
                        delay=1.5,
                    )
                    cache.put(query, args.search_limit, raw_results)
                    counters["network_searches"] += 1
                    if args.sleep:
                        time.sleep(args.sleep)
                else:
                    raw_results = cached
                    counters["cache_hits"] += 1
            except Exception as exc:
                counters["search_errors"] += 1
                review_handle.write(
                    json.dumps(
                        {
                            "candidate_id": candidate_id,
                            "artist": artist,
                            "title": title,
                            "status": "search_error",
                            "query": query,
                            "error": str(exc),
                            "source": record,
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                review_handle.flush()
                continue

            scored: list[dict[str, Any]] = []
            for rank, result in enumerate(raw_results):
                candidate = score_result(artist, title, result, rank)
                if candidate is not None:
                    scored.append(candidate)

            scored.sort(key=lambda item: item["score"], reverse=True)

            if not scored:
                counters["no_candidates"] += 1
                review_handle.write(
                    json.dumps(
                        {
                            "candidate_id": candidate_id,
                            "artist": artist,
                            "title": title,
                            "status": "no_match",
                            "query": query,
                            "candidates": [],
                            "source": record,
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                review_handle.flush()
                continue

            best = scored[0]
            second_score = scored[1]["score"] if len(scored) > 1 else None
            gap = (
                round(best["score"] - second_score, 2)
                if second_score is not None
                else None
            )

            strong = (
                best["score"] >= args.accept_score
                and best["title_score"] >= args.accept_title
                and best["artist_score"] >= args.accept_artist
            )

            # If there is only one usable result, require a little extra total
            # confidence instead of relying on a score-gap test.
            separated = (
                gap is not None and gap >= args.accept_gap
            ) or (
                gap is None and best["score"] >= max(args.accept_score + 3.0, 93.0)
            )

            if strong and separated:
                counters["resolved"] += 1
                score_samples.append(float(best["score"]))

                output = dict(record)
                output["audio_source"] = {
                    "provider": "youtube_music",
                    "status": "resolved",
                    "query": query,
                    "video_id": best["video_id"],
                    "title": best["title"],
                    "artists": best["artists"],
                    "album": best["album"],
                    "duration": best["duration"],
                    "duration_seconds": best["duration_seconds"],
                    "video_type": best["video_type"],
                    "score": best["score"],
                    "title_score": best["title_score"],
                    "artist_score": best["artist_score"],
                    "score_gap": gap,
                    "version": best["version"],
                }

                resolved_handle.write(
                    json.dumps(output, ensure_ascii=False, separators=(",", ":"))
                    + "\n"
                )
                resolved_handle.flush()

                print(
                    f"  resolved -> {best['title']} / "
                    f"{', '.join(best['artists'])} "
                    f"[{best['video_id']}] score={best['score']:.2f}"
                )
            else:
                counters["review"] += 1
                review_handle.write(
                    json.dumps(
                        {
                            "candidate_id": candidate_id,
                            "artist": artist,
                            "title": title,
                            "labels": record.get("labels", []),
                            "status": "review",
                            "query": query,
                            "best_score": best["score"],
                            "score_gap": gap,
                            "candidates": scored[:3],
                            "source": record,
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                review_handle.flush()

                print(
                    f"  review -> best={best['score']:.2f} "
                    f"title={best['title_score']:.2f} "
                    f"artist={best['artist_score']:.2f} gap={gap}"
                )
    finally:
        resolved_handle.close()
        review_handle.close()

    total_completed = len(load_previous(resolved_path)) + len(load_previous(review_path))
    mean_score = (
        round(sum(score_samples) / len(score_samples), 3)
        if score_samples
        else None
    )

    report = {
        "input": str(args.input),
        "training_tracks": len(records),
        "completed_tracks": total_completed,
        "this_run": {
            "requested": len(pending),
            "resolved": counters["resolved"],
            "review": counters["review"],
            "no_candidates": counters["no_candidates"],
            "search_errors": counters["search_errors"],
            "invalid_input": counters["invalid_input"],
            "network_searches": counters["network_searches"],
            "cache_hits": counters["cache_hits"],
            "mean_resolved_score": mean_score,
        },
        "thresholds": {
            "accept_score": args.accept_score,
            "accept_title": args.accept_title,
            "accept_artist": args.accept_artist,
            "accept_gap": args.accept_gap,
        },
        "outputs": {
            "resolved": str(resolved_path),
            "review": str(review_path),
            "cache": str(cache_path),
        },
    }

    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print()
    print("Resolution run complete")
    print(f"  resolved:       {counters['resolved']}")
    print(f"  review:         {counters['review']}")
    print(f"  no candidates:  {counters['no_candidates']}")
    print(f"  search errors:  {counters['search_errors']}")
    print(f"  cache hits:     {counters['cache_hits']}")
    print(f"  network calls:  {counters['network_searches']}")
    print()
    print(f"Resolved manifest: {resolved_path}")
    print(f"Review queue:      {review_path}")
    print(f"Report:            {report_path}")


if __name__ == "__main__":
    main()
