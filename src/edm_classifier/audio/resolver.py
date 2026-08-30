"""V2 YouTube Music resolver: song-first, video fallback, resumable retries.

This replaces the earlier song-only resolver.

Inputs:
    data/v2/validation/accepted_candidates.jsonl
    data/v2/validation/class_support.csv
    data/v2/validation/provisional_active_classes.json
    config/taxonomy.yaml

Outputs:
    data/v2/audio_resolution/
      resolution_manifest.jsonl
      resolved_candidates.jsonl
      unresolved_candidates.jsonl
      resolution_report.json
      cache/ytm_search_cache.jsonl

Resolution policy:
1. Search YouTube Music songs first.
2. If no acceptable song match exists, search YouTube Music videos.
3. Normalize harmless metadata differences:
   - featured-artist suffixes
   - "Official Video", "Official Audio", "Visualizer", lyric-video decorations
   - collaborator formatting
   - "Original Mix" versus an unqualified catalog title
4. Preserve remix/version identity. A requested remix is not replaced by the
   original or by a different remix.
5. Reject obvious cover/karaoke/reaction/DJ-set/live mismatches.
6. Existing resolved manifest rows remain final by default.
7. Existing unresolved/error rows are automatically retried.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import json
import re
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml
from ytmusicapi import YTMusic


RESOLVER_VERSION = "v3_song_video_fallback"

DEFAULT_ACCEPTED = Path("data/v2/validation/accepted_candidates.jsonl")
DEFAULT_CLASS_SUPPORT = Path("data/v2/validation/class_support.csv")
DEFAULT_ACTIVE_CLASSES = Path(
    "data/v2/validation/provisional_active_classes.json"
)
DEFAULT_TAXONOMY = Path("config/taxonomy.yaml")
DEFAULT_OUTPUT_DIR = Path("data/v2/audio_resolution")


FEATURE_RE = re.compile(
    r"""
    (?:
        \s*[\(\[\{]\s*
        (?:feat(?:uring)?|ft)\.?\s+[^)\]\}]+
        [\)\]\}]
      |
        \s+(?:feat(?:uring)?|ft)\.?\s+.+$
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

VIDEO_DECORATION_RE = re.compile(
    r"""
    [\(\[\{]\s*
    (?:
        official\s+(?:music\s+)?video
      | official\s+audio
      | official\s+visualizer
      | music\s+video
      | lyric(?:s)?\s+video
      | lyric(?:s)?
      | visualizer
      | audio
      | hd
      | 4k
    )
    \s*[\)\]\}]
    """,
    re.IGNORECASE | re.VERBOSE,
)

TRAILING_VIDEO_DECORATION_RE = re.compile(
    r"""
    \s*[-|:]\s*
    (?:
        official\s+(?:music\s+)?video
      | official\s+audio
      | official\s+visualizer
      | music\s+video
      | lyric(?:s)?\s+video
      | visualizer
      | audio
    )\s*$
    """,
    re.IGNORECASE | re.VERBOSE,
)

COLLAB_SPLIT_RE = re.compile(
    r"""
    \s+
    (?:
        feat(?:uring)?\.?
      | ft\.?
      | vs\.?
      | versus
      | x
    )
    \s+
    |
    \s*&\s*
    |
    \s*,\s*
    """,
    re.IGNORECASE | re.VERBOSE,
)

BAD_VIDEO_TERMS = (
    "karaoke",
    "reaction",
    "tutorial",
    "drum cover",
    "piano cover",
    "guitar cover",
    "vocal cover",
    "dance cover",
    "fan made",
    "fanmade",
    "nightcore",
    "sped up",
    "slowed reverb",
    "slowed + reverb",
    "slowed and reverb",
    "8d audio",
    "1 hour",
    "10 hours",
    "full album",
    "dj set",
    "live set",
    "festival set",
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    rows: list[dict[str, Any]] = []

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


def normalize(value: str) -> str:
    text = unicodedata.normalize("NFKD", value)
    text = "".join(
        char
        for char in text
        if not unicodedata.combining(char)
    )
    text = text.casefold()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def sequence_similarity(left: str, right: str) -> float:
    a = normalize(left)
    b = normalize(right)

    if not a or not b:
        return 0.0
    if a == b:
        return 1.0

    return difflib.SequenceMatcher(None, a, b).ratio()


def token_similarity(left: str, right: str) -> float:
    a = set(normalize(left).split())
    b = set(normalize(right).split())

    if not a or not b:
        return 0.0
    if a == b:
        return 1.0

    intersection = len(a & b)
    union = len(a | b)

    return intersection / union if union else 0.0


def flexible_similarity(left: str, right: str) -> float:
    a = normalize(left)
    b = normalize(right)

    if not a or not b:
        return 0.0
    if a == b:
        return 1.0

    # Exact containment is common with channel suffixes and metadata credits.
    containment = 0.0
    if len(a) >= 4 and len(b) >= 4 and (a in b or b in a):
        shorter = min(len(a), len(b))
        longer = max(len(a), len(b))
        containment = 0.90 + 0.10 * (shorter / longer)

    return max(
        sequence_similarity(a, b),
        token_similarity(a, b),
        containment,
    )


def strip_feature_credit(title: str) -> str:
    return FEATURE_RE.sub("", title).strip()


def strip_video_decorations(title: str) -> str:
    text = VIDEO_DECORATION_RE.sub("", title)
    text = TRAILING_VIDEO_DECORATION_RE.sub("", text)
    return re.sub(r"\s+", " ", text).strip(" -|:[](){}")


def version_descriptor(title: str) -> dict[str, Any]:
    text = normalize(strip_video_decorations(strip_feature_credit(title)))

    remix = "remix" in text
    original_mix = "original mix" in text
    radio = "radio edit" in text or "radio mix" in text
    extended = "extended mix" in text or "extended version" in text
    live = "live" in text
    acoustic = "acoustic" in text
    instrumental = "instrumental" in text
    vip = "vip" in text
    bootleg = "bootleg" in text
    mashup = "mashup" in text

    remix_descriptor = None

    match = re.search(
        r"(?:\(|\[|-)\s*([^()\[\]-]*?)\s+remix(?:\)|\]|$)",
        strip_video_decorations(strip_feature_credit(title)),
        flags=re.IGNORECASE,
    )
    if match:
        remix_descriptor = normalize(match.group(1))

    return {
        "remix": remix,
        "remix_descriptor": remix_descriptor,
        "original_mix": original_mix,
        "radio": radio,
        "extended": extended,
        "live": live,
        "acoustic": acoustic,
        "instrumental": instrumental,
        "vip": vip,
        "bootleg": bootleg,
        "mashup": mashup,
    }


def base_title(title: str) -> str:
    text = strip_video_decorations(strip_feature_credit(title))

    # Remove version suffixes for BASE-title comparison. Version compatibility
    # is checked separately below.
    patterns = (
        r"\s*[\(\[][^)\]]*remix[\)\]]\s*$",
        r"\s*-\s*[^-]*remix\s*$",
        r"\s*[\(\[]\s*original\s+mix\s*[\)\]]\s*$",
        r"\s*-\s*original\s+mix\s*$",
        r"\s*[\(\[]\s*radio\s+(?:edit|mix)\s*[\)\]]\s*$",
        r"\s*-\s*radio\s+(?:edit|mix)\s*$",
        r"\s*[\(\[]\s*extended\s+(?:mix|version)\s*[\)\]]\s*$",
        r"\s*-\s*extended\s+(?:mix|version)\s*$",
        r"\s*[\(\[]\s*(?:vip|instrumental|acoustic|live)\s*[\)\]]\s*$",
    )

    for pattern in patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    return normalize(text)


def title_similarity(source_title: str, target_title: str) -> float:
    source = base_title(source_title)
    target = base_title(target_title)

    return flexible_similarity(source, target)


def version_compatibility(
    source_title: str,
    target_title: str,
) -> tuple[bool, float, str | None]:
    source = version_descriptor(source_title)
    target = version_descriptor(target_title)

    # Remix identity is musically important for an EDM classifier.
    if source["remix"] != target["remix"]:
        return False, 0.40, "remix_original_mismatch"

    if source["remix"] and target["remix"]:
        left = source.get("remix_descriptor")
        right = target.get("remix_descriptor")

        if left and right and flexible_similarity(left, right) < 0.55:
            return False, 0.40, "different_remix"

    for key in ("live", "acoustic", "instrumental", "vip", "bootleg", "mashup"):
        if source[key] != target[key]:
            return False, 0.35, f"{key}_mismatch"

    # Explicit radio-vs-extended is a meaningful conflict.
    if source["radio"] and target["extended"]:
        return False, 0.30, "radio_extended_mismatch"
    if source["extended"] and target["radio"]:
        return False, 0.30, "extended_radio_mismatch"

    # YTM often omits "Original Mix", "Radio Edit", or "Extended Mix" from
    # catalog/video metadata. Missing qualification receives only a penalty.
    penalty = 0.0
    reason = None

    if source["radio"] != target["radio"]:
        penalty += 0.06
        reason = "radio_qualification_difference"

    if source["extended"] != target["extended"]:
        penalty += 0.08
        reason = "extended_qualification_difference"

    # Original Mix versus bare title is considered equivalent.
    return True, penalty, reason


def artist_components(value: str) -> list[str]:
    parts = [
        normalize(part)
        for part in COLLAB_SPLIT_RE.split(value)
    ]

    return [
        part
        for part in parts
        if part
    ]


def result_artists(result: dict[str, Any]) -> list[str]:
    raw = result.get("artists")

    if not isinstance(raw, list):
        return []

    names = []

    for item in raw:
        if isinstance(item, dict) and isinstance(item.get("name"), str):
            names.append(item["name"])

    return names


def artist_similarity(
    source_artist: str,
    result: dict[str, Any],
) -> float:
    source_parts = artist_components(source_artist)

    targets = result_artists(result)
    target_parts = []

    for artist in targets:
        target_parts.extend(artist_components(artist))

    # Video titles frequently include "Artist - Track" even when YTM artist
    # metadata is absent or uses an uploader/channel name.
    result_title = result.get("title")
    if isinstance(result_title, str):
        target_parts.append(normalize(result_title))

    if not source_parts or not target_parts:
        return 0.0

    per_source = []

    for source in source_parts:
        best = max(
            flexible_similarity(source, target)
            for target in target_parts
        )
        per_source.append(best)

    # The primary/any credited artist matching strongly is enough. Missing a
    # featured collaborator should not reject an otherwise exact recording.
    return max(per_source, default=0.0)


def is_bad_video_result(
    source_title: str,
    result: dict[str, Any],
) -> tuple[bool, str | None]:
    title = result.get("title")

    if not isinstance(title, str):
        return True, "missing_video_title"

    normalized_target = normalize(title)
    normalized_source = normalize(source_title)

    for term in BAD_VIDEO_TERMS:
        normalized_term = normalize(term)

        if normalized_term in normalized_target and normalized_term not in normalized_source:
            return True, f"video_{normalized_term.replace(' ', '_')}"

    # Reject unrequested live versions even when they do not contain one of
    # the longer live-set phrases above.
    if "live" in normalized_target.split() and "live" not in normalized_source.split():
        return True, "video_live_mismatch"

    # Generic covers are risky, except when the source itself says cover.
    if "cover" in normalized_target.split() and "cover" not in normalized_source.split():
        return True, "video_cover"

    return False, None


def result_album(result: dict[str, Any]) -> dict[str, Any] | None:
    album = result.get("album")

    if not isinstance(album, dict):
        return None

    return {
        "name": album.get("name"),
        "id": album.get("id"),
    }


def duration_seconds(result: dict[str, Any]) -> int | None:
    value = result.get("duration_seconds")

    if isinstance(value, int):
        return value

    if isinstance(value, str) and value.isdigit():
        return int(value)

    duration = result.get("duration")

    if not isinstance(duration, str):
        return None

    pieces = duration.split(":")

    try:
        nums = [int(piece) for piece in pieces]
    except ValueError:
        return None

    if len(nums) == 2:
        return nums[0] * 60 + nums[1]

    if len(nums) == 3:
        return nums[0] * 3600 + nums[1] * 60 + nums[2]

    return None


def score_result(
    source_artist: str,
    source_title: str,
    result: dict[str, Any],
    *,
    source_type: str,
) -> dict[str, Any]:
    target_title = result.get("title")

    if not isinstance(target_title, str):
        target_title = ""

    t_score = title_similarity(source_title, target_title)
    a_score = artist_similarity(source_artist, result)

    compatible, version_penalty, version_reason = version_compatibility(
        source_title,
        target_title,
    )

    bad_video = False
    bad_video_reason = None

    if source_type == "video":
        bad_video, bad_video_reason = is_bad_video_result(
            source_title,
            result,
        )

    # Title is the strongest signal. Artist metadata is allowed to be weaker
    # for videos because uploader/channel metadata is inconsistent.
    artist_weight = 0.30 if source_type == "song" else 0.24
    title_weight = 1.0 - artist_weight

    combined = (
        title_weight * t_score
        + artist_weight * a_score
        - version_penalty
    )

    return {
        "score": max(0.0, min(1.0, combined)),
        "title_score": t_score,
        "artist_score": a_score,
        "version_compatible": compatible,
        "version_reason": version_reason,
        "bad_video": bad_video,
        "bad_video_reason": bad_video_reason,
    }


class SearchCache:
    def __init__(self, path: Path):
        self.path = path
        self.cache: dict[str, list[dict[str, Any]]] = {}
        self.hits = 0
        self.misses = 0
        self.hits_by_filter: Counter[str] = Counter()
        self.misses_by_filter: Counter[str] = Counter()

        path.parent.mkdir(parents=True, exist_ok=True)

        for row in load_jsonl(path):
            key = row.get("key")
            results = row.get("results")

            if isinstance(key, str) and isinstance(results, list):
                self.cache[key] = results

    @staticmethod
    def make_key(
        query: str,
        limit: int,
        search_filter: str,
    ) -> str:
        return json.dumps(
            {
                "query": query,
                "filter": search_filter,
                "limit": limit,
            },
            sort_keys=True,
            separators=(",", ":"),
        )

    def get(
        self,
        ytmusic: YTMusic,
        *,
        query: str,
        limit: int,
        search_filter: str,
    ) -> list[dict[str, Any]]:
        key = self.make_key(
            query,
            limit,
            search_filter,
        )

        if key in self.cache:
            self.hits += 1
            self.hits_by_filter[search_filter] += 1
            return self.cache[key]

        self.misses += 1
        self.misses_by_filter[search_filter] += 1

        results = ytmusic.search(
            query,
            filter=search_filter,
            limit=limit,
        )

        if not isinstance(results, list):
            results = []

        cleaned = [
            item
            for item in results
            if isinstance(item, dict)
        ]

        self.cache[key] = cleaned

        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "key": key,
                        "filter": search_filter,
                        "results": cleaned,
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )

        return cleaned


def query_variants(artist: str, title: str) -> list[str]:
    # Avoid excessive searches: two forms catch nearly all useful YTM cases.
    candidates = [
        f"{artist} {title}",
        f"{title} {artist}",
    ]

    result = []
    seen = set()

    for query in candidates:
        key = normalize(query)

        if key and key not in seen:
            seen.add(key)
            result.append(query)

    return result


def rank_candidates(
    source_artist: str,
    source_title: str,
    search_results: list[tuple[str, dict[str, Any]]],
    *,
    source_type: str,
) -> list[dict[str, Any]]:
    by_video: dict[str, dict[str, Any]] = {}

    for query, result in search_results:
        video_id = result.get("videoId")

        if not isinstance(video_id, str) or not video_id:
            continue

        scored = score_result(
            source_artist,
            source_title,
            result,
            source_type=source_type,
        )

        if not scored["version_compatible"]:
            continue

        if source_type == "video" and scored["bad_video"]:
            continue

        candidate = {
            "video_id": video_id,
            "query": query,
            "result": result,
            "source_type": source_type,
            **scored,
        }

        existing = by_video.get(video_id)

        if existing is None or candidate["score"] > existing["score"]:
            by_video[video_id] = candidate

    return sorted(
        by_video.values(),
        key=lambda item: (
            -float(item["score"]),
            -float(item["title_score"]),
            -float(item["artist_score"]),
            item["video_id"],
        ),
    )


def evaluate_ranked(
    ranked: list[dict[str, Any]],
    *,
    source_type: str,
    min_score: float,
    min_title_score: float,
    min_artist_score: float,
    min_margin: float,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    if not ranked:
        return None, {
            "reason": f"no_{source_type}_candidates",
        }

    best = ranked[0]
    second_score = (
        float(ranked[1]["score"])
        if len(ranked) > 1
        else 0.0
    )
    margin = float(best["score"]) - second_score

    diagnostic = {
        "reason": f"{source_type}_below_threshold",
        "best_score": round(float(best["score"]), 6),
        "best_title_score": round(float(best["title_score"]), 6),
        "best_artist_score": round(float(best["artist_score"]), 6),
        "second_score": round(second_score, 6),
        "margin": round(margin, 6),
        "best_video_id": best["video_id"],
        "best_title": best["result"].get("title"),
        "best_artists": result_artists(best["result"]),
        "best_video_type": best["result"].get("videoType"),
    }

    if float(best["title_score"]) < min_title_score:
        diagnostic["reason"] = f"{source_type}_title_below_threshold"
        return None, diagnostic

    if float(best["artist_score"]) < min_artist_score:
        diagnostic["reason"] = f"{source_type}_artist_below_threshold"
        return None, diagnostic

    if float(best["score"]) < min_score:
        diagnostic["reason"] = f"{source_type}_combined_below_threshold"
        return None, diagnostic

    # If the best candidate is very strong, ambiguity among duplicate uploads
    # is harmless. Otherwise insist on a small score margin.
    if float(best["score"]) < 0.94 and margin < min_margin:
        diagnostic["reason"] = f"{source_type}_ambiguous"
        return None, diagnostic

    return best, diagnostic


def resolved_payload(
    best: dict[str, Any],
    *,
    attempted_queries: list[dict[str, str]],
) -> dict[str, Any]:
    result = best["result"]
    source_type = best["source_type"]

    return {
        "status": "resolved",
        "reason": (
            "matched_song"
            if source_type == "song"
            else "matched_video_fallback"
        ),
        "resolver_version": RESOLVER_VERSION,
        "resolution_source": source_type,
        "video_id": best["video_id"],
        "score": round(float(best["score"]), 6),
        "title_score": round(float(best["title_score"]), 6),
        "artist_score": round(float(best["artist_score"]), 6),
        "matched_query": best["query"],
        "result_type": result.get("resultType"),
        "video_type": result.get("videoType"),
        "ytm_title": result.get("title"),
        "ytm_artists": result_artists(result),
        "ytm_artist_ids": [
            item.get("id")
            for item in result.get("artists", [])
            if isinstance(item, dict)
            and isinstance(item.get("id"), str)
        ] if isinstance(result.get("artists"), list) else [],
        "album": result_album(result),
        "duration_seconds": duration_seconds(result),
        "is_explicit": result.get("isExplicit"),
        "attempted_queries": attempted_queries,
    }


def resolve_one(
    ytmusic: YTMusic,
    cache: SearchCache,
    row: dict[str, Any],
    *,
    song_limit: int,
    video_limit: int,
    song_min_score: float,
    song_min_title: float,
    song_min_artist: float,
    song_min_margin: float,
    video_min_score: float,
    video_min_title: float,
    video_min_artist: float,
    video_min_margin: float,
) -> dict[str, Any]:
    source_artist = row.get("artist")
    source_title = row.get("title")

    if not isinstance(source_artist, str) or not isinstance(source_title, str):
        return {
            "status": "unresolved",
            "reason": "missing_artist_or_title",
            "resolver_version": RESOLVER_VERSION,
            "attempted_queries": [],
        }

    queries = query_variants(
        source_artist,
        source_title,
    )

    attempted: list[dict[str, str]] = []

    # ----- Song pass ------------------------------------------------------
    song_results: list[tuple[str, dict[str, Any]]] = []

    for query in queries:
        attempted.append({
            "filter": "songs",
            "query": query,
        })

        try:
            results = cache.get(
                ytmusic,
                query=query,
                limit=song_limit,
                search_filter="songs",
            )
        except Exception as exc:
            return {
                "status": "error",
                "reason": f"{type(exc).__name__}: {exc}",
                "resolver_version": RESOLVER_VERSION,
                "attempted_queries": attempted,
            }

        song_results.extend(
            (query, result)
            for result in results
        )

        ranked_now = rank_candidates(
            source_artist,
            source_title,
            song_results,
            source_type="song",
        )

        best_now, _ = evaluate_ranked(
            ranked_now,
            source_type="song",
            min_score=song_min_score,
            min_title_score=song_min_title,
            min_artist_score=song_min_artist,
            min_margin=song_min_margin,
        )

        if (
            best_now is not None
            and float(best_now["score"]) >= 0.97
            and float(best_now["title_score"]) >= 0.96
            and float(best_now["artist_score"]) >= 0.90
        ):
            return resolved_payload(
                best_now,
                attempted_queries=attempted,
            )

    ranked_songs = rank_candidates(
        source_artist,
        source_title,
        song_results,
        source_type="song",
    )

    best_song, song_diagnostic = evaluate_ranked(
        ranked_songs,
        source_type="song",
        min_score=song_min_score,
        min_title_score=song_min_title,
        min_artist_score=song_min_artist,
        min_margin=song_min_margin,
    )

    if best_song is not None:
        return resolved_payload(
            best_song,
            attempted_queries=attempted,
        )

    # ----- Video fallback -------------------------------------------------
    video_results: list[tuple[str, dict[str, Any]]] = []

    for query in queries:
        attempted.append({
            "filter": "videos",
            "query": query,
        })

        try:
            results = cache.get(
                ytmusic,
                query=query,
                limit=video_limit,
                search_filter="videos",
            )
        except Exception as exc:
            return {
                "status": "error",
                "reason": f"{type(exc).__name__}: {exc}",
                "resolver_version": RESOLVER_VERSION,
                "attempted_queries": attempted,
                "song_diagnostic": song_diagnostic,
            }

        video_results.extend(
            (query, result)
            for result in results
        )

        ranked_now = rank_candidates(
            source_artist,
            source_title,
            video_results,
            source_type="video",
        )

        best_now, _ = evaluate_ranked(
            ranked_now,
            source_type="video",
            min_score=video_min_score,
            min_title_score=video_min_title,
            min_artist_score=video_min_artist,
            min_margin=video_min_margin,
        )

        if (
            best_now is not None
            and float(best_now["score"]) >= 0.96
            and float(best_now["title_score"]) >= 0.95
            and float(best_now["artist_score"]) >= 0.85
        ):
            return resolved_payload(
                best_now,
                attempted_queries=attempted,
            )

    ranked_videos = rank_candidates(
        source_artist,
        source_title,
        video_results,
        source_type="video",
    )

    best_video, video_diagnostic = evaluate_ranked(
        ranked_videos,
        source_type="video",
        min_score=video_min_score,
        min_title_score=video_min_title,
        min_artist_score=video_min_artist,
        min_margin=video_min_margin,
    )

    if best_video is not None:
        return resolved_payload(
            best_video,
            attempted_queries=attempted,
        )

    return {
        "status": "unresolved",
        "reason": video_diagnostic.get(
            "reason",
            song_diagnostic.get("reason", "no_match"),
        ),
        "resolver_version": RESOLVER_VERSION,
        "song_diagnostic": song_diagnostic,
        "video_diagnostic": video_diagnostic,
        "attempted_queries": attempted,
    }


# ----- Priority machinery -------------------------------------------------

def load_taxonomy(
    path: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, set[str]]]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    genres = raw.get("genres") if isinstance(raw, dict) else None

    if not isinstance(genres, list):
        raise ValueError(f"{path}: expected genres list")

    taxonomy: dict[str, dict[str, Any]] = {}
    children: dict[str, set[str]] = defaultdict(set)

    for item in genres:
        if not isinstance(item, dict):
            continue

        genre_id = item.get("id")

        if isinstance(genre_id, str):
            taxonomy[genre_id] = dict(item)

    for genre_id, item in taxonomy.items():
        parent = item.get("parent")

        if isinstance(parent, str) and parent:
            children[parent].add(genre_id)

    return taxonomy, children


def ancestors(
    genre_id: str,
    taxonomy: dict[str, dict[str, Any]],
) -> list[str]:
    result = []
    current = genre_id
    seen = set()

    while current in taxonomy:
        if current in seen:
            break

        seen.add(current)

        parent = taxonomy[current].get("parent")

        if not isinstance(parent, str) or not parent:
            break

        result.append(parent)
        current = parent

    return result


def assigned_label_ids(row: dict[str, Any]) -> list[str]:
    raw = row.get("labels")

    if not isinstance(raw, list):
        return []

    result = []

    for item in raw:
        if isinstance(item, dict) and isinstance(item.get("id"), str):
            result.append(item["id"])
        elif isinstance(item, str):
            result.append(item)

    return result


def load_class_support(path: Path) -> dict[str, dict[str, Any]]:
    result = {}

    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            genre_id = row.get("id")

            if not genre_id:
                continue

            parsed = dict(row)

            for field in (
                "v1_direct_tracks",
                "v2_direct_tracks",
                "combined_direct_tracks",
                "v1_unique_artists",
                "v2_unique_artists",
                "combined_unique_artists",
            ):
                try:
                    parsed[field] = int(row.get(field, "0"))
                except (TypeError, ValueError):
                    parsed[field] = 0

            result[genre_id] = parsed

    return result


def load_active_ids(path: Path) -> set[str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    classes = raw.get("classes") if isinstance(raw, dict) else None

    if not isinstance(classes, list):
        raise ValueError(f"{path}: expected classes list")

    return {
        item["id"]
        for item in classes
        if isinstance(item, dict)
        and isinstance(item.get("id"), str)
    }


def candidate_priority(
    row: dict[str, Any],
    *,
    active_ids: set[str],
    class_support: dict[str, dict[str, Any]],
    taxonomy: dict[str, dict[str, Any]],
    children: dict[str, set[str]],
) -> tuple[int, int, int, str]:
    labels = assigned_label_ids(row)
    active_leaf_supports = []
    active_parent_supports = []
    has_dropped_leaf = False

    for genre_id in labels:
        if genre_id not in taxonomy:
            continue

        is_leaf = not children.get(genre_id)

        if genre_id in active_ids:
            support = int(
                class_support.get(genre_id, {}).get(
                    "combined_direct_tracks",
                    10**9,
                )
            )

            if is_leaf:
                active_leaf_supports.append(support)
            else:
                active_parent_supports.append(support)

        elif is_leaf:
            has_dropped_leaf = True

        for parent_id in ancestors(genre_id, taxonomy):
            if parent_id in active_ids:
                active_parent_supports.append(
                    int(
                        class_support.get(parent_id, {}).get(
                            "combined_direct_tracks",
                            10**9,
                        )
                    )
                )

    candidate_id = str(row.get("candidate_id", ""))

    if active_leaf_supports:
        return (
            0,
            min(active_leaf_supports),
            min(active_parent_supports, default=10**9),
            candidate_id,
        )

    if not has_dropped_leaf and active_parent_supports:
        return (
            1,
            min(active_parent_supports),
            10**9,
            candidate_id,
        )

    return (
        2,
        min(active_parent_supports, default=10**9),
        10**9,
        candidate_id,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resolve V2 tracks with YTM song-first/video-fallback search."
    )

    parser.add_argument("--accepted", type=Path, default=DEFAULT_ACCEPTED)
    parser.add_argument("--class-support", type=Path, default=DEFAULT_CLASS_SUPPORT)
    parser.add_argument("--active-classes", type=Path, default=DEFAULT_ACTIVE_CLASSES)
    parser.add_argument("--taxonomy", type=Path, default=DEFAULT_TAXONOMY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)

    parser.add_argument("--song-limit", type=int, default=20)
    parser.add_argument("--video-limit", type=int, default=20)

    parser.add_argument("--song-min-score", type=float, default=0.78)
    parser.add_argument("--song-min-title", type=float, default=0.78)
    parser.add_argument("--song-min-artist", type=float, default=0.55)
    parser.add_argument("--song-min-margin", type=float, default=0.025)

    parser.add_argument("--video-min-score", type=float, default=0.80)
    parser.add_argument("--video-min-title", type=float, default=0.83)
    parser.add_argument("--video-min-artist", type=float, default=0.55)
    parser.add_argument("--video-min-margin", type=float, default=0.03)

    parser.add_argument(
        "--sleep",
        type=float,
        default=0.15,
        help="Seconds between candidate-resolution attempts.",
    )

    parser.add_argument(
        "--max-records",
        type=int,
        help="Optional cap after retry/new priority sorting.",
    )

    parser.add_argument(
        "--retry-resolved",
        action="store_true",
        help="Re-resolve records already marked resolved. Default keeps them.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    accepted = load_jsonl(args.accepted)
    class_support = load_class_support(args.class_support)
    active_ids = load_active_ids(args.active_classes)
    taxonomy, children = load_taxonomy(args.taxonomy)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = args.output_dir / "resolution_manifest.jsonl"
    resolved_path = args.output_dir / "resolved_candidates.jsonl"
    unresolved_path = args.output_dir / "unresolved_candidates.jsonl"
    report_path = args.output_dir / "resolution_report.json"
    cache_path = args.output_dir / "cache" / "ytm_search_cache.jsonl"

    prior_rows = load_jsonl(manifest_path)

    # Last row wins if the manifest contains append-time retry records.
    prior_by_id = {
        row["candidate_id"]: row
        for row in prior_rows
        if isinstance(row.get("candidate_id"), str)
    }

    accepted_by_id = {
        row["candidate_id"]: row
        for row in accepted
        if isinstance(row.get("candidate_id"), str)
    }

    previous_resolved = {
        candidate_id
        for candidate_id, row in prior_by_id.items()
        if row.get("status") == "resolved"
    }

    previous_unresolved = {
        candidate_id
        for candidate_id, row in prior_by_id.items()
        if row.get("status") != "resolved"
    }

    queue = []

    for row in accepted:
        candidate_id = row.get("candidate_id")

        if not isinstance(candidate_id, str):
            continue

        prior = prior_by_id.get(candidate_id)

        if (
            prior is not None
            and prior.get("status") == "resolved"
            and not args.retry_resolved
        ):
            continue

        # New rows and ALL prior unresolved/error rows enter the retry queue.
        queue.append(row)

    queue.sort(
        key=lambda row: (
            # Retry previous failures before processing brand-new candidates.
            0 if row["candidate_id"] in previous_unresolved else 1,
            *candidate_priority(
                row,
                active_ids=active_ids,
                class_support=class_support,
                taxonomy=taxonomy,
                children=children,
            ),
        )
    )

    if args.max_records is not None:
        queue = queue[: args.max_records]

    ytmusic = YTMusic()
    cache = SearchCache(cache_path)

    started = time.monotonic()
    new_records: list[dict[str, Any]] = []

    print("V2 YouTube Music resolver")
    print(f"  resolver version:          {RESOLVER_VERSION}")
    print(f"  accepted candidates:       {len(accepted)}")
    print(f"  previous resolved kept:    {len(previous_resolved) if not args.retry_resolved else 0}")
    print(f"  previous failures retried: {len(previous_unresolved)}")
    print(f"  queued this run:           {len(queue)}")
    print()

    for index, row in enumerate(queue, 1):
        candidate_id = row["candidate_id"]

        if index == 1 or index % 100 == 0 or index == len(queue):
            print(
                f"  resolving {index}/{len(queue)} "
                f"(cache hits={cache.hits}, misses={cache.misses})"
            )

        priority = candidate_priority(
            row,
            active_ids=active_ids,
            class_support=class_support,
            taxonomy=taxonomy,
            children=children,
        )

        result = resolve_one(
            ytmusic,
            cache,
            row,
            song_limit=args.song_limit,
            video_limit=args.video_limit,
            song_min_score=args.song_min_score,
            song_min_title=args.song_min_title,
            song_min_artist=args.song_min_artist,
            song_min_margin=args.song_min_margin,
            video_min_score=args.video_min_score,
            video_min_title=args.video_min_title,
            video_min_artist=args.video_min_artist,
            video_min_margin=args.video_min_margin,
        )

        record = {
            "candidate_id": candidate_id,
            "artist": row.get("artist"),
            "title": row.get("title"),
            "labels": row.get("labels", []),
            "priority_tier": priority[0],
            "priority_support": priority[1],
            **result,
        }

        new_records.append(record)

        # Append immediately for crash-safe progress. At a clean exit the file
        # is compacted back to one deterministic row per candidate.
        with manifest_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )

        if args.sleep > 0:
            time.sleep(args.sleep)

    combined_by_id = dict(prior_by_id)

    for record in new_records:
        combined_by_id[record["candidate_id"]] = record

    combined = [
        combined_by_id[row["candidate_id"]]
        for row in accepted
        if isinstance(row.get("candidate_id"), str)
        and row["candidate_id"] in combined_by_id
    ]

    write_jsonl(manifest_path, combined)

    resolved = [
        row
        for row in combined
        if row.get("status") == "resolved"
    ]

    unresolved = [
        row
        for row in combined
        if row.get("status") != "resolved"
    ]

    write_jsonl(resolved_path, resolved)
    write_jsonl(unresolved_path, unresolved)

    unique_video_ids = {
        row["video_id"]
        for row in resolved
        if isinstance(row.get("video_id"), str)
    }

    video_counts = Counter(
        row["video_id"]
        for row in resolved
        if isinstance(row.get("video_id"), str)
    )

    status_counts = Counter(
        row.get("status", "unknown")
        for row in combined
    )

    unresolved_reasons = Counter(
        row.get("reason", "unknown")
        for row in unresolved
    )

    source_counts = Counter(
        row.get("resolution_source", "legacy_song")
        for row in resolved
    )

    priority_counts = Counter(
        int(row.get("priority_tier", -1))
        for row in combined
    )

    elapsed = time.monotonic() - started

    report = {
        "resolver_version": RESOLVER_VERSION,
        "input_candidates": len(accepted),
        "processed_manifest_records": len(combined),
        "status_counts": dict(status_counts),
        "resolved_candidate_records": len(resolved),
        "unique_resolved_video_ids": len(unique_video_ids),
        "duplicate_video_id_groups": sum(
            1
            for count in video_counts.values()
            if count > 1
        ),
        "resolution_rate": (
            len(resolved) / len(combined)
            if combined
            else 0.0
        ),
        "resolution_source_counts": dict(source_counts),
        "unresolved_reasons": dict(unresolved_reasons),
        "resume": {
            "previous_resolved_kept": (
                len(previous_resolved)
                if not args.retry_resolved
                else 0
            ),
            "previous_unresolved_retried": len(previous_unresolved),
            "records_attempted_this_run": len(new_records),
        },
        "priority_tiers": {
            "0_active_leaf": priority_counts.get(0, 0),
            "1_parent_only": priority_counts.get(1, 0),
            "2_dropped_leaf_parent_use_only": priority_counts.get(2, 0),
        },
        "matching_thresholds": {
            "song": {
                "score": args.song_min_score,
                "title": args.song_min_title,
                "artist": args.song_min_artist,
                "margin": args.song_min_margin,
            },
            "video": {
                "score": args.video_min_score,
                "title": args.video_min_title,
                "artist": args.video_min_artist,
                "margin": args.video_min_margin,
            },
        },
        "search": {
            "filters": ["songs", "videos"],
            "song_limit": args.song_limit,
            "video_limit": args.video_limit,
            "cache_hits_this_run": cache.hits,
            "cache_misses_this_run": cache.misses,
            "cache_hits_by_filter": dict(cache.hits_by_filter),
            "cache_misses_by_filter": dict(cache.misses_by_filter),
        },
        "elapsed_seconds_this_run": round(elapsed, 2),
        "outputs": {
            "manifest": str(manifest_path),
            "resolved": str(resolved_path),
            "unresolved": str(unresolved_path),
            "cache": str(cache_path),
        },
    }

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print("Resolution complete")
    print(f"  manifest records:    {len(combined)}")
    print(f"  resolved candidates: {len(resolved)}")
    print(f"  unique video IDs:    {len(unique_video_ids)}")
    print(f"  unresolved/error:    {len(unresolved)}")
    print(f"  resolution rate:     {report['resolution_rate']:.2%}")
    print(f"  song resolutions:    {source_counts.get('song', 0)}")
    print(f"  video fallbacks:     {source_counts.get('video', 0)}")
    print(f"  cache hits:          {cache.hits}")
    print(f"  cache misses:        {cache.misses}")
    print(f"  elapsed this run:    {elapsed / 3600:.2f} h")
    print()
    print(f"Report:     {report_path}")
    print(f"Resolved:   {resolved_path}")
    print(f"Unresolved: {unresolved_path}")


if __name__ == "__main__":
    main()
