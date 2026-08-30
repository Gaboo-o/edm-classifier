"""Stage 1B: rescue V2 genres that missed their Last.fm track-tag quota.

This pass only targets genres with shortfall in:
    data/v2/candidates/coverage.json

Retrieval tiers, in order:
1. leaf_tag_artist:
   tag.getTopArtists(leaf/alias) -> artist.getTopTracks
   Medium evidence: artist is associated with the leaf, track is not guaranteed.

2. similar_tag:
   tag.getSimilar(leaf/alias) -> tag.getTopTracks(similar tag)
   Weak/medium evidence.

3. parent_fallback:
   tag.getTopTracks(parent genre)
   Weak evidence used only for remaining quota.

New candidates carry retrieval_evidence so the V2 LLM labeling prompt can
distinguish exact Stage-1 retrieval from rescue/fallback retrieval.

The script reuses the Stage-1 Last.fm cache and appends new candidates to
data/v2/candidates/candidate_tracks.jsonl.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from edm_classifier.dataset.candidate_collector import (
    LastFm,
    candidate_id,
    exclusion_keys,
    load_jsonl,
    load_query_config,
    load_taxonomy,
    norm,
    queries_for,
    text_key,
    tkey,
    top_tags,
    top_tracks,
    write_jsonl,
)


DEFAULT_CANDIDATES_DIR = Path("data/v2/candidates")
DEFAULT_COVERAGE = DEFAULT_CANDIDATES_DIR / "coverage.json"
DEFAULT_CANDIDATES = DEFAULT_CANDIDATES_DIR / "candidate_tracks.jsonl"
DEFAULT_CACHE = DEFAULT_CANDIDATES_DIR / "cache" / "lastfm_cache.jsonl"
DEFAULT_V1_SAMPLES = Path("data/splits/samples.jsonl")
DEFAULT_V1_CANDIDATES = Path("data/candidates/candidate_tracks.jsonl")
DEFAULT_TAXONOMY = Path("config/taxonomy.yaml")
DEFAULT_QUERY_CONFIG = Path("config/candidate_queries.yaml")


def load_coverage(path: Path) -> dict[str, dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    genres = raw.get("genres") if isinstance(raw, dict) else None
    if not isinstance(genres, dict):
        raise ValueError(f"{path}: missing genres object")
    return {
        key: value
        for key, value in genres.items()
        if isinstance(key, str) and isinstance(value, dict)
    }


def parse_top_artists(data: Any) -> list[str]:
    obj = data.get("topartists") if isinstance(data, dict) else None
    raw = obj.get("artist") if isinstance(obj, dict) else None
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list):
        return []

    result = []
    seen = set()
    for item in raw:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        key = norm(name)
        if key and key not in seen:
            seen.add(key)
            result.append(name.strip())
    return result


def parse_artist_top_tracks(data: Any) -> list[dict[str, Any]]:
    obj = data.get("toptracks") if isinstance(data, dict) else None
    raw = obj.get("track") if isinstance(obj, dict) else None
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list):
        return []

    result = []
    for rank, item in enumerate(raw, 1):
        if not isinstance(item, dict):
            continue
        title = item.get("name")
        artist_obj = item.get("artist")
        artist = (
            artist_obj.get("name")
            if isinstance(artist_obj, dict)
            else artist_obj
        )
        if not isinstance(title, str) or not isinstance(artist, str):
            continue
        mbid = item.get("mbid")
        result.append(
            {
                "artist": artist.strip(),
                "title": title.strip(),
                "mbid": (
                    mbid.strip()
                    if isinstance(mbid, str) and mbid.strip()
                    else None
                ),
                "lastfm_url": (
                    item.get("url")
                    if isinstance(item.get("url"), str)
                    else None
                ),
                "rank": rank,
            }
        )
    return result


def parse_similar_tags(data: Any, maximum: int) -> list[str]:
    obj = data.get("similartags") if isinstance(data, dict) else None
    raw = obj.get("tag") if isinstance(obj, dict) else None
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list):
        return []

    result = []
    seen = set()
    for item in raw:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        key = norm(name)
        if key and key not in seen:
            seen.add(key)
            result.append(name.strip())
        if len(result) >= maximum:
            break
    return result


def current_artists_for_genre(
    candidates: list[dict[str, Any]],
    genre_id: str,
) -> set[str]:
    result = set()
    for row in candidates:
        discovered = row.get("discovered_for")
        if not isinstance(discovered, list) or genre_id not in discovered:
            continue
        artist = row.get("artist")
        if isinstance(artist, str) and artist.strip():
            result.add(norm(artist))
    return result


def make_candidate(
    row: dict[str, Any],
    *,
    genre_id: str,
    genre_label: str,
    evidence_type: str,
    query: str,
    strength: str,
) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id(
            row["artist"],
            row["title"],
            row.get("mbid"),
        ),
        "artist": row["artist"],
        "title": row["title"],
        "mbid": row.get("mbid"),
        "lastfm_url": row.get("lastfm_url"),
        "discovered_for": [genre_id],
        "discovered_for_labels": [genre_label],
        "discovery": [],
        "retrieval_evidence": [
            {
                "genre_id": genre_id,
                "genre_label": genre_label,
                "type": evidence_type,
                "query": query,
                "strength": strength,
            }
        ],
        "top_tags": [],
    }


def add_if_new(
    selected: list[dict[str, Any]],
    row: dict[str, Any],
    *,
    excluded: set[str],
    selected_artist_counts: dict[str, int],
    max_per_artist: int,
    genre_id: str,
    genre_label: str,
    evidence_type: str,
    query: str,
    strength: str,
) -> bool:
    artist = row.get("artist")
    title = row.get("title")
    if not isinstance(artist, str) or not isinstance(title, str):
        return False

    mbid = row.get("mbid")
    primary = tkey(
        artist,
        title,
        mbid if isinstance(mbid, str) else None,
    )
    text = text_key(artist, title)

    if primary in excluded or text in excluded:
        return False

    artist_key = norm(artist)
    if selected_artist_counts[artist_key] >= max_per_artist:
        return False

    candidate = make_candidate(
        row,
        genre_id=genre_id,
        genre_label=genre_label,
        evidence_type=evidence_type,
        query=query,
        strength=strength,
    )
    selected.append(candidate)
    selected_artist_counts[artist_key] += 1
    excluded.add(primary)
    excluded.add(text)
    return True


def rescue_from_tagged_artists(
    client: LastFm,
    *,
    queries: list[str],
    needed: int,
    excluded: set[str],
    existing_artists: set[str],
    max_tracks_per_artist: int,
    artists_per_query: int,
    tracks_per_artist_fetch: int,
    genre_id: str,
    genre_label: str,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    selected_artist_counts: dict[str, int] = defaultdict(int)

    # Prefer leaf-tagged artists that were not already represented for the
    # genre in Stage 1.
    artist_sources: dict[str, tuple[str, str]] = {}

    for query in queries:
        data = client.request(
            "tag.getTopArtists",
            tag=query,
            limit=artists_per_query,
            page=1,
        )
        for artist in parse_top_artists(data):
            key = norm(artist)
            if key:
                artist_sources.setdefault(key, (artist, query))

    ordered = sorted(
        artist_sources.items(),
        key=lambda pair: (
            pair[0] in existing_artists,
            pair[0],
        ),
    )

    for artist_key, (artist_name, query) in ordered:
        if len(selected) >= needed:
            break

        data = client.request(
            "artist.getTopTracks",
            artist=artist_name,
            autocorrect=1,
            limit=tracks_per_artist_fetch,
            page=1,
        )

        rows = parse_artist_top_tracks(data)

        for row in rows:
            if len(selected) >= needed:
                break
            if add_if_new(
                selected,
                row,
                excluded=excluded,
                selected_artist_counts=selected_artist_counts,
                max_per_artist=max_tracks_per_artist,
                genre_id=genre_id,
                genre_label=genre_label,
                evidence_type="leaf_tag_artist",
                query=query,
                strength="medium",
            ):
                # Prefer breadth: once one track is chosen from an artist,
                # move to the next artist before considering second tracks.
                break

    # Second pass allows another track per tagged artist if quota remains.
    if len(selected) < needed and max_tracks_per_artist > 1:
        for artist_key, (artist_name, query) in ordered:
            if len(selected) >= needed:
                break
            if selected_artist_counts[artist_key] >= max_tracks_per_artist:
                continue

            data = client.request(
                "artist.getTopTracks",
                artist=artist_name,
                autocorrect=1,
                limit=tracks_per_artist_fetch,
                page=1,
            )
            for row in parse_artist_top_tracks(data):
                if len(selected) >= needed:
                    break
                if add_if_new(
                    selected,
                    row,
                    excluded=excluded,
                    selected_artist_counts=selected_artist_counts,
                    max_per_artist=max_tracks_per_artist,
                    genre_id=genre_id,
                    genre_label=genre_label,
                    evidence_type="leaf_tag_artist",
                    query=query,
                    strength="medium",
                ):
                    break

    return selected


def rescue_from_similar_tags(
    client: LastFm,
    *,
    queries: list[str],
    needed: int,
    excluded: set[str],
    max_tracks_per_artist: int,
    similar_tags_per_query: int,
    tracks_per_tag: int,
    genre_id: str,
    genre_label: str,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    artist_counts: dict[str, int] = defaultdict(int)
    similar: list[tuple[str, str]] = []
    seen = set()

    for query in queries:
        data = client.request("tag.getSimilar", tag=query)
        for tag in parse_similar_tags(data, similar_tags_per_query):
            key = norm(tag)
            if key and key not in seen and key != norm(query):
                seen.add(key)
                similar.append((tag, query))

    for tag, source_query in similar:
        if len(selected) >= needed:
            break

        rows = top_tracks(
            client,
            tag,
            page=1,
            limit=tracks_per_tag,
        )

        for row in rows:
            if len(selected) >= needed:
                break
            add_if_new(
                selected,
                row,
                excluded=excluded,
                selected_artist_counts=artist_counts,
                max_per_artist=max_tracks_per_artist,
                genre_id=genre_id,
                genre_label=genre_label,
                evidence_type="similar_tag",
                query=f"{source_query} -> {tag}",
                strength="weak",
            )

    return selected


def rescue_from_parent(
    client: LastFm,
    *,
    parent_queries: list[str],
    needed: int,
    excluded: set[str],
    max_tracks_per_artist: int,
    tracks_per_tag: int,
    pages: int,
    genre_id: str,
    genre_label: str,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    artist_counts: dict[str, int] = defaultdict(int)

    for query in parent_queries:
        for page in range(1, pages + 1):
            if len(selected) >= needed:
                return selected

            rows = top_tracks(
                client,
                query,
                page=page,
                limit=tracks_per_tag,
            )

            for row in rows:
                if len(selected) >= needed:
                    return selected
                add_if_new(
                    selected,
                    row,
                    excluded=excluded,
                    selected_artist_counts=artist_counts,
                    max_per_artist=max_tracks_per_artist,
                    genre_id=genre_id,
                    genre_label=genre_label,
                    evidence_type="parent_fallback",
                    query=query,
                    strength="weak",
                )

    return selected


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Rescue V2 Last.fm shortfall genres using broader retrieval."
    )
    p.add_argument("--coverage", type=Path, default=DEFAULT_COVERAGE)
    p.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    p.add_argument("--taxonomy", type=Path, default=DEFAULT_TAXONOMY)
    p.add_argument("--query-config", type=Path, default=DEFAULT_QUERY_CONFIG)
    p.add_argument("--api-key", default=os.getenv("LASTFM_API_KEY"))
    p.add_argument("--request-delay", type=float, default=0.20)
    p.add_argument("--request-retries", type=int, default=4)
    p.add_argument("--max-tracks-per-artist", type=int, default=2)
    p.add_argument("--artists-per-query", type=int, default=200)
    p.add_argument("--tracks-per-artist-fetch", type=int, default=10)
    p.add_argument("--similar-tags-per-query", type=int, default=5)
    p.add_argument("--tracks-per-similar-tag", type=int, default=200)
    p.add_argument("--parent-pages", type=int, default=3)
    p.add_argument("--parent-page-size", type=int, default=200)
    p.add_argument("--top-tags", type=int, default=12)
    p.add_argument("--max-genres", type=int)
    p.add_argument(
        "--no-parent-fallback",
        action="store_true",
        help="Do not use broad parent-tag fallback.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.api_key:
        raise SystemExit("Set LASTFM_API_KEY or pass --api-key.")

    coverage = load_coverage(args.coverage)
    taxonomy = load_taxonomy(args.taxonomy)
    query_config = load_query_config(args.query_config)
    existing_candidates = load_jsonl(args.candidates)

    shortfalls = [
        (genre_id, data)
        for genre_id, data in coverage.items()
        if int(data.get("shortfall", 0)) > 0
    ]

    # Preserve the original Stage-1 priority order from coverage.json.
    if args.max_genres is not None:
        shortfalls = shortfalls[: args.max_genres]

    v1_records = (
        load_jsonl(DEFAULT_V1_SAMPLES)
        + load_jsonl(DEFAULT_V1_CANDIDATES)
    )

    # Rescue must add genuinely new tracks, not rediscover Stage-1 candidates.
    excluded = exclusion_keys(v1_records + existing_candidates)

    client = LastFm(
        args.api_key,
        DEFAULT_CACHE,
        args.request_delay,
        args.request_retries,
    )

    all_new: list[dict[str, Any]] = []
    rescue_summary: dict[str, Any] = {}

    start = time.monotonic()

    print(f"Shortfall genres to rescue: {len(shortfalls)}")
    print(f"Existing V2 candidates:     {len(existing_candidates)}")
    print()

    for index, (genre_id, data) in enumerate(shortfalls, 1):
        if genre_id not in taxonomy:
            continue

        needed = int(data.get("shortfall", 0))
        genre_label = str(taxonomy[genre_id].get("label", genre_id))
        queries = queries_for(genre_id, taxonomy, query_config)

        existing_artists = current_artists_for_genre(
            existing_candidates + all_new,
            genre_id,
        )

        print(
            f"[{index}/{len(shortfalls)}] {genre_id}: "
            f"need={needed}, existing_artists={len(existing_artists)}"
        )

        rescued: list[dict[str, Any]] = []
        by_tier = {
            "leaf_tag_artist": 0,
            "similar_tag": 0,
            "parent_fallback": 0,
        }

        try:
            first = rescue_from_tagged_artists(
                client,
                queries=queries,
                needed=needed,
                excluded=excluded,
                existing_artists=existing_artists,
                max_tracks_per_artist=args.max_tracks_per_artist,
                artists_per_query=args.artists_per_query,
                tracks_per_artist_fetch=args.tracks_per_artist_fetch,
                genre_id=genre_id,
                genre_label=genre_label,
            )
            rescued.extend(first)
            by_tier["leaf_tag_artist"] = len(first)

            remaining = needed - len(rescued)

            if remaining > 0:
                second = rescue_from_similar_tags(
                    client,
                    queries=queries,
                    needed=remaining,
                    excluded=excluded,
                    max_tracks_per_artist=args.max_tracks_per_artist,
                    similar_tags_per_query=args.similar_tags_per_query,
                    tracks_per_tag=args.tracks_per_similar_tag,
                    genre_id=genre_id,
                    genre_label=genre_label,
                )
                rescued.extend(second)
                by_tier["similar_tag"] = len(second)

            remaining = needed - len(rescued)

            if remaining > 0 and not args.no_parent_fallback:
                parent_id = taxonomy[genre_id].get("parent")
                parent_queries: list[str] = []

                if isinstance(parent_id, str) and parent_id in taxonomy:
                    parent_queries = queries_for(
                        parent_id,
                        taxonomy,
                        query_config,
                    )

                if parent_queries:
                    third = rescue_from_parent(
                        client,
                        parent_queries=parent_queries,
                        needed=remaining,
                        excluded=excluded,
                        max_tracks_per_artist=args.max_tracks_per_artist,
                        tracks_per_tag=args.parent_page_size,
                        pages=args.parent_pages,
                        genre_id=genre_id,
                        genre_label=genre_label,
                    )
                    rescued.extend(third)
                    by_tier["parent_fallback"] = len(third)

        except Exception as exc:
            rescue_summary[genre_id] = {
                "needed": needed,
                "rescued": len(rescued),
                "remaining_shortfall": max(0, needed - len(rescued)),
                "by_tier": by_tier,
                "error": f"{type(exc).__name__}: {exc}",
            }
            all_new.extend(rescued)
            print(f"  ERROR after {len(rescued)} rescued: {exc}")
            continue

        all_new.extend(rescued)

        rescue_summary[genre_id] = {
            "needed": needed,
            "rescued": len(rescued),
            "remaining_shortfall": max(0, needed - len(rescued)),
            "by_tier": by_tier,
            "error": None,
        }

        print(
            f"  rescued={len(rescued)} "
            f"(artist={by_tier['leaf_tag_artist']}, "
            f"similar={by_tier['similar_tag']}, "
            f"parent={by_tier['parent_fallback']}) "
            f"remaining={max(0, needed-len(rescued))}"
        )

    # Enrich only newly added rescue tracks.
    print()
    print(f"Top-tag enrichment for {len(all_new)} rescue candidates...")
    tag_errors = []

    for index, row in enumerate(all_new, 1):
        if index == 1 or index % 100 == 0 or index == len(all_new):
            print(f"  {index}/{len(all_new)}")

        try:
            row["top_tags"] = top_tags(
                client,
                row["artist"],
                row["title"],
                args.top_tags,
            )
        except Exception as exc:
            row["top_tags"] = []
            tag_errors.append(
                {
                    "candidate_id": row["candidate_id"],
                    "artist": row["artist"],
                    "title": row["title"],
                    "error": str(exc),
                }
            )

    merged = existing_candidates + all_new
    write_jsonl(args.candidates, merged)

    output_dir = args.candidates.parent
    write_jsonl(
        output_dir / "rescue_top_tag_errors.jsonl",
        tag_errors,
    )

    original_selected = sum(
        int(data.get("selected", 0))
        for data in coverage.values()
    )

    final_genres = {}
    total_remaining = 0

    for genre_id, data in coverage.items():
        rescue = rescue_summary.get(genre_id)
        rescued = int(rescue.get("rescued", 0)) if rescue else 0
        planned = int(data.get("planned", 0))
        original = int(data.get("selected", 0))
        final_selected = original + rescued
        remaining = max(0, planned - final_selected)
        total_remaining += remaining

        final_genres[genre_id] = {
            "planned": planned,
            "stage1_selected": original,
            "rescue_selected": rescued,
            "final_selected_assignments": final_selected,
            "remaining_shortfall": remaining,
            "rescue_by_tier": (
                rescue.get("by_tier")
                if rescue
                else None
            ),
        }

    final_coverage = {
        "planned_candidate_assignments": sum(
            int(data.get("planned", 0))
            for data in coverage.values()
        ),
        "stage1_selected_assignments": original_selected,
        "rescue_new_unique_candidates": len(all_new),
        "final_unique_candidate_tracks": len(merged),
        "remaining_assignment_shortfall": total_remaining,
        "genres_with_remaining_shortfall": sum(
            1
            for data in final_genres.values()
            if data["remaining_shortfall"] > 0
        ),
        "genres": final_genres,
    }

    final_coverage_path = output_dir / "coverage_after_rescue.json"
    final_coverage_path.write_text(
        json.dumps(final_coverage, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    elapsed = time.monotonic() - start
    report = {
        "shortfall_genres_processed": len(shortfalls),
        "stage1_unique_candidates": len(existing_candidates),
        "rescue_unique_candidates_added": len(all_new),
        "final_unique_candidates": len(merged),
        "remaining_assignment_shortfall": total_remaining,
        "genres_with_remaining_shortfall": final_coverage[
            "genres_with_remaining_shortfall"
        ],
        "rescue_tier_totals": {
            tier: sum(
                int(data.get("by_tier", {}).get(tier, 0))
                for data in rescue_summary.values()
            )
            for tier in (
                "leaf_tag_artist",
                "similar_tag",
                "parent_fallback",
            )
        },
        "top_tag_errors": len(tag_errors),
        "lastfm": {
            "network_requests_this_run": client.network,
            "cache_hits_this_run": client.cache_hits,
            "request_retries_this_run": client.retry_count,
        },
        "elapsed_seconds": round(elapsed, 2),
        "outputs": {
            "candidate_tracks": str(args.candidates),
            "coverage_after_rescue": str(final_coverage_path),
        },
    }

    report_path = output_dir / "rescue_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print()
    print("Stage 1B rescue complete")
    print(f"  Stage-1 candidates:       {len(existing_candidates)}")
    print(f"  rescue candidates added:  {len(all_new)}")
    print(f"  final unique candidates:  {len(merged)}")
    print(f"  remaining shortfall:      {total_remaining}")
    print(
        f"  genres still short:       "
        f"{final_coverage['genres_with_remaining_shortfall']}"
    )
    print(f"  elapsed:                  {elapsed/3600:.2f} h")
    print()
    print(f"Report:   {report_path}")
    print(f"Coverage: {final_coverage_path}")


if __name__ == "__main__":
    main()
