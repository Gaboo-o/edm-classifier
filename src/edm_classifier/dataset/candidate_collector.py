"""Targeted V2 Last.fm candidate collector.

Reads the V2 collection plan in priority order, excludes V1 tracks, maximizes
new artists per genre, globally deduplicates selected tracks, and enriches
selected tracks with Last.fm top tags.

Outputs:
    data/v2/candidates/candidate_tracks.jsonl
    data/v2/candidates/coverage.json
    data/v2/candidates/collector_report.json
    data/v2/candidates/top_tag_errors.jsonl
    data/v2/candidates/cache/lastfm_cache.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml


API_URL = "https://ws.audioscrobbler.com/2.0/"
DEFAULT_TAXONOMY = Path("config/taxonomy.yaml")
DEFAULT_QUERY_CONFIG = Path("config/candidate_queries.yaml")
DEFAULT_PLAN = Path("data/v2/collection_plan/collection_plan.json")
DEFAULT_V1_SAMPLES = Path("data/splits/samples.jsonl")
DEFAULT_V1_CANDIDATES = Path("data/candidates/candidate_tracks.jsonl")
DEFAULT_OUTPUT = Path("data/v2/candidates")


def norm(value: str) -> str:
    text = unicodedata.normalize("NFKD", value)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", text.casefold()).strip()


def tkey(artist: str, title: str, mbid: str | None = None) -> str:
    if mbid and mbid.strip():
        return "mbid:" + mbid.strip().casefold()
    return "text:" + norm(artist) + "\0" + norm(title)


def text_key(artist: str, title: str) -> str:
    return "text:" + norm(artist) + "\0" + norm(title)


def candidate_id(artist: str, title: str, mbid: str | None) -> str:
    raw = tkey(artist, title, mbid)
    return "v2_" + hashlib.sha256(raw.encode()).hexdigest()[:20]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out = []
    with path.open("r", encoding="utf-8") as f:
        for n, raw in enumerate(f, 1):
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{n}: invalid JSON: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{n}: expected JSON object")
            out.append(obj)
    return out


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def load_taxonomy(path: Path) -> dict[str, dict[str, Any]]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    genres = raw.get("genres") if isinstance(raw, dict) else None
    if not isinstance(genres, list):
        raise ValueError(f"{path}: expected genres list")
    return {
        row["id"]: dict(row)
        for row in genres
        if isinstance(row, dict) and isinstance(row.get("id"), str)
    }


def load_plan(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    leaves = raw.get("leaves") if isinstance(raw, dict) else None
    if not isinstance(leaves, list):
        raise ValueError(f"{path}: missing leaves list")
    return [x for x in leaves if isinstance(x, dict) and isinstance(x.get("id"), str)]


def as_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, list):
        return [x.strip() for x in value if isinstance(x, str) and x.strip()]
    return []


def load_query_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return raw if isinstance(raw, dict) else {}


def extra_queries(config: dict[str, Any], genre_id: str) -> list[str]:
    values: list[Any] = [config.get(genre_id)]
    for name in ("queries", "overrides", "genres"):
        obj = config.get(name)
        if isinstance(obj, dict):
            values.append(obj.get(genre_id))

    out: list[str] = []
    for value in values:
        if isinstance(value, dict):
            for key in ("queries", "tags", "aliases"):
                out.extend(as_strings(value.get(key)))
        else:
            out.extend(as_strings(value))
    return out


def queries_for(genre_id: str, taxonomy: dict[str, dict[str, Any]], config: dict[str, Any]) -> list[str]:
    item = taxonomy[genre_id]
    values = []
    if isinstance(item.get("label"), str):
        values.append(item["label"])
    values.extend(as_strings(item.get("aliases")))
    values.extend(extra_queries(config, genre_id))

    out, seen = [], set()
    for value in values:
        key = norm(value)
        if key and key not in seen:
            seen.add(key)
            out.append(value)
    return out


def artist_keys(record: dict[str, Any]) -> set[str]:
    values = []
    if isinstance(record.get("artist_keys"), list):
        values.extend(record["artist_keys"])
    if isinstance(record.get("artists"), list):
        values.extend(record["artists"])
    if isinstance(record.get("artist"), str):
        values.append(record["artist"])
    return {norm(x) for x in values if isinstance(x, str) and norm(x)}


def labels(record: dict[str, Any]) -> set[str]:
    raw = record.get("labels")
    if not isinstance(raw, list):
        return set()
    return {x for x in raw if isinstance(x, str) and x}


def exclusion_keys(records: list[dict[str, Any]]) -> set[str]:
    out = set()
    for row in records:
        artist, title = row.get("artist"), row.get("title")
        if not isinstance(artist, str) or not isinstance(title, str):
            continue
        mbid = row.get("mbid")
        out.add(text_key(artist, title))
        out.add(tkey(artist, title, mbid if isinstance(mbid, str) else None))
    return out


class LastFm:
    def __init__(self, api_key: str, cache_path: Path, delay: float, retries: int):
        self.api_key = api_key
        self.cache_path = cache_path
        self.delay = delay
        self.retries = retries
        self.cache: dict[str, Any] = {}
        self.network = 0
        self.cache_hits = 0
        self.retry_count = 0
        self.last_request = 0.0

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        for row in load_jsonl(cache_path):
            if isinstance(row.get("key"), str) and "response" in row:
                self.cache[row["key"]] = row["response"]

    def request(self, method: str, **params: Any) -> Any:
        cache_key = json.dumps(
            {"method": method, "params": params},
            sort_keys=True,
            separators=(",", ":"),
        )
        if cache_key in self.cache:
            self.cache_hits += 1
            return self.cache[cache_key]

        query_params = dict(params)
        query_params.update(method=method, api_key=self.api_key, format="json")
        url = API_URL + "?" + urllib.parse.urlencode(query_params)

        last_exc: Exception | None = None
        for attempt in range(1, self.retries + 1):
            remaining = self.delay - (time.monotonic() - self.last_request)
            if remaining > 0:
                time.sleep(remaining)

            try:
                req = urllib.request.Request(
                    url, headers={"User-Agent": "edm-classifier-v2/0.1"}
                )
                self.network += 1
                self.last_request = time.monotonic()
                with urllib.request.urlopen(req, timeout=45) as response:
                    data = json.loads(response.read().decode("utf-8"))

                if isinstance(data, dict) and "error" in data:
                    if data.get("error") == 29:
                        raise RuntimeError("Last.fm rate limited: " + str(data.get("message")))
                    raise ValueError(
                        f"Last.fm API error {data.get('error')}: {data.get('message')}"
                    )

                self.cache[cache_key] = data
                with self.cache_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(
                        {"key": cache_key, "response": data},
                        ensure_ascii=False, separators=(",", ":")
                    ) + "\n")
                return data

            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError,
                    RuntimeError, json.JSONDecodeError) as exc:
                last_exc = exc
                if attempt == self.retries:
                    break
                self.retry_count += 1
                wait = min(60.0, (2 ** attempt) + random.random())
                time.sleep(wait)

        assert last_exc is not None
        raise last_exc


def top_tracks(client: LastFm, tag: str, page: int, limit: int) -> list[dict[str, Any]]:
    data = client.request("tag.getTopTracks", tag=tag, page=page, limit=limit)
    obj = data.get("tracks") if isinstance(data, dict) else None
    raw = obj.get("track") if isinstance(obj, dict) else None
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list):
        return []

    out = []
    for offset, row in enumerate(raw, 1):
        if not isinstance(row, dict):
            continue
        title = row.get("name")
        a = row.get("artist")
        artist = a.get("name") if isinstance(a, dict) else a
        if not isinstance(title, str) or not isinstance(artist, str):
            continue
        mbid = row.get("mbid")
        out.append({
            "artist": artist.strip(),
            "title": title.strip(),
            "mbid": mbid.strip() if isinstance(mbid, str) and mbid.strip() else None,
            "lastfm_url": row.get("url") if isinstance(row.get("url"), str) else None,
            "query": tag,
            "page": page,
            "rank": (page - 1) * limit + offset,
        })
    return out


def top_tags(client: LastFm, artist: str, title: str, maximum: int) -> list[dict[str, Any]]:
    data = client.request(
        "track.getTopTags", artist=artist, track=title, autocorrect=1
    )
    obj = data.get("toptags") if isinstance(data, dict) else None
    raw = obj.get("tag") if isinstance(obj, dict) else None
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list):
        return []

    out = []
    for row in raw:
        if not isinstance(row, dict) or not isinstance(row.get("name"), str):
            continue
        try:
            count = int(row.get("count", 0))
        except (TypeError, ValueError):
            count = 0
        out.append({"name": row["name"].strip(), "count": count})
    out.sort(key=lambda x: x["count"], reverse=True)
    return out[:maximum]


def collect_pool(
    client: LastFm,
    queries: list[str],
    target: int,
    page_size: int,
    max_pages: int,
    excluded: set[str],
) -> dict[str, dict[str, Any]]:
    pool: dict[str, dict[str, Any]] = {}
    exhausted = set()

    for page in range(1, max_pages + 1):
        for query in queries:
            if query in exhausted:
                continue
            rows = top_tracks(client, query, page, page_size)
            if not rows:
                exhausted.add(query)
                continue

            for row in rows:
                primary = tkey(row["artist"], row["title"], row["mbid"])
                text = text_key(row["artist"], row["title"])
                if primary in excluded or text in excluded:
                    continue

                existing = pool.get(primary) or pool.get(text)
                if existing is None:
                    pool[primary] = {
                        "artist": row["artist"],
                        "title": row["title"],
                        "mbid": row["mbid"],
                        "lastfm_url": row["lastfm_url"],
                        "discovery": [{
                            "query": query,
                            "page": page,
                            "rank": row["rank"],
                        }],
                    }
                else:
                    existing["discovery"].append({
                        "query": query,
                        "page": page,
                        "rank": row["rank"],
                    })

            if len(pool) >= target:
                return pool

        if len(exhausted) == len(queries):
            break
    return pool


def best_rank(row: dict[str, Any]) -> int:
    vals = [
        x.get("rank")
        for x in row.get("discovery", [])
        if isinstance(x, dict) and isinstance(x.get("rank"), int)
    ]
    return min(vals) if vals else 10**9


def select_artist_diverse(
    pool: dict[str, dict[str, Any]],
    quota: int,
    existing_artists: set[str],
    max_per_artist: int,
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pool.values():
        groups[norm(row["artist"]) or candidate_id(row["artist"], row["title"], row["mbid"])].append(row)

    for group in groups.values():
        group.sort(key=lambda r: (-len(r["discovery"]), best_rank(r), norm(r["title"])))

    unseen = [a for a in groups if a not in existing_artists]
    seen = [a for a in groups if a in existing_artists]
    unseen.sort(key=lambda a: (best_rank(groups[a][0]), a))
    seen.sort(key=lambda a: (best_rank(groups[a][0]), a))

    selected = []
    for layer in range(max_per_artist):
        for artist_list in (unseen, seen):
            for artist in artist_list:
                if layer < len(groups[artist]):
                    selected.append(groups[artist][layer])
                    if len(selected) >= quota:
                        return selected
    return selected


def merge_candidate(
    global_rows: dict[str, dict[str, Any]],
    row: dict[str, Any],
    genre_id: str,
    genre_label: str,
) -> None:
    primary = tkey(row["artist"], row["title"], row["mbid"])
    text = text_key(row["artist"], row["title"])
    existing = global_rows.get(primary) or global_rows.get(text)

    if existing is None:
        global_rows[primary] = {
            "candidate_id": candidate_id(row["artist"], row["title"], row["mbid"]),
            "artist": row["artist"],
            "title": row["title"],
            "mbid": row["mbid"],
            "lastfm_url": row.get("lastfm_url"),
            "discovered_for": [genre_id],
            "discovered_for_labels": [genre_label],
            "discovery": row["discovery"],
            "top_tags": [],
        }
        return

    if genre_id not in existing["discovered_for"]:
        existing["discovered_for"].append(genre_id)
        existing["discovered_for_labels"].append(genre_label)

    markers = {
        (x.get("query"), x.get("page"), x.get("rank"))
        for x in existing["discovery"] if isinstance(x, dict)
    }
    for x in row["discovery"]:
        marker = (x.get("query"), x.get("page"), x.get("rank"))
        if marker not in markers:
            existing["discovery"].append(x)
            markers.add(marker)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--taxonomy", type=Path, default=DEFAULT_TAXONOMY)
    p.add_argument("--query-config", type=Path, default=DEFAULT_QUERY_CONFIG)
    p.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--api-key", default=os.getenv("LASTFM_API_KEY"))
    p.add_argument("--pool-multiplier", type=float, default=4.0)
    p.add_argument("--page-size", type=int, default=200)
    p.add_argument("--max-pages-per-query", type=int, default=6)
    p.add_argument("--max-tracks-per-artist", type=int, default=2)
    p.add_argument("--top-tags", type=int, default=12)
    p.add_argument("--request-delay", type=float, default=0.20)
    p.add_argument("--request-retries", type=int, default=4)
    p.add_argument("--max-labels", type=int)
    p.add_argument("--skip-top-tags", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.api_key:
        raise SystemExit("Set LASTFM_API_KEY or pass --api-key.")

    taxonomy = load_taxonomy(args.taxonomy)
    plan = load_plan(args.plan)
    config = load_query_config(args.query_config)
    if args.max_labels is not None:
        plan = plan[:args.max_labels]

    v1_samples = load_jsonl(DEFAULT_V1_SAMPLES)
    v1_candidates = load_jsonl(DEFAULT_V1_CANDIDATES)
    excluded = exclusion_keys(v1_samples + v1_candidates)

    existing_artists: dict[str, set[str]] = defaultdict(set)
    for sample in v1_samples:
        akeys = artist_keys(sample)
        for label in labels(sample):
            existing_artists[label].update(akeys)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    client = LastFm(
        args.api_key,
        args.output_dir / "cache" / "lastfm_cache.jsonl",
        args.request_delay,
        args.request_retries,
    )

    global_rows: dict[str, dict[str, Any]] = {}
    coverage: dict[str, Any] = {}
    start = time.monotonic()

    print(f"Priority leaves: {len(plan)}")
    print(f"Excluded V1 keys: {len(excluded)}")
    print()

    for i, pitem in enumerate(plan, 1):
        gid = pitem["id"]
        if gid not in taxonomy:
            continue
        quota = int(pitem.get("planned_candidate_requests", 0))
        if quota <= 0:
            continue

        glabel = taxonomy[gid].get("label", gid)
        queries = queries_for(gid, taxonomy, config)
        pool_target = max(quota, int(round(quota * args.pool_multiplier)))

        print(f"[{i}/{len(plan)}] {gid}: quota={quota}, pool={pool_target}")

        try:
            pool = collect_pool(
                client, queries, pool_target,
                args.page_size, args.max_pages_per_query, excluded
            )
            selected = select_artist_diverse(
                pool, quota, existing_artists[gid], args.max_tracks_per_artist
            )
        except Exception as exc:
            coverage[gid] = {
                "label": glabel, "planned": quota, "selected": 0,
                "pool": 0, "shortfall": quota,
                "queries": queries,
                "error": f"{type(exc).__name__}: {exc}",
            }
            print(f"  ERROR: {exc}")
            continue

        selected_artists = {norm(x["artist"]) for x in selected}
        new_artists = selected_artists - existing_artists[gid]

        for row in selected:
            merge_candidate(global_rows, row, gid, str(glabel))

        coverage[gid] = {
            "label": glabel,
            "planned": quota,
            "selected": len(selected),
            "pool": len(pool),
            "selected_unique_artists": len(selected_artists),
            "selected_new_artists": len(new_artists),
            "shortfall": max(0, quota - len(selected)),
            "queries": queries,
            "error": None,
        }

        print(
            f"  selected={len(selected)}, artists={len(selected_artists)}, "
            f"new_artists={len(new_artists)}, shortfall={max(0, quota-len(selected))}"
        )

    rows = sorted(global_rows.values(), key=lambda r: r["candidate_id"])
    tag_errors: list[dict[str, Any]] = []

    if not args.skip_top_tags:
        print(f"\nTop-tag enrichment for {len(rows)} unique candidates...")
        for i, row in enumerate(rows, 1):
            if i == 1 or i % 100 == 0 or i == len(rows):
                print(f"  {i}/{len(rows)}")
            try:
                row["top_tags"] = top_tags(
                    client, row["artist"], row["title"], args.top_tags
                )
            except Exception as exc:
                row["top_tags"] = []
                tag_errors.append({
                    "candidate_id": row["candidate_id"],
                    "artist": row["artist"],
                    "title": row["title"],
                    "error": str(exc),
                })

    write_jsonl(args.output_dir / "candidate_tracks.jsonl", rows)
    write_jsonl(args.output_dir / "top_tag_errors.jsonl", tag_errors)

    planned = sum(int(x.get("planned_candidate_requests", 0)) for x in plan)
    selected = sum(int(x.get("selected", 0)) for x in coverage.values())
    shortfall = sum(int(x.get("shortfall", 0)) for x in coverage.values())

    coverage_payload = {
        "planned_candidate_assignments": planned,
        "selected_candidate_assignments": selected,
        "globally_unique_candidates": len(rows),
        "assignment_shortfall": shortfall,
        "genres": coverage,
    }
    (args.output_dir / "coverage.json").write_text(
        json.dumps(coverage_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    artist_count = len({norm(x["artist"]) for x in rows if norm(x["artist"])})
    elapsed = time.monotonic() - start
    report = {
        "version": "v2",
        "priority_leaves_processed": len(plan),
        "planned_candidate_assignments": planned,
        "selected_candidate_assignments": selected,
        "globally_unique_candidates": len(rows),
        "globally_unique_candidate_artists": artist_count,
        "assignment_shortfall": shortfall,
        "genres_with_shortfall": sum(
            1 for x in coverage.values() if int(x.get("shortfall", 0)) > 0
        ),
        "top_tag_errors": len(tag_errors),
        "lastfm": {
            "network_requests": client.network,
            "cache_hits": client.cache_hits,
            "request_retries": client.retry_count,
            "request_delay_seconds": args.request_delay,
        },
        "elapsed_seconds": round(elapsed, 2),
    }
    (args.output_dir / "collector_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("\nV2 Stage 1 complete")
    print(f"  planned assignments: {planned}")
    print(f"  selected assignments:{selected}")
    print(f"  unique tracks:       {len(rows)}")
    print(f"  unique artists:      {artist_count}")
    print(f"  shortfall:           {shortfall}")
    print(f"  elapsed:             {elapsed/3600:.2f} h")


if __name__ == "__main__":
    main()
