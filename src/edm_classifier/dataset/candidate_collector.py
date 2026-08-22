"""Collect Last.fm track candidates for every label in the EDM taxonomy.

The collector deliberately does not assign final genres. It discovers tracks from
Last.fm tag charts, deduplicates them globally, enriches each unique track with
Last.fm community top tags, and writes JSONL for later LLM weak labeling.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import requests
import yaml

API_ROOT = "https://ws.audioscrobbler.com/2.0/"
RETRYABLE_API_ERRORS = {11, 16, 29}  # service offline, temporary error, rate limit
RETRYABLE_HTTP = {429, 500, 502, 503, 504}


class LastFmError(RuntimeError):
    pass


@dataclass
class LastFmClient:
    api_key: str
    cache_dir: Path
    min_interval: float = 0.25
    max_attempts: int = 5

    def __post_init__(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "edm-classifier-candidate-collector/0.1"})
        self._last_request_at = 0.0

    def _cache_path(self, params: dict[str, Any]) -> Path:
        cache_params = {k: v for k, v in params.items() if k != "api_key"}
        encoded = json.dumps(cache_params, sort_keys=True, ensure_ascii=False).encode("utf-8")
        digest = hashlib.sha256(encoded).hexdigest()
        method = str(params.get("method", "request")).replace(".", "_")
        return self.cache_dir / f"{method}-{digest}.json"

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request_at
        remaining = self.min_interval - elapsed
        if remaining > 0:
            time.sleep(remaining)

    def call(self, method: str, **kwargs: Any) -> dict[str, Any]:
        params = {
            "method": method,
            "api_key": self.api_key,
            "format": "json",
            **kwargs,
        }
        cache_path = self._cache_path(params)
        if cache_path.exists():
            return json.loads(cache_path.read_text(encoding="utf-8"))

        last_error: Exception | None = None
        for attempt in range(self.max_attempts):
            self._throttle()
            try:
                response = self.session.get(API_ROOT, params=params, timeout=30)
                self._last_request_at = time.monotonic()

                if response.status_code in RETRYABLE_HTTP:
                    raise LastFmError(f"HTTP {response.status_code}: {response.text[:200]}")
                response.raise_for_status()
                payload = response.json()

                if "error" in payload:
                    code = int(payload.get("error", -1))
                    message = payload.get("message", "Unknown Last.fm API error")
                    if code in RETRYABLE_API_ERRORS:
                        raise LastFmError(f"Last.fm error {code}: {message}")
                    raise LastFmError(f"Last.fm error {code}: {message}")

                cache_path.write_text(
                    json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
                )
                return payload
            except (requests.RequestException, ValueError, LastFmError) as exc:
                last_error = exc
                if attempt + 1 >= self.max_attempts:
                    break
                delay = 2**attempt
                print(f"warning: {method} failed ({exc}); retrying in {delay}s")
                time.sleep(delay)

        raise LastFmError(f"{method} failed after {self.max_attempts} attempts: {last_error}")

    def top_tracks(self, tag: str, *, page: int, limit: int) -> list[dict[str, Any]]:
        payload = self.call("tag.getTopTracks", tag=tag, page=page, limit=limit)
        container = payload.get("tracks") or payload.get("toptracks") or {}
        tracks = container.get("track", [])
        if isinstance(tracks, dict):
            tracks = [tracks]
        return tracks

    def track_top_tags(
        self, *, artist: str, title: str, mbid: str | None = None
    ) -> list[dict[str, Any]]:
        kwargs: dict[str, Any]
        if mbid:
            kwargs = {"mbid": mbid, "autocorrect": 1}
        else:
            kwargs = {"artist": artist, "track": title, "autocorrect": 1}
        payload = self.call("track.getTopTags", **kwargs)
        tags = (payload.get("toptags") or {}).get("tag", [])
        if isinstance(tags, dict):
            tags = [tags]
        return tags


def normalize_text(value: str) -> str:
    text = unicodedata.normalize("NFKD", value)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.casefold()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def candidate_key(artist: str, title: str, mbid: str | None) -> str:
    if mbid:
        return f"mbid:{mbid.strip().lower()}"
    material = f"{normalize_text(artist)}\x00{normalize_text(title)}"
    digest = hashlib.sha1(material.encode("utf-8")).hexdigest()[:16]
    return f"name:{digest}"


def candidate_id(key: str) -> str:
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
    return f"lastfm:{digest}"


def load_taxonomy(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("genres"), list):
        raise ValueError(f"Invalid taxonomy file: {path}")
    return data


def load_query_overrides(path: Path | None) -> dict[str, list[str]]:
    if path is None or not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    queries = data.get("queries", data)
    if not isinstance(queries, dict):
        raise ValueError("Query override file must contain a mapping or a top-level 'queries' mapping")

    result: dict[str, list[str]] = {}
    for label_id, values in queries.items():
        if isinstance(values, str):
            values = [values]
        if not isinstance(values, list):
            raise ValueError(f"Queries for {label_id!r} must be a list of strings")
        result[str(label_id)] = [str(v).strip() for v in values if str(v).strip()]
    return result


def unique_strings(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        value = value.strip()
        if not value:
            continue
        key = value.casefold()
        if key not in seen:
            seen.add(key)
            result.append(value)
    return result


def query_terms(label: dict[str, Any], overrides: dict[str, list[str]]) -> list[str]:
    label_id = label["id"]
    return unique_strings(
        [label["label"], *label.get("aliases", []), *overrides.get(label_id, [])]
    )


def parse_track(raw: dict[str, Any], *, page: int, page_size: int, index: int) -> dict[str, Any] | None:
    title = str(raw.get("name") or "").strip()
    artist_obj = raw.get("artist") or {}
    artist = str(artist_obj.get("name") if isinstance(artist_obj, dict) else artist_obj).strip()
    if not artist or not title:
        return None

    mbid = str(raw.get("mbid") or "").strip() or None
    artist_mbid = None
    if isinstance(artist_obj, dict):
        artist_mbid = str(artist_obj.get("mbid") or "").strip() or None

    rank = None
    attrs = raw.get("@attr") or {}
    if isinstance(attrs, dict):
        try:
            rank = int(attrs.get("rank"))
        except (TypeError, ValueError):
            pass
    if rank is None:
        rank = (page - 1) * page_size + index + 1

    return {
        "artist": artist,
        "title": title,
        "mbid": mbid,
        "artist_mbid": artist_mbid,
        "lastfm_url": raw.get("url"),
        "rank": rank,
    }


def add_discovery(candidate: dict[str, Any], discovery: dict[str, Any]) -> None:
    existing = candidate["discovered_for"]
    for item in existing:
        if item["label_id"] == discovery["label_id"]:
            # Keep the best rank but preserve which query produced it.
            if discovery["rank"] < item["rank"]:
                item.update(discovery)
            elif discovery["query"].casefold() not in [q.casefold() for q in item["matched_queries"]]:
                item["matched_queries"].append(discovery["query"])
            return
    discovery = dict(discovery)
    discovery["matched_queries"] = [discovery["query"]]
    existing.append(discovery)


def collect_candidates(
    *,
    taxonomy: dict[str, Any],
    client: LastFmClient,
    overrides: dict[str, list[str]],
    per_label: int,
    page_size: int,
    max_pages: int,
    roles: set[str],
    only_ids: set[str] | None,
    max_labels: int | None,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    labels = [g for g in taxonomy["genres"] if g.get("role") in roles]
    if only_ids:
        labels = [g for g in labels if g["id"] in only_ids]
    if max_labels is not None:
        labels = labels[:max_labels]

    candidates: dict[str, dict[str, Any]] = {}
    coverage: dict[str, Any] = {
        "taxonomy_version": (taxonomy.get("taxonomy") or {}).get("version"),
        "requested_per_label": per_label,
        "label_count": len(labels),
        "labels": {},
    }

    for label_index, label in enumerate(labels, start=1):
        label_id = label["id"]
        terms = query_terms(label, overrides)
        found_keys: set[str] = set()
        errors: list[str] = []
        queries_used: list[str] = []

        print(f"[{label_index}/{len(labels)}] {label['label']} ({label_id})")
        for term in terms:
            if len(found_keys) >= per_label:
                break
            queries_used.append(term)
            for page in range(1, max_pages + 1):
                if len(found_keys) >= per_label:
                    break
                try:
                    raw_tracks = client.top_tracks(term, page=page, limit=page_size)
                except LastFmError as exc:
                    errors.append(f"{term!r} page {page}: {exc}")
                    break
                if not raw_tracks:
                    break

                for i, raw in enumerate(raw_tracks):
                    parsed = parse_track(raw, page=page, page_size=page_size, index=i)
                    if parsed is None:
                        continue
                    key = candidate_key(parsed["artist"], parsed["title"], parsed["mbid"])
                    if key in found_keys:
                        continue

                    if key not in candidates:
                        candidates[key] = {
                            "candidate_id": candidate_id(key),
                            "artist": parsed["artist"],
                            "title": parsed["title"],
                            "mbid": parsed["mbid"],
                            "artist_mbid": parsed["artist_mbid"],
                            "lastfm_url": parsed["lastfm_url"],
                            "source": "lastfm",
                            "discovered_for": [],
                            "top_tags": [],
                        }

                    add_discovery(
                        candidates[key],
                        {
                            "label_id": label_id,
                            "label": label["label"],
                            "query": term,
                            "rank": parsed["rank"],
                        },
                    )
                    found_keys.add(key)
                    if len(found_keys) >= per_label:
                        break

        coverage["labels"][label_id] = {
            "label": label["label"],
            "role": label.get("role"),
            "target": per_label,
            "collected": len(found_keys),
            "shortfall": max(0, per_label - len(found_keys)),
            "queries": queries_used,
            "errors": errors,
        }
        suffix = "" if len(found_keys) >= per_label else "  <-- shortfall"
        print(f"    collected {len(found_keys)}/{per_label}{suffix}")

    coverage["candidate_assignments"] = sum(
        item["collected"] for item in coverage["labels"].values()
    )
    coverage["unique_tracks"] = len(candidates)
    coverage["labels_with_shortfall"] = sum(
        1 for item in coverage["labels"].values() if item["shortfall"] > 0
    )
    return candidates, coverage


def enrich_top_tags(
    candidates: dict[str, dict[str, Any]],
    *,
    client: LastFmClient,
    top_tags: int,
) -> None:
    total = len(candidates)
    for index, candidate in enumerate(candidates.values(), start=1):
        if index == 1 or index % 50 == 0 or index == total:
            print(f"enriching top tags: {index}/{total}")
        try:
            tags = client.track_top_tags(
                artist=candidate["artist"],
                title=candidate["title"],
                mbid=candidate.get("mbid"),
            )
        except LastFmError as exc:
            candidate["top_tags_error"] = str(exc)
            continue

        parsed_tags: list[dict[str, Any]] = []
        for tag in tags[:top_tags]:
            name = str(tag.get("name") or "").strip()
            if not name:
                continue
            try:
                count = int(tag.get("count", 0))
            except (TypeError, ValueError):
                count = 0
            parsed_tags.append(
                {
                    "name": name,
                    "count": count,
                    "url": tag.get("url"),
                }
            )
        candidate["top_tags"] = parsed_tags


def write_jsonl(path: Path, candidates: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(candidates.values(), key=lambda c: (c["artist"].casefold(), c["title"].casefold()))
    with path.open("w", encoding="utf-8") as handle:
        for candidate in ordered:
            handle.write(json.dumps(candidate, ensure_ascii=False) + "\n")


def write_coverage(path: Path, coverage: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(coverage, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect Last.fm track candidates for taxonomy labels."
    )
    parser.add_argument("--taxonomy", type=Path, default=Path("config/taxonomy.yaml"))
    parser.add_argument(
        "--query-overrides",
        type=Path,
        default=Path("config/candidate_queries.yaml"),
        help="Optional YAML mapping label IDs to additional Last.fm tag queries.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/candidates/candidate_tracks.jsonl"),
    )
    parser.add_argument(
        "--coverage",
        type=Path,
        default=Path("data/candidates/coverage.json"),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/candidates/cache"),
    )
    parser.add_argument("--api-key", default=os.getenv("LASTFM_API_KEY"))
    parser.add_argument("--per-label", type=int, default=30)
    parser.add_argument("--page-size", type=int, default=50)
    parser.add_argument("--max-pages", type=int, default=3)
    parser.add_argument("--top-tags", type=int, default=15)
    parser.add_argument("--min-interval", type=float, default=0.25)
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument(
        "--roles",
        choices=["all", "root", "subgenre"],
        default="all",
        help="Which taxonomy roles to collect. 'all' targets every table label.",
    )
    parser.add_argument(
        "--only",
        action="append",
        help="Collect only this taxonomy ID. Repeat for multiple labels.",
    )
    parser.add_argument("--max-labels", type=int)
    parser.add_argument("--skip-top-tags", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.api_key:
        raise SystemExit(
            "LASTFM_API_KEY is required. Set it in the environment or pass --api-key."
        )
    if args.per_label <= 0 or args.page_size <= 0 or args.max_pages <= 0:
        raise SystemExit("--per-label, --page-size, and --max-pages must be positive")

    taxonomy = load_taxonomy(args.taxonomy)
    overrides = load_query_overrides(args.query_overrides)
    known_ids = {genre["id"] for genre in taxonomy["genres"]}
    unknown_override_ids = sorted(set(overrides) - known_ids)
    if unknown_override_ids:
        raise SystemExit(
            "Unknown taxonomy IDs in query overrides: " + ", ".join(unknown_override_ids)
        )
    if args.only:
        unknown_only_ids = sorted(set(args.only) - known_ids)
        if unknown_only_ids:
            raise SystemExit(
                "Unknown taxonomy IDs passed to --only: " + ", ".join(unknown_only_ids)
            )
    roles = {"root", "subgenre"} if args.roles == "all" else {args.roles}

    client = LastFmClient(
        api_key=args.api_key,
        cache_dir=args.cache_dir,
        min_interval=args.min_interval,
        max_attempts=args.max_attempts,
    )

    candidates, coverage = collect_candidates(
        taxonomy=taxonomy,
        client=client,
        overrides=overrides,
        per_label=args.per_label,
        page_size=args.page_size,
        max_pages=args.max_pages,
        roles=roles,
        only_ids=set(args.only) if args.only else None,
        max_labels=args.max_labels,
    )

    # Save discovery results before the much longer top-tag enrichment pass.
    write_jsonl(args.output, candidates)
    write_coverage(args.coverage, coverage)
    print(
        f"discovery complete: {coverage['candidate_assignments']} assignments, "
        f"{coverage['unique_tracks']} unique tracks, "
        f"{coverage['labels_with_shortfall']} labels with shortfalls"
    )

    if not args.skip_top_tags:
        enrich_top_tags(candidates, client=client, top_tags=args.top_tags)
        write_jsonl(args.output, candidates)
        print(f"wrote enriched candidates to {args.output}")

    print(f"wrote coverage report to {args.coverage}")


if __name__ == "__main__":
    main()
