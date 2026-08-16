#!/usr/bin/env python3
"""Collect weak-label evidence for one music track.

Outputs JSON conforming to weak_label_input.schema.json.

Sources:
- yt-dlp .info.json (identity + low-reliability source tags)
- MusicBrainz recording genres/tags (no API key; rate-limited)
- Last.fm track top tags (optional LASTFM_API_KEY)
- Discogs release/master genres/styles (optional DISCOGS_TOKEN)

The collector does NOT assign taxonomy labels. It only gathers evidence for the
separate LLM weak-labeling stage.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import unicodedata
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote

import requests

try:
    import jsonschema
except ImportError:  # pragma: no cover - friendly runtime error
    jsonschema = None


SCHEMA_VERSION = "0.1.0"
TAXONOMY_VERSION = "0.1.0"
DEFAULT_USER_AGENT = "ytm-edm-evidence-collector/0.1 (personal music metadata project)"


# ---------------------------------------------------------------------------
# Normalization / matching
# ---------------------------------------------------------------------------


def normalize_text(value: str | None) -> str:
    if not value:
        return ""
    value = unicodedata.normalize("NFKD", value)
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = value.casefold()
    value = value.replace("&", " and ")
    value = re.sub(r"\b(feat(?:uring)?|ft)\.?\b", " feat ", value)
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return " ".join(value.split())


def similarity(a: str | None, b: str | None) -> float:
    return SequenceMatcher(None, normalize_text(a), normalize_text(b)).ratio()


def first_nonempty(*values: Any) -> Any:
    for value in values:
        if value not in (None, "", [], {}):
            return value
    return None


def trim_list(values: Iterable[Any], limit: int = 20) -> list[Any]:
    result = []
    for value in values:
        if value in result:
            continue
        result.append(value)
        if len(result) >= limit:
            break
    return result


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


@dataclass
class Track:
    track_id: str
    artist: str
    title: str
    album: str | None = None
    release_date: str | None = None
    duration_seconds: float | None = None
    bpm: float | None = None
    youtube_video_id: str | None = None

    def to_schema(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "artist": self.artist,
            "title": self.title,
            "album": self.album,
            "release_date": self.release_date,
            "duration_seconds": self.duration_seconds,
            "bpm": self.bpm,
            "youtube_video_id": self.youtube_video_id,
        }


@dataclass
class Evidence:
    id: str
    source: str
    type: str
    claim: str
    source_item_id: str | None = None
    source_url: str | None = None
    independence_group: str | None = None
    reliability: str = "unknown"

    def to_schema(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source": self.source,
            "type": self.type,
            "claim": self.claim,
            "source_item_id": self.source_item_id,
            "source_url": self.source_url,
            "independence_group": self.independence_group,
            "reliability": self.reliability,
        }


@dataclass
class CollectionResult:
    track: Track
    evidence: list[Evidence] = field(default_factory=list)

    def add(self, *items: Evidence) -> None:
        existing = {item.id for item in self.evidence}
        for item in items:
            if item.id not in existing:
                self.evidence.append(item)
                existing.add(item.id)

    def to_schema(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "taxonomy_version": TAXONOMY_VERSION,
            "track": self.track.to_schema(),
            "evidence": [item.to_schema() for item in self.evidence],
        }


# ---------------------------------------------------------------------------
# yt-dlp local metadata
# ---------------------------------------------------------------------------


def load_ytdlp_track(path: Path) -> tuple[Track, list[Evidence]]:
    data = json.loads(path.read_text(encoding="utf-8"))

    video_id = str(data.get("id") or "").strip() or None
    artist = first_nonempty(
        data.get("artist"),
        data.get("creator"),
        data.get("uploader"),
        data.get("channel"),
    )
    title = first_nonempty(data.get("track"), data.get("title"))

    if not artist or not title:
        raise ValueError(f"Could not derive artist/title from {path}")

    track_id = video_id or path.name.removesuffix(".info.json")
    release_date = first_nonempty(data.get("release_date"), data.get("upload_date"))
    if isinstance(release_date, str) and len(release_date) == 8 and release_date.isdigit():
        release_date = f"{release_date[:4]}-{release_date[4:6]}-{release_date[6:8]}"

    track = Track(
        track_id=track_id,
        artist=str(artist),
        title=str(title),
        album=data.get("album"),
        release_date=release_date,
        duration_seconds=data.get("duration"),
        youtube_video_id=video_id,
    )

    evidence: list[Evidence] = []
    raw_tags = trim_list([str(x) for x in data.get("tags") or [] if str(x).strip()], 25)
    if raw_tags:
        evidence.append(
            Evidence(
                id="youtube_source_tags",
                source="YouTube via yt-dlp info.json",
                type="tags",
                claim="Source-provided YouTube tags: " + ", ".join(raw_tags),
                source_item_id=video_id,
                source_url=data.get("webpage_url"),
                independence_group=f"youtube:{video_id}" if video_id else "youtube:local",
                reliability="low",
            )
        )

    categories = trim_list([str(x) for x in data.get("categories") or [] if str(x).strip()], 10)
    if categories:
        evidence.append(
            Evidence(
                id="youtube_categories",
                source="YouTube via yt-dlp info.json",
                type="genre",
                claim="YouTube categories: " + ", ".join(categories),
                source_item_id=video_id,
                source_url=data.get("webpage_url"),
                independence_group=f"youtube:{video_id}" if video_id else "youtube:local",
                reliability="low",
            )
        )

    return track, evidence


# ---------------------------------------------------------------------------
# MusicBrainz
# ---------------------------------------------------------------------------


class MusicBrainzCollector:
    ROOT = "https://musicbrainz.org/ws/2"

    def __init__(self, session: requests.Session, user_agent: str, min_score: int = 85):
        self.session = session
        self.user_agent = user_agent
        self.min_score = min_score
        self._last_request_at = 0.0

    def _get(self, url: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        # MusicBrainz asks clients to stay at or below one request/second.
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < 1.05:
            time.sleep(1.05 - elapsed)
        response = self.session.get(
            url,
            params=params,
            headers={"User-Agent": self.user_agent, "Accept": "application/json"},
            timeout=20,
        )
        self._last_request_at = time.monotonic()
        response.raise_for_status()
        return response.json()

    def collect(self, track: Track) -> list[Evidence]:
        query = f'recording:"{track.title}" AND artist:"{track.artist}"'
        search = self._get(
            f"{self.ROOT}/recording/",
            {"query": query, "fmt": "json", "limit": 5},
        )
        candidates = search.get("recordings") or []
        if not candidates:
            return []

        ranked: list[tuple[float, dict[str, Any]]] = []
        for candidate in candidates:
            api_score = float(candidate.get("score") or 0)
            title_score = similarity(track.title, candidate.get("title"))
            artist_credit = "".join(
                str(part.get("name") or part.get("artist", {}).get("name") or "")
                + str(part.get("joinphrase") or "")
                for part in candidate.get("artist-credit") or []
                if isinstance(part, dict)
            )
            artist_score = similarity(track.artist, artist_credit)
            duration_score = 1.0
            if track.duration_seconds and candidate.get("length"):
                delta = abs(track.duration_seconds - (float(candidate["length"]) / 1000.0))
                duration_score = max(0.0, 1.0 - delta / 20.0)

            composite = (
                (api_score / 100.0) * 0.50
                + title_score * 0.25
                + artist_score * 0.20
                + duration_score * 0.05
            )
            ranked.append((composite, candidate))

        ranked.sort(key=lambda x: x[0], reverse=True)
        composite, best = ranked[0]
        api_score = int(best.get("score") or 0)
        if api_score < self.min_score or composite < 0.80:
            return []

        mbid = best["id"]
        detail = self._get(
            f"{self.ROOT}/recording/{mbid}",
            {"inc": "genres+tags+artist-credits+isrcs", "fmt": "json"},
        )
        group = f"musicbrainz:{mbid}"
        url = f"https://musicbrainz.org/recording/{mbid}"
        reliability = "high" if api_score >= 95 and composite >= 0.90 else "medium"

        evidence: list[Evidence] = []
        genres = detail.get("genres") or []
        if genres:
            rendered = ", ".join(
                f"{item.get('name')} (votes={item.get('count', 0)})" for item in genres[:15]
            )
            evidence.append(
                Evidence(
                    id="musicbrainz_recording_genres",
                    source="MusicBrainz recording",
                    type="genre",
                    claim=f"Recording genres: {rendered}. Match score={api_score}; composite identity score={composite:.3f}.",
                    source_item_id=mbid,
                    source_url=url,
                    independence_group=group,
                    reliability=reliability,
                )
            )

        tags = detail.get("tags") or []
        if tags:
            tags = sorted(tags, key=lambda x: int(x.get("count") or 0), reverse=True)
            rendered = ", ".join(
                f"{item.get('name')} (votes={item.get('count', 0)})" for item in tags[:20]
            )
            evidence.append(
                Evidence(
                    id="musicbrainz_recording_tags",
                    source="MusicBrainz recording",
                    type="tags",
                    claim=f"Recording community tags: {rendered}. Match score={api_score}; composite identity score={composite:.3f}.",
                    source_item_id=mbid,
                    source_url=url,
                    independence_group=group,
                    reliability=reliability,
                )
            )

        # Metadata is useful for identity/debugging even when no genre tags exist.
        isrcs = detail.get("isrcs") or []
        evidence.append(
            Evidence(
                id="musicbrainz_identity",
                source="MusicBrainz recording",
                type="release_metadata",
                claim=(
                    f"Matched recording '{detail.get('title')}' with MusicBrainz ID {mbid}; "
                    f"ISRCs={isrcs or 'none'}; MusicBrainz search score={api_score}; "
                    f"composite identity score={composite:.3f}."
                ),
                source_item_id=mbid,
                source_url=url,
                independence_group=group,
                reliability=reliability,
            )
        )
        return evidence


# ---------------------------------------------------------------------------
# Last.fm
# ---------------------------------------------------------------------------


class LastFmCollector:
    ROOT = "https://ws.audioscrobbler.com/2.0/"

    def __init__(self, session: requests.Session, api_key: str):
        self.session = session
        self.api_key = api_key

    def collect(self, track: Track) -> list[Evidence]:
        response = self.session.get(
            self.ROOT,
            params={
                "method": "track.getTopTags",
                "artist": track.artist,
                "track": track.title,
                "autocorrect": 1,
                "api_key": self.api_key,
                "format": "json",
            },
            timeout=20,
        )
        response.raise_for_status()
        data = response.json()
        if "error" in data:
            return []

        toptags = (data.get("toptags") or {}).get("tag") or []
        if isinstance(toptags, dict):
            toptags = [toptags]
        parsed = []
        for item in toptags:
            name = str(item.get("name") or "").strip()
            if not name:
                continue
            count = item.get("count")
            try:
                count = int(count)
            except (TypeError, ValueError):
                count = 0
            parsed.append((name, count))
        parsed.sort(key=lambda x: x[1], reverse=True)
        parsed = parsed[:20]
        if not parsed:
            return []

        claim = ", ".join(f"{name} (count={count})" for name, count in parsed)
        return [
            Evidence(
                id="lastfm_track_top_tags",
                source="Last.fm track top tags",
                type="tags",
                claim="Track-level community tags ordered by popularity: " + claim,
                source_item_id=None,
                source_url=None,
                independence_group=f"lastfm:{normalize_text(track.artist)}:{normalize_text(track.title)}",
                reliability="medium",
            )
        ]


# ---------------------------------------------------------------------------
# Discogs
# ---------------------------------------------------------------------------


class DiscogsCollector:
    ROOT = "https://api.discogs.com"

    def __init__(self, session: requests.Session, token: str, user_agent: str):
        self.session = session
        self.token = token
        self.user_agent = user_agent

    def _get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        params = dict(params or {})
        params["token"] = self.token
        response = self.session.get(
            self.ROOT + path,
            params=params,
            headers={"User-Agent": self.user_agent},
            timeout=20,
        )
        response.raise_for_status()
        return response.json()

    @staticmethod
    def _tracklist_score(track: Track, payload: dict[str, Any]) -> float:
        best = 0.0
        for item in payload.get("tracklist") or []:
            candidate = item.get("title")
            best = max(best, similarity(track.title, candidate))
        return best

    def collect(self, track: Track) -> list[Evidence]:
        # Search broadly for release/master candidates, then require the target
        # track title to appear in the fetched tracklist. This prevents a release
        # style from being attached merely because the artist is associated with it.
        search = self._get(
            "/database/search",
            {
                "artist": track.artist,
                "q": track.title,
                "type": "master",
                "per_page": 8,
                "page": 1,
            },
        )
        results = search.get("results") or []
        if not results:
            search = self._get(
                "/database/search",
                {
                    "artist": track.artist,
                    "q": track.title,
                    "type": "release",
                    "per_page": 8,
                    "page": 1,
                },
            )
            results = search.get("results") or []

        best: tuple[float, dict[str, Any], str] | None = None
        for result in results[:8]:
            result_type = result.get("type")
            result_id = result.get("id")
            if not result_type or not result_id:
                continue
            if result_type == "master":
                payload = self._get(f"/masters/{result_id}")
            elif result_type == "release":
                payload = self._get(f"/releases/{result_id}")
            else:
                continue
            track_score = self._tracklist_score(track, payload)
            if track_score < 0.90:
                continue
            # Artist match is release-level; keep a modest requirement.
            artist_names = " ".join(
                str(a.get("name") or "") for a in payload.get("artists") or []
            )
            artist_score = similarity(track.artist, artist_names)
            score = track_score * 0.75 + artist_score * 0.25
            if best is None or score > best[0]:
                best = (score, payload, str(result_type))

        if best is None:
            return []

        score, payload, result_type = best
        item_id = str(payload.get("id"))
        group = f"discogs:{result_type}:{item_id}"
        url = payload.get("uri") or payload.get("resource_url")
        genres = [str(x) for x in payload.get("genres") or []]
        styles = [str(x) for x in payload.get("styles") or []]
        evidence: list[Evidence] = []

        if genres:
            evidence.append(
                Evidence(
                    id="discogs_release_genres",
                    source=f"Discogs {result_type}",
                    type="genre",
                    claim=f"Discogs release/master genres: {', '.join(genres)}. Tracklist identity score={score:.3f}.",
                    source_item_id=item_id,
                    source_url=url,
                    independence_group=group,
                    reliability="medium",
                )
            )
        if styles:
            evidence.append(
                Evidence(
                    id="discogs_release_styles",
                    source=f"Discogs {result_type}",
                    type="style",
                    claim=f"Discogs release/master styles: {', '.join(styles)}. Tracklist identity score={score:.3f}.",
                    source_item_id=item_id,
                    source_url=url,
                    independence_group=group,
                    reliability="medium",
                )
            )
        return evidence


# ---------------------------------------------------------------------------
# Validation / CLI
# ---------------------------------------------------------------------------


def validate_output(data: dict[str, Any], schema_path: Path | None) -> None:
    if schema_path is None:
        return
    if jsonschema is None:
        raise RuntimeError("Install jsonschema to validate output: pip install jsonschema")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator(schema).validate(data)


def build_track_from_args(args: argparse.Namespace) -> tuple[Track, list[Evidence]]:
    if args.info_json:
        track, local_evidence = load_ytdlp_track(Path(args.info_json))
        # Explicit CLI values override sidecar values.
        if args.track_id:
            track.track_id = args.track_id
        if args.artist:
            track.artist = args.artist
        if args.title:
            track.title = args.title
        if args.album:
            track.album = args.album
        if args.duration is not None:
            track.duration_seconds = args.duration
        if args.bpm is not None:
            track.bpm = args.bpm
        return track, local_evidence

    if not args.artist or not args.title:
        raise ValueError("--artist and --title are required unless --info-json is supplied")
    track_id = args.track_id or args.youtube_id or f"{normalize_text(args.artist)}::{normalize_text(args.title)}"
    return (
        Track(
            track_id=track_id,
            artist=args.artist,
            title=args.title,
            album=args.album,
            release_date=args.release_date,
            duration_seconds=args.duration,
            bpm=args.bpm,
            youtube_video_id=args.youtube_id,
        ),
        [],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--info-json", help="yt-dlp .info.json for the track")
    parser.add_argument("--track-id")
    parser.add_argument("--artist")
    parser.add_argument("--title")
    parser.add_argument("--album")
    parser.add_argument("--release-date")
    parser.add_argument("--duration", type=float)
    parser.add_argument("--bpm", type=float)
    parser.add_argument("--youtube-id")
    parser.add_argument("--output", "-o", required=True, help="Output weak-label input JSON")
    parser.add_argument(
        "--schema",
        default=str(Path(__file__).with_name("weak_label_input.schema.json")),
        help="Input JSON Schema used to validate output",
    )
    parser.add_argument(
        "--user-agent",
        default=os.environ.get("MUSIC_METADATA_USER_AGENT", DEFAULT_USER_AGENT),
    )
    parser.add_argument("--no-musicbrainz", action="store_true")
    parser.add_argument("--no-lastfm", action="store_true")
    parser.add_argument("--no-discogs", action="store_true")
    return parser.parse_args()


def safe_collect(name: str, fn) -> list[Evidence]:
    try:
        return fn()
    except requests.RequestException as exc:
        print(f"warning: {name} request failed: {exc}", file=sys.stderr)
        return []
    except Exception as exc:
        print(f"warning: {name} collector failed: {exc}", file=sys.stderr)
        return []


def main() -> int:
    args = parse_args()
    track, local_evidence = build_track_from_args(args)
    result = CollectionResult(track=track)
    result.add(*local_evidence)

    session = requests.Session()

    if not args.no_musicbrainz:
        mb = MusicBrainzCollector(session, args.user_agent)
        result.add(*safe_collect("MusicBrainz", lambda: mb.collect(track)))

    lastfm_key = os.environ.get("LASTFM_API_KEY")
    if not args.no_lastfm and lastfm_key:
        lf = LastFmCollector(session, lastfm_key)
        result.add(*safe_collect("Last.fm", lambda: lf.collect(track)))
    elif not args.no_lastfm:
        print("note: LASTFM_API_KEY not set; skipping Last.fm", file=sys.stderr)

    discogs_token = os.environ.get("DISCOGS_TOKEN")
    if not args.no_discogs and discogs_token:
        dg = DiscogsCollector(session, discogs_token, args.user_agent)
        result.add(*safe_collect("Discogs", lambda: dg.collect(track)))
    elif not args.no_discogs:
        print("note: DISCOGS_TOKEN not set; skipping Discogs", file=sys.stderr)

    output = result.to_schema()
    schema_path = Path(args.schema) if args.schema else None
    if schema_path and schema_path.exists():
        validate_output(output, schema_path)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"wrote {out_path} with {len(result.evidence)} evidence items")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
