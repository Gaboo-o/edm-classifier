"""V2 audio downloader that can follow a live resolver manifest.

Consumes data/v2/audio_resolution/resolution_manifest.jsonl and downloads each
resolved unique video_id once. The manifest is rescanned on every poll, so it
is safe if the resolver appends to it and later compacts/rewrites it.
"""

from __future__ import annotations

import argparse
import json
import re
import signal
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yt_dlp

DEFAULT_RESOLUTION_MANIFEST = Path("data/v2/audio_resolution/resolution_manifest.jsonl")
DEFAULT_OUTPUT_DIR = Path("data/v2/audio")

PARTIAL_SUFFIXES = (".part", ".ytdl", ".tmp", ".temp")

TERMINAL_PATTERNS = (
    ("age_restricted", re.compile(r"age[- ]?restricted|confirm your age|age restriction", re.I)),
    ("private", re.compile(r"private video|video is private", re.I)),
    ("members_only", re.compile(r"members[- ]only|members only", re.I)),
    ("deleted", re.compile(r"video (?:has been )?removed|deleted video", re.I)),
    ("region_blocked", re.compile(r"not available in your country|geo restriction", re.I)),
    ("authentication_required", re.compile(r"sign in to view|authentication required|login required", re.I)),
)

GLOBAL_PATTERNS = (
    re.compile(r"sign in to confirm you(?:'|’)re not a bot", re.I),
    re.compile(r"this content isn(?:'|’)t available, try again later", re.I),
    re.compile(r"http error 429|too many requests|rate limit", re.I),
)

TRANSIENT_PATTERNS = (
    re.compile(r"http error 403", re.I),
    re.compile(r"connection reset|connection aborted|connection refused", re.I),
    re.compile(r"timed? out|timeout", re.I),
    re.compile(r"fragment .* unavailable|fragment .* error", re.I),
    re.compile(r"remote end closed connection", re.I),
    re.compile(r"temporary failure|temporarily unavailable", re.I),
)


def load_jsonl_tolerant(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for i, raw in enumerate(lines):
        line = raw.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            if i == len(lines) - 1:
                continue
            raise
        if isinstance(value, dict):
            rows.append(value)
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def resolved_by_video(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_candidate: dict[str, dict[str, Any]] = {}
    for row in rows:
        cid = row.get("candidate_id")
        if isinstance(cid, str):
            by_candidate[cid] = row
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in by_candidate.values():
        if row.get("status") != "resolved":
            continue
        vid = row.get("video_id")
        if isinstance(vid, str) and vid:
            grouped[vid].append(row)
    return grouped


def physical_audio_file(files_dir: Path, video_id: str) -> Path | None:
    found = []
    for path in files_dir.glob(f"{video_id}.*"):
        if not path.is_file():
            continue
        lower = path.name.casefold()
        if any(lower.endswith(suffix) for suffix in PARTIAL_SUFFIXES):
            continue
        found.append(path)
    if not found:
        return None
    return max(found, key=lambda p: p.stat().st_size)


def classify_error(message: str) -> tuple[str, str]:
    for code, pattern in TERMINAL_PATTERNS:
        if pattern.search(message):
            return "terminal", code
    for pattern in GLOBAL_PATTERNS:
        if pattern.search(message):
            return "global", "youtube_rate_limit"
    for pattern in TRANSIENT_PATTERNS:
        if pattern.search(message):
            return "transient", "transient_download_error"
    if re.search(r"video unavailable|this video is unavailable", message, re.I):
        return "terminal", "unavailable"
    return "transient", "unclassified_download_error"


def load_state(path: Path) -> dict[str, dict[str, Any]]:
    state: dict[str, dict[str, Any]] = {}
    for row in load_jsonl_tolerant(path):
        vid = row.get("video_id")
        if isinstance(vid, str):
            state[vid] = row
    return state


def ydl_options(files_dir: Path, socket_timeout: int) -> dict[str, Any]:
    return {
        "format": "bestaudio/best",
        "outtmpl": str(files_dir / "%(id)s.%(ext)s"),
        "noplaylist": True,
        "continuedl": True,
        "overwrites": False,
        "retries": 2,
        "fragment_retries": 2,
        "socket_timeout": socket_timeout,
        "concurrent_fragment_downloads": 1,
        "postprocessors": [],
    }


def download_once(video_id: str, files_dir: Path, socket_timeout: int) -> tuple[bool, str | None]:
    url = f"https://www.youtube.com/watch?v={video_id}"
    try:
        with yt_dlp.YoutubeDL(ydl_options(files_dir, socket_timeout)) as ydl:
            ydl.download([url])
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    if physical_audio_file(files_dir, video_id) is None:
        return False, "yt-dlp returned without a completed audio file"
    return True, None


def union_labels(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for row in rows:
        labels = row.get("labels")
        if not isinstance(labels, list):
            continue
        for label in labels:
            if not isinstance(label, dict) or not isinstance(label.get("id"), str):
                continue
            gid = label["id"]
            old = best.get(gid)
            if old is None:
                best[gid] = dict(label)
                continue
            new_c = label.get("confidence")
            old_c = old.get("confidence")
            if isinstance(new_c, (int, float)) and (not isinstance(old_c, (int, float)) or new_c > old_c):
                best[gid] = dict(label)
    return [best[k] for k in sorted(best)]


def rebuild_audio_manifest(
    resolution_manifest: Path,
    state: dict[str, dict[str, Any]],
    files_dir: Path,
    output_path: Path,
) -> list[dict[str, Any]]:
    grouped = resolved_by_video(load_jsonl_tolerant(resolution_manifest))
    rows = []
    for vid, resolution_rows in sorted(grouped.items()):
        audio = physical_audio_file(files_dir, vid)
        if audio is not None:
            status = "downloaded"
        else:
            status = state.get(vid, {}).get("status", "pending")
        rows.append({
            "video_id": vid,
            "status": status,
            "audio_path": str(audio) if audio else None,
            "candidate_ids": sorted({r["candidate_id"] for r in resolution_rows if isinstance(r.get("candidate_id"), str)}),
            "labels": union_labels(resolution_rows),
            "resolution_sources": sorted({str(r.get("resolution_source", "legacy_song")) for r in resolution_rows}),
            "artists": sorted({r["artist"] for r in resolution_rows if isinstance(r.get("artist"), str)}),
            "titles": sorted({r["title"] for r in resolution_rows if isinstance(r.get("title"), str)}),
        })
    write_jsonl(output_path, rows)
    return rows


def write_report(
    report_path: Path,
    resolution_rows: list[dict[str, Any]],
    grouped: dict[str, list[dict[str, Any]]],
    audio_rows: list[dict[str, Any]],
    *,
    successful: int,
    attempts: int,
    transient_retries: int,
    global_cooldowns: int,
    started: float,
    args: argparse.Namespace,
) -> None:
    report = {
        "resolution_manifest_records_seen": len(resolution_rows),
        "unique_resolved_video_ids_seen": len(grouped),
        "audio_manifest_records": len(audio_rows),
        "audio_status_counts": dict(Counter(row.get("status", "unknown") for row in audio_rows)),
        "this_run": {
            "successful_downloads": successful,
            "yt_dlp_attempts": attempts,
            "transient_retries": transient_retries,
            "global_cooldowns": global_cooldowns,
            "elapsed_seconds": round(time.monotonic() - started, 2),
        },
        "configuration": {
            "follow": args.follow,
            "poll_seconds": args.poll_seconds,
            "pacing_seconds": args.sleep,
            "transient_retries": args.transient_retries,
            "workers": 1,
            "concurrent_fragments": 1,
            "format": "bestaudio/best",
            "transcode": False,
        },
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download V2 audio while optionally following a live resolver.")
    p.add_argument("--resolution-manifest", type=Path, default=DEFAULT_RESOLUTION_MANIFEST)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--follow", action="store_true", help="Poll for newly resolved IDs until Ctrl+C.")
    p.add_argument("--poll-seconds", type=float, default=30.0)
    p.add_argument("--sleep", type=float, default=12.0, help="Pacing between ordinary video attempts.")
    p.add_argument("--transient-retries", type=int, default=2)
    p.add_argument("--socket-timeout", type=int, default=30)
    p.add_argument("--max-downloads", type=int, help="Smoke-test cap on successful downloads.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    files_dir = args.output_dir / "files"
    state_path = args.output_dir / "download_state.jsonl"
    errors_path = args.output_dir / "download_errors.jsonl"
    audio_manifest_path = args.output_dir / "audio_manifest.jsonl"
    report_path = args.output_dir / "download_report.json"
    files_dir.mkdir(parents=True, exist_ok=True)

    state = load_state(state_path)
    stop = False

    def request_stop(signum: int, frame: Any) -> None:
        nonlocal stop
        stop = True
        print("\nStop requested; finishing current operation...")

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    cooldown_sequence = (45, 20, 20, 20, 20, 20, 20, 20, 20, 20)
    cooldown_stage = 0
    successful = 0
    attempts = 0
    transient_retries = 0
    global_cooldowns = 0
    started = time.monotonic()

    print("V2 audio downloader")
    print(f"  follow mode: {args.follow}")
    print(f"  pacing:     {args.sleep:.1f}s")
    print(f"  files:      {files_dir}")

    while not stop:
        resolution_rows = load_jsonl_tolerant(args.resolution_manifest)
        grouped = resolved_by_video(resolution_rows)
        pending = []

        for vid in grouped:
            existing = physical_audio_file(files_dir, vid)
            if existing is not None:
                if state.get(vid, {}).get("status") != "downloaded":
                    record = {"video_id": vid, "status": "downloaded", "reason": "physical_audio_exists", "audio_path": str(existing), "timestamp": time.time()}
                    state[vid] = record
                    append_jsonl(state_path, record)
                continue
            if state.get(vid, {}).get("status") == "terminal":
                continue
            pending.append(vid)

        if pending:
            print(f"Resolved IDs available={len(grouped)}; pending={len(pending)}")

        for vid in pending:
            if stop:
                break
            if args.max_downloads is not None and successful >= args.max_downloads:
                stop = True
                break

            per_video_attempt = 0
            while not stop:
                per_video_attempt += 1
                attempts += 1
                print(f"Downloading {vid} (attempt {per_video_attempt})")
                ok, error = download_once(vid, files_dir, args.socket_timeout)

                if ok:
                    audio = physical_audio_file(files_dir, vid)
                    record = {"video_id": vid, "status": "downloaded", "reason": "success", "audio_path": str(audio) if audio else None, "attempt": per_video_attempt, "timestamp": time.time()}
                    state[vid] = record
                    append_jsonl(state_path, record)
                    successful += 1
                    cooldown_stage = 0
                    if args.sleep > 0:
                        time.sleep(args.sleep)
                    break

                text = error or "unknown yt-dlp error"
                kind, reason = classify_error(text)

                if kind == "terminal":
                    record = {"video_id": vid, "status": "terminal", "reason": reason, "error": text, "attempt": per_video_attempt, "timestamp": time.time()}
                    state[vid] = record
                    append_jsonl(state_path, record)
                    append_jsonl(errors_path, record)
                    print(f"  terminal: {reason}; no retry")
                    if args.sleep > 0:
                        time.sleep(args.sleep)
                    break

                if kind == "global":
                    wait_min = cooldown_sequence[min(cooldown_stage, len(cooldown_sequence) - 1)]
                    cooldown_stage += 1
                    global_cooldowns += 1
                    print(f"GLOBAL YouTube rate limit: waiting {wait_min} minutes; same video will be the probe.")
                    deadline = time.monotonic() + wait_min * 60
                    while not stop and time.monotonic() < deadline:
                        time.sleep(min(5.0, deadline - time.monotonic()))
                    continue

                if per_video_attempt <= args.transient_retries:
                    transient_retries += 1
                    print(f"  transient: {reason}; fresh retry")
                    time.sleep(min(30.0, 5.0 * per_video_attempt))
                    continue

                record = {"video_id": vid, "status": "transient_exhausted", "reason": reason, "error": text, "attempt": per_video_attempt, "timestamp": time.time()}
                state[vid] = record
                append_jsonl(state_path, record)
                append_jsonl(errors_path, record)
                print("  transient retries exhausted; will retry next program run")
                if args.sleep > 0:
                    time.sleep(args.sleep)
                break

        audio_rows = rebuild_audio_manifest(args.resolution_manifest, state, files_dir, audio_manifest_path)
        write_report(report_path, resolution_rows, grouped, audio_rows, successful=successful, attempts=attempts, transient_retries=transient_retries, global_cooldowns=global_cooldowns, started=started, args=args)

        if not args.follow or stop:
            break

        print(f"Caught up. Polling resolver again in {args.poll_seconds:.0f}s...")
        deadline = time.monotonic() + args.poll_seconds
        while not stop and time.monotonic() < deadline:
            time.sleep(min(1.0, deadline - time.monotonic()))

    resolution_rows = load_jsonl_tolerant(args.resolution_manifest)
    grouped = resolved_by_video(resolution_rows)
    audio_rows = rebuild_audio_manifest(args.resolution_manifest, state, files_dir, audio_manifest_path)
    write_report(report_path, resolution_rows, grouped, audio_rows, successful=successful, attempts=attempts, transient_retries=transient_retries, global_cooldowns=global_cooldowns, started=started, args=args)

    print("\nDownloader stopped")
    print(f"  resolved IDs seen:   {len(grouped)}")
    print(f"  downloaded this run: {successful}")
    print(f"  status counts:       {dict(Counter(r.get('status', 'unknown') for r in audio_rows))}")
    print(f"  report:              {report_path}")


if __name__ == "__main__":
    main()
