from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yt_dlp
from yt_dlp.utils import DownloadError

DEFAULT_INPUT = Path("data/audio_resolution/audio_manifest.jsonl")
DEFAULT_OUTPUT_DIR = Path("data/audio")

RATE_LIMIT_MARKERS = (
    "sign in to confirm you're not a bot",
    "sign in to confirm you’re not a bot",
    "confirm you're not a bot",
    "confirm you’re not a bot",
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
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
            records.append(obj)
    return records


def extract_video_id(record: dict[str, Any]) -> str | None:
    src = record.get("audio_source")
    if not isinstance(src, dict):
        return None
    vid = src.get("video_id")
    return vid if isinstance(vid, str) and vid else None


def find_audio_file(files_dir: Path, video_id: str) -> Path | None:
    excluded = {
        ".json", ".part", ".ytdl", ".jpg", ".jpeg", ".png", ".webp",
        ".description", ".vtt", ".srt", ".temp",
    }
    files = []
    for p in files_dir.glob(f"{video_id}.*"):
        if not p.is_file() or p.name.endswith(".info.json"):
            continue
        if p.suffix.lower() in excluded:
            continue
        files.append(p)
    return max(files, key=lambda p: p.stat().st_size) if files else None


def is_rate_limit_error(exc: Exception) -> bool:
    msg = str(exc).casefold()
    return any(marker in msg for marker in RATE_LIMIT_MARKERS)


def build_ydl_options(files_dir: Path, archive_path: Path, quiet: bool,
                      http_retries: int, fragment_retries: int,
                      extractor_retries: int) -> dict[str, Any]:
    return {
        "format": "bestaudio/best",
        "outtmpl": str(files_dir / "%(id)s.%(ext)s"),
        "writeinfojson": True,
        "clean_infojson": True,
        "download_archive": str(archive_path),
        "noplaylist": True,
        "continuedl": True,
        "retries": http_retries,
        "fragment_retries": fragment_retries,
        "extractor_retries": extractor_retries,
        "file_access_retries": 3,
        "ignoreerrors": False,
        "quiet": quiet,
        "noprogress": quiet,
    }


def fresh_download_attempt(url: str, video_id: str, files_dir: Path,
                           archive_path: Path, quiet: bool,
                           http_retries: int, fragment_retries: int,
                           extractor_retries: int) -> Path:
    opts = build_ydl_options(
        files_dir, archive_path, quiet,
        http_retries, fragment_retries, extractor_retries,
    )
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
    if info is None:
        raise DownloadError("yt-dlp returned no metadata")
    audio_path = find_audio_file(files_dir, video_id)
    if audio_path is None:
        raise DownloadError("download completed but no local audio file was found")
    return audio_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--max-downloads", type=int)
    p.add_argument("--quiet-yt-dlp", action="store_true")

    p.add_argument("--fresh-attempts", type=int, default=3)
    p.add_argument("--retry-delay", type=float, default=5.0)
    p.add_argument("--http-retries", type=int, default=5)
    p.add_argument("--fragment-retries", type=int, default=5)
    p.add_argument("--extractor-retries", type=int, default=3)

    p.add_argument("--sleep-min", type=float, default=5.0)
    p.add_argument("--sleep-max", type=float, default=20.0)

    p.add_argument("--rate-limit-cooldown", type=float, default=3960.0)
    p.add_argument("--max-cooldowns", type=int, default=10)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.sleep_min < 0 or args.sleep_max < args.sleep_min:
        raise SystemExit("invalid sleep range")
    if args.fresh_attempts < 1:
        raise SystemExit("--fresh-attempts must be >= 1")

    records = load_jsonl(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    files_dir = args.output_dir / "files"
    files_dir.mkdir(parents=True, exist_ok=True)

    archive_path = args.output_dir / "downloaded.txt"
    manifest_path = args.output_dir / "download_manifest.jsonl"
    errors_path = args.output_dir / "download_errors.jsonl"
    report_path = args.output_dir / "download_report.json"

    by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        vid = extract_video_id(r)
        if vid:
            by_video[vid].append(r)

    unique_ids = list(by_video)
    already = {vid for vid in unique_ids if find_audio_file(files_dir, vid)}
    pending = [vid for vid in unique_ids if vid not in already]
    if args.max_downloads is not None:
        pending = pending[:args.max_downloads]

    print(f"Resolved training records: {len(records)}")
    print(f"Unique YouTube IDs:        {len(unique_ids)}")
    print(f"Already on disk:           {len(already)}")
    print(f"Attempting this run:       {len(pending)}")
    print(f"Inter-video pacing:        {args.sleep_min:.1f}-{args.sleep_max:.1f}s")
    print(f"Rate-limit cooldown:       {args.rate_limit_cooldown / 3600:.2f} h")
    print()

    counters = Counter()
    cooldowns = 0
    started = time.monotonic()
    stop_all = False

    for i, vid in enumerate(pending, 1):
        if stop_all:
            break

        url = f"https://www.youtube.com/watch?v={vid}"

        while True:
            print(f"[{i}/{len(pending)}] {vid}")
            errors = []
            downloaded = None
            rate_limited = False

            for attempt in range(1, args.fresh_attempts + 1):
                if attempt > 1:
                    delay = args.retry_delay * (2 ** (attempt - 2))
                    print(f"  fresh retry {attempt}/{args.fresh_attempts} in {delay:.1f}s")
                    time.sleep(delay)

                try:
                    downloaded = fresh_download_attempt(
                        url, vid, files_dir, archive_path, args.quiet_yt_dlp,
                        args.http_retries, args.fragment_retries,
                        args.extractor_retries,
                    )
                    counters["downloaded"] += 1
                    if attempt > 1:
                        counters["recovered_by_fresh_retry"] += 1
                    print(f"  -> {downloaded.name}")
                    break
                except Exception as exc:
                    if is_rate_limit_error(exc):
                        rate_limited = True
                        counters["rate_limit_events"] += 1
                        print("  YouTube bot/rate-limit check detected.")
                        break
                    errors.append({
                        "attempt": attempt,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    })
                    print(f"  attempt {attempt}/{args.fresh_attempts} failed: {exc}")

            if downloaded is not None:
                break

            if rate_limited:
                cooldowns += 1
                if cooldowns > args.max_cooldowns:
                    print(f"Exceeded --max-cooldowns={args.max_cooldowns}; stopping.")
                    stop_all = True
                    break
                print(
                    f"  Cooling down {args.rate_limit_cooldown / 3600:.2f} h "
                    f"({cooldowns}/{args.max_cooldowns}); retrying same video afterward."
                )
                time.sleep(args.rate_limit_cooldown)
                counters["cooldowns_completed"] += 1
                continue

            counters["persistent_failures"] += 1
            with errors_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "video_id": vid,
                    "url": url,
                    "candidate_ids": [
                        r.get("candidate_id") for r in by_video[vid]
                        if isinstance(r.get("candidate_id"), str)
                    ],
                    "attempts": errors,
                }, ensure_ascii=False, separators=(",", ":")) + "\n")
            break

        if stop_all:
            break

        if i < len(pending) and args.sleep_max > 0:
            delay = random.uniform(args.sleep_min, args.sleep_max)
            print(f"  pacing: sleeping {delay:.1f}s")
            time.sleep(delay)

    mapped = 0
    with manifest_path.open("w", encoding="utf-8") as f:
        for r in records:
            vid = extract_video_id(r)
            if not vid:
                continue
            audio = find_audio_file(files_dir, vid)
            if not audio:
                continue
            out = dict(r)
            out["local_audio"] = {
                "video_id": vid,
                "path": str(audio),
                "filename": audio.name,
                "extension": audio.suffix.lstrip("."),
                "bytes": audio.stat().st_size,
                "info_json": str(files_dir / f"{vid}.info.json"),
            }
            f.write(json.dumps(out, ensure_ascii=False, separators=(",", ":")) + "\n")
            mapped += 1

    present = {vid for vid in unique_ids if find_audio_file(files_dir, vid)}
    elapsed = time.monotonic() - started

    report = {
        "input": str(args.input),
        "resolved_training_records": len(records),
        "unique_video_ids": len(unique_ids),
        "already_present_before_run": len(already),
        "requested_this_run": len(pending),
        "downloaded_this_run": counters["downloaded"],
        "recovered_by_fresh_retry": counters["recovered_by_fresh_retry"],
        "persistent_failures_this_run": counters["persistent_failures"],
        "rate_limit_events_this_run": counters["rate_limit_events"],
        "cooldowns_completed_this_run": counters["cooldowns_completed"],
        "unique_audio_files_present": len(present),
        "candidate_records_with_audio": mapped,
        "candidate_records_without_audio": len(records) - mapped,
        "elapsed_seconds_this_run": round(elapsed, 2),
        "settings": {
            "sleep_min": args.sleep_min,
            "sleep_max": args.sleep_max,
            "rate_limit_cooldown": args.rate_limit_cooldown,
            "max_cooldowns": args.max_cooldowns,
            "fresh_attempts": args.fresh_attempts,
        },
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print()
    print("Download pass complete")
    print(f"  downloaded this run:      {counters['downloaded']}")
    print(f"  persistent failures:      {counters['persistent_failures']}")
    print(f"  rate-limit events:        {counters['rate_limit_events']}")
    print(f"  cooldowns completed:      {counters['cooldowns_completed']}")
    print(f"  unique audio files:       {len(present)} / {len(unique_ids)}")
    print(f"  training records w/audio: {mapped} / {len(records)}")
    print(f"  elapsed:                  {elapsed / 3600:.2f} h")


if __name__ == "__main__":
    main()
