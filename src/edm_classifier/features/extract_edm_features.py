"""Extract EDM rhythm descriptors and MERT-aligned section scores in one pass.

The extractor decodes each physical audio track once at a low analysis sample
rate, computes global rhythm descriptors plus 5-second activity descriptors,
and writes section indices that align directly with the existing MERT window
files. It never runs MERT itself.

Outputs (default ``data/features/edm``):
    rhythm/<video_id>.npy       fixed-size float32 rhythm vector
    sections/<video_id>.npz    per-window descriptors + selected bounds
    feature_manifest.jsonl
    feature_state.jsonl
    feature_errors.jsonl
    section_diagnostics.csv
    feature_report.json
    feature_info.json
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from pathlib import Path
from typing import Any

import librosa
import numpy as np

from .section_selection import (
    best_contiguous_bounds,
    center_fraction_bounds,
    robust_unit_scale,
)

EXTRACTOR_VERSION = "edm-rhythm-sections-v1"
DEFAULT_AUDIO_MANIFEST = Path("data/audio/audio_manifest.jsonl")
DEFAULT_AUDIO_FILES_DIR = Path("data/audio/files")
DEFAULT_MERT_WINDOWS_DIR = Path("data/embeddings/mert95m/windows")
DEFAULT_OUTPUT_DIR = Path("data/features/edm")
DEFAULT_SAMPLE_RATE = 11025
DEFAULT_WINDOW_SECONDS = 5.0
DEFAULT_ENERGY_SECONDS = 20.0
DEFAULT_TEMPO_BINS = 24
AUDIO_EXTENSIONS = (".webm", ".m4a", ".mp3", ".opus", ".wav", ".flac", ".ogg")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            line = raw.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if isinstance(row, dict):
                rows.append(row)
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def atomic_save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
    tmp.replace(path)


def atomic_save_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    tmp.replace(path)


def resolve_audio_path(row: dict[str, Any], files_dir: Path) -> Path | None:
    raw = row.get("audio_path")
    if isinstance(raw, str) and raw:
        path = Path(raw)
        if path.is_file():
            return path
    video_id = row.get("video_id")
    if not isinstance(video_id, str) or not video_id:
        return None
    for ext in AUDIO_EXTENSIONS:
        path = files_dir / f"{video_id}{ext}"
        if path.is_file():
            return path
    matches = [p for p in files_dir.glob(f"{video_id}.*") if p.is_file()]
    return sorted(matches)[0] if matches else None


def decode_audio(path: Path, sample_rate: int, ffmpeg: str) -> np.ndarray:
    command = [
        ffmpeg,
        "-v", "error",
        "-i", str(path),
        "-f", "f32le",
        "-ac", "1",
        "-ar", str(sample_rate),
        "pipe:1",
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if result.returncode != 0:
        message = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg failed ({result.returncode}): {message[-1000:]}")
    y = np.frombuffer(result.stdout, dtype="<f4").astype(np.float32, copy=False)
    if y.size == 0:
        raise RuntimeError("decoded audio is empty")
    if not np.isfinite(y).all():
        raise RuntimeError("decoded audio contains non-finite samples")
    return y


def _window_mean(values: np.ndarray, times: np.ndarray, window_count: int, window_seconds: float) -> np.ndarray:
    if values.size == 0:
        return np.zeros(window_count, dtype=np.float32)
    indices = np.floor(times / window_seconds).astype(np.int64)
    indices = np.clip(indices, 0, window_count - 1)
    sums = np.bincount(indices, weights=values.astype(np.float64), minlength=window_count)
    counts = np.bincount(indices, minlength=window_count)
    return np.divide(sums, counts, out=np.zeros(window_count, dtype=np.float64), where=counts > 0).astype(np.float32)


def _window_onset_density(onset_times: np.ndarray, window_count: int, window_seconds: float) -> np.ndarray:
    if onset_times.size == 0:
        return np.zeros(window_count, dtype=np.float32)
    indices = np.floor(onset_times / window_seconds).astype(np.int64)
    indices = np.clip(indices, 0, window_count - 1)
    counts = np.bincount(indices, minlength=window_count).astype(np.float32)
    return counts / float(window_seconds)


def _fold_tempo(bpm: float) -> float:
    if not np.isfinite(bpm) or bpm <= 0:
        return 0.0
    value = float(bpm)
    while value > 120.0:
        value /= 2.0
    while value < 60.0:
        value *= 2.0
    return value


def _safe_stats(x: np.ndarray, percentiles: tuple[float, ...]) -> list[float]:
    values = np.asarray(x, dtype=np.float64)
    if values.size == 0:
        return [0.0, 0.0, *([0.0] * len(percentiles))]
    return [float(values.mean()), float(values.std()), *[float(np.percentile(values, p)) for p in percentiles]]


def rhythm_feature_names(tempo_bins: int) -> list[str]:
    names = [
        "tempo_bpm",
        "tempo_folded_60_120",
        "beat_interval_mean_s",
        "beat_interval_std_s",
        "beat_interval_cv",
        "beat_regularity",
        "beat_density_hz",
        "onset_mean",
        "onset_std",
        "onset_p75",
        "onset_p90",
        "onset_p95",
        "rms_mean",
        "rms_std",
        "rms_p75",
        "rms_p90",
        "rms_p95",
        "rms_dynamic_range_p90_p10",
    ]
    names.extend(f"tempogram_logbin_{i:02d}" for i in range(tempo_bins))
    return names


def compute_features(
    y: np.ndarray,
    *,
    sample_rate: int,
    window_count: int,
    window_seconds: float,
    energy_seconds: float,
    tempo_bins: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    if window_count < 1:
        raise ValueError("window_count must be positive")

    hop_length = 512
    frame_length = 2048
    onset = librosa.onset.onset_strength(y=y, sr=sample_rate, hop_length=hop_length)
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length, center=True)[0]

    onset_times_frames = librosa.frames_to_time(np.arange(onset.size), sr=sample_rate, hop_length=hop_length)
    rms_times_frames = librosa.frames_to_time(np.arange(rms.size), sr=sample_rate, hop_length=hop_length)

    onset_times = librosa.onset.onset_detect(
        onset_envelope=onset,
        sr=sample_rate,
        hop_length=hop_length,
        units="time",
        backtrack=False,
    )
    onset_times = np.asarray(onset_times, dtype=np.float64)

    window_onset = _window_mean(onset, onset_times_frames, window_count, window_seconds)
    window_rms = _window_mean(rms, rms_times_frames, window_count, window_seconds)
    window_density = _window_onset_density(onset_times, window_count, window_seconds)

    activity_score = (
        0.50 * robust_unit_scale(window_rms)
        + 0.30 * robust_unit_scale(window_onset)
        + 0.20 * robust_unit_scale(window_density)
    ).astype(np.float32)

    center_start, center_end = center_fraction_bounds(window_count)
    energy_windows = max(1, int(round(energy_seconds / window_seconds)))
    energy_start, energy_end = best_contiguous_bounds(activity_score, block_windows=energy_windows)

    tempo_values = librosa.feature.tempo(
        onset_envelope=onset,
        sr=sample_rate,
        hop_length=hop_length,
        aggregate=np.median,
    )
    tempo_bpm = float(np.asarray(tempo_values).reshape(-1)[0]) if np.asarray(tempo_values).size else 0.0

    _, beat_frames = librosa.beat.beat_track(
        onset_envelope=onset,
        sr=sample_rate,
        hop_length=hop_length,
        sparse=True,
    )
    beat_times = librosa.frames_to_time(np.asarray(beat_frames), sr=sample_rate, hop_length=hop_length)
    intervals = np.diff(beat_times)
    if intervals.size:
        interval_mean = float(intervals.mean())
        interval_std = float(intervals.std())
        interval_cv = interval_std / interval_mean if interval_mean > 1e-12 else 0.0
    else:
        interval_mean = interval_std = interval_cv = 0.0
    beat_regularity = float(math.exp(-interval_cv)) if interval_cv >= 0 else 0.0
    duration_seconds = y.size / float(sample_rate)
    beat_density = float(len(beat_times) / duration_seconds) if duration_seconds > 0 else 0.0

    onset_stats = _safe_stats(onset, (75.0, 90.0, 95.0))
    rms_stats = _safe_stats(rms, (75.0, 90.0, 95.0))
    rms_p10 = float(np.percentile(rms, 10.0)) if rms.size else 0.0
    rms_dynamic = float(rms_stats[-2] - rms_p10)  # p90 - p10

    # A compact global tempogram signature on fixed log-spaced BPM bins.
    win_length = max(2, min(384, int(onset.size)))
    tempogram = librosa.feature.tempogram(
        onset_envelope=onset,
        sr=sample_rate,
        hop_length=hop_length,
        win_length=win_length,
    )
    global_tempogram = np.mean(tempogram, axis=1) if tempogram.size else np.zeros(1)
    tempo_freqs = librosa.tempo_frequencies(global_tempogram.size, sr=sample_rate, hop_length=hop_length)
    target_bpms = np.geomspace(60.0, 220.0, tempo_bins)
    valid = np.isfinite(tempo_freqs) & (tempo_freqs > 0)
    if np.any(valid):
        freqs = tempo_freqs[valid]
        powers = global_tempogram[valid]
        sampled = np.asarray([powers[int(np.argmin(np.abs(freqs - bpm)))] for bpm in target_bpms], dtype=np.float64)
    else:
        sampled = np.zeros(tempo_bins, dtype=np.float64)
    sampled = np.maximum(sampled, 0.0)
    denom = float(sampled.sum())
    if denom > 0:
        sampled /= denom

    rhythm = np.asarray(
        [
            tempo_bpm,
            _fold_tempo(tempo_bpm),
            interval_mean,
            interval_std,
            interval_cv,
            beat_regularity,
            beat_density,
            *onset_stats,
            *rms_stats,
            rms_dynamic,
            *sampled.tolist(),
        ],
        dtype=np.float32,
    )
    expected_dim = len(rhythm_feature_names(tempo_bins))
    if rhythm.shape != (expected_dim,):
        raise AssertionError(f"rhythm feature dimension mismatch: {rhythm.shape} vs {expected_dim}")
    if not np.isfinite(rhythm).all():
        raise RuntimeError("rhythm vector contains non-finite values")

    section = {
        "window_rms": window_rms,
        "window_onset_strength": window_onset,
        "window_onset_density": window_density,
        "activity_score": activity_score,
        "window_count": np.asarray(window_count, dtype=np.int32),
        "window_seconds": np.asarray(window_seconds, dtype=np.float32),
        "center30_start": np.asarray(center_start, dtype=np.int32),
        "center30_end": np.asarray(center_end, dtype=np.int32),
        "energy20_start": np.asarray(energy_start, dtype=np.int32),
        "energy20_end": np.asarray(energy_end, dtype=np.int32),
        "duration_seconds": np.asarray(duration_seconds, dtype=np.float32),
        "extractor_version": np.asarray(EXTRACTOR_VERSION),
    }
    return rhythm, section


def valid_outputs(rhythm_path: Path, section_path: Path, expected_windows: int, rhythm_dim: int) -> bool:
    try:
        r = np.load(rhythm_path, mmap_mode="r", allow_pickle=False)
        if r.shape != (rhythm_dim,) or r.dtype != np.float32 or not np.isfinite(r).all():
            return False
        with np.load(section_path, allow_pickle=False) as s:
            version = str(np.asarray(s["extractor_version"]).item())
            count = int(np.asarray(s["window_count"]).item())
            score = np.asarray(s["activity_score"])
            if version != EXTRACTOR_VERSION or count != expected_windows or score.shape != (expected_windows,):
                return False
            for key in ("center30_start", "center30_end", "energy20_start", "energy20_end"):
                int(np.asarray(s[key]).item())
        return True
    except Exception:
        return False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract EDM rhythm features and MERT-aligned section indices.")
    p.add_argument("--audio-manifest", type=Path, default=DEFAULT_AUDIO_MANIFEST)
    p.add_argument("--audio-files-dir", type=Path, default=DEFAULT_AUDIO_FILES_DIR)
    p.add_argument("--mert-windows-dir", type=Path, default=DEFAULT_MERT_WINDOWS_DIR)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE)
    p.add_argument("--window-seconds", type=float, default=DEFAULT_WINDOW_SECONDS)
    p.add_argument("--energy-seconds", type=float, default=DEFAULT_ENERGY_SECONDS)
    p.add_argument("--tempo-bins", type=int, default=DEFAULT_TEMPO_BINS)
    p.add_argument("--ffmpeg", default="ffmpeg")
    p.add_argument("--max-tracks", type=int, default=None)
    p.add_argument("--workers", type=int, default=1, help="Concurrent tracks; start with 2-4 on a CPU with enough RAM.")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.audio_manifest.is_file():
        raise SystemExit(f"missing audio manifest: {args.audio_manifest}")
    if not args.mert_windows_dir.is_dir():
        raise SystemExit(f"missing MERT windows directory: {args.mert_windows_dir}")

    rhythm_dir = args.output_dir / "rhythm"
    sections_dir = args.output_dir / "sections"
    rhythm_dir.mkdir(parents=True, exist_ok=True)
    sections_dir.mkdir(parents=True, exist_ok=True)
    state_path = args.output_dir / "feature_state.jsonl"
    error_path = args.output_dir / "feature_errors.jsonl"
    manifest_path = args.output_dir / "feature_manifest.jsonl"
    report_path = args.output_dir / "feature_report.json"
    info_path = args.output_dir / "feature_info.json"
    diagnostics_path = args.output_dir / "section_diagnostics.csv"

    feature_names = rhythm_feature_names(args.tempo_bins)
    info = {
        "extractor_version": EXTRACTOR_VERSION,
        "analysis_sample_rate": args.sample_rate,
        "window_seconds": args.window_seconds,
        "energy_seconds": args.energy_seconds,
        "activity_score": "0.50*rms + 0.30*onset_strength + 0.20*onset_density after per-track robust scaling",
        "rhythm_dimension": len(feature_names),
        "rhythm_feature_names": feature_names,
        "tempogram_bpm_range": [60.0, 220.0],
        "tempogram_bins": args.tempo_bins,
    }
    info_path.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")

    # Last row wins for older append-only audio manifests.
    by_id: dict[str, dict[str, Any]] = {}
    for row in load_jsonl(args.audio_manifest):
        video_id = row.get("video_id")
        if isinstance(video_id, str) and video_id:
            by_id[video_id] = row

    candidates: list[tuple[str, dict[str, Any], Path, Path, int]] = []
    for video_id in sorted(by_id):
        audio_path = resolve_audio_path(by_id[video_id], args.audio_files_dir)
        windows_path = args.mert_windows_dir / f"{video_id}.npy"
        if audio_path is None or not windows_path.is_file():
            continue
        a = np.load(windows_path, mmap_mode="r", allow_pickle=False)
        if a.ndim != 2 or a.shape[0] < 1:
            continue
        candidates.append((video_id, by_id[video_id], audio_path, windows_path, int(a.shape[0])))
    if args.max_tracks is not None:
        candidates = candidates[: args.max_tracks]
    if not candidates:
        raise SystemExit("no tracks have both usable audio and MERT windows")

    started = time.monotonic()
    status = Counter()

    def extract_candidate(candidate):
        video_id, _, audio_path, windows_path, window_count = candidate
        rhythm_path = rhythm_dir / f"{video_id}.npy"
        section_path = sections_dir / f"{video_id}.npz"
        if not args.overwrite and valid_outputs(rhythm_path, section_path, window_count, len(feature_names)):
            return {"kind": "skipped_valid", "video_id": video_id}
        track_started = time.monotonic()
        try:
            y = decode_audio(audio_path, args.sample_rate, args.ffmpeg)
            rhythm, section = compute_features(
                y,
                sample_rate=args.sample_rate,
                window_count=window_count,
                window_seconds=args.window_seconds,
                energy_seconds=args.energy_seconds,
                tempo_bins=args.tempo_bins,
            )
            atomic_save_npy(rhythm_path, rhythm)
            atomic_save_npz(section_path, **section)
            return {
                "kind": "extracted",
                "video_id": video_id,
                "window_count": window_count,
                "rhythm_path": str(rhythm_path),
                "section_path": str(section_path),
                "elapsed_seconds": round(time.monotonic() - track_started, 3),
            }
        except Exception as exc:
            return {
                "kind": "error",
                "video_id": video_id,
                "audio_path": str(audio_path),
                "windows_path": str(windows_path),
                "error": f"{type(exc).__name__}: {exc}",
            }

    workers = max(1, int(args.workers))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(extract_candidate, candidate) for candidate in candidates]
        for index, future in enumerate(as_completed(futures), 1):
            result = future.result()
            kind = result.pop("kind")
            status[kind] += 1
            if kind == "extracted":
                append_jsonl(state_path, {"status": "extracted", **result})
            elif kind == "error":
                append_jsonl(error_path, {"status": "error", **result})
            if index == 1 or index % 100 == 0 or index == len(candidates):
                print(
                    f"[{index}/{len(candidates)}] extracted={status['extracted']} "
                    f"skipped={status['skipped_valid']} errors={status['error']}"
                )

    manifest_rows = []
    diagnostic_rows = []
    for video_id, _, audio_path, windows_path, window_count in candidates:
        rhythm_path = rhythm_dir / f"{video_id}.npy"
        section_path = sections_dir / f"{video_id}.npz"
        if not valid_outputs(rhythm_path, section_path, window_count, len(feature_names)):
            continue
        with np.load(section_path, allow_pickle=False) as s:
            cs, ce = int(s["center30_start"]), int(s["center30_end"])
            es, ee = int(s["energy20_start"]), int(s["energy20_end"])
            duration = float(s["duration_seconds"])
            score = np.asarray(s["activity_score"], dtype=np.float32)
            rms = np.asarray(s["window_rms"], dtype=np.float32)
            onset = np.asarray(s["window_onset_strength"], dtype=np.float32)
            density = np.asarray(s["window_onset_density"], dtype=np.float32)
        rhythm = np.asarray(np.load(rhythm_path, allow_pickle=False), dtype=np.float32)
        manifest_rows.append({
            "video_id": video_id,
            "audio_path": str(audio_path),
            "mert_windows_path": str(windows_path),
            "window_count": window_count,
            "rhythm_path": str(rhythm_path),
            "section_path": str(section_path),
            "center30": [cs, ce],
            "energy20": [es, ee],
        })
        diagnostic_rows.append({
            "video_id": video_id,
            "duration_seconds": round(duration, 3),
            "window_count": window_count,
            "estimated_bpm": round(float(rhythm[0]), 3),
            "center30_start_seconds": round(cs * args.window_seconds, 3),
            "center30_end_seconds": round(min(ce * args.window_seconds, duration), 3),
            "energy20_start_seconds": round(es * args.window_seconds, 3),
            "energy20_end_seconds": round(min(ee * args.window_seconds, duration), 3),
            "energy20_mean_score": round(float(score[es:ee].mean()), 6),
            "track_mean_score": round(float(score.mean()), 6),
            "energy20_mean_rms": round(float(rms[es:ee].mean()), 6),
            "track_mean_rms": round(float(rms.mean()), 6),
            "energy20_mean_onset": round(float(onset[es:ee].mean()), 6),
            "track_mean_onset": round(float(onset.mean()), 6),
            "energy20_mean_onset_density": round(float(density[es:ee].mean()), 6),
            "track_mean_onset_density": round(float(density.mean()), 6),
        })

    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in manifest_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with diagnostics_path.open("w", encoding="utf-8", newline="") as handle:
        if diagnostic_rows:
            writer = csv.DictWriter(handle, fieldnames=list(diagnostic_rows[0]))
            writer.writeheader()
            writer.writerows(diagnostic_rows)

    report = {
        "extractor_version": EXTRACTOR_VERSION,
        "eligible_tracks": len(candidates),
        "complete_feature_tracks": len(manifest_rows),
        "status_counts": dict(status),
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "configuration": {
            "audio_manifest": str(args.audio_manifest),
            "audio_files_dir": str(args.audio_files_dir),
            "mert_windows_dir": str(args.mert_windows_dir),
            "output_dir": str(args.output_dir),
            "sample_rate": args.sample_rate,
            "window_seconds": args.window_seconds,
            "energy_seconds": args.energy_seconds,
            "tempo_bins": args.tempo_bins,
            "workers": max(1, int(args.workers)),
        },
        "outputs": {
            "rhythm_dir": str(rhythm_dir),
            "sections_dir": str(sections_dir),
            "manifest": str(manifest_path),
            "diagnostics": str(diagnostics_path),
            "feature_info": str(info_path),
        },
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
