"""Extract Discogs-EffNet embeddings from downloaded training audio.

Pipeline:
    audio file
      -> FFmpeg CLI (decode to mono float32 PCM at 16 kHz)
      -> TensorflowPredictEffnetDiscogs
      -> [num_patches, 1280] patch embeddings
      -> mean across patches
      -> [1280] track embedding

Input:
    data/audio/download_manifest.jsonl

Outputs:
    data/embeddings/
      pooled/<video_id>.npy
      patches/<video_id>.npy          # optional: --save-patches
      audio_embeddings.jsonl          # one row per unique audio/video_id
      embedding_manifest.jsonl        # one row per candidate/training record
      embedding_errors.jsonl
      embedding_report.json
      model_info.json

The script is resumable. Existing pooled embeddings are skipped unless
--overwrite is used. If --save-patches is enabled later, tracks that have a
pooled vector but no patch file are processed again to create the patch file.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_INPUT = Path("data/audio/download_manifest.jsonl")
DEFAULT_OUTPUT_DIR = Path("data/embeddings")
DEFAULT_MODEL = Path("data/models/discogs-effnet-bs64-1.pb")
DEFAULT_MODEL_METADATA = Path("data/models/discogs-effnet-bs64-1.json")

MODEL_URL = (
    "https://essentia.upf.edu/models/feature-extractors/"
    "discogs-effnet/discogs-effnet-bs64-1.pb"
)
MODEL_METADATA_URL = (
    "https://essentia.upf.edu/models/feature-extractors/"
    "discogs-effnet/discogs-effnet-bs64-1.json"
)

SAMPLE_RATE = 16000
EMBEDDING_DIM = 1280
MODEL_OUTPUT = "PartitionedCall:1"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            records.append(value)
    return records


def get_video_id(record: dict[str, Any]) -> str | None:
    local_audio = record.get("local_audio")
    if isinstance(local_audio, dict):
        video_id = local_audio.get("video_id")
        if isinstance(video_id, str) and video_id:
            return video_id

    audio_source = record.get("audio_source")
    if isinstance(audio_source, dict):
        video_id = audio_source.get("video_id")
        if isinstance(video_id, str) and video_id:
            return video_id
    return None


def get_audio_path(record: dict[str, Any]) -> Path | None:
    local_audio = record.get("local_audio")
    if not isinstance(local_audio, dict):
        return None
    raw_path = local_audio.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        return None
    return Path(raw_path)


def decode_audio_ffmpeg(path: Path, *, ffmpeg: str = "ffmpeg") -> np.ndarray:
    """Decode media to mono float32 PCM at 16 kHz using system FFmpeg."""
    command = [
        ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-loglevel", "error",
        "-i", str(path),
        "-vn",
        "-ac", "1",
        "-ar", str(SAMPLE_RATE),
        "-f", "f32le",
        "-acodec", "pcm_f32le",
        "pipe:1",
    ]
    completed = subprocess.run(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False
    )
    if completed.returncode != 0:
        stderr = completed.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            f"FFmpeg decode failed with exit code {completed.returncode}: {stderr}"
        )
    if not completed.stdout:
        raise RuntimeError("FFmpeg produced an empty decoded waveform")
    audio = np.frombuffer(completed.stdout, dtype="<f4").copy()
    if audio.size == 0:
        raise RuntimeError("FFmpeg produced zero audio samples")
    if not np.isfinite(audio).all():
        raise RuntimeError("Decoded waveform contains NaN or infinity")
    return audio


def download_file(url: str, destination: Path, *, minimum_bytes: int = 1) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=destination.name + ".",
        suffix=".part",
        dir=destination.parent,
        delete=False,
    ) as tmp:
        temp_path = Path(tmp.name)

    try:
        print(f"Downloading {url}")
        with urllib.request.urlopen(url, timeout=120) as response:
            with temp_path.open("wb") as handle:
                shutil.copyfileobj(response, handle)
        size = temp_path.stat().st_size
        if size < minimum_bytes:
            raise RuntimeError(f"Downloaded file is unexpectedly small: {size} bytes")
        os.replace(temp_path, destination)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def ensure_model(model_path: Path, metadata_path: Path, *, auto_download: bool) -> None:
    if not model_path.exists():
        if not auto_download:
            raise FileNotFoundError(
                f"Discogs-EffNet model not found: {model_path}\n"
                f"Download it from:\n{MODEL_URL}"
            )
        download_file(MODEL_URL, model_path, minimum_bytes=10_000_000)

    if not metadata_path.exists() and auto_download:
        try:
            download_file(MODEL_METADATA_URL, metadata_path, minimum_bytes=1000)
        except Exception as exc:
            print(f"warning: could not download model metadata: {exc}")


def atomic_save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=path.stem + ".",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    ) as tmp:
        temp_path = Path(tmp.name)
    try:
        with temp_path.open("wb") as handle:
            np.save(handle, array, allow_pickle=False)
        os.replace(temp_path, path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def validate_embedding_matrix(embeddings: np.ndarray, *, video_id: str) -> np.ndarray:
    matrix = np.asarray(embeddings, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"{video_id}: expected 2-D embedding matrix, got shape {matrix.shape}")
    if matrix.shape[0] < 1:
        raise ValueError(f"{video_id}: model returned zero patches")
    if matrix.shape[1] != EMBEDDING_DIM:
        raise ValueError(
            f"{video_id}: expected embedding dimension {EMBEDDING_DIM}, got shape {matrix.shape}"
        )
    if not np.isfinite(matrix).all():
        raise ValueError(f"{video_id}: patch embeddings contain NaN or infinity")
    return matrix


def pooled_file_is_valid(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        array = np.load(path, mmap_mode="r", allow_pickle=False)
    except Exception:
        return False
    return array.shape == (EMBEDDING_DIM,) and array.dtype.kind in {"f", "i", "u"}


def rebuild_manifests(
    *,
    input_records: list[dict[str, Any]],
    pooled_dir: Path,
    patches_dir: Path,
    audio_index_path: Path,
    candidate_manifest_path: Path,
) -> tuple[int, int]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in input_records:
        video_id = get_video_id(record)
        if video_id:
            grouped[video_id].append(record)

    unique_written = 0
    with audio_index_path.open("w", encoding="utf-8") as audio_handle:
        for video_id, group in grouped.items():
            pooled_path = pooled_dir / f"{video_id}.npy"
            if not pooled_file_is_valid(pooled_path):
                continue
            pooled = np.load(pooled_path, mmap_mode="r", allow_pickle=False)
            patch_path = patches_dir / f"{video_id}.npy"
            first = group[0]
            audio_path = get_audio_path(first)
            item = {
                "video_id": video_id,
                "audio_path": str(audio_path) if audio_path else None,
                "pooled_embedding": str(pooled_path),
                "embedding_dim": int(pooled.shape[0]),
                "patch_embedding": str(patch_path) if patch_path.exists() else None,
                "candidate_count": len(group),
                "candidate_ids": [
                    record.get("candidate_id")
                    for record in group
                    if isinstance(record.get("candidate_id"), str)
                ],
            }
            audio_handle.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")
            unique_written += 1

    candidate_written = 0
    with candidate_manifest_path.open("w", encoding="utf-8") as candidate_handle:
        for record in input_records:
            video_id = get_video_id(record)
            if not video_id:
                continue
            pooled_path = pooled_dir / f"{video_id}.npy"
            if not pooled_file_is_valid(pooled_path):
                continue
            patch_path = patches_dir / f"{video_id}.npy"
            output = dict(record)
            output["embedding"] = {
                "model": "discogs-effnet-bs64-1",
                "video_id": video_id,
                "pooled_path": str(pooled_path),
                "patches_path": str(patch_path) if patch_path.exists() else None,
                "dimensions": EMBEDDING_DIM,
                "pooling": "mean",
                "sample_rate": SAMPLE_RATE,
            }
            candidate_handle.write(
                json.dumps(output, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
            candidate_written += 1

    return unique_written, candidate_written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract mean-pooled Discogs-EffNet embeddings.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--model-metadata", type=Path, default=DEFAULT_MODEL_METADATA)
    parser.add_argument("--no-auto-download", action="store_true")
    parser.add_argument(
        "--save-patches",
        action="store_true",
        help="Also save the complete [patches, 1280] matrix for every track.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-tracks", type=int, help="Process at most N unique pending audio files.")
    parser.add_argument("--ffmpeg", default="ffmpeg", help="FFmpeg executable (default: ffmpeg).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_tracks is not None and args.max_tracks < 1:
        raise SystemExit("--max-tracks must be >= 1")

    try:
        from essentia.standard import TensorflowPredictEffnetDiscogs
    except ImportError as exc:
        raise SystemExit(
            "Essentia with TensorFlow support is required.\n"
            "Install it with:\n\n"
            "  python -m pip install essentia-tensorflow\n\n"
            "Then verify:\n\n"
            "  python -c \"from essentia.standard import TensorflowPredictEffnetDiscogs; print('OK')\"\n"
        ) from exc

    input_records = load_jsonl(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pooled_dir = args.output_dir / "pooled"
    patches_dir = args.output_dir / "patches"
    pooled_dir.mkdir(parents=True, exist_ok=True)
    if args.save_patches:
        patches_dir.mkdir(parents=True, exist_ok=True)

    errors_path = args.output_dir / "embedding_errors.jsonl"
    report_path = args.output_dir / "embedding_report.json"
    audio_index_path = args.output_dir / "audio_embeddings.jsonl"
    candidate_manifest_path = args.output_dir / "embedding_manifest.jsonl"
    model_info_path = args.output_dir / "model_info.json"

    ensure_model(args.model, args.model_metadata, auto_download=not args.no_auto_download)

    model = TensorflowPredictEffnetDiscogs(
        graphFilename=str(args.model),
        output=MODEL_OUTPUT,
    )

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    invalid_records = 0
    for record in input_records:
        video_id = get_video_id(record)
        if not video_id:
            invalid_records += 1
            continue
        grouped[video_id].append(record)

    unique_video_ids = list(grouped)
    pending: list[str] = []
    already_complete = 0

    for video_id in unique_video_ids:
        pooled_path = pooled_dir / f"{video_id}.npy"
        patch_path = patches_dir / f"{video_id}.npy"
        complete = (
            pooled_file_is_valid(pooled_path)
            and not args.overwrite
            and (not args.save_patches or patch_path.exists())
        )
        if complete:
            already_complete += 1
        else:
            pending.append(video_id)

    if args.max_tracks is not None:
        pending = pending[: args.max_tracks]

    print(f"Candidate records:          {len(input_records)}")
    print(f"Unique audio/video IDs:     {len(unique_video_ids)}")
    print(f"Already embedded:           {already_complete}")
    print(f"Processing this run:        {len(pending)}")
    print(f"Model:                      {args.model}")
    print(f"Sample rate:                {SAMPLE_RATE} Hz")
    print(f"Embedding dimension:        {EMBEDDING_DIM}")
    print(f"Save full patches:          {args.save_patches}")
    print()

    counters: Counter[str] = Counter()
    patch_counts: list[int] = []
    extraction_times: list[float] = []
    errors_this_run: list[dict[str, Any]] = []
    started = time.monotonic()

    for index, video_id in enumerate(pending, start=1):
        group = grouped[video_id]
        audio_path: Path | None = None
        for record in group:
            candidate_path = get_audio_path(record)
            if candidate_path is not None and candidate_path.exists():
                audio_path = candidate_path
                break

        print(f"[{index}/{len(pending)}] {video_id}")

        if audio_path is None:
            counters["missing_audio"] += 1
            errors_this_run.append({
                "video_id": video_id,
                "type": "missing_audio",
                "candidate_ids": [
                    record.get("candidate_id")
                    for record in group
                    if isinstance(record.get("candidate_id"), str)
                ],
            })
            print("  ERROR: local audio file not found")
            continue

        track_started = time.monotonic()
        try:
            audio = decode_audio_ffmpeg(
                audio_path,
                ffmpeg=args.ffmpeg,
            )

            matrix = validate_embedding_matrix(model(audio), video_id=video_id)
            pooled = np.asarray(matrix.mean(axis=0), dtype=np.float32)

            if pooled.shape != (EMBEDDING_DIM,):
                raise ValueError(f"mean-pooled embedding has unexpected shape {pooled.shape}")
            if not np.isfinite(pooled).all():
                raise ValueError("mean-pooled embedding contains NaN or infinity")

            atomic_save_npy(pooled_dir / f"{video_id}.npy", pooled)
            if args.save_patches:
                atomic_save_npy(
                    patches_dir / f"{video_id}.npy",
                    np.asarray(matrix, dtype=np.float32),
                )

            elapsed = time.monotonic() - track_started
            counters["embedded"] += 1
            patch_counts.append(int(matrix.shape[0]))
            extraction_times.append(elapsed)
            print(f"  -> {matrix.shape[0]} patches -> {pooled.shape} in {elapsed:.2f}s")

        except Exception as exc:
            counters["errors"] += 1
            errors_this_run.append({
                "video_id": video_id,
                "audio_path": str(audio_path),
                "type": type(exc).__name__,
                "error": str(exc),
                "candidate_ids": [
                    record.get("candidate_id")
                    for record in group
                    if isinstance(record.get("candidate_id"), str)
                ],
            })
            print(f"  ERROR: {type(exc).__name__}: {exc}")

    with errors_path.open("w", encoding="utf-8") as handle:
        for item in errors_this_run:
            handle.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")

    unique_embedded, candidate_embedded = rebuild_manifests(
        input_records=input_records,
        pooled_dir=pooled_dir,
        patches_dir=patches_dir,
        audio_index_path=audio_index_path,
        candidate_manifest_path=candidate_manifest_path,
    )

    total_elapsed = time.monotonic() - started

    model_info = {
        "name": "discogs-effnet-bs64-1",
        "framework": "tensorflow",
        "graph_filename": str(args.model),
        "metadata_filename": str(args.model_metadata) if args.model_metadata.exists() else None,
        "official_model_url": MODEL_URL,
        "output": MODEL_OUTPUT,
        "sample_rate": SAMPLE_RATE,
        "embedding_dimension": EMBEDDING_DIM,
        "pooling": "mean",
        "decoder": "ffmpeg_cli",
        "ffmpeg": args.ffmpeg,
        "full_patch_embeddings_saved": args.save_patches,
    }
    model_info_path.write_text(json.dumps(model_info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    report = {
        "input": str(args.input),
        "candidate_records": len(input_records),
        "invalid_records_without_video_id": invalid_records,
        "unique_audio_video_ids": len(unique_video_ids),
        "already_complete_before_run": already_complete,
        "requested_this_run": len(pending),
        "embedded_this_run": counters["embedded"],
        "missing_audio_this_run": counters["missing_audio"],
        "errors_this_run": counters["errors"],
        "unique_embeddings_present": unique_embedded,
        "candidate_records_with_embeddings": candidate_embedded,
        "candidate_records_without_embeddings": len(input_records) - candidate_embedded,
        "patch_statistics_this_run": {
            "minimum": min(patch_counts) if patch_counts else None,
            "maximum": max(patch_counts) if patch_counts else None,
            "mean": round(sum(patch_counts) / len(patch_counts), 3) if patch_counts else None,
        },
        "timing_this_run": {
            "elapsed_seconds": round(total_elapsed, 2),
            "mean_seconds_per_embedded_track": (
                round(sum(extraction_times) / len(extraction_times), 3)
                if extraction_times else None
            ),
        },
        "settings": {
            "sample_rate": SAMPLE_RATE,
            "embedding_dimension": EMBEDDING_DIM,
            "model_output": MODEL_OUTPUT,
            "pooling": "mean",
            "save_patches": args.save_patches,
            "decoder": "ffmpeg_cli",
        "ffmpeg": args.ffmpeg,
        },
        "outputs": {
            "pooled_dir": str(pooled_dir),
            "patches_dir": str(patches_dir) if args.save_patches or patches_dir.exists() else None,
            "audio_embeddings": str(audio_index_path),
            "embedding_manifest": str(candidate_manifest_path),
            "errors": str(errors_path),
            "model_info": str(model_info_path),
        },
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print()
    print("Embedding extraction complete")
    print(f"  embedded this run:         {counters['embedded']}")
    print(f"  missing audio:             {counters['missing_audio']}")
    print(f"  extraction errors:         {counters['errors']}")
    print(f"  unique embeddings present: {unique_embedded} / {len(unique_video_ids)}")
    print(f"  candidate records covered: {candidate_embedded} / {len(input_records)}")
    print(f"  elapsed:                   {total_elapsed / 3600:.2f} h")
    print()
    print(f"Manifest: {candidate_manifest_path}")
    print(f"Report:   {report_path}")
    print(f"Errors:   {errors_path}")


if __name__ == "__main__":
    main()
