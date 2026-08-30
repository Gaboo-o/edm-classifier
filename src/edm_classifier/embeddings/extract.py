"""Extract Discogs-EffNet embeddings for the unified audio corpus.

The unified corpus already contains the authoritative mean-pooled baseline in:

    data/embeddings/pooled/<video_id>.npy

For temporal/attention models this script can additionally persist the complete
ordered Discogs-EffNet patch sequence:

    data/embeddings/patches/<video_id>.npy      # [T, 1280], float32

Important invariants:
- one physical extraction per unique video_id
- full patch sequences are saved in temporal order; no subsampling is done here
- existing valid pooled embeddings are NEVER overwritten unless
  --overwrite-pooled is explicitly supplied
- existing valid patch embeddings are skipped unless --overwrite-patches is
  explicitly supplied
- the canonical candidate-level embedding manifest is NOT rebuilt or rewritten
- patch files are model-agnostic for any downstream model that uses the frozen
  Discogs-EffNet patch representation (scalar attention, gated attention,
  multi-head attention pooling, temporal CNN/RNN, or a transformer over patches)

Inputs:
    data/audio/audio_manifest.jsonl
    data/models/discogs-effnet-bs64-1.pb

Outputs:
    data/embeddings/
      pooled/<video_id>.npy          # preserved baseline; filled only if missing
      patches/<video_id>.npy         # full ordered [T,1280] sequence
      patch_manifest.jsonl           # one row per available physical recording
      patch_state.jsonl              # append-only extraction/resume state
      patch_errors.jsonl             # extraction failures
      patch_report.json              # current summary

Typical attention extraction:

    python -m edm_classifier.embeddings.extract --save-patches

Smoke test:

    python -m edm_classifier.embeddings.extract --save-patches --max-tracks 10
"""

from __future__ import annotations

import argparse
import json
import signal
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_AUDIO_MANIFEST = Path("data/audio/audio_manifest.jsonl")
DEFAULT_MODEL = Path("data/models/discogs-effnet-bs64-1.pb")
DEFAULT_OUTPUT_DIR = Path("data/embeddings")

EXPECTED_DIM = 1280
SAMPLE_RATE = 16000
EXTRACTOR_VERSION = "unified-patches-v2"


def load_jsonl_tolerant(path: Path) -> list[dict[str, Any]]:
    """Read complete JSONL records, tolerating a final partial line."""
    if not path.exists():
        return []

    rows: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        lines = handle.readlines()

    for index, raw in enumerate(lines):
        line = raw.strip()
        if not line:
            continue

        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines) - 1:
                continue
            raise

        if isinstance(value, dict):
            rows.append(value)

    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                row,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n"
        )
        handle.flush()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")

    with temp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )

    temp.replace(path)


def atomic_save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp")

    with temp.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)

    temp.replace(path)


def load_state(path: Path) -> dict[str, dict[str, Any]]:
    state: dict[str, dict[str, Any]] = {}

    for row in load_jsonl_tolerant(path):
        video_id = row.get("video_id")
        if isinstance(video_id, str):
            state[video_id] = row

    return state


def valid_pooled_embedding(path: Path) -> bool:
    if not path.is_file():
        return False

    try:
        array = np.load(path, mmap_mode="r", allow_pickle=False)
    except Exception:
        return False

    return (
        array.ndim == 1
        and array.shape == (EXPECTED_DIM,)
        and np.issubdtype(array.dtype, np.floating)
        and np.all(np.isfinite(array))
    )


def patch_shape(path: Path) -> tuple[int, int] | None:
    """Return a valid patch shape without loading the full sequence into RAM."""
    if not path.is_file():
        return None

    try:
        array = np.load(path, mmap_mode="r", allow_pickle=False)
    except Exception:
        return None

    if (
        array.ndim != 2
        or array.shape[0] < 1
        or array.shape[1] != EXPECTED_DIM
        or not np.issubdtype(array.dtype, np.floating)
    ):
        return None

    return int(array.shape[0]), int(array.shape[1])


def valid_patch_embedding(path: Path) -> bool:
    return patch_shape(path) is not None


def decode_audio_ffmpeg(
    audio_path: Path,
    *,
    ffmpeg: str,
    sample_rate: int,
) -> np.ndarray:
    """Decode arbitrary audio to mono float32 PCM using system FFmpeg."""
    command = [
        ffmpeg,
        "-v",
        "error",
        "-nostdin",
        "-i",
        str(audio_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "f32le",
        "pipe:1",
    ]

    process = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    if process.returncode != 0:
        message = process.stderr.decode(
            "utf-8",
            errors="replace",
        ).strip()
        raise RuntimeError(
            f"ffmpeg decode failed ({process.returncode}): {message}"
        )

    waveform = np.frombuffer(
        process.stdout,
        dtype="<f4",
    ).astype(np.float32, copy=True)

    if waveform.size == 0:
        raise RuntimeError("ffmpeg produced empty waveform")

    if not np.all(np.isfinite(waveform)):
        raise RuntimeError("decoded waveform contains non-finite values")

    return waveform


def create_effnet(model_path: Path):
    """Import Essentia lazily so dependency errors remain understandable."""
    try:
        from essentia.standard import TensorflowPredictEffnetDiscogs
    except Exception as exc:
        raise RuntimeError(
            "Could not import Essentia TensorflowPredictEffnetDiscogs. "
            "Use the same environment/dependencies as the pooled extractor."
        ) from exc

    return TensorflowPredictEffnetDiscogs(
        graphFilename=str(model_path),
        output="PartitionedCall:1",
    )


def extract_effnet(
    predictor: Any,
    waveform: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return full ordered patches [T,1280] and mean-pooled [1280]."""
    patches = np.asarray(
        predictor(waveform),
        dtype=np.float32,
    )

    if patches.ndim == 1:
        if patches.shape[0] != EXPECTED_DIM:
            raise RuntimeError(
                f"unexpected 1D EffNet output shape {patches.shape}"
            )
        patches = patches[None, :]

    if patches.ndim != 2 or patches.shape[1] != EXPECTED_DIM:
        raise RuntimeError(
            f"unexpected EffNet output shape {patches.shape}; "
            f"expected [T,{EXPECTED_DIM}]"
        )

    if patches.shape[0] == 0:
        raise RuntimeError("EffNet returned zero patches")

    if not np.all(np.isfinite(patches)):
        raise RuntimeError("EffNet output contains non-finite values")

    # Mean in float64 for stable reproduction of the existing pooled pipeline.
    pooled = patches.mean(
        axis=0,
        dtype=np.float64,
    ).astype(np.float32)

    if pooled.shape != (EXPECTED_DIM,):
        raise RuntimeError(f"unexpected pooled shape {pooled.shape}")

    if not np.all(np.isfinite(pooled)):
        raise RuntimeError("mean-pooled embedding contains non-finite values")

    return patches, pooled


def available_audio_rows(
    audio_manifest: Path,
) -> dict[str, dict[str, Any]]:
    """Return one usable physical-audio row per video ID.

    The unified merge writes one row per physical recording, but the last-row
    rule also makes this tolerant of older append-only manifests.
    """
    result: dict[str, dict[str, Any]] = {}

    for row in load_jsonl_tolerant(audio_manifest):
        video_id = row.get("video_id")
        audio_path = row.get("audio_path")

        if not isinstance(video_id, str) or not video_id:
            continue
        if not isinstance(audio_path, str) or not audio_path:
            continue

        path = Path(audio_path)
        if not path.is_file():
            continue

        result[video_id] = row

    return result


def build_patch_manifest(
    *,
    audio_rows: dict[str, dict[str, Any]],
    pooled_dir: Path,
    patches_dir: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for video_id in sorted(audio_rows):
        audio_row = audio_rows[video_id]
        pooled_path = pooled_dir / f"{video_id}.npy"
        patches_path = patches_dir / f"{video_id}.npy"
        shape = patch_shape(patches_path)

        rows.append(
            {
                "video_id": video_id,
                "audio_path": audio_row.get("audio_path"),
                "pooled_path": (
                    str(pooled_path)
                    if valid_pooled_embedding(pooled_path)
                    else None
                ),
                "patches_path": str(patches_path) if shape else None,
                "patch_shape": list(shape) if shape else None,
                "patch_count": shape[0] if shape else None,
                "embedding_dimension": EXPECTED_DIM,
                "patch_dtype": "float32" if shape else None,
                "temporal_order_preserved": bool(shape),
                "candidate_ids": audio_row.get("candidate_ids", []),
                "labels": audio_row.get("labels", []),
                "artists": audio_row.get("artists", []),
                "ytm_artists": audio_row.get("ytm_artists", []),
                "dataset_sources": audio_row.get("dataset_sources", []),
            }
        )

    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract unified Discogs-EffNet pooled and/or full patch "
            "embeddings without rewriting canonical manifests."
        )
    )

    parser.add_argument(
        "--audio-manifest",
        type=Path,
        default=DEFAULT_AUDIO_MANIFEST,
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--ffmpeg",
        default="ffmpeg",
    )
    parser.add_argument(
        "--save-patches",
        action="store_true",
        help=(
            "Persist the full ordered [T,1280] EffNet sequence for temporal "
            "models. Without this flag, only missing pooled embeddings are "
            "generated."
        ),
    )
    parser.add_argument(
        "--overwrite-pooled",
        action="store_true",
        help="Explicitly replace existing valid pooled embeddings.",
    )
    parser.add_argument(
        "--overwrite-patches",
        action="store_true",
        help="Explicitly replace existing valid patch embeddings.",
    )
    parser.add_argument(
        "--max-tracks",
        type=int,
        help="Optional successful-extraction cap for smoke testing.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.audio_manifest.is_file():
        raise SystemExit(f"Audio manifest not found: {args.audio_manifest}")

    if not args.model.is_file():
        raise SystemExit(f"EffNet model not found: {args.model}")

    if args.max_tracks is not None and args.max_tracks < 1:
        raise SystemExit("--max-tracks must be >= 1")

    output_dir = args.output_dir
    pooled_dir = output_dir / "pooled"
    patches_dir = output_dir / "patches"
    state_path = output_dir / "patch_state.jsonl"
    errors_path = output_dir / "patch_errors.jsonl"
    manifest_path = output_dir / "patch_manifest.jsonl"
    report_path = output_dir / "patch_report.json"

    pooled_dir.mkdir(parents=True, exist_ok=True)
    if args.save_patches:
        patches_dir.mkdir(parents=True, exist_ok=True)

    state = load_state(state_path)
    audio_rows = available_audio_rows(args.audio_manifest)

    if not audio_rows:
        raise SystemExit(
            f"No usable audio files found through {args.audio_manifest}"
        )

    pending: list[tuple[str, Path, bool, bool]] = []
    already_complete = 0

    for video_id, row in sorted(audio_rows.items()):
        audio_path = Path(str(row["audio_path"]))
        pooled_path = pooled_dir / f"{video_id}.npy"
        patches_path = patches_dir / f"{video_id}.npy"

        pooled_ok = valid_pooled_embedding(pooled_path)
        patches_ok = valid_patch_embedding(patches_path)

        need_pooled = args.overwrite_pooled or not pooled_ok
        need_patches = args.save_patches and (
            args.overwrite_patches or not patches_ok
        )

        if need_pooled or need_patches:
            pending.append((video_id, audio_path, need_pooled, need_patches))
        else:
            already_complete += 1

    predictor = create_effnet(args.model) if pending else None

    stop_requested = False

    def request_stop(signum: int, frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True
        print()
        print("Stop requested; finishing current track...")

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    extracted_this_run = 0
    failed_this_run = 0
    pooled_written_this_run = 0
    patches_written_this_run = 0
    total_patches_this_run = 0
    started = time.monotonic()

    print(f"Unified Discogs-EffNet extractor [{EXTRACTOR_VERSION}]")
    print(f"  audio manifest:     {args.audio_manifest}")
    print(f"  physical audio IDs: {len(audio_rows)}")
    print(f"  model:              {args.model}")
    print(f"  pooled dir:         {pooled_dir}")
    print(f"  save patches:       {args.save_patches}")
    if args.save_patches:
        print(f"  patches dir:        {patches_dir}")
        print("  patch shape:        [T,1280], full temporal sequence")
    print(f"  pending extraction: {len(pending)}")
    print(f"  already complete:   {already_complete}")
    print(f"  overwrite pooled:   {args.overwrite_pooled}")
    print(f"  overwrite patches:  {args.overwrite_patches}")
    print()

    for video_id, audio_path, need_pooled, need_patches in pending:
        if stop_requested:
            break

        if (
            args.max_tracks is not None
            and extracted_this_run >= args.max_tracks
        ):
            break

        pooled_path = pooled_dir / f"{video_id}.npy"
        patches_path = patches_dir / f"{video_id}.npy"

        reasons = []
        if need_pooled:
            reasons.append("pooled")
        if need_patches:
            reasons.append("patches")

        print(
            f"[{extracted_this_run + 1}/{len(pending)}] "
            f"{video_id}: {audio_path.name} -> {'+'.join(reasons)}"
        )

        try:
            waveform = decode_audio_ffmpeg(
                audio_path,
                ffmpeg=args.ffmpeg,
                sample_rate=SAMPLE_RATE,
            )

            if predictor is None:
                raise RuntimeError("EffNet predictor was not initialized")

            patches, pooled = extract_effnet(predictor, waveform)

            # The core preservation rule: a patch-only pass does not touch a
            # valid pooled baseline unless --overwrite-pooled was requested.
            if need_pooled:
                atomic_save_npy(pooled_path, pooled)
                pooled_written_this_run += 1

            if need_patches:
                atomic_save_npy(
                    patches_path,
                    np.asarray(patches, dtype=np.float32),
                )
                patches_written_this_run += 1

            record = {
                "video_id": video_id,
                "status": "embedded",
                "reason": "success",
                "audio_path": str(audio_path),
                "pooled_path": str(pooled_path),
                "pooled_written": need_pooled,
                "patches_path": str(patches_path) if args.save_patches else None,
                "patches_written": need_patches,
                "patch_shape": [int(patches.shape[0]), EXPECTED_DIM],
                "patch_count": int(patches.shape[0]),
                "sample_rate": SAMPLE_RATE,
                "audio_samples": int(waveform.shape[0]),
                "timestamp": time.time(),
            }

            state[video_id] = record
            append_jsonl(state_path, record)

            extracted_this_run += 1
            total_patches_this_run += int(patches.shape[0])

        except Exception as exc:
            failed_this_run += 1

            record = {
                "video_id": video_id,
                "status": "error",
                "reason": f"{type(exc).__name__}: {exc}",
                "audio_path": str(audio_path),
                "timestamp": time.time(),
            }

            state[video_id] = record
            append_jsonl(state_path, record)
            append_jsonl(errors_path, record)

            print(f"  ERROR: {record['reason']}")

    patch_manifest = build_patch_manifest(
        audio_rows=audio_rows,
        pooled_dir=pooled_dir,
        patches_dir=patches_dir,
    )
    write_jsonl(manifest_path, patch_manifest)

    pooled_available = sum(
        1
        for row in patch_manifest
        if row.get("pooled_path") is not None
    )
    patches_available = sum(
        1
        for row in patch_manifest
        if row.get("patches_path") is not None
    )

    status_counts = Counter(
        row.get("status", "unknown")
        for row in state.values()
    )

    elapsed = time.monotonic() - started

    report = {
        "extractor": "unified_discogs_effnet",
        "sample_rate": SAMPLE_RATE,
        "embedding_dimension": EXPECTED_DIM,
        "physical_audio_records": len(audio_rows),
        "pooled_embeddings_available": pooled_available,
        "patch_embeddings_available": patches_available,
        "patch_embeddings_complete": patches_available == len(audio_rows),
        "patch_representation": {
            "shape": "[T,1280]",
            "dtype": "float32",
            "temporal_order_preserved": True,
            "subsampled_during_extraction": False,
            "compatible_frozen_effnet_models": [
                "scalar_attention_pooling",
                "gated_attention_pooling",
                "multihead_attention_pooling",
                "temporal_cnn_or_rnn_over_patches",
                "transformer_over_effnet_patches",
            ],
            "not_sufficient_for": [
                "end_to_end_finetuning_of_discogs_effnet",
                "a_different_pretrained_audio_encoder",
                "raw_waveform_or_logmel_models_without_reencoding",
            ],
        },
        "preservation_policy": {
            "existing_valid_pooled_embeddings_overwritten": bool(
                args.overwrite_pooled
            ),
            "canonical_embedding_manifest_rewritten": False,
            "canonical_audio_manifest_rewritten": False,
        },
        "state_status_counts": dict(status_counts),
        "this_run": {
            "pending_before_run": len(pending),
            "already_complete_before_run": already_complete,
            "successful_extractions": extracted_this_run,
            "failed_extractions": failed_this_run,
            "pooled_files_written": pooled_written_this_run,
            "patch_files_written": patches_written_this_run,
            "effnet_patches_processed": total_patches_this_run,
            "elapsed_seconds": round(elapsed, 2),
        },
        "configuration": {
            "audio_manifest": str(args.audio_manifest),
            "model": str(args.model),
            "output_dir": str(output_dir),
            "save_patches": args.save_patches,
            "overwrite_pooled": args.overwrite_pooled,
            "overwrite_patches": args.overwrite_patches,
            "max_tracks": args.max_tracks,
            "ffmpeg": args.ffmpeg,
        },
        "outputs": {
            "pooled_dir": str(pooled_dir),
            "patches_dir": str(patches_dir),
            "patch_manifest": str(manifest_path),
            "patch_state": str(state_path),
            "patch_errors": str(errors_path),
        },
    }

    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print()
    print("Extraction pass complete")
    print(f"  physical audio:      {len(audio_rows)}")
    print(f"  pooled available:    {pooled_available}")
    print(f"  patches available:   {patches_available}")
    print(f"  extracted this run:  {extracted_this_run}")
    print(f"  failures this run:   {failed_this_run}")
    print(f"  pooled files written:{pooled_written_this_run}")
    print(f"  patch files written: {patches_written_this_run}")
    print(f"  elapsed:             {elapsed / 3600:.2f} h")
    print()
    print(f"Report:   {report_path}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
