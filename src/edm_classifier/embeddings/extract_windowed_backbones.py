"""Extract fixed-duration MERT-95M or MuQ embeddings for complete tracks.

Each physical track is split into consecutive, non-overlapping windows. The
final logical window may be shorter than ``--window-seconds``; it is padded
only for batched model inference and accompanied by an attention mask, so no
source sample is discarded or duplicated.

For every ``video_id`` this module writes:

    windows/<video_id>.npy   # [number_of_windows, embedding_dimension]
    pooled/<video_id>.npy    # [embedding_dimension]
    metadata/<video_id>.json # coverage, masking, and reproducibility metadata

The ordered window matrix is suitable for attention pooling. The pooled vector
is suitable for linear, MLP, tree, nearest-neighbor, or other fixed-size models.

Typical usage from the repository root:

    python -m edm_classifier.embeddings.extract_windowed_backbones \
        --encoder mert95m

    python -m edm_classifier.embeddings.extract_windowed_backbones \
        --encoder muq
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import subprocess
import tempfile
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np


DEFAULT_AUDIO_MANIFEST = Path("data/audio/audio_manifest.jsonl")
SAMPLE_RATE = 24_000
EXTRACTOR_VERSION = "windowed-music-backbones-v1"


@dataclass(frozen=True)
class ModelSpec:
    key: str
    display_name: str
    default_model_id: str
    embedding_dimension: int
    default_batch_size: int


MODEL_SPECS: dict[str, ModelSpec] = {
    "mert95m": ModelSpec(
        key="mert95m",
        display_name="MERT-v1-95M",
        default_model_id="m-a-p/MERT-v1-95M",
        embedding_dimension=768,
        default_batch_size=8,
    ),
    "muq": ModelSpec(
        key="muq",
        display_name="MuQ-large-msd-iter",
        default_model_id="OpenMuQ/MuQ-large-msd-iter",
        embedding_dimension=1024,
        default_batch_size=2,
    ),
}


def load_jsonl_tolerant(path: Path) -> list[dict[str, Any]]:
    """Load complete JSONL records, tolerating only a partial final line."""
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
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
        )
        handle.flush()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def atomic_write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(
                    json.dumps(
                        row,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def atomic_save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{path.stem}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            np.save(handle, array, allow_pickle=False)
        os.replace(temporary, path)
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def load_json_object(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def load_state(path: Path) -> dict[str, dict[str, Any]]:
    state: dict[str, dict[str, Any]] = {}
    for row in load_jsonl_tolerant(path):
        video_id = row.get("video_id")
        if isinstance(video_id, str) and video_id:
            state[video_id] = row
    return state


def available_audio_rows(path: Path) -> dict[str, dict[str, Any]]:
    """Return the last usable physical-audio row for each video ID."""
    rows: dict[str, dict[str, Any]] = {}
    for row in load_jsonl_tolerant(path):
        video_id = row.get("video_id")
        audio_path = row.get("audio_path")
        if not isinstance(video_id, str) or not video_id:
            continue
        if not isinstance(audio_path, str) or not audio_path:
            continue
        if not Path(audio_path).is_file():
            continue
        rows[video_id] = row
    return rows


def decode_audio_ffmpeg(
    path: Path,
    *,
    ffmpeg: str,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """Decode arbitrary media to mono float32 PCM at the target sample rate."""
    command = [
        ffmpeg,
        "-v",
        "error",
        "-nostdin",
        "-i",
        str(path),
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
        message = process.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            f"ffmpeg decode failed ({process.returncode}): {message}"
        )

    waveform = np.frombuffer(process.stdout, dtype="<f4").astype(
        np.float32,
        copy=True,
    )
    if waveform.size == 0:
        raise RuntimeError("ffmpeg produced an empty waveform")
    if not np.isfinite(waveform).all():
        raise RuntimeError("decoded waveform contains NaN or infinity")
    return waveform


def logical_window_count(total_samples: int, window_samples: int) -> int:
    if total_samples < 1:
        raise ValueError("total_samples must be positive")
    if window_samples < 1:
        raise ValueError("window_samples must be positive")
    return (total_samples + window_samples - 1) // window_samples


def valid_samples_per_window(
    total_samples: int,
    window_samples: int,
) -> np.ndarray:
    count = logical_window_count(total_samples, window_samples)
    values = np.full(count, window_samples, dtype=np.int64)
    values[-1] = total_samples - (count - 1) * window_samples
    if values[-1] < 1 or values[-1] > window_samples:
        raise AssertionError("invalid final-window coverage")
    if int(values.sum()) != total_samples:
        raise AssertionError("window coverage does not equal source length")
    return values


def iter_window_batches(
    waveform: np.ndarray,
    *,
    window_samples: int,
    batch_size: int,
) -> Iterator[tuple[np.ndarray, np.ndarray, int]]:
    """Yield zero-padded batches and their real sample counts.

    The third return value is the global index of the first logical window in
    the batch. Source samples occur in exactly one batch/window.
    """
    if waveform.ndim != 1:
        raise ValueError(f"expected mono waveform, got shape {waveform.shape}")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    valid = valid_samples_per_window(int(waveform.size), window_samples)
    for first in range(0, len(valid), batch_size):
        counts = valid[first : first + batch_size]
        batch = np.zeros((len(counts), window_samples), dtype=np.float32)
        for local_index, real_count_value in enumerate(counts):
            real_count = int(real_count_value)
            global_index = first + local_index
            source_start = global_index * window_samples
            source_stop = source_start + real_count
            batch[local_index, :real_count] = waveform[source_start:source_stop]
        yield batch, counts.copy(), first


def mean_pool_windows(
    windows: np.ndarray,
    frame_counts: np.ndarray,
    *,
    mode: str,
) -> np.ndarray:
    """Pool window vectors into one whole-track vector.

    ``frame_mean`` weights each already-pooled window by its number of valid
    encoder frames. This is equivalent to a mean across all valid encoder
    frames and prevents a short tail from receiving a full five-second vote.
    ``window_mean`` is the unweighted arithmetic mean across logical windows.
    """
    matrix = np.asarray(windows, dtype=np.float32)
    counts = np.asarray(frame_counts, dtype=np.int64)
    if matrix.ndim != 2 or matrix.shape[0] < 1:
        raise ValueError(f"invalid window matrix shape {matrix.shape}")
    if counts.shape != (matrix.shape[0],):
        raise ValueError(
            f"frame-count shape {counts.shape} does not match {matrix.shape}"
        )
    if not np.isfinite(matrix).all():
        raise ValueError("window matrix contains NaN or infinity")
    if np.any(counts < 1):
        raise ValueError("every logical window must contain a valid frame")

    if mode == "window_mean":
        pooled = matrix.mean(axis=0, dtype=np.float64)
    elif mode == "frame_mean":
        weights = counts.astype(np.float64)
        pooled = np.average(
            matrix.astype(np.float64),
            axis=0,
            weights=weights,
        )
    else:
        raise ValueError(f"unsupported pooling mode {mode!r}")

    result = np.asarray(pooled, dtype=np.float32)
    if result.shape != (matrix.shape[1],):
        raise AssertionError(f"unexpected pooled shape {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError("pooled vector contains NaN or infinity")
    return result


def valid_window_file(path: Path, embedding_dimension: int) -> tuple[int, int] | None:
    if not path.is_file():
        return None
    try:
        array = np.load(path, mmap_mode="r", allow_pickle=False)
    except Exception:
        return None
    if (
        array.ndim != 2
        or array.shape[0] < 1
        or array.shape[1] != embedding_dimension
        or not np.issubdtype(array.dtype, np.floating)
    ):
        return None
    if not np.isfinite(array).all():
        return None
    return int(array.shape[0]), int(array.shape[1])


def valid_pooled_file(path: Path, embedding_dimension: int) -> bool:
    if not path.is_file():
        return False
    try:
        array = np.load(path, mmap_mode="r", allow_pickle=False)
    except Exception:
        return False
    return (
        array.shape == (embedding_dimension,)
        and np.issubdtype(array.dtype, np.floating)
        and bool(np.isfinite(array).all())
    )


def configuration_fingerprint(configuration: dict[str, Any]) -> str:
    serialized = json.dumps(
        configuration,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def resolve_hugging_face_revision(model_id: str, revision: str | None) -> str:
    """Resolve a moving model reference to a concrete Hub commit when possible."""
    local = Path(model_id)
    if local.exists():
        return revision or "local-files"

    try:
        from huggingface_hub import model_info
    except Exception as exc:
        raise RuntimeError(
            "huggingface_hub is required to resolve a reproducible model revision"
        ) from exc

    try:
        info = model_info(model_id, revision=revision)
    except Exception as exc:
        if revision:
            # A commit hash supplied by the user is already immutable. Permit
            # cached/offline model loading even if the metadata request failed.
            return revision
        raise RuntimeError(
            f"could not resolve the current revision for {model_id!r}; "
            "pass --revision with a commit hash for an offline run"
        ) from exc

    sha = getattr(info, "sha", None)
    if not isinstance(sha, str) or not sha:
        raise RuntimeError(f"Hugging Face returned no commit SHA for {model_id!r}")
    return sha


def choose_device(requested: str) -> str:
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("PyTorch is required for MERT and MuQ extraction") from exc

    if requested != "auto":
        return str(torch.device(requested))
    if torch.cuda.is_available():
        return "cuda"
    # CPU is the conservative fallback. MuQ's audio front end and dependency
    # stack have historically been less predictable on MPS.
    return "cpu"


def set_inference_seed(seed: int) -> None:
    import torch

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _select_hidden_state(outputs: Any, layer: int) -> Any:
    if layer == -1:
        hidden = getattr(outputs, "last_hidden_state", None)
        if hidden is None:
            raise RuntimeError("model output has no last_hidden_state")
        return hidden

    states = getattr(outputs, "hidden_states", None)
    if states is None:
        raise RuntimeError("model did not return hidden states")
    try:
        return states[layer]
    except IndexError as exc:
        raise ValueError(
            f"layer {layer} is outside the returned hidden-state range "
            f"({len(states)} states)"
        ) from exc


def _masked_mean_torch(hidden: Any, frame_mask: Any) -> tuple[np.ndarray, np.ndarray]:
    import torch

    if hidden.ndim != 3:
        raise RuntimeError(f"expected hidden states [B,T,D], got {hidden.shape}")
    if frame_mask.shape != hidden.shape[:2]:
        raise RuntimeError(
            f"frame mask {frame_mask.shape} does not match hidden states {hidden.shape}"
        )

    valid = frame_mask.to(device=hidden.device, dtype=torch.bool)
    counts = valid.sum(dim=1)
    if bool(torch.any(counts < 1).item()):
        raise RuntimeError("a logical window produced zero valid encoder frames")

    weights = valid.unsqueeze(-1).to(dtype=hidden.dtype)
    vectors = (hidden * weights).sum(dim=1) / counts.unsqueeze(-1).to(hidden.dtype)
    vectors_np = vectors.detach().float().cpu().numpy().astype(np.float32, copy=False)
    counts_np = counts.detach().cpu().numpy().astype(np.int64, copy=False)
    if not np.isfinite(vectors_np).all():
        raise RuntimeError("model produced NaN or infinity")
    return vectors_np, counts_np


class MertAdapter:
    def __init__(
        self,
        *,
        model_id: str,
        revision: str | None,
        device: str,
        expected_dimension: int,
    ) -> None:
        try:
            import torch
            from transformers import AutoModel, Wav2Vec2FeatureExtractor
        except Exception as exc:
            raise RuntimeError(
                "MERT extraction requires torch and transformers. "
                "Install requirements-mert-muq.txt first."
            ) from exc

        self.torch = torch
        self.device = torch.device(device)
        self.expected_dimension = expected_dimension
        load_kwargs: dict[str, Any] = {}
        if revision is not None:
            load_kwargs["revision"] = revision
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            model_id,
            **load_kwargs,
        )
        self.model = AutoModel.from_pretrained(
            model_id,
            trust_remote_code=True,
            **load_kwargs,
        )
        self.model = self.model.to(self.device).float().eval()

        configured_rate = getattr(self.model.config, "sample_rate", None)
        if configured_rate is not None and int(configured_rate) != SAMPLE_RATE:
            raise RuntimeError(
                f"MERT checkpoint requests {configured_rate} Hz, expected {SAMPLE_RATE}"
            )
        configured_dimension = getattr(self.model.config, "hidden_size", None)
        if configured_dimension is not None and int(configured_dimension) != expected_dimension:
            raise RuntimeError(
                f"MERT hidden size is {configured_dimension}, expected {expected_dimension}"
            )

    def _frame_mask(self, raw_mask: Any, output_length: int) -> Any:
        torch = self.torch
        method = getattr(self.model, "_get_feature_vector_attention_mask", None)
        if callable(method):
            mask = method(output_length, raw_mask)
            return mask.to(dtype=torch.bool)

        length_method = getattr(self.model, "_get_feat_extract_output_lengths", None)
        if callable(length_method):
            raw_lengths = raw_mask.sum(dim=-1)
            output_lengths = length_method(raw_lengths).to(dtype=torch.long)
            indices = torch.arange(output_length, device=raw_mask.device)
            return indices.unsqueeze(0) < output_lengths.unsqueeze(1)

        raise RuntimeError(
            "MERT model exposes neither feature-vector mask nor output-length helper"
        )

    def embed(
        self,
        padded_windows: np.ndarray,
        valid_samples: np.ndarray,
        *,
        layer: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        clips = [
            np.asarray(padded_windows[i, : int(length)], dtype=np.float32)
            for i, length in enumerate(valid_samples)
        ]
        encoded = self.feature_extractor(
            clips,
            sampling_rate=SAMPLE_RATE,
            padding="max_length",
            max_length=int(padded_windows.shape[1]),
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_values = encoded["input_values"].to(
            device=self.device,
            dtype=self.torch.float32,
        )
        raw_mask = encoded["attention_mask"].to(device=self.device)
        need_states = layer != -1

        with self.torch.inference_mode():
            outputs = self.model(
                input_values,
                attention_mask=raw_mask,
                output_hidden_states=need_states,
                return_dict=True,
            )
            hidden = _select_hidden_state(outputs, layer)
            if int(hidden.shape[-1]) != self.expected_dimension:
                raise RuntimeError(
                    f"MERT returned dimension {hidden.shape[-1]}, "
                    f"expected {self.expected_dimension}"
                )
            frame_mask = self._frame_mask(raw_mask, int(hidden.shape[1]))
            return _masked_mean_torch(hidden, frame_mask)

    def description(self) -> dict[str, Any]:
        return {
            "implementation": type(self.model).__name__,
            "configured_hidden_size": int(self.model.config.hidden_size),
            "configured_layer_count": int(self.model.config.num_hidden_layers),
            "dtype": "float32",
            "trust_remote_code": True,
        }


class MuqAdapter:
    def __init__(
        self,
        *,
        model_id: str,
        revision: str | None,
        device: str,
        expected_dimension: int,
        allow_flash: bool,
    ) -> None:
        try:
            import torch
            from muq import MuQ
        except Exception as exc:
            raise RuntimeError(
                "MuQ extraction requires the official muq package and PyTorch. "
                "Install requirements-mert-muq.txt first."
            ) from exc

        self.torch = torch
        self.device = torch.device(device)
        self.expected_dimension = expected_dimension
        load_kwargs: dict[str, Any] = {}
        if revision is not None:
            load_kwargs["revision"] = revision
        self.model = MuQ.from_pretrained(model_id, **load_kwargs)
        self.model = self.model.to(self.device).float().eval()

        configured_dimension = int(getattr(self.model.config, "encoder_dim", -1))
        if configured_dimension != expected_dimension:
            raise RuntimeError(
                f"MuQ encoder dimension is {configured_dimension}, "
                f"expected {expected_dimension}"
            )

        self.is_flash = bool(getattr(self.model.config, "is_flash", False))
        if self.is_flash and not allow_flash:
            raise RuntimeError(
                "this MuQ checkpoint selects the flash conformer path, which has "
                "a reported eval-time dropout defect; use a non-flash checkpoint "
                "or pass --allow-muq-flash after independently validating it"
            )

    def _frame_mask(self, raw_mask: Any, output_length: int) -> Any:
        """Reproduce MuQ's own raw-mask downsampling rule."""
        torch = self.torch
        skip = int(raw_mask.shape[-1] / output_length)
        if skip < 1:
            raise RuntimeError(
                f"cannot downsample {raw_mask.shape[-1]} samples to {output_length} frames"
            )
        mask = raw_mask.to(dtype=torch.bool)[:, ::skip]
        mask = mask[:, :output_length]
        if int(mask.shape[1]) < output_length:
            missing = output_length - int(mask.shape[1])
            mask = torch.nn.functional.pad(mask, (0, missing), value=False)
        return mask

    def embed(
        self,
        padded_windows: np.ndarray,
        valid_samples: np.ndarray,
        *,
        layer: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        waveforms = self.torch.from_numpy(padded_windows).to(
            device=self.device,
            dtype=self.torch.float32,
        )
        sample_indices = self.torch.arange(
            padded_windows.shape[1],
            device=self.device,
        ).unsqueeze(0)
        lengths = self.torch.from_numpy(valid_samples).to(device=self.device)
        raw_mask = sample_indices < lengths.unsqueeze(1)
        need_states = layer != -1

        with self.torch.inference_mode():
            outputs = self.model(
                waveforms,
                attention_mask=raw_mask,
                output_hidden_states=need_states,
            )
            hidden = _select_hidden_state(outputs, layer)
            if int(hidden.shape[-1]) != self.expected_dimension:
                raise RuntimeError(
                    f"MuQ returned dimension {hidden.shape[-1]}, "
                    f"expected {self.expected_dimension}"
                )
            frame_mask = self._frame_mask(raw_mask, int(hidden.shape[1]))
            return _masked_mean_torch(hidden, frame_mask)

    def description(self) -> dict[str, Any]:
        return {
            "implementation": type(self.model).__name__,
            "configured_hidden_size": int(self.model.config.encoder_dim),
            "configured_layer_count": int(self.model.config.encoder_depth),
            "configured_is_flash": self.is_flash,
            "dtype": "float32",
        }


def create_adapter(
    encoder: str,
    *,
    model_id: str,
    revision: str | None,
    device: str,
    expected_dimension: int,
    allow_muq_flash: bool,
) -> MertAdapter | MuqAdapter:
    if encoder == "mert95m":
        return MertAdapter(
            model_id=model_id,
            revision=revision,
            device=device,
            expected_dimension=expected_dimension,
        )
    if encoder == "muq":
        return MuqAdapter(
            model_id=model_id,
            revision=revision,
            device=device,
            expected_dimension=expected_dimension,
            allow_flash=allow_muq_flash,
        )
    raise ValueError(encoder)


def track_is_complete(
    *,
    video_id: str,
    fingerprint: str,
    embedding_dimension: int,
    windows_dir: Path,
    pooled_dir: Path,
    metadata_dir: Path,
) -> bool:
    shape = valid_window_file(
        windows_dir / f"{video_id}.npy",
        embedding_dimension,
    )
    if shape is None:
        return False
    if not valid_pooled_file(
        pooled_dir / f"{video_id}.npy",
        embedding_dimension,
    ):
        return False
    metadata = load_json_object(metadata_dir / f"{video_id}.json")
    if metadata is None:
        return False
    if metadata.get("configuration_fingerprint") != fingerprint:
        return False
    if metadata.get("status") != "embedded":
        return False
    if metadata.get("windows_shape") != list(shape):
        return False
    valid_samples = metadata.get("window_valid_samples")
    if not isinstance(valid_samples, list) or len(valid_samples) != shape[0]:
        return False
    return True


def compatible_model_info(path: Path, fingerprint: str) -> None:
    existing = load_json_object(path)
    if existing is None:
        return
    previous = existing.get("configuration_fingerprint")
    if previous == fingerprint:
        return
    raise SystemExit(
        f"{path.parent} contains embeddings from an incompatible configuration.\n"
        "Use a different --output-dir so model/layer/window variants cannot mix.\n"
        f"Existing fingerprint: {previous}\nRequested fingerprint: {fingerprint}"
    )


def extract_track(
    adapter: MertAdapter | MuqAdapter,
    waveform: np.ndarray,
    *,
    window_samples: int,
    batch_size: int,
    layer: int,
    song_pooling: str,
    repeatability_check: bool,
    allow_nondeterministic: bool,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    vectors: list[np.ndarray] = []
    frame_counts: list[np.ndarray] = []
    valid_sample_counts: list[np.ndarray] = []
    repeatability: dict[str, Any] | None = None

    for padded, valid, first_index in iter_window_batches(
        waveform,
        window_samples=window_samples,
        batch_size=batch_size,
    ):
        batch_vectors, batch_frames = adapter.embed(
            padded,
            valid,
            layer=layer,
        )

        if repeatability_check and repeatability is None:
            repeated_vectors, repeated_frames = adapter.embed(
                padded,
                valid,
                layer=layer,
            )
            maximum_absolute_difference = float(
                np.max(np.abs(batch_vectors - repeated_vectors))
            )
            counts_equal = bool(np.array_equal(batch_frames, repeated_frames))
            close = bool(
                counts_equal
                and np.allclose(
                    batch_vectors,
                    repeated_vectors,
                    rtol=1e-5,
                    atol=1e-6,
                )
            )
            repeatability = {
                "checked_at_first_window_index": first_index,
                "allclose": close,
                "frame_counts_equal": counts_equal,
                "maximum_absolute_difference": maximum_absolute_difference,
                "relative_tolerance": 1e-5,
                "absolute_tolerance": 1e-6,
            }
            if not close and not allow_nondeterministic:
                raise RuntimeError(
                    "repeated inference on the same first batch was not stable; "
                    f"maximum absolute difference={maximum_absolute_difference:.8g}. "
                    "Pass --allow-nondeterministic only after deciding this is acceptable."
                )

        if batch_vectors.shape[0] != len(valid):
            raise RuntimeError("model returned the wrong batch size")
        vectors.append(batch_vectors)
        frame_counts.append(batch_frames)
        valid_sample_counts.append(valid)

    window_matrix = np.concatenate(vectors, axis=0).astype(np.float32, copy=False)
    encoder_frame_counts = np.concatenate(frame_counts).astype(np.int64, copy=False)
    source_sample_counts = np.concatenate(valid_sample_counts).astype(
        np.int64,
        copy=False,
    )

    if int(source_sample_counts.sum()) != int(waveform.size):
        raise AssertionError("logical windows do not cover the complete waveform")

    pooled = mean_pool_windows(
        window_matrix,
        encoder_frame_counts,
        mode=song_pooling,
    )
    details = {
        "window_valid_samples": source_sample_counts.tolist(),
        "encoder_frame_counts": encoder_frame_counts.tolist(),
        "repeatability_check": repeatability,
    }
    return window_matrix, pooled, details


def build_manifests(
    *,
    audio_rows: dict[str, dict[str, Any]],
    state: dict[str, dict[str, Any]],
    fingerprint: str,
    spec: ModelSpec,
    model_id: str,
    revision: str,
    layer: int,
    window_seconds: float,
    song_pooling: str,
    windows_dir: Path,
    pooled_dir: Path,
    metadata_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    all_rows: list[dict[str, Any]] = []
    complete_rows: list[dict[str, Any]] = []

    for video_id in sorted(audio_rows):
        audio_row = audio_rows[video_id]
        windows_path = windows_dir / f"{video_id}.npy"
        pooled_path = pooled_dir / f"{video_id}.npy"
        metadata_path = metadata_dir / f"{video_id}.json"
        complete = track_is_complete(
            video_id=video_id,
            fingerprint=fingerprint,
            embedding_dimension=spec.embedding_dimension,
            windows_dir=windows_dir,
            pooled_dir=pooled_dir,
            metadata_dir=metadata_dir,
        )
        metadata = load_json_object(metadata_path) if complete else None
        last_state = state.get(video_id, {})
        shape = (
            valid_window_file(windows_path, spec.embedding_dimension)
            if complete
            else None
        )
        status = "embedded" if complete else str(last_state.get("status", "pending"))

        row = {
            "video_id": video_id,
            "status": status,
            "audio_path": audio_row.get("audio_path"),
            "encoder": spec.key,
            "model_name": spec.display_name,
            "model_id": model_id,
            "model_revision": revision,
            "layer": layer,
            "sample_rate": SAMPLE_RATE,
            "window_seconds": window_seconds,
            "window_hop_seconds": window_seconds,
            "overlap_seconds": 0.0,
            "window_count": shape[0] if shape else None,
            "last_window_valid_seconds": (
                metadata.get("last_window_valid_seconds")
                if metadata is not None
                else None
            ),
            "windows_path": str(windows_path) if complete else None,
            "windows_shape": list(shape) if shape else None,
            "patches_path": str(windows_path) if complete else None,
            "pooled_path": str(pooled_path) if complete else None,
            "embedding_path": str(pooled_path) if complete else None,
            "pooled_shape": [spec.embedding_dimension] if complete else None,
            "embedding_dimension": spec.embedding_dimension,
            "pooling": song_pooling,
            "temporal_order_preserved": complete,
            "tail_padding_excluded_by_attention_mask": complete,
            "configuration_fingerprint": fingerprint,
            "candidate_ids": audio_row.get("candidate_ids", []),
            "labels": audio_row.get("labels", []),
            "artists": audio_row.get("artists", []),
            "ytm_artists": audio_row.get("ytm_artists", []),
            "dataset_sources": audio_row.get(
                "dataset_sources",
                audio_row.get("sources", []),
            ),
        }
        if not complete and last_state.get("reason"):
            row["reason"] = last_state["reason"]
        all_rows.append(row)
        if complete:
            complete_rows.append(row)

    return all_rows, complete_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract non-overlapping fixed-duration MERT-95M or MuQ window "
            "embeddings plus one mean-pooled vector per complete track."
        )
    )
    parser.add_argument("--encoder", choices=sorted(MODEL_SPECS), required=True)
    parser.add_argument(
        "--audio-manifest",
        type=Path,
        default=DEFAULT_AUDIO_MANIFEST,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Default: data/embeddings/<encoder>",
    )
    parser.add_argument(
        "--model-id",
        help="Hugging Face model ID or local model directory.",
    )
    parser.add_argument(
        "--revision",
        help=(
            "Hugging Face branch/tag/commit. A moving reference is resolved to "
            "a concrete commit SHA before extraction."
        ),
    )
    parser.add_argument("--window-seconds", type=float, default=5.0)
    parser.add_argument(
        "--song-pooling",
        choices=["frame_mean", "window_mean"],
        default="frame_mean",
        help=(
            "frame_mean weights each window by valid encoder-frame count; "
            "window_mean gives every logical window equal weight."
        ),
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=-1,
        help=(
            "Hidden-state index. -1 uses last_hidden_state without retaining "
            "all layers; 0 is the pre-encoder representation for these models."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="Windows per forward pass. 0 selects a model-specific default.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--max-tracks",
        type=int,
        help="Optional pending-track cap for a smoke test.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute valid outputs under the same extraction configuration.",
    )
    parser.add_argument(
        "--skip-repeatability-check",
        action="store_true",
        help="Do not run the first model batch twice to check inference stability.",
    )
    parser.add_argument(
        "--allow-nondeterministic",
        action="store_true",
        help="Record but do not fail a repeatability-check mismatch.",
    )
    parser.add_argument(
        "--allow-muq-flash",
        action="store_true",
        help="Permit a MuQ checkpoint configured for its reported flash path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec = MODEL_SPECS[args.encoder]

    if not args.audio_manifest.is_file():
        raise SystemExit(f"Audio manifest not found: {args.audio_manifest}")
    if not math.isfinite(args.window_seconds) or args.window_seconds <= 0:
        raise SystemExit("--window-seconds must be a positive finite number")
    if args.batch_size < 0:
        raise SystemExit("--batch-size must be >= 0")
    if args.max_tracks is not None and args.max_tracks < 1:
        raise SystemExit("--max-tracks must be >= 1")

    window_samples_float = args.window_seconds * SAMPLE_RATE
    window_samples = int(round(window_samples_float))
    if not math.isclose(window_samples, window_samples_float, abs_tol=1e-8):
        raise SystemExit(
            "--window-seconds must map to an integer number of 24 kHz samples"
        )

    batch_size = args.batch_size or spec.default_batch_size
    model_id = args.model_id or spec.default_model_id
    output_dir = args.output_dir or Path("data/embeddings") / args.encoder
    device = choose_device(args.device)

    print(f"Resolving model revision for {model_id} ...")
    revision = resolve_hugging_face_revision(model_id, args.revision)
    load_revision = None if Path(model_id).exists() else revision

    configuration = {
        "extractor_version": EXTRACTOR_VERSION,
        "encoder": spec.key,
        "model_id": model_id,
        "resolved_model_revision": revision,
        "sample_rate": SAMPLE_RATE,
        "window_samples": window_samples,
        "window_seconds": args.window_seconds,
        "window_hop_samples": window_samples,
        "overlap_samples": 0,
        "layer": args.layer,
        "song_pooling": args.song_pooling,
        "embedding_dimension": spec.embedding_dimension,
        "dtype": "float32",
        "seed": args.seed,
    }
    fingerprint = configuration_fingerprint(configuration)

    windows_dir = output_dir / "windows"
    pooled_dir = output_dir / "pooled"
    metadata_dir = output_dir / "metadata"
    state_path = output_dir / "embedding_state.jsonl"
    errors_path = output_dir / "embedding_errors.jsonl"
    manifest_path = output_dir / "embedding_manifest.jsonl"
    audio_embeddings_path = output_dir / "audio_embeddings.jsonl"
    report_path = output_dir / "embedding_report.json"
    model_info_path = output_dir / "model_info.json"

    output_dir.mkdir(parents=True, exist_ok=True)
    compatible_model_info(model_info_path, fingerprint)
    windows_dir.mkdir(parents=True, exist_ok=True)
    pooled_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    audio_rows = available_audio_rows(args.audio_manifest)
    if not audio_rows:
        raise SystemExit(
            f"No usable physical audio files found through {args.audio_manifest}"
        )

    state = load_state(state_path)
    pending: list[tuple[str, Path]] = []
    already_complete = 0
    for video_id, row in sorted(audio_rows.items()):
        complete = track_is_complete(
            video_id=video_id,
            fingerprint=fingerprint,
            embedding_dimension=spec.embedding_dimension,
            windows_dir=windows_dir,
            pooled_dir=pooled_dir,
            metadata_dir=metadata_dir,
        )
        if complete and not args.overwrite:
            already_complete += 1
        else:
            pending.append((video_id, Path(str(row["audio_path"]))))

    if args.max_tracks is not None:
        pending = pending[: args.max_tracks]

    print(f"Windowed {spec.display_name} extraction")
    print(f"  physical audio tracks:  {len(audio_rows)}")
    print(f"  already complete:       {already_complete}")
    print(f"  processing this run:    {len(pending)}")
    print(f"  model revision:         {revision}")
    print(f"  device:                 {device}")
    print(f"  sample rate:            {SAMPLE_RATE} Hz")
    print(f"  window/hop:             {args.window_seconds:g}s / {args.window_seconds:g}s")
    print(f"  embedding dimension:    {spec.embedding_dimension}")
    print(f"  hidden layer:           {args.layer}")
    print(f"  song pooling:           {args.song_pooling}")
    print(f"  window batch size:      {batch_size}")
    print()

    base_model_info = {
        "configuration_fingerprint": fingerprint,
        "configuration": configuration,
        "model_spec": asdict(spec),
        "device_requested": args.device,
        "device_selected": device,
        "ffmpeg": args.ffmpeg,
        "seed": args.seed,
        "repeatability_check_enabled": not args.skip_repeatability_check,
    }

    adapter: MertAdapter | MuqAdapter | None = None
    if pending:
        set_inference_seed(args.seed)
        print("Loading model ...")
        adapter = create_adapter(
            args.encoder,
            model_id=model_id,
            revision=load_revision,
            device=device,
            expected_dimension=spec.embedding_dimension,
            allow_muq_flash=args.allow_muq_flash,
        )
        base_model_info["loaded_model"] = adapter.description()
    elif model_info_path.is_file():
        previous_info = load_json_object(model_info_path)
        if previous_info is not None and "loaded_model" in previous_info:
            base_model_info["loaded_model"] = previous_info["loaded_model"]

    atomic_write_json(model_info_path, base_model_info)

    stop_requested = False

    def request_stop(_signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True
        print("\nStop requested; finishing the current track.")

    signal.signal(signal.SIGINT, request_stop)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, request_stop)

    counters: Counter[str] = Counter()
    window_counts: list[int] = []
    durations: list[float] = []
    extraction_seconds: list[float] = []
    repeatability_checked = not args.skip_repeatability_check
    started = time.monotonic()

    if adapter is not None:
        for index, (video_id, audio_path) in enumerate(pending, 1):
            if stop_requested:
                break
            print(f"[{index}/{len(pending)}] {video_id}")
            track_started = time.monotonic()
            try:
                waveform = decode_audio_ffmpeg(
                    audio_path,
                    ffmpeg=args.ffmpeg,
                    sample_rate=SAMPLE_RATE,
                )
                windows, pooled, details = extract_track(
                    adapter,
                    waveform,
                    window_samples=window_samples,
                    batch_size=batch_size,
                    layer=args.layer,
                    song_pooling=args.song_pooling,
                    repeatability_check=(
                        repeatability_checked
                        and counters["repeatability_checks"] == 0
                    ),
                    allow_nondeterministic=args.allow_nondeterministic,
                )
                repeatability = details["repeatability_check"]
                if repeatability is not None:
                    counters["repeatability_checks"] += 1
                    if not repeatability["allclose"]:
                        counters["repeatability_mismatches"] += 1

                windows_path = windows_dir / f"{video_id}.npy"
                pooled_path = pooled_dir / f"{video_id}.npy"
                metadata_path = metadata_dir / f"{video_id}.json"
                atomic_save_npy(windows_path, windows)
                atomic_save_npy(pooled_path, pooled)

                valid_samples = details["window_valid_samples"]
                elapsed = time.monotonic() - track_started
                metadata = {
                    "video_id": video_id,
                    "status": "embedded",
                    "configuration_fingerprint": fingerprint,
                    "encoder": spec.key,
                    "model_id": model_id,
                    "model_revision": revision,
                    "layer": args.layer,
                    "sample_rate": SAMPLE_RATE,
                    "source_audio_samples": int(waveform.size),
                    "source_duration_seconds": float(waveform.size / SAMPLE_RATE),
                    "window_samples": window_samples,
                    "window_seconds": args.window_seconds,
                    "window_hop_samples": window_samples,
                    "overlap_samples": 0,
                    "window_valid_samples": valid_samples,
                    "window_valid_seconds": [
                        float(value / SAMPLE_RATE) for value in valid_samples
                    ],
                    "last_window_valid_seconds": float(
                        valid_samples[-1] / SAMPLE_RATE
                    ),
                    "tail_padding_samples": int(window_samples - valid_samples[-1]),
                    "tail_padding_excluded_by_attention_mask": True,
                    "encoder_frame_counts": details["encoder_frame_counts"],
                    "windows_path": str(windows_path),
                    "windows_shape": list(windows.shape),
                    "pooled_path": str(pooled_path),
                    "pooled_shape": list(pooled.shape),
                    "embedding_dimension": spec.embedding_dimension,
                    "pooling": args.song_pooling,
                    "repeatability_check": repeatability,
                    "elapsed_seconds": round(elapsed, 6),
                    "timestamp": time.time(),
                }
                atomic_write_json(metadata_path, metadata)

                record = {
                    "video_id": video_id,
                    "status": "embedded",
                    "reason": "success",
                    "audio_path": str(audio_path),
                    "windows_path": str(windows_path),
                    "pooled_path": str(pooled_path),
                    "metadata_path": str(metadata_path),
                    "window_count": int(windows.shape[0]),
                    "embedding_dimension": int(windows.shape[1]),
                    "configuration_fingerprint": fingerprint,
                    "elapsed_seconds": round(elapsed, 6),
                    "timestamp": time.time(),
                }
                state[video_id] = record
                append_jsonl(state_path, record)
                counters["embedded"] += 1
                window_counts.append(int(windows.shape[0]))
                durations.append(float(waveform.size / SAMPLE_RATE))
                extraction_seconds.append(elapsed)
                print(
                    f"  -> windows={windows.shape} pooled={pooled.shape} "
                    f"duration={durations[-1]:.2f}s time={elapsed:.2f}s"
                )
            except Exception as exc:
                elapsed = time.monotonic() - track_started
                record = {
                    "video_id": video_id,
                    "status": "error",
                    "reason": f"{type(exc).__name__}: {exc}",
                    "audio_path": str(audio_path),
                    "configuration_fingerprint": fingerprint,
                    "elapsed_seconds": round(elapsed, 6),
                    "timestamp": time.time(),
                }
                state[video_id] = record
                append_jsonl(state_path, record)
                append_jsonl(errors_path, record)
                counters["errors"] += 1
                print(f"  ERROR: {record['reason']}")

    manifest_rows, complete_rows = build_manifests(
        audio_rows=audio_rows,
        state=state,
        fingerprint=fingerprint,
        spec=spec,
        model_id=model_id,
        revision=revision,
        layer=args.layer,
        window_seconds=args.window_seconds,
        song_pooling=args.song_pooling,
        windows_dir=windows_dir,
        pooled_dir=pooled_dir,
        metadata_dir=metadata_dir,
    )
    atomic_write_jsonl(manifest_path, manifest_rows)
    atomic_write_jsonl(audio_embeddings_path, complete_rows)

    total_elapsed = time.monotonic() - started
    status_counts = Counter(row["status"] for row in manifest_rows)
    all_window_counts = [
        int(row["window_count"])
        for row in complete_rows
        if isinstance(row.get("window_count"), int)
    ]
    report = {
        "configuration_fingerprint": fingerprint,
        "configuration": configuration,
        "audio_manifest": str(args.audio_manifest),
        "physical_audio_tracks": len(audio_rows),
        "complete_embeddings": len(complete_rows),
        "status_counts": dict(status_counts),
        "this_run": {
            "requested": len(pending),
            "embedded": counters["embedded"],
            "errors": counters["errors"],
            "interrupted": stop_requested,
            "repeatability_checks": counters["repeatability_checks"],
            "repeatability_mismatches": counters["repeatability_mismatches"],
            "elapsed_seconds": round(total_elapsed, 3),
            "mean_track_extraction_seconds": (
                float(np.mean(extraction_seconds))
                if extraction_seconds
                else None
            ),
        },
        "window_count_statistics_complete": {
            "minimum": min(all_window_counts) if all_window_counts else None,
            "maximum": max(all_window_counts) if all_window_counts else None,
            "mean": (
                float(np.mean(all_window_counts))
                if all_window_counts
                else None
            ),
        },
        "coverage_invariant": (
            "non-overlapping logical windows cover each decoded source sample "
            "exactly once; final right-padding is excluded by the model attention mask"
        ),
        "outputs": {
            "windows": str(windows_dir),
            "pooled": str(pooled_dir),
            "metadata": str(metadata_dir),
            "embedding_manifest": str(manifest_path),
            "audio_embeddings": str(audio_embeddings_path),
            "state": str(state_path),
            "errors": str(errors_path),
            "model_info": str(model_info_path),
        },
    }
    atomic_write_json(report_path, report)

    print()
    print("Windowed embedding extraction complete")
    print(f"  complete embeddings: {len(complete_rows)}/{len(audio_rows)}")
    print(f"  new embeddings:      {counters['embedded']}")
    print(f"  errors this run:     {counters['errors']}")
    print(f"  windows:             {windows_dir}")
    print(f"  pooled:              {pooled_dir}")
    print(f"  manifest:            {manifest_path}")
    print(f"  report:              {report_path}")


if __name__ == "__main__":
    main()
