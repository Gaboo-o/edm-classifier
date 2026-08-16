"""Weak EDM genre labeling from collected metadata evidence.

Default mode uses a local Ollama model, so no paid LLM API is required.
A manual mode can emit a self-contained bundle for use in ChatGPT or another LLM.

Expected project layout::

    edm-classifier/
      config/
        taxonomy.yaml
        weak_label_prompt.md
        weak_label_input.schema.json
        weak_label_output.schema.json
      src/edm_classifier/labeling/weak_labeler.py
      data/evidence/
      data/labels/
      data/review/

Example::

    python -m edm_classifier.labeling.weak_labeler \
      data/evidence/take_you_down.json \
      --backend ollama \
      --model gpt-oss:20b

Manual bundle::

    python -m edm_classifier.labeling.weak_labeler \
      data/evidence/take_you_down.json \
      --backend manual
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator


DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "gpt-oss:20b"
DEFAULT_TIMEOUT_SECONDS = 900


class WeakLabelerError(RuntimeError):
    """Base exception for weak-labeling failures."""


class ValidationError(WeakLabelerError):
    """Raised when input or model output does not match its schema."""


class BackendError(WeakLabelerError):
    """Raised when an LLM backend fails."""


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    config: Path
    labels: Path
    review: Path

    @classmethod
    def discover(cls) -> "ProjectPaths":
        # .../src/edm_classifier/labeling/weak_labeler.py -> project root
        root = Path(__file__).resolve().parents[3]
        return cls(
            root=root,
            config=root / "config",
            labels=root / "data" / "labels",
            review=root / "data" / "review",
        )


@dataclass(frozen=True)
class LabelingResources:
    taxonomy_text: str
    taxonomy: dict[str, Any]
    prompt: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]


@dataclass(frozen=True)
class LabelResult:
    data: dict[str, Any]
    backend: str
    model: str | None


# ---------------------------------------------------------------------------
# File / schema helpers
# ---------------------------------------------------------------------------


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise WeakLabelerError(f"File not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise WeakLabelerError(f"Invalid JSON in {path}: {exc}") from exc

    if not isinstance(value, dict):
        raise WeakLabelerError(f"Expected a JSON object in {path}")
    return value


def load_resources(config_dir: Path) -> LabelingResources:
    taxonomy_path = config_dir / "taxonomy.yaml"
    prompt_path = config_dir / "weak_label_prompt.md"
    input_schema_path = config_dir / "weak_label_input.schema.json"
    output_schema_path = config_dir / "weak_label_output.schema.json"

    try:
        taxonomy_text = taxonomy_path.read_text(encoding="utf-8")
        taxonomy = yaml.safe_load(taxonomy_text)
        prompt = prompt_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise WeakLabelerError(f"Missing config file: {exc.filename}") from exc
    except yaml.YAMLError as exc:
        raise WeakLabelerError(f"Invalid taxonomy YAML: {exc}") from exc

    if not isinstance(taxonomy, dict):
        raise WeakLabelerError("taxonomy.yaml must contain a YAML mapping/object")

    return LabelingResources(
        taxonomy_text=taxonomy_text,
        taxonomy=taxonomy,
        prompt=prompt,
        input_schema=load_json(input_schema_path),
        output_schema=load_json(output_schema_path),
    )


def schema_errors(instance: Any, schema: dict[str, Any]) -> list[str]:
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(instance), key=lambda e: list(e.absolute_path))

    formatted: list[str] = []
    for error in errors:
        location = ".".join(str(part) for part in error.absolute_path) or "<root>"
        formatted.append(f"{location}: {error.message}")
    return formatted


def validate_or_raise(instance: Any, schema: dict[str, Any], label: str) -> None:
    errors = schema_errors(instance, schema)
    if errors:
        details = "\n  - " + "\n  - ".join(errors[:20])
        if len(errors) > 20:
            details += f"\n  - ... {len(errors) - 20} more"
        raise ValidationError(f"{label} failed schema validation:{details}")


def taxonomy_label_ids(taxonomy: dict[str, Any]) -> set[str]:
    genres = taxonomy.get("genres")
    if not isinstance(genres, list):
        raise WeakLabelerError("taxonomy.yaml is missing a 'genres' list")

    result: set[str] = set()
    for genre in genres:
        if isinstance(genre, dict) and isinstance(genre.get("id"), str):
            result.add(genre["id"])
    return result


def validate_semantics(
    output: dict[str, Any],
    evidence_record: dict[str, Any],
    taxonomy: dict[str, Any],
) -> None:
    """Check constraints that are awkward or intentionally duplicated outside JSON Schema."""

    problems: list[str] = []

    expected_track_id = evidence_record["track"]["track_id"]
    if output.get("track_id") != expected_track_id:
        problems.append(
            f"track_id mismatch: expected {expected_track_id!r}, got {output.get('track_id')!r}"
        )

    expected_taxonomy_version = evidence_record.get("taxonomy_version")
    if output.get("taxonomy_version") != expected_taxonomy_version:
        problems.append(
            "taxonomy_version mismatch: "
            f"expected {expected_taxonomy_version!r}, got {output.get('taxonomy_version')!r}"
        )

    evidence_ids = {
        item.get("id")
        for item in evidence_record.get("evidence", [])
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    }
    allowed_labels = taxonomy_label_ids(taxonomy)

    emitted_ids: list[str] = []
    for bucket in ("labels", "candidates"):
        for item in output.get(bucket, []):
            label_id = item.get("id")
            if isinstance(label_id, str):
                emitted_ids.append(label_id)
                if label_id not in allowed_labels:
                    problems.append(f"unknown taxonomy label: {label_id}")

            for ref in item.get("evidence_refs", []):
                if ref not in evidence_ids:
                    problems.append(f"unknown evidence reference: {ref}")

    for item in output.get("ignored_evidence", []):
        ref = item.get("evidence_ref")
        if ref not in evidence_ids:
            problems.append(f"unknown ignored evidence reference: {ref}")

    if len(emitted_ids) != len(set(emitted_ids)):
        problems.append("the same label appears more than once across labels/candidates")

    for item in output.get("labels", []):
        confidence = item.get("confidence")
        if isinstance(confidence, (int, float)) and confidence < 0.85:
            problems.append(
                f"accepted label {item.get('id')} has confidence {confidence}; minimum is 0.85"
            )

    if problems:
        raise ValidationError(
            "Weak-label output failed semantic validation:\n  - " + "\n  - ".join(problems)
        )


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def build_user_message(
    evidence_record: dict[str, Any],
    resources: LabelingResources,
) -> str:
    """Build the per-track message.

    The behavioral instructions stay in weak_label_prompt.md. The user message contains
    only the controlled taxonomy, current evidence record, and output schema.
    """

    return "\n".join(
        [
            "Classify the following single recording using the supplied evidence only.",
            "",
            "## TAXONOMY.YAML",
            "```yaml",
            resources.taxonomy_text.rstrip(),
            "```",
            "",
            "## TRACK EVIDENCE JSON",
            "```json",
            json.dumps(evidence_record, ensure_ascii=False, indent=2),
            "```",
            "",
            "## REQUIRED OUTPUT JSON SCHEMA",
            "```json",
            json.dumps(resources.output_schema, ensure_ascii=False, indent=2),
            "```",
            "",
            "Return only the JSON object required by the schema.",
        ]
    )


# ---------------------------------------------------------------------------
# Ollama backend
# ---------------------------------------------------------------------------


def ollama_chat(
    *,
    system_prompt: str,
    user_message: str,
    output_schema: dict[str, Any],
    model: str,
    host: str,
    timeout_seconds: int,
    think: str,
) -> str:
    """Call a local Ollama server using JSON-schema structured output."""

    endpoint = f"{host.rstrip('/')}/api/chat"
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "stream": False,
        "format": output_schema,
        "options": {
            # Conservative labeling benefits from deterministic-ish decoding.
            "temperature": 0,
        },
    }

    # GPT-OSS uses string-valued reasoning effort in Ollama. Other models may reject
    # this option, so callers can pass --think none to omit it.
    if think != "none":
        payload["think"] = think

    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise BackendError(
            f"Ollama returned HTTP {exc.code}. Response: {details[:1000]}"
        ) from exc
    except urllib.error.URLError as exc:
        raise BackendError(
            "Could not connect to Ollama at "
            f"{host}. Make sure Ollama is installed/running and the model is pulled. "
            f"Original error: {exc.reason}"
        ) from exc

    try:
        envelope = json.loads(body)
        content = envelope["message"]["content"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise BackendError(f"Unexpected Ollama response: {body[:1000]}") from exc

    if not isinstance(content, str) or not content.strip():
        raise BackendError("Ollama returned an empty response")

    return content


def parse_model_json(text: str) -> dict[str, Any]:
    """Parse a model's final answer, tolerating accidental Markdown fences."""

    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)

    try:
        value = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise ValidationError(f"Model returned invalid JSON: {exc}\n{text[:1000]}") from exc

    if not isinstance(value, dict):
        raise ValidationError("Model output must be a JSON object")
    return value


def label_with_ollama(
    evidence_record: dict[str, Any],
    resources: LabelingResources,
    *,
    model: str = DEFAULT_OLLAMA_MODEL,
    host: str = DEFAULT_OLLAMA_HOST,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    think: str = "low",
) -> LabelResult:
    user_message = build_user_message(evidence_record, resources)

    raw = ollama_chat(
        system_prompt=resources.prompt,
        user_message=user_message,
        output_schema=resources.output_schema,
        model=model,
        host=host,
        timeout_seconds=timeout_seconds,
        think=think,
    )
    output = parse_model_json(raw)
    validate_or_raise(output, resources.output_schema, "LLM output")
    validate_semantics(output, evidence_record, resources.taxonomy)

    return LabelResult(data=output, backend="ollama", model=model)


# ---------------------------------------------------------------------------
# Manual backend
# ---------------------------------------------------------------------------


def build_manual_bundle(
    evidence_record: dict[str, Any],
    resources: LabelingResources,
) -> str:
    """Produce one self-contained Markdown artifact for manual LLM use."""

    return "\n".join(
        [
            "# Manual EDM Weak-Labeling Bundle",
            "",
            "Use the instructions below to label exactly one track.",
            "Return only the requested JSON object.",
            "",
            "---",
            "",
            resources.prompt.rstrip(),
            "",
            "---",
            "",
            build_user_message(evidence_record, resources),
            "",
        ]
    )


def write_manual_bundle(
    evidence_record: dict[str, Any],
    resources: LabelingResources,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        build_manual_bundle(evidence_record, resources),
        encoding="utf-8",
    )


def validate_manual_result(
    result_path: Path,
    evidence_record: dict[str, Any],
    resources: LabelingResources,
) -> LabelResult:
    output = load_json(result_path)
    validate_or_raise(output, resources.output_schema, "Manual LLM output")
    validate_semantics(output, evidence_record, resources.taxonomy)
    return LabelResult(data=output, backend="manual", model=None)


# ---------------------------------------------------------------------------
# Output routing
# ---------------------------------------------------------------------------


def safe_track_name(track_id: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", track_id).strip("._-")
    return cleaned or "track"


def default_result_path(
    result: dict[str, Any],
    paths: ProjectPaths,
) -> Path:
    track_name = safe_track_name(result["track_id"])
    base = paths.labels if result["status"] == "labeled" else paths.review
    return base / f"{track_name}.json"


def write_result(result: LabelResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result.data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Map collected music metadata evidence into the EDM taxonomy.",
    )
    parser.add_argument("evidence", type=Path, help="Input evidence JSON")
    parser.add_argument(
        "--backend",
        choices=("ollama", "manual"),
        default="ollama",
        help="Label locally with Ollama (default) or emit a manual LLM bundle.",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=None,
        help="Config directory containing taxonomy/prompt/schemas.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Explicit output path. Otherwise results route to data/labels or data/review.",
    )

    # Ollama options
    parser.add_argument("--model", default=DEFAULT_OLLAMA_MODEL)
    parser.add_argument("--ollama-host", default=DEFAULT_OLLAMA_HOST)
    parser.add_argument(
        "--think",
        choices=("none", "low", "medium", "high"),
        default="low",
        help="Reasoning effort for compatible Ollama models; GPT-OSS supports low/medium/high.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="Ollama request timeout in seconds.",
    )

    # Manual options
    parser.add_argument(
        "--manual-result",
        type=Path,
        default=None,
        help="Validate/import a JSON response previously produced in manual mode.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = ProjectPaths.discover()
    config_dir = args.config_dir or paths.config

    try:
        resources = load_resources(config_dir)
        evidence_record = load_json(args.evidence)
        validate_or_raise(evidence_record, resources.input_schema, "Evidence input")

        if evidence_record.get("taxonomy_version") != resources.taxonomy.get("taxonomy", {}).get(
            "version"
        ):
            raise ValidationError(
                "Evidence taxonomy_version does not match config/taxonomy.yaml"
            )

        if args.backend == "manual":
            if args.manual_result:
                result = validate_manual_result(
                    args.manual_result,
                    evidence_record,
                    resources,
                )
                output_path = args.output or default_result_path(result.data, paths)
                write_result(result, output_path)
                print(f"Validated manual result -> {output_path}")
                return 0

            track_id = evidence_record["track"]["track_id"]
            output_path = args.output or (
                paths.review / f"{safe_track_name(track_id)}.manual.md"
            )
            write_manual_bundle(evidence_record, resources, output_path)
            print(f"Manual labeling bundle -> {output_path}")
            return 0

        result = label_with_ollama(
            evidence_record,
            resources,
            model=args.model,
            host=args.ollama_host,
            timeout_seconds=args.timeout,
            think=args.think,
        )
        output_path = args.output or default_result_path(result.data, paths)
        write_result(result, output_path)

        labels = ", ".join(item["id"] for item in result.data.get("labels", [])) or "none"
        print(
            f"{result.data['track_id']}: status={result.data['status']} "
            f"labels={labels} -> {output_path}"
        )
        return 0

    except WeakLabelerError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
