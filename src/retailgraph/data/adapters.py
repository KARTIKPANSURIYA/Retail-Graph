"""Strict adapters for RetailGraph normalized JSON metadata.

These do not claim compatibility with unverified upstream annotation schemas. Convert
upstream data explicitly after comparing it with the pinned manifest and documentation.
"""

import json
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from retailgraph.schema.records import (
    ActionEvaluationManifest,
    RetailActionLabel,
    RetailGazeLabel,
    VideoSample,
)

T = TypeVar("T", bound=BaseModel)


def _read_jsonl(path: Path, model: type[T]) -> list[T]:
    records: list[T] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            payload: Any = json.loads(line)
            records.append(model.model_validate(payload))
        except (json.JSONDecodeError, ValidationError) as error:
            raise ValueError(f"{path}:{line_number}: invalid {model.__name__}: {error}") from error
    return records


def load_samples(path: Path) -> list[VideoSample]:
    return _read_jsonl(path, VideoSample)


def load_retail_action_labels(path: Path) -> list[RetailActionLabel]:
    return _read_jsonl(path, RetailActionLabel)


def load_retail_gaze_labels(path: Path) -> list[RetailGazeLabel]:
    return _read_jsonl(path, RetailGazeLabel)


def load_action_evaluation_manifest(path: Path) -> ActionEvaluationManifest:
    """Load the authoritative sample universe for an action evaluation split."""
    try:
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
        return ActionEvaluationManifest.model_validate(payload)
    except (json.JSONDecodeError, ValidationError) as error:
        raise ValueError(f"{path}: invalid ActionEvaluationManifest: {error}") from error
