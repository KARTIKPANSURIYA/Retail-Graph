"""Versioned normalized records; dataset-specific annotations remain distinct."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = "1.0"
ACTION_SCHEMA_VERSION = "2.0"


class StrictRecord(BaseModel):
    """Reject unknown fields so upstream schema changes cannot pass silently."""

    model_config = ConfigDict(extra="forbid")
    schema_version: str = Field(default=SCHEMA_VERSION, pattern=r"^1\.")


class DatasetName(StrEnum):
    RETAIL_ACTION = "retail_action"
    RETAIL_GAZE = "retail_gaze"


class Split(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class ActionClass(StrEnum):
    TAKE = "take"
    PUT = "put"
    TOUCH = "touch"


class AttentionType(StrEnum):
    GAZE_POINT = "gaze_point"
    SHELF_REGION = "shelf_region"
    UNKNOWN = "unknown"


class DatasetProvenance(StrictRecord):
    dataset: DatasetName
    source_url: str
    version: str
    split: Split
    store_id: str | None = None
    subject_id: str | None = None
    session_id: str | None = None


class NormalizedPoint(StrictRecord):
    """Image point in [0,1], origin top-left, x rightward and y downward."""

    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)


class PixelBox(StrictRecord):
    """Half-open pixel box [x_min,y_min,x_max,y_max]."""

    x_min: int = Field(ge=0)
    y_min: int = Field(ge=0)
    x_max: int = Field(gt=0)
    y_max: int = Field(gt=0)

    @model_validator(mode="after")
    def ordered(self) -> PixelBox:
        if self.x_min >= self.x_max or self.y_min >= self.y_max:
            raise ValueError("box minima must be strictly less than maxima")
        return self


class TemporalInterval(StrictRecord):
    """Half-open interval [start_s,end_s) in seconds."""

    start_s: float = Field(ge=0)
    end_s: float = Field(gt=0)

    @model_validator(mode="after")
    def ordered(self) -> TemporalInterval:
        if self.start_s >= self.end_s:
            raise ValueError("start_s must be less than end_s")
        return self


class CameraView(StrictRecord):
    view_id: str
    video_path: Path
    width: int | None = Field(default=None, gt=0)
    height: int | None = Field(default=None, gt=0)
    fps: float | None = Field(default=None, gt=0)
    synchronization_group: str | None = None
    time_offset_s: float = 0.0


class VideoSample(StrictRecord):
    sample_id: str
    provenance: DatasetProvenance
    views: list[CameraView] = Field(min_length=1)
    timestamp_s: float | None = Field(default=None, ge=0)


class ViewPoint(StrictRecord):
    view_id: str
    point: NormalizedPoint


class PoseReference(StrictRecord):
    view_id: str
    path: Path
    format: str


class ActionEvaluationManifest(StrictRecord):
    """Authoritative sample universe for one RetailAction evaluation split."""

    dataset_version: str = Field(min_length=1)
    split: Split
    sample_ids: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_sample_ids(self) -> ActionEvaluationManifest:
        if any(not sample_id.strip() for sample_id in self.sample_ids):
            raise ValueError("sample_ids must contain non-empty identifiers")
        if len(self.sample_ids) != len(set(self.sample_ids)):
            raise ValueError("sample_ids must be unique")
        return self


class RetailActionLabel(StrictRecord):
    """Normalized action event schema v2; v1 lacked a sample identifier."""

    schema_version: Literal["2.0"]
    sample_id: str = Field(min_length=1)
    event_id: str = Field(min_length=1)
    action: ActionClass
    interval: TemporalInterval
    points: list[ViewPoint] = Field(min_length=1)
    pose: list[PoseReference] = Field(default_factory=list)


class RetailGazeLabel(StrictRecord):
    sample_id: str
    head_box: PixelBox
    gaze_target: NormalizedPoint
    region_mask_path: Path | None = None
    region_id: str | None = None


class ModelPrediction(StrictRecord):
    """Prediction schema v2 adds stable IDs for action predictions."""

    schema_version: Literal["2.0"]
    sample_id: str = Field(min_length=1)
    prediction_id: str | None = Field(default=None, min_length=1)
    prediction_type: ActionClass | AttentionType
    model_version: str
    confidence: float = Field(ge=0, le=1)
    point: NormalizedPoint | None = None
    region_id: str | None = None
    timestamp_s: float | None = Field(default=None, ge=0)
    interval: TemporalInterval | None = None
    view_id: str | None = None

    @model_validator(mode="after")
    def payload_is_coherent(self) -> ModelPrediction:
        if self.prediction_type == AttentionType.UNKNOWN:
            if self.point is not None or self.region_id is not None:
                raise ValueError("unknown/abstained predictions cannot carry a target")
        elif isinstance(self.prediction_type, ActionClass):
            if self.interval is None:
                raise ValueError("action predictions require an interval")
            if self.prediction_id is None:
                raise ValueError("action predictions require a prediction_id")
        elif self.prediction_type == AttentionType.GAZE_POINT and self.point is None:
            raise ValueError("gaze-point predictions require a point")
        elif self.prediction_type == AttentionType.SHELF_REGION and self.region_id is None:
            raise ValueError("shelf-region predictions require region_id")
        return self
