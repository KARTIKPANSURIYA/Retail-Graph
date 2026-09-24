"""Deterministic contract smoke predictors, not research baselines."""

from retailgraph.schema.records import (
    AttentionType,
    ModelPrediction,
    NormalizedPoint,
    RetailActionLabel,
    RetailGazeLabel,
)

SMOKE_MODEL_VERSION = "deterministic-contract-smoke-v1"


def predict_gaze(labels: list[RetailGazeLabel]) -> list[ModelPrediction]:
    """Return the image center; labels are used only to enumerate sample IDs."""
    return [
        ModelPrediction(
            schema_version="2.0",
            sample_id=label.sample_id,
            prediction_type=AttentionType.GAZE_POINT,
            point=NormalizedPoint(x=0.5, y=0.5),
            confidence=0.5,
            model_version=SMOKE_MODEL_VERSION,
        )
        for label in labels
    ]


def predict_actions(labels: list[RetailActionLabel]) -> list[ModelPrediction]:
    """Copy fixture intervals/classes solely to exercise evaluation plumbing."""
    return [
        ModelPrediction(
            schema_version="2.0",
            sample_id=label.sample_id,
            prediction_id=f"smoke-{label.event_id}",
            prediction_type=label.action,
            interval=label.interval,
            confidence=0.5,
            model_version=SMOKE_MODEL_VERSION,
        )
        for label in labels
    ]
