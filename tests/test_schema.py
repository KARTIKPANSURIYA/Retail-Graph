import pytest
from pydantic import ValidationError

from retailgraph.schema.coordinates import normalized_to_pixel, pixel_to_normalized
from retailgraph.schema.records import (
    ActionClass,
    ModelPrediction,
    NormalizedPoint,
    PixelBox,
    TemporalInterval,
)


def test_coordinate_round_trip() -> None:
    point = pixel_to_normalized(320, 240, 640, 480)
    assert point == NormalizedPoint(x=0.5, y=0.5)
    assert normalized_to_pixel(point, 640, 480) == (320.0, 240.0)


@pytest.mark.parametrize("payload", [{"x": -0.1, "y": 0.5}, {"x": 0.5, "y": 1.1}])
def test_normalized_coordinates_reject_out_of_range(payload: dict[str, float]) -> None:
    with pytest.raises(ValidationError):
        NormalizedPoint.model_validate(payload)


def test_invalid_interval_and_box_are_rejected() -> None:
    with pytest.raises(ValidationError, match="start_s"):
        TemporalInterval(start_s=2, end_s=1)
    with pytest.raises(ValidationError, match="minima"):
        PixelBox(x_min=10, y_min=2, x_max=5, y_max=4)


def test_action_prediction_requires_interval() -> None:
    with pytest.raises(ValidationError, match="require an interval"):
        ModelPrediction(
            schema_version="2.0",
            sample_id="x",
            prediction_type=ActionClass.TAKE,
            model_version="v1",
            confidence=0.5,
        )


def test_action_v1_record_requires_explicit_migration() -> None:
    from retailgraph.schema.records import RetailActionLabel

    with pytest.raises(ValidationError, match="schema_version"):
        RetailActionLabel.model_validate(
            {
                "sample_id": "sample-a",
                "event_id": "event-a",
                "action": "take",
                "interval": {"start_s": 1, "end_s": 2},
                "points": [{"view_id": "v", "point": {"x": 0.5, "y": 0.5}}],
            }
        )


def test_prediction_v1_record_requires_explicit_migration() -> None:
    with pytest.raises(ValidationError, match="schema_version"):
        ModelPrediction.model_validate(
            {
                "sample_id": "sample-a",
                "prediction_type": "unknown",
                "model_version": "old",
                "confidence": 0.5,
            }
        )


def test_action_evaluation_manifest_rejects_duplicate_samples() -> None:
    from retailgraph.schema.records import ActionEvaluationManifest

    with pytest.raises(ValidationError, match="sample_ids must be unique"):
        ActionEvaluationManifest(
            dataset_version="synthetic-v1",
            split="test",
            sample_ids=["sample-a", "sample-a"],
        )
