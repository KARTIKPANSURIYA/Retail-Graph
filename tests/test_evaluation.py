import pytest

from retailgraph.evaluation.action import evaluate_actions, temporal_iou
from retailgraph.evaluation.gaze import evaluate_gaze
from retailgraph.schema.records import (
    ActionClass,
    AttentionType,
    ModelPrediction,
    NormalizedPoint,
    PixelBox,
    RetailActionLabel,
    RetailGazeLabel,
    TemporalInterval,
    ViewPoint,
)


def test_temporal_iou_and_per_class_counts() -> None:
    interval = TemporalInterval(start_s=0, end_s=2)
    assert temporal_iou(interval, TemporalInterval(start_s=1, end_s=3)) == pytest.approx(1 / 3)
    label = RetailActionLabel(
        event_id="e",
        action=ActionClass.TAKE,
        interval=interval,
        points=[ViewPoint(view_id="v", point=NormalizedPoint(x=0.5, y=0.5))],
    )
    prediction = ModelPrediction(
        sample_id="e",
        prediction_type=ActionClass.TAKE,
        interval=interval,
        confidence=1,
        model_version="test",
    )
    result = evaluate_actions([label], [prediction])
    assert result["metric"] == "simple_temporal_event_pr_not_official_map"
    assert result["per_class"]["take"]["precision"] == 1
    assert result["per_class"]["put"]["recall"] == 0


def test_gaze_reports_evaluated_and_skipped() -> None:
    labels = [
        RetailGazeLabel(
            sample_id="a",
            head_box=PixelBox(x_min=0, y_min=0, x_max=2, y_max=2),
            gaze_target=NormalizedPoint(x=0.5, y=0.5),
            region_id="one",
        ),
        RetailGazeLabel(
            sample_id="b",
            head_box=PixelBox(x_min=0, y_min=0, x_max=2, y_max=2),
            gaze_target=NormalizedPoint(x=0, y=0),
        ),
    ]
    predictions = [
        ModelPrediction(
            sample_id="a",
            prediction_type=AttentionType.GAZE_POINT,
            point=NormalizedPoint(x=0.5, y=0.5),
            region_id="one",
            confidence=0.8,
            model_version="test",
        )
    ]
    metrics = evaluate_gaze(labels, predictions)
    assert metrics.mean_normalized_distance == 0
    assert metrics.region_accuracy == 1
    assert (metrics.sample_count, metrics.point_skipped, metrics.region_skipped) == (2, 1, 1)
    assert metrics.calibration_status.startswith("not_computed")
