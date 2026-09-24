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
        schema_version="2.0",
        sample_id="sample-a",
        event_id="e",
        action=ActionClass.TAKE,
        interval=interval,
        points=[ViewPoint(view_id="v", point=NormalizedPoint(x=0.5, y=0.5))],
    )
    prediction = ModelPrediction(
        schema_version="2.0",
        sample_id="sample-a",
        prediction_id="prediction-e",
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
            schema_version="2.0",
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


def _action_label(sample_id: str, event_id: str) -> RetailActionLabel:
    return RetailActionLabel(
        schema_version="2.0",
        sample_id=sample_id,
        event_id=event_id,
        action=ActionClass.TAKE,
        interval=TemporalInterval(start_s=1, end_s=2),
        points=[ViewPoint(view_id="v", point=NormalizedPoint(x=0.5, y=0.5))],
    )


def _action_prediction(sample_id: str, prediction_id: str) -> ModelPrediction:
    return ModelPrediction(
        schema_version="2.0",
        sample_id=sample_id,
        prediction_id=prediction_id,
        prediction_type=ActionClass.TAKE,
        interval=TemporalInterval(start_s=1, end_s=2),
        confidence=1,
        model_version="test",
    )


def test_action_matching_never_crosses_sample_boundary() -> None:
    labels = [_action_label("sample-a", "event-a"), _action_label("sample-b", "event-b")]
    result = evaluate_actions(labels, [_action_prediction("sample-a", "prediction-a")])
    take = result["per_class"]["take"]
    assert take == {
        "true_positives": 1,
        "false_positives": 0,
        "false_negatives": 1,
        "precision": 1.0,
        "recall": 0.5,
        "ground_truth_count": 2,
        "prediction_count": 1,
    }


def test_action_prediction_for_missing_sample_is_false_positive() -> None:
    result = evaluate_actions(
        [_action_label("sample-a", "event-a")],
        [_action_prediction("missing-sample", "prediction-a")],
    )
    assert result["predictions_for_unlabeled_samples"] == 1
    assert result["per_class"]["take"]["false_positives"] == 1
    assert result["per_class"]["take"]["false_negatives"] == 1
    assert result["samples_without_action_predictions"] == ["sample-a"]


def test_action_evaluation_reports_abstention_and_empty_classes() -> None:
    abstention = ModelPrediction(
        schema_version="2.0",
        sample_id="sample-a",
        prediction_type=AttentionType.UNKNOWN,
        confidence=0.2,
        model_version="test",
    )
    result = evaluate_actions([], [abstention])
    assert result["abstention_count"] == 1
    assert result["per_class"]["touch"]["ground_truth_count"] == 0
    assert result["per_class"]["touch"]["prediction_count"] == 0


@pytest.mark.parametrize("duplicate_kind", ["event", "prediction"])
def test_action_evaluation_rejects_duplicate_ids(duplicate_kind: str) -> None:
    labels = [_action_label("sample-a", "event-a")]
    predictions = [_action_prediction("sample-a", "prediction-a")]
    if duplicate_kind == "event":
        labels.append(_action_label("sample-b", "event-a"))
    else:
        predictions.append(_action_prediction("sample-b", "prediction-a"))
    with pytest.raises(ValueError, match=f"duplicate .*{duplicate_kind}_id"):
        evaluate_actions(labels, predictions)
