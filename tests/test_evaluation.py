import pytest

from retailgraph.evaluation.action import evaluate_actions, temporal_iou
from retailgraph.evaluation.gaze import evaluate_gaze
from retailgraph.schema.records import (
    ActionClass,
    ActionEvaluationManifest,
    AttentionType,
    ModelPrediction,
    NormalizedPoint,
    PixelBox,
    RetailActionLabel,
    RetailGazeLabel,
    TemporalInterval,
    ViewPoint,
)


def _manifest(*sample_ids: str) -> ActionEvaluationManifest:
    return ActionEvaluationManifest(
        dataset_version="synthetic-v1", split="test", sample_ids=list(sample_ids)
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
    result = evaluate_actions([label], [prediction], _manifest("sample-a"))
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
    result = evaluate_actions(
        labels,
        [_action_prediction("sample-a", "prediction-a")],
        _manifest("sample-a", "sample-b"),
    )
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


def test_action_prediction_on_empty_evaluated_sample_is_false_positive() -> None:
    result = evaluate_actions(
        [_action_label("sample-a", "event-a")],
        [_action_prediction("empty-sample", "prediction-a")],
        _manifest("sample-a", "empty-sample"),
    )
    assert result["samples_without_ground_truth_actions"] == ["empty-sample"]
    assert result["per_class"]["take"]["false_positives"] == 1
    assert result["per_class"]["take"]["false_negatives"] == 1
    assert result["samples_without_action_predictions"] == ["sample-a"]


def test_action_prediction_outside_manifest_is_rejected() -> None:
    with pytest.raises(ValueError, match="predictions outside evaluation manifest"):
        evaluate_actions(
            [],
            [_action_prediction("out-of-scope", "prediction-a")],
            _manifest("evaluated-empty-sample"),
        )


def test_action_label_outside_manifest_is_rejected() -> None:
    with pytest.raises(ValueError, match="labels outside evaluation manifest"):
        evaluate_actions(
            [_action_label("out-of-scope", "event-a")],
            [],
            _manifest("evaluated-empty-sample"),
        )


def test_action_evaluation_reports_abstention_and_empty_classes() -> None:
    abstention = ModelPrediction(
        schema_version="2.0",
        sample_id="sample-a",
        prediction_type=AttentionType.UNKNOWN,
        confidence=0.2,
        model_version="test",
    )
    result = evaluate_actions([], [abstention], _manifest("sample-a", "empty-sample"))
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
        evaluate_actions(labels, predictions, _manifest("sample-a", "sample-b"))


def test_action_label_without_predictions_is_false_negative() -> None:
    result = evaluate_actions(
        [_action_label("sample-a", "event-a")], [], _manifest("sample-a", "empty-sample")
    )
    assert result["per_class"]["take"]["false_negatives"] == 1
    assert result["samples_without_action_predictions"] == ["empty-sample", "sample-a"]
    assert result["evaluated_sample_count"] == 2


def test_gaze_rejects_duplicate_ground_truth_samples() -> None:
    label = RetailGazeLabel(
        sample_id="a",
        head_box=PixelBox(x_min=0, y_min=0, x_max=2, y_max=2),
        gaze_target=NormalizedPoint(x=0.5, y=0.5),
    )
    duplicate = RetailGazeLabel(
        sample_id="a",
        head_box=PixelBox(x_min=0, y_min=0, x_max=2, y_max=2),
        gaze_target=NormalizedPoint(x=0.2, y=0.2),
    )
    with pytest.raises(ValueError, match="duplicate ground-truth sample_id values: \\['a'\\]"):
        evaluate_gaze([label, duplicate], [])


def test_gaze_rejects_duplicate_predictions_for_same_sample() -> None:
    label = RetailGazeLabel(
        sample_id="a",
        head_box=PixelBox(x_min=0, y_min=0, x_max=2, y_max=2),
        gaze_target=NormalizedPoint(x=0.5, y=0.5),
    )
    p1 = ModelPrediction(
        schema_version="2.0",
        sample_id="a",
        prediction_type=AttentionType.GAZE_POINT,
        point=NormalizedPoint(x=0.5, y=0.5),
        confidence=0.8,
        model_version="test",
    )
    p2 = ModelPrediction(
        schema_version="2.0",
        sample_id="a",
        prediction_type=AttentionType.GAZE_POINT,
        point=NormalizedPoint(x=0.4, y=0.4),
        confidence=0.7,
        model_version="test",
    )
    with pytest.raises(ValueError, match="duplicate predictions for sample_id: \\['a'\\]"):
        evaluate_gaze([label], [p1, p2])


def test_gaze_rejects_duplicate_prediction_ids() -> None:
    l1 = RetailGazeLabel(
        sample_id="a",
        head_box=PixelBox(x_min=0, y_min=0, x_max=2, y_max=2),
        gaze_target=NormalizedPoint(x=0.5, y=0.5),
    )
    l2 = RetailGazeLabel(
        sample_id="b",
        head_box=PixelBox(x_min=0, y_min=0, x_max=2, y_max=2),
        gaze_target=NormalizedPoint(x=0.5, y=0.5),
    )
    p1 = ModelPrediction(
        schema_version="2.0",
        sample_id="a",
        prediction_id="dup-pred",
        prediction_type=AttentionType.GAZE_POINT,
        point=NormalizedPoint(x=0.5, y=0.5),
        confidence=0.8,
        model_version="test",
    )
    p2 = ModelPrediction(
        schema_version="2.0",
        sample_id="b",
        prediction_id="dup-pred",
        prediction_type=AttentionType.GAZE_POINT,
        point=NormalizedPoint(x=0.5, y=0.5),
        confidence=0.8,
        model_version="test",
    )
    with pytest.raises(ValueError, match="duplicate prediction_id values: \\['dup-pred'\\]"):
        evaluate_gaze([l1, l2], [p1, p2])


def test_gaze_rejects_predictions_outside_evaluated_sample_set() -> None:
    label = RetailGazeLabel(
        sample_id="a",
        head_box=PixelBox(x_min=0, y_min=0, x_max=2, y_max=2),
        gaze_target=NormalizedPoint(x=0.5, y=0.5),
    )
    pred_extra = ModelPrediction(
        schema_version="2.0",
        sample_id="extra-sample",
        prediction_type=AttentionType.GAZE_POINT,
        point=NormalizedPoint(x=0.5, y=0.5),
        confidence=0.8,
        model_version="test",
    )
    with pytest.raises(
        ValueError, match="predictions outside evaluated sample set: \\['extra-sample'\\]"
    ):
        evaluate_gaze([label], [pred_extra])


def test_gaze_rejects_invalid_prediction_types() -> None:
    label = RetailGazeLabel(
        sample_id="a",
        head_box=PixelBox(x_min=0, y_min=0, x_max=2, y_max=2),
        gaze_target=NormalizedPoint(x=0.5, y=0.5),
    )
    action_pred = ModelPrediction(
        schema_version="2.0",
        sample_id="a",
        prediction_id="p-1",
        prediction_type=ActionClass.TAKE,
        interval=TemporalInterval(start_s=0, end_s=1),
        confidence=0.9,
        model_version="test",
    )
    with pytest.raises(ValueError, match="non-gaze prediction types passed to gaze evaluation"):
        evaluate_gaze([label], [action_pred])


def test_gaze_reports_absent_predictions_and_abstentions() -> None:
    labels = [
        RetailGazeLabel(
            sample_id="a",
            head_box=PixelBox(x_min=0, y_min=0, x_max=2, y_max=2),
            gaze_target=NormalizedPoint(x=0.5, y=0.5),
            region_id="reg-1",
        ),
        RetailGazeLabel(
            sample_id="b",
            head_box=PixelBox(x_min=0, y_min=0, x_max=2, y_max=2),
            gaze_target=NormalizedPoint(x=0.5, y=0.5),
            region_id="reg-1",
        ),
        RetailGazeLabel(
            sample_id="c",
            head_box=PixelBox(x_min=0, y_min=0, x_max=2, y_max=2),
            gaze_target=NormalizedPoint(x=0.5, y=0.5),
            region_id="reg-1",
        ),
    ]
    predictions = [
        ModelPrediction(
            schema_version="2.0",
            sample_id="a",
            prediction_type=AttentionType.GAZE_POINT,
            point=NormalizedPoint(x=0.5, y=0.5),
            region_id="reg-1",
            confidence=0.9,
            model_version="test",
        ),
        ModelPrediction(
            schema_version="2.0",
            sample_id="b",
            prediction_type=AttentionType.UNKNOWN,
            confidence=0.1,
            model_version="test",
        ),
        # sample "c" has NO prediction (absent)
    ]
    metrics = evaluate_gaze(labels, predictions)
    assert metrics.sample_count == 3
    assert metrics.point_evaluated == 1
    assert metrics.point_skipped == 2
    assert metrics.region_evaluated == 1
    assert metrics.region_skipped == 2
    assert metrics.missing_prediction_count == 1
    assert metrics.abstention_count == 1
    assert metrics.region_accuracy == 1.0
    assert metrics.mean_normalized_distance == 0.0


def test_gaze_empty_inputs() -> None:
    metrics = evaluate_gaze([], [])
    assert metrics.sample_count == 0
    assert metrics.point_evaluated == 0
    assert metrics.point_skipped == 0
    assert metrics.region_evaluated == 0
    assert metrics.region_skipped == 0
    assert metrics.missing_prediction_count == 0
    assert metrics.abstention_count == 0
    assert metrics.mean_normalized_distance is None
    assert metrics.region_accuracy is None
