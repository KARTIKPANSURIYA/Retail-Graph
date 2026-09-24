"""Sample-aware initial event matching, explicitly not official spatio-temporal mAP."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass

from retailgraph.schema.records import (
    ActionClass,
    AttentionType,
    ModelPrediction,
    RetailActionLabel,
    TemporalInterval,
)

METRIC_NAME = "simple_temporal_event_pr_not_official_map"


@dataclass(frozen=True)
class ClassMetrics:
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    ground_truth_count: int
    prediction_count: int

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def temporal_iou(left: TemporalInterval, right: TemporalInterval) -> float:
    intersection = max(0.0, min(left.end_s, right.end_s) - max(left.start_s, right.start_s))
    union = max(left.end_s, right.end_s) - min(left.start_s, right.start_s)
    return intersection / union


def _duplicates(values: list[str]) -> list[str]:
    return sorted(value for value, count in Counter(values).items() if count > 1)


def evaluate_actions(
    labels: list[RetailActionLabel],
    predictions: list[ModelPrediction],
    temporal_iou_threshold: float = 0.5,
) -> dict[str, object]:
    """Match events one-to-one within the same sample and class.

    Unknown predictions are explicit abstentions and are reported but not counted as false
    positives. Action predictions for sample IDs absent from ground truth are false positives and
    reported separately. Duplicate event or action-prediction IDs are rejected. This metric ignores
    spatial localization and must not be called official mAP.
    """
    if not 0 < temporal_iou_threshold <= 1:
        raise ValueError("temporal_iou_threshold must be in (0, 1]")

    duplicate_events = _duplicates([label.event_id for label in labels])
    if duplicate_events:
        raise ValueError(f"duplicate ground-truth event_id values: {duplicate_events}")

    action_predictions = [
        prediction
        for prediction in predictions
        if isinstance(prediction.prediction_type, ActionClass)
    ]
    prediction_ids = [
        prediction.prediction_id
        for prediction in action_predictions
        if prediction.prediction_id is not None
    ]
    duplicate_predictions = _duplicates(prediction_ids)
    if duplicate_predictions:
        raise ValueError(f"duplicate action prediction_id values: {duplicate_predictions}")

    abstentions = [
        prediction
        for prediction in predictions
        if prediction.prediction_type == AttentionType.UNKNOWN
    ]
    invalid_types = [
        prediction
        for prediction in predictions
        if not isinstance(prediction.prediction_type, ActionClass)
        and prediction.prediction_type != AttentionType.UNKNOWN
    ]
    if invalid_types:
        kinds = sorted({str(prediction.prediction_type) for prediction in invalid_types})
        raise ValueError(f"non-action prediction types passed to action evaluation: {kinds}")

    labeled_samples = {label.sample_id for label in labels}
    predicted_samples = {prediction.sample_id for prediction in action_predictions}
    predictions_for_unlabeled_samples = sum(
        prediction.sample_id not in labeled_samples for prediction in action_predictions
    )
    samples_without_action_predictions = sorted(labeled_samples - predicted_samples)
    output: dict[str, object] = {
        "metric": METRIC_NAME,
        "temporal_iou_threshold": temporal_iou_threshold,
        "ground_truth_samples": len(labeled_samples),
        "abstention_count": len(abstentions),
        "predictions_for_unlabeled_samples": predictions_for_unlabeled_samples,
        "samples_without_action_predictions": samples_without_action_predictions,
        "per_class": {},
    }
    per_class: dict[str, dict[str, int | float]] = {}
    for action in ActionClass:
        truths = [label for label in labels if label.action == action]
        preds = [
            prediction for prediction in action_predictions if prediction.prediction_type == action
        ]
        candidates = sorted(
            (
                (temporal_iou(truth.interval, prediction.interval), truth_idx, prediction_idx)
                for truth_idx, truth in enumerate(truths)
                for prediction_idx, prediction in enumerate(preds)
                if prediction.interval is not None and truth.sample_id == prediction.sample_id
            ),
            reverse=True,
        )
        used_truths: set[int] = set()
        used_predictions: set[int] = set()
        for overlap, truth_idx, prediction_idx in candidates:
            if overlap < temporal_iou_threshold:
                break
            if truth_idx not in used_truths and prediction_idx not in used_predictions:
                used_truths.add(truth_idx)
                used_predictions.add(prediction_idx)
        tp = len(used_truths)
        fp = len(preds) - tp
        fn = len(truths) - tp
        metrics = ClassMetrics(
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
            precision=tp / (tp + fp) if tp + fp else 0.0,
            recall=tp / (tp + fn) if tp + fn else 0.0,
            ground_truth_count=len(truths),
            prediction_count=len(preds),
        )
        per_class[action.value] = metrics.to_dict()
    output["per_class"] = per_class
    return output


class OfficialRetailActionEvaluator:
    """Boundary for a future confirmed implementation of the published protocol."""

    def evaluate(self) -> None:
        raise NotImplementedError(
            "TODO: validate and reproduce the official spatio-temporal protocol"
        )
