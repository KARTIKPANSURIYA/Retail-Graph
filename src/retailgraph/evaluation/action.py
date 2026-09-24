"""Sample-aware initial event matching, explicitly not official spatio-temporal mAP."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass

from retailgraph.schema.records import (
    ActionClass,
    ActionEvaluationManifest,
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


def _validate_ids(labels: list[RetailActionLabel], predictions: list[ModelPrediction]) -> None:
    duplicate_events = _duplicates([label.event_id for label in labels])
    if duplicate_events:
        raise ValueError(f"duplicate ground-truth event_id values: {duplicate_events}")
    action_prediction_ids = [
        prediction.prediction_id
        for prediction in predictions
        if isinstance(prediction.prediction_type, ActionClass)
        and prediction.prediction_id is not None
    ]
    duplicate_predictions = _duplicates(action_prediction_ids)
    if duplicate_predictions:
        raise ValueError(f"duplicate action prediction_id values: {duplicate_predictions}")


def evaluate_actions(
    labels: list[RetailActionLabel],
    predictions: list[ModelPrediction],
    manifest: ActionEvaluationManifest,
    *,
    temporal_iou_threshold: float = 0.5,
) -> dict[str, object]:
    """Match events one-to-one within the manifest sample universe and action class.

    Unknown predictions are explicit abstentions and do not count as false positives. Predictions
    for an evaluated sample with no ground-truth actions do count as false positives. Any label or
    prediction outside the authoritative manifest is an error. This temporal-only smoke metric
    ignores spatial localization and must not be called official mAP.
    """
    if not 0 < temporal_iou_threshold <= 1:
        raise ValueError("temporal_iou_threshold must be in (0, 1]")
    _validate_ids(labels, predictions)

    evaluated_samples = set(manifest.sample_ids)
    out_of_scope_labels = sorted({label.sample_id for label in labels} - evaluated_samples)
    if out_of_scope_labels:
        raise ValueError(f"ground-truth labels outside evaluation manifest: {out_of_scope_labels}")
    out_of_scope_predictions = sorted(
        {prediction.sample_id for prediction in predictions} - evaluated_samples
    )
    if out_of_scope_predictions:
        raise ValueError(f"predictions outside evaluation manifest: {out_of_scope_predictions}")

    action_predictions = [
        prediction
        for prediction in predictions
        if isinstance(prediction.prediction_type, ActionClass)
    ]
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

    samples_with_ground_truth = {label.sample_id for label in labels}
    predicted_samples = {prediction.sample_id for prediction in action_predictions}
    output: dict[str, object] = {
        "metric": METRIC_NAME,
        "dataset_version": manifest.dataset_version,
        "split": manifest.split.value,
        "temporal_iou_threshold": temporal_iou_threshold,
        "evaluated_sample_count": len(evaluated_samples),
        "samples_with_ground_truth_actions": len(samples_with_ground_truth),
        "samples_without_ground_truth_actions": sorted(
            evaluated_samples - samples_with_ground_truth
        ),
        "abstention_count": len(abstentions),
        "samples_without_action_predictions": sorted(evaluated_samples - predicted_samples),
        "per_class": {},
    }
    per_class: dict[str, dict[str, int | float]] = {}
    for action in ActionClass:
        truths = [label for label in labels if label.action == action]
        action_preds = [
            prediction for prediction in action_predictions if prediction.prediction_type == action
        ]
        candidates = sorted(
            (
                (temporal_iou(truth.interval, prediction.interval), truth_idx, prediction_idx)
                for truth_idx, truth in enumerate(truths)
                for prediction_idx, prediction in enumerate(action_preds)
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
        fp = len(action_preds) - tp
        fn = len(truths) - tp
        per_class[action.value] = ClassMetrics(
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
            precision=tp / (tp + fp) if tp + fp else 0.0,
            recall=tp / (tp + fn) if tp + fn else 0.0,
            ground_truth_count=len(truths),
            prediction_count=len(action_preds),
        ).to_dict()
    output["per_class"] = per_class
    return output


class OfficialRetailActionEvaluator:
    """Boundary for a future confirmed implementation of the published protocol."""

    def evaluate(self) -> None:
        raise NotImplementedError(
            "TODO: validate and reproduce the official spatio-temporal protocol"
        )
