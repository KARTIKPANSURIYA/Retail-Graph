"""Transparent initial event matching, explicitly not official spatio-temporal mAP."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from retailgraph.schema.records import (
    ActionClass,
    ModelPrediction,
    RetailActionLabel,
    TemporalInterval,
)


@dataclass(frozen=True)
class ClassMetrics:
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def temporal_iou(left: TemporalInterval, right: TemporalInterval) -> float:
    intersection = max(0.0, min(left.end_s, right.end_s) - max(left.start_s, right.start_s))
    union = max(left.end_s, right.end_s) - min(left.start_s, right.start_s)
    return intersection / union


def evaluate_actions(
    labels: list[RetailActionLabel],
    predictions: list[ModelPrediction],
    temporal_iou_threshold: float = 0.5,
) -> dict[str, object]:
    """Greedily match same-class events at temporal IoU >= threshold.

    This smoke metric ignores spatial localization and must not be called official mAP.
    """
    if not 0 < temporal_iou_threshold <= 1:
        raise ValueError("temporal_iou_threshold must be in (0, 1]")
    output: dict[str, object] = {
        "metric": "simple_temporal_event_pr_not_official_map",
        "temporal_iou_threshold": temporal_iou_threshold,
        "per_class": {},
    }
    per_class: dict[str, dict[str, int | float]] = {}
    for action in ActionClass:
        truths = [label for label in labels if label.action == action]
        preds = [p for p in predictions if p.prediction_type == action and p.interval is not None]
        candidates = sorted(
            (
                (temporal_iou(truth.interval, pred.interval), truth_idx, pred_idx)
                for truth_idx, truth in enumerate(truths)
                for pred_idx, pred in enumerate(preds)
                if pred.interval is not None
            ),
            reverse=True,
        )
        used_truths: set[int] = set()
        used_preds: set[int] = set()
        for overlap, truth_idx, pred_idx in candidates:
            if overlap < temporal_iou_threshold:
                break
            if truth_idx not in used_truths and pred_idx not in used_preds:
                used_truths.add(truth_idx)
                used_preds.add(pred_idx)
        tp = len(used_truths)
        fp = len(preds) - tp
        fn = len(truths) - tp
        metrics = ClassMetrics(
            tp, fp, fn, tp / (tp + fp) if tp + fp else 0.0, tp / (tp + fn) if tp + fn else 0.0
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
