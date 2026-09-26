"""Initial Retail Gaze evaluation; not a claim of upstream official metrics."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict, dataclass

from retailgraph.schema.records import (
    AttentionType,
    ModelPrediction,
    RetailGazeLabel,
)


@dataclass(frozen=True)
class GazeMetrics:
    sample_count: int
    point_evaluated: int
    point_skipped: int
    mean_normalized_distance: float | None
    region_evaluated: int
    region_skipped: int
    region_accuracy: float | None
    calibration_status: str
    missing_prediction_count: int = 0
    abstention_count: int = 0

    def to_dict(self) -> dict[str, int | float | str | None]:
        return asdict(self)


def _duplicates(values: list[str]) -> list[str]:
    return sorted(value for value, count in Counter(values).items() if count > 1)


def evaluate_gaze(labels: list[RetailGazeLabel], predictions: list[ModelPrediction]) -> GazeMetrics:
    """Evaluate gaze point distance and region accuracy.

    This temporal/spatial evaluation is an initial deterministic metric, not verified official
    benchmark metrics. Unknown predictions are treated as explicit abstentions.
    """
    # 1. Reject duplicate ground-truth sample records
    duplicate_labels = _duplicates([label.sample_id for label in labels])
    if duplicate_labels:
        raise ValueError(f"duplicate ground-truth sample_id values: {duplicate_labels}")

    # 2. Reject duplicate predictions for the same evaluated sample or duplicate prediction_ids
    duplicate_pred_samples = _duplicates([prediction.sample_id for prediction in predictions])
    if duplicate_pred_samples:
        raise ValueError(f"duplicate predictions for sample_id: {duplicate_pred_samples}")

    pred_ids = [p.prediction_id for p in predictions if p.prediction_id is not None]
    duplicate_pred_ids = _duplicates(pred_ids)
    if duplicate_pred_ids:
        raise ValueError(f"duplicate prediction_id values: {duplicate_pred_ids}")

    # 3. Policy for predictions outside the evaluated sample set: reject them
    evaluated_samples = {label.sample_id for label in labels}
    out_of_scope_predictions = sorted(
        {prediction.sample_id for prediction in predictions} - evaluated_samples
    )
    if out_of_scope_predictions:
        raise ValueError(f"predictions outside evaluated sample set: {out_of_scope_predictions}")

    # 4. Validate prediction types appropriate for gaze evaluation
    invalid_types = [
        prediction
        for prediction in predictions
        if prediction.prediction_type
        not in (AttentionType.GAZE_POINT, AttentionType.SHELF_REGION, AttentionType.UNKNOWN)
    ]
    if invalid_types:
        kinds = sorted({str(prediction.prediction_type) for prediction in invalid_types})
        raise ValueError(f"non-gaze prediction types passed to gaze evaluation: {kinds}")

    by_id: Mapping[str, ModelPrediction] = {
        prediction.sample_id: prediction for prediction in predictions
    }
    distances: list[float] = []
    region_hits: list[bool] = []
    point_skipped = region_skipped = 0
    missing_prediction_count = abstention_count = 0

    for label in labels:
        prediction = by_id.get(label.sample_id)
        if prediction is None:
            missing_prediction_count += 1
            point_skipped += 1
            region_skipped += 1
            continue

        if prediction.prediction_type == AttentionType.UNKNOWN:
            abstention_count += 1
            point_skipped += 1
            region_skipped += 1
            continue

        if prediction.point is None:
            point_skipped += 1
        else:
            distances.append(
                math.hypot(
                    prediction.point.x - label.gaze_target.x,
                    prediction.point.y - label.gaze_target.y,
                )
            )

        if label.region_id is None or prediction.region_id is None:
            region_skipped += 1
        else:
            region_hits.append(label.region_id == prediction.region_id)

    return GazeMetrics(
        sample_count=len(labels),
        point_evaluated=len(distances),
        point_skipped=point_skipped,
        mean_normalized_distance=sum(distances) / len(distances) if distances else None,
        region_evaluated=len(region_hits),
        region_skipped=region_skipped,
        region_accuracy=sum(region_hits) / len(region_hits) if region_hits else None,
        calibration_status="not_computed: probabilistic target distributions are required",
        missing_prediction_count=missing_prediction_count,
        abstention_count=abstention_count,
    )
