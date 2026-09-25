"""Initial Retail Gaze evaluation; not a claim of upstream official metrics."""

from __future__ import annotations

import math
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

    def to_dict(self) -> dict[str, int | float | str | None]:
        return asdict(self)


def evaluate_gaze(labels: list[RetailGazeLabel], predictions: list[ModelPrediction]) -> GazeMetrics:
    by_id: Mapping[str, ModelPrediction] = {
        prediction.sample_id: prediction for prediction in predictions
    }
    distances: list[float] = []
    region_hits: list[bool] = []
    point_skipped = region_skipped = 0
    for label in labels:
        prediction = by_id.get(label.sample_id)
        if prediction is None or prediction.prediction_type == AttentionType.UNKNOWN:
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
    )
