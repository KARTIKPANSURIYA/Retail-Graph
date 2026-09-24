"""Strict verified converter for upstream RetailAction dataset.

Converts verified upstream metadata (from split tar archives or directories) into
normalized RetailGraph contracts: VideoSample, RetailActionLabel, and ActionEvaluationManifest.
"""

from __future__ import annotations

import contextlib
import datetime
import hashlib
import json
import tarfile
from collections import Counter
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from retailgraph.schema.records import (
    ActionClass,
    ActionEvaluationManifest,
    CameraView,
    DatasetName,
    DatasetProvenance,
    NormalizedPoint,
    RetailActionLabel,
    Split,
    TemporalInterval,
    VideoSample,
    ViewPoint,
)

PINNED_DATASET_REVISION = "49cb590723db921a4bd5a38adea92f6abd3f7a00"
SOURCE_URL = "https://huggingface.co/datasets/standard-cognition/RetailAction"
CONVERTER_VERSION = "0.1.0"


class RetailActionConversionError(ValueError):
    """Raised when upstream RetailAction data cannot be converted to normalized contracts."""

    def __init__(
        self, message: str, *, sample_id: str | None = None, source_ref: str | None = None
    ) -> None:
        self.sample_id = sample_id
        self.source_ref = source_ref
        context = []
        if source_ref:
            context.append(f"source='{source_ref}'")
        if sample_id:
            context.append(f"sample_id='{sample_id}'")
        prefix = f"[{', '.join(context)}] " if context else ""
        super().__init__(f"{prefix}{message}")


@dataclass(frozen=True)
class ConversionSummary:
    dataset_revision: str
    converter_version: str
    split: str
    total_samples: int
    zero_action_sample_count: int
    samples_with_actions_count: int
    total_action_count: int
    counts_by_class: dict[str, int]
    action_count_per_sample_histogram: dict[int, int]
    skipped_count: int
    error_count: int
    complete: bool  # False when --max-samples was used
    output_paths: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _parse_iso_timestamp(
    timestamp_str: Any, *, field_name: str, sample_id: str, source_ref: str
) -> datetime.datetime:
    if not isinstance(timestamp_str, str):
        raise RetailActionConversionError(
            f"expected ISO timestamp string for '{field_name}', got {type(timestamp_str).__name__}",
            sample_id=sample_id,
            source_ref=source_ref,
        )
    try:
        return datetime.datetime.fromisoformat(timestamp_str)
    except ValueError as err:
        raise RetailActionConversionError(
            f"invalid ISO timestamp format for '{field_name}': '{timestamp_str}'",
            sample_id=sample_id,
            source_ref=source_ref,
        ) from err


def convert_sample_metadata(
    payload: dict[str, Any],
    sample_id: str,
    split: Split,
    *,
    revision: str = PINNED_DATASET_REVISION,
    source_ref: str = "metadata.json",
) -> tuple[VideoSample, list[RetailActionLabel]]:
    """Convert a single upstream RetailAction metadata dictionary to normalized records."""
    if not isinstance(payload, dict):
        raise RetailActionConversionError(
            f"expected dict payload, got {type(payload).__name__}",
            sample_id=sample_id,
            source_ref=source_ref,
        )

    content = payload.get("content")
    if not isinstance(content, dict):
        raise RetailActionConversionError(
            f"missing or invalid 'content' dict in metadata: {type(content).__name__}",
            sample_id=sample_id,
            source_ref=source_ref,
        )

    # 1. Validate segment_info
    seg = content.get("segment_info")
    if not isinstance(seg, dict):
        raise RetailActionConversionError(
            "missing or invalid 'segment_info' in content",
            sample_id=sample_id,
            source_ref=source_ref,
        )

    t_start = _parse_iso_timestamp(
        seg.get("sampled_at_start"),
        field_name="segment_info.sampled_at_start",
        sample_id=sample_id,
        source_ref=source_ref,
    )
    t_end = _parse_iso_timestamp(
        seg.get("sampled_at_end"),
        field_name="segment_info.sampled_at_end",
        sample_id=sample_id,
        source_ref=source_ref,
    )

    duration_s = (t_end - t_start).total_seconds()
    if duration_s <= 0:
        raise RetailActionConversionError(
            f"segment duration must be positive, got {duration_s}s (from '{t_start}' to '{t_end}')",
            sample_id=sample_id,
            source_ref=source_ref,
        )

    # 2. Camera views from action_cam
    action_cam = content.get("action_cam")
    if not isinstance(action_cam, dict):
        raise RetailActionConversionError(
            "missing or invalid 'action_cam' dict in content",
            sample_id=sample_id,
            source_ref=source_ref,
        )

    views: list[CameraView] = []
    # Observed views are rank0 and rank1
    for view_id in sorted(action_cam.keys()):
        view_path = Path(split.value) / sample_id / f"{view_id}_video.mp4"
        views.append(
            CameraView(
                view_id=view_id,
                video_path=view_path,
                synchronization_group=f"retail_action:{split.value}:{sample_id}",
                time_offset_s=0.0,
            )
        )

    if not views:
        raise RetailActionConversionError(
            "no camera views found in 'action_cam'",
            sample_id=sample_id,
            source_ref=source_ref,
        )
    if len(views) < 2:
        raise RetailActionConversionError(
            f"at least 2 camera views required, found {len(views)}: {[v.view_id for v in views]}",
            sample_id=sample_id,
            source_ref=source_ref,
        )

    provenance = DatasetProvenance(
        dataset=DatasetName.RETAIL_ACTION,
        source_url=SOURCE_URL,
        version=revision,
        split=split,
    )

    sample = VideoSample(
        sample_id=sample_id,
        provenance=provenance,
        views=views,
        timestamp_s=0.0,
    )

    # 3. Actions from labels
    labels_container = content.get("labels")
    if not isinstance(labels_container, dict):
        raise RetailActionConversionError(
            "missing or invalid 'labels' dict in content",
            sample_id=sample_id,
            source_ref=source_ref,
        )

    # Require explicit 'action' key; a missing key is not the same as an empty list.
    if "action" not in labels_container:
        raise RetailActionConversionError(
            "'labels' dict is missing required 'action' key (expected list, even if empty)",
            sample_id=sample_id,
            source_ref=source_ref,
        )
    raw_actions = labels_container["action"]
    if not isinstance(raw_actions, list):
        raise RetailActionConversionError(
            f"expected 'action' list in labels, got {type(raw_actions).__name__}",
            sample_id=sample_id,
            source_ref=source_ref,
        )

    converted_labels: list[RetailActionLabel] = []
    for act_idx, raw_act in enumerate(raw_actions):
        if not isinstance(raw_act, dict):
            raise RetailActionConversionError(
                f"action item #{act_idx} is not a dict: {type(raw_act).__name__}",
                sample_id=sample_id,
                source_ref=source_ref,
            )

        label_name = raw_act.get("label")
        if not isinstance(label_name, str):
            raise RetailActionConversionError(
                f"action #{act_idx} 'label' must be a string, got {type(label_name).__name__}",
                sample_id=sample_id,
                source_ref=source_ref,
            )
        try:
            action_class = ActionClass(label_name)
        except ValueError as err:
            raise RetailActionConversionError(
                f"unrecognized action class '{label_name}' in action #{act_idx}",
                sample_id=sample_id,
                source_ref=source_ref,
            ) from err

        start_norm = raw_act.get("start")
        end_norm = raw_act.get("end")
        if not isinstance(start_norm, (int, float)) or not isinstance(end_norm, (int, float)):
            raise RetailActionConversionError(
                f"action #{act_idx} start/end must be numeric, got start={start_norm}, end={end_norm}",
                sample_id=sample_id,
                source_ref=source_ref,
            )

        if not (0.0 <= start_norm < end_norm <= 1.0):
            raise RetailActionConversionError(
                f"action #{act_idx} normalized interval [start, end] must satisfy 0.0 <= start < end <= 1.0; "
                f"got [{start_norm}, {end_norm}]",
                sample_id=sample_id,
                source_ref=source_ref,
            )

        start_s = round(float(start_norm) * duration_s, 6)
        end_s = round(float(end_norm) * duration_s, 6)
        if start_s >= end_s:
            # If rounding collapsed seconds, make end_s strictly greater
            end_s = start_s + 0.000001

        spatial = raw_act.get("spatial")
        if not isinstance(spatial, dict):
            raise RetailActionConversionError(
                f"action #{act_idx} missing 'spatial' dict",
                sample_id=sample_id,
                source_ref=source_ref,
            )

        sp_cam = spatial.get("action_cam")
        if not isinstance(sp_cam, dict) or not sp_cam:
            raise RetailActionConversionError(
                f"action #{act_idx} missing or empty 'spatial.action_cam' dict",
                sample_id=sample_id,
                source_ref=source_ref,
            )

        points: list[ViewPoint] = []
        for v_id, pt_dict in sorted(sp_cam.items()):
            if not isinstance(pt_dict, dict):
                raise RetailActionConversionError(
                    f"action #{act_idx} spatial point for view '{v_id}' must be a dict",
                    sample_id=sample_id,
                    source_ref=source_ref,
                )
            x = pt_dict.get("x")
            y = pt_dict.get("y")
            if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
                raise RetailActionConversionError(
                    f"action #{act_idx} spatial coordinates for view '{v_id}' must be numeric (x={x}, y={y})",
                    sample_id=sample_id,
                    source_ref=source_ref,
                )
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                raise RetailActionConversionError(
                    f"action #{act_idx} coordinates out of [0, 1] range: x={x}, y={y}",
                    sample_id=sample_id,
                    source_ref=source_ref,
                )
            points.append(
                ViewPoint(
                    view_id=str(v_id),
                    point=NormalizedPoint(x=float(x), y=float(y)),
                )
            )

        converted_labels.append(
            RetailActionLabel(
                schema_version="2.0",
                sample_id=sample_id,
                event_id=f"{sample_id}-e{act_idx}",
                action=action_class,
                interval=TemporalInterval(start_s=start_s, end_s=end_s),
                points=points,
                pose=[],
            )
        )

    return sample, converted_labels


def _detect_split_from_name(name: str) -> Split:
    lower = name.lower()
    for s in Split:
        if s.value in lower:
            return s
    raise ValueError(f"could not determine split from '{name}'. Expected one of: {list(Split)}")


def iter_archive_sample_metadata(
    archive_path: Path,
) -> Iterator[tuple[str, Split, dict[str, Any], str]]:
    """Iterate over sample metadata files in a TAR archive without full extraction.

    Yields (sample_id, split, metadata_dict, member_name).
    """
    detected_split: Split | None = None
    with contextlib.suppress(ValueError):
        detected_split = _detect_split_from_name(archive_path.name)

    with tarfile.open(archive_path, "r") as tar:
        for member in tar:
            if not member.isfile() or not member.name.endswith("metadata.json"):
                continue

            parts = member.name.strip("./").split("/")
            # Expected format: <split>/<sample_id>/metadata.json or <sample_id>/metadata.json
            if len(parts) >= 3:
                split_name = parts[-3]
                sample_id = parts[-2]
                split = detected_split or Split(split_name)
            elif len(parts) == 2:
                sample_id = parts[-2]
                if detected_split is None:
                    raise RetailActionConversionError(
                        f"cannot determine split from archive entry '{member.name}'",
                        sample_id=sample_id,
                        source_ref=str(archive_path),
                    )
                split = detected_split
            else:
                continue

            extracted = tar.extractfile(member)
            if extracted is None:
                raise RetailActionConversionError(
                    f"failed to extract '{member.name}'",
                    sample_id=sample_id,
                    source_ref=str(archive_path),
                )

            try:
                payload = json.load(extracted)
            except json.JSONDecodeError as err:
                raise RetailActionConversionError(
                    f"malformed JSON in '{member.name}': {err}",
                    sample_id=sample_id,
                    source_ref=member.name,
                ) from err

            yield sample_id, split, payload, member.name


def iter_directory_sample_metadata(
    split_dir: Path,
    split_override: Split | None = None,
) -> Iterator[tuple[str, Split, dict[str, Any], str]]:
    """Iterate over sample metadata files in a directory of sample folders."""
    split = split_override or _detect_split_from_name(split_dir.name)

    # Search for metadata.json in immediate subdirectories
    for metadata_path in sorted(split_dir.glob("*/metadata.json")):
        sample_id = metadata_path.parent.name
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as err:
            raise RetailActionConversionError(
                f"cannot read JSON from '{metadata_path}': {err}",
                sample_id=sample_id,
                source_ref=str(metadata_path),
            ) from err

        yield sample_id, split, payload, str(metadata_path)


def _verify_archive_checksum(archive_path: Path, expected_sha256: str, *, source_ref: str) -> None:
    """Raise RetailActionConversionError if the archive SHA-256 does not match."""
    actual = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    if actual != expected_sha256:
        raise RetailActionConversionError(
            f"archive checksum mismatch for '{archive_path.name}': "
            f"expected {expected_sha256!r}, got {actual!r}",
            source_ref=source_ref,
        )


def convert_retail_action_split(
    source_path: Path,
    output_dir: Path | None = None,
    *,
    split_override: Split | None = None,
    revision: str = PINNED_DATASET_REVISION,
    max_samples: int | None = None,
    expected_sha256: str | None = None,
) -> ConversionSummary:
    """Convert an official RetailAction split archive or directory into normalized contracts.

    Emits ``samples.jsonl``, ``labels.jsonl``, ``report.json``, and either
    ``evaluation_manifest.json`` (complete split) or ``inspection_subset.json``
    (partial run with ``--max-samples``) into *output_dir*.

    Pass ``expected_sha256`` to verify the archive before parsing.
    The evaluation sample manifest is derived from all observed sample entries,
    including zero-action samples.
    """
    if not source_path.exists():
        raise FileNotFoundError(f"source path does not exist: {source_path}")

    is_partial = max_samples is not None

    if source_path.is_file() and source_path.suffix == ".tar":
        if expected_sha256 is not None:
            _verify_archive_checksum(
                source_path,
                expected_sha256,
                source_ref=str(source_path),
            )
        iterator = iter_archive_sample_metadata(source_path)
    elif source_path.is_dir():
        iterator = iter_directory_sample_metadata(source_path, split_override=split_override)
    else:
        raise ValueError(
            f"unsupported source path '{source_path}'; expected .tar archive or directory"
        )

    samples: list[VideoSample] = []
    labels: list[RetailActionLabel] = []
    sample_ids: list[str] = []
    counts_by_class: Counter[str] = Counter()
    actions_per_sample: Counter[int] = Counter()
    zero_action_count = 0
    resolved_split: Split | None = split_override

    for sample_id, item_split, payload, source_ref in iterator:
        if resolved_split is None:
            resolved_split = item_split
        elif resolved_split != item_split:
            raise RetailActionConversionError(
                f"split mismatch in stream: expected {resolved_split}, got {item_split}",
                sample_id=sample_id,
                source_ref=source_ref,
            )

        sample, sample_labels = convert_sample_metadata(
            payload,
            sample_id=sample_id,
            split=resolved_split,
            revision=revision,
            source_ref=source_ref,
        )

        samples.append(sample)
        labels.extend(sample_labels)
        sample_ids.append(sample_id)

        action_count = len(sample_labels)
        actions_per_sample[action_count] += 1
        if action_count == 0:
            zero_action_count += 1
        for lbl in sample_labels:
            counts_by_class[lbl.action.value] += 1

        if max_samples is not None and len(samples) >= max_samples:
            break

    if not samples:
        raise RetailActionConversionError(
            "no valid samples found in source",
            source_ref=str(source_path),
        )

    assert resolved_split is not None
    manifest = ActionEvaluationManifest(
        dataset_version=revision,
        split=resolved_split,
        sample_ids=sample_ids,
        complete=not is_partial,
    )

    output_paths: dict[str, str] = {}
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        samples_file = output_dir / "samples.jsonl"
        labels_file = output_dir / "labels.jsonl"
        # Partial runs write an inspection artifact, not the authoritative manifest,
        # so that benchmark evaluation cannot accidentally load it as the full split.
        if is_partial:
            manifest_file = output_dir / "inspection_subset.json"
        else:
            manifest_file = output_dir / "evaluation_manifest.json"
        report_file = output_dir / "report.json"

        samples_file.write_text(
            "\n".join(s.model_dump_json() for s in samples) + "\n",
            encoding="utf-8",
        )
        labels_file.write_text(
            "\n".join(lbl.model_dump_json() for lbl in labels) + "\n",
            encoding="utf-8",
        )
        manifest_file.write_text(
            json.dumps(manifest.model_dump(mode="json"), indent=2) + "\n",
            encoding="utf-8",
        )

        output_paths = {
            "samples": str(samples_file),
            "labels": str(labels_file),
            "manifest": str(manifest_file),
            "report": str(report_file),
        }

    summary = ConversionSummary(
        dataset_revision=revision,
        converter_version=CONVERTER_VERSION,
        split=resolved_split.value,
        total_samples=len(samples),
        zero_action_sample_count=zero_action_count,
        samples_with_actions_count=len(samples) - zero_action_count,
        total_action_count=len(labels),
        counts_by_class={act.value: counts_by_class[act.value] for act in ActionClass},
        action_count_per_sample_histogram=dict(sorted(actions_per_sample.items())),
        skipped_count=0,
        error_count=0,
        complete=not is_partial,
        output_paths=output_paths,
    )

    if output_dir is not None:
        report_path = output_dir / "report.json"
        report_path.write_text(
            json.dumps(summary.to_dict(), indent=2) + "\n",
            encoding="utf-8",
        )

    return summary
