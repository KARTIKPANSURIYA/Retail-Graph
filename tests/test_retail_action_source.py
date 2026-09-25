"""Tests for verified RetailAction upstream converter using synthetic data fixtures."""

import io
import json
import tarfile
from pathlib import Path
from typing import Any

import pytest

from retailgraph.data.adapters import (
    load_action_evaluation_manifest,
    load_retail_action_labels,
    load_samples,
)
from retailgraph.data.integrity import assert_no_group_leakage
from retailgraph.data.retail_action_source import (
    PINNED_ARCHIVE_SHA256,
    PINNED_DATASET_REVISION,
    RetailActionConversionError,
    convert_retail_action_split,
    convert_sample_metadata,
)
from retailgraph.schema.records import ActionClass, Split


@pytest.fixture(autouse=True)
def _restore_pinned_archive_digests() -> Any:
    """Keep tests that pin generated fixture archives isolated from one another."""
    original = dict(PINNED_ARCHIVE_SHA256)
    yield
    PINNED_ARCHIVE_SHA256.clear()
    PINNED_ARCHIVE_SHA256.update(original)


def _make_synthetic_sample_metadata(
    actions: list[dict[str, Any]],
    *,
    start_time: str = "1970-01-01T00:00:00",
    end_time: str = "1970-01-01T00:00:05",
) -> dict[str, Any]:
    """Construct synthetic metadata matching the verified upstream RetailAction schema."""
    return {
        "content": {
            "segment_info": {
                "sampled_at_start": start_time,
                "sampled_at_end": end_time,
            },
            "action_cam": {
                "rank0": {
                    "face_positions": [],
                    "frame_timestamps": [start_time, end_time],
                    "sampling_scores": [[start_time, 0.5]],
                    "poses": [],
                },
                "rank1": {
                    "face_positions": [],
                    "frame_timestamps": [start_time, end_time],
                    "sampling_scores": [[start_time, 0.5]],
                    "poses": [],
                },
            },
            "labels": {
                "action": actions,
            },
        }
    }


def test_convert_single_take_sample() -> None:
    raw = _make_synthetic_sample_metadata(
        [
            {
                "label": "take",
                "start": 0.2,
                "end": 0.8,
                "spatial": {
                    "action_cam": {
                        "rank0": {"x": 0.25, "y": 0.35},
                        "rank1": {"x": 0.65, "y": 0.75},
                    }
                },
            }
        ],
        start_time="1970-01-01T00:00:00",
        end_time="1970-01-01T00:00:10",
    )
    sample, labels = convert_sample_metadata(
        raw,
        sample_id="synth-001",
        split=Split.VALIDATION,
        revision=PINNED_DATASET_REVISION,
    )

    assert sample.sample_id == "synth-001"
    assert sample.provenance.split == Split.VALIDATION
    assert sample.provenance.version == PINNED_DATASET_REVISION
    assert len(sample.views) == 2
    view_ids = {v.view_id for v in sample.views}
    assert view_ids == {"rank0", "rank1"}
    assert sample.views[0].width is None
    assert sample.views[0].height is None
    assert sample.views[0].fps is None

    assert len(labels) == 1
    lbl = labels[0]
    assert lbl.schema_version == "2.0"
    assert lbl.sample_id == "synth-001"
    assert lbl.action == ActionClass.TAKE
    assert lbl.interval.start_s == pytest.approx(2.0)
    assert lbl.interval.end_s == pytest.approx(8.0)
    assert len(lbl.points) == 2
    pt_map = {p.view_id: (p.point.x, p.point.y) for p in lbl.points}
    assert pt_map["rank0"] == (0.25, 0.35)
    assert pt_map["rank1"] == (0.65, 0.75)


def test_convert_zero_action_sample() -> None:
    raw = _make_synthetic_sample_metadata([])
    sample, labels = convert_sample_metadata(
        raw,
        sample_id="synth-zero",
        split=Split.VALIDATION,
    )
    assert sample.sample_id == "synth-zero"
    assert len(labels) == 0


@pytest.mark.parametrize("field_name", ["frame_timestamps", "sampling_scores"])
def test_documented_nullable_camera_metadata_is_accepted(field_name: str) -> None:
    """Only fields observed as null in the pinned validation metadata accept null."""
    raw = _make_synthetic_sample_metadata([])
    raw["content"]["action_cam"]["rank0"][field_name] = None

    sample, labels = convert_sample_metadata(
        raw,
        sample_id="synthetic-nullable-camera",
        split=Split.VALIDATION,
        source_ref="validation/synthetic-nullable-camera/metadata.json",
    )

    assert sample.sample_id == "synthetic-nullable-camera"
    assert labels == []


@pytest.mark.parametrize(
    ("field_name", "invalid_value", "expected_type"),
    [
        ("face_positions", None, "list"),
        ("poses", None, "list"),
        ("frame_timestamps", "not-a-list", "list or null"),
        ("sampling_scores", {"not": "a list"}, "list or null"),
    ],
)
def test_invalid_camera_metadata_type_names_sample_and_member(
    field_name: str, invalid_value: Any, expected_type: str
) -> None:
    raw = _make_synthetic_sample_metadata([])
    raw["content"]["action_cam"]["rank1"][field_name] = invalid_value
    source_ref = "validation/bad-camera/metadata.json"

    with pytest.raises(RetailActionConversionError) as exc_info:
        convert_sample_metadata(
            raw,
            sample_id="bad-camera",
            split=Split.VALIDATION,
            source_ref=source_ref,
        )

    message = str(exc_info.value)
    assert "sample_id='bad-camera'" in message
    assert f"source='{source_ref}'" in message
    assert f"field '{field_name}' must be {expected_type}" in message


def test_convert_multi_action_sample_put_and_touch() -> None:
    raw = _make_synthetic_sample_metadata(
        [
            {
                "label": "put",
                "start": 0.1,
                "end": 0.3,
                "spatial": {
                    "action_cam": {
                        "rank0": {"x": 0.1, "y": 0.2},
                        "rank1": {"x": 0.3, "y": 0.4},
                    }
                },
            },
            {
                "label": "touch",
                "start": 0.5,
                "end": 0.9,
                "spatial": {
                    "action_cam": {
                        "rank0": {"x": 0.5, "y": 0.6},
                        "rank1": {"x": 0.7, "y": 0.8},
                    }
                },
            },
        ],
        start_time="1970-01-01T00:00:00",
        end_time="1970-01-01T00:00:10",
    )
    sample, labels = convert_sample_metadata(
        raw,
        sample_id="synth-multi",
        split=Split.TRAIN,
    )
    assert sample.sample_id == "synth-multi"
    assert len(labels) == 2
    assert labels[0].action == ActionClass.PUT
    assert labels[0].interval.start_s == pytest.approx(1.0)
    assert labels[0].interval.end_s == pytest.approx(3.0)
    assert labels[1].action == ActionClass.TOUCH
    assert labels[1].interval.start_s == pytest.approx(5.0)
    assert labels[1].interval.end_s == pytest.approx(9.0)


def _create_synthetic_tar(archive_path: Path, samples_data: dict[str, dict[str, Any]]) -> None:
    """Create a synthetic TAR archive with multiple sample directories."""
    with tarfile.open(archive_path, "w") as tar:
        for entry_path, data in samples_data.items():
            payload_bytes = json.dumps(data).encode("utf-8")
            ti = tarfile.TarInfo(name=entry_path)
            ti.size = len(payload_bytes)
            tar.addfile(ti, io.BytesIO(payload_bytes))
            sample_dir = entry_path.rsplit("/", 1)[0]
            for view_id in ("rank0", "rank1"):
                video = tarfile.TarInfo(name=f"{sample_dir}/{view_id}_video.mp4")
                video.size = 0
                tar.addfile(video, io.BytesIO())


def _trust_synthetic_archive(archive_path: Path) -> str:
    """Pin a generated fixture digest for a complete synthetic conversion."""
    import hashlib

    digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    PINNED_ARCHIVE_SHA256[Split.VALIDATION] = digest
    return digest


def test_convert_synthetic_tar_archive(tmp_path: Path) -> None:
    tar_path = tmp_path / "validation.tar"
    samples_data = {
        "validation/000001/metadata.json": _make_synthetic_sample_metadata(
            [
                {
                    "label": "take",
                    "start": 0.1,
                    "end": 0.5,
                    "spatial": {
                        "action_cam": {
                            "rank0": {"x": 0.2, "y": 0.3},
                            "rank1": {"x": 0.4, "y": 0.5},
                        }
                    },
                }
            ]
        ),
        "validation/000002/metadata.json": _make_synthetic_sample_metadata([]),  # zero action
        "validation/000003/metadata.json": _make_synthetic_sample_metadata(
            [
                {
                    "label": "put",
                    "start": 0.2,
                    "end": 0.4,
                    "spatial": {
                        "action_cam": {
                            "rank0": {"x": 0.1, "y": 0.2},
                            "rank1": {"x": 0.3, "y": 0.4},
                        }
                    },
                },
                {
                    "label": "touch",
                    "start": 0.6,
                    "end": 0.8,
                    "spatial": {
                        "action_cam": {
                            "rank0": {"x": 0.5, "y": 0.6},
                            "rank1": {"x": 0.7, "y": 0.8},
                        }
                    },
                },
            ]
        ),
    }
    samples_data["validation/000002/metadata.json"]["content"]["action_cam"]["rank0"][
        "frame_timestamps"
    ] = None
    samples_data["validation/000002/metadata.json"]["content"]["action_cam"]["rank1"][
        "sampling_scores"
    ] = None
    _create_synthetic_tar(tar_path, samples_data)

    out_dir = tmp_path / "converted_val"
    digest = _trust_synthetic_archive(tar_path)
    summary = convert_retail_action_split(tar_path, output_dir=out_dir, expected_sha256=digest)

    assert summary.total_samples == 3
    assert summary.zero_action_sample_count == 1
    assert summary.samples_with_actions_count == 2
    assert summary.total_action_count == 3
    assert summary.counts_by_class == {"take": 1, "put": 1, "touch": 1}
    assert summary.split == "validation"
    assert summary.dataset_revision == PINNED_DATASET_REVISION
    assert summary.unavailable_camera_metadata_counts == {
        "face_positions": 0,
        "poses": 0,
        "frame_timestamps": 1,
        "sampling_scores": 1,
    }

    # Verify output files can be loaded by standard adapters
    loaded_samples = load_samples(out_dir / "samples.jsonl")
    loaded_labels = load_retail_action_labels(out_dir / "labels.jsonl")
    loaded_manifest = load_action_evaluation_manifest(out_dir / "evaluation_manifest.json")

    assert len(loaded_samples) == 3
    assert len(loaded_labels) == 3
    assert loaded_manifest.sample_ids == ["000001", "000002", "000003"]
    assert "000002" in loaded_manifest.sample_ids  # zero-action sample in manifest!
    assert_no_group_leakage(loaded_samples)

    report = json.loads((out_dir / "report.json").read_text(encoding="utf-8"))
    assert report["total_samples"] == 3
    assert report["zero_action_sample_count"] == 1


def test_convert_synthetic_directory(tmp_path: Path) -> None:
    split_dir = tmp_path / "test"
    (split_dir / "000010").mkdir(parents=True)
    (split_dir / "000020").mkdir(parents=True)

    (split_dir / "000010" / "metadata.json").write_text(
        json.dumps(
            _make_synthetic_sample_metadata(
                [
                    {
                        "label": "take",
                        "start": 0.1,
                        "end": 0.5,
                        "spatial": {
                            "action_cam": {
                                "rank0": {"x": 0.2, "y": 0.3},
                                "rank1": {"x": 0.4, "y": 0.5},
                            }
                        },
                    }
                ]
            )
        )
    )
    (split_dir / "000020" / "metadata.json").write_text(
        json.dumps(_make_synthetic_sample_metadata([]))
    )
    for sample_id in ("000010", "000020"):
        for view_id in ("rank0", "rank1"):
            (split_dir / sample_id / f"{view_id}_video.mp4").touch()

    out_dir = tmp_path / "converted_dir"
    summary = convert_retail_action_split(split_dir, output_dir=out_dir, max_samples=2)
    assert summary.total_samples == 2
    assert summary.split == "test"
    assert summary.zero_action_sample_count == 1
    assert summary.total_action_count == 1
    assert summary.complete is False
    assert not (out_dir / "evaluation_manifest.json").exists()


@pytest.mark.parametrize(
    ("corrupt_key", "corrupt_val", "expected_match"),
    [
        ("content", "not-a-dict", "missing or invalid 'content' dict"),
        ("segment_info", "not-a-dict", "missing or invalid 'segment_info'"),
        ("sampled_at_start", "invalid-date", "invalid ISO timestamp"),
        ("action_cam", {}, "must contain exactly rank0 and rank1"),
        ("action_cam_type", "not-a-dict", "missing or invalid 'action_cam'"),
        ("labels", [], "missing or invalid 'labels'"),
        ("label", "unknown_action", "unrecognized action class"),
        ("start", 0.9, "must satisfy 0.0 <= start < end <= 1.0"),
        ("x", 1.5, "coordinates out of \\[0, 1\\] range"),
    ],
)
def test_malformed_metadata_rejected_with_useful_error(
    corrupt_key: str, corrupt_val: Any, expected_match: str
) -> None:
    data = _make_synthetic_sample_metadata(
        [
            {
                "label": "take",
                "start": 0.2,
                "end": 0.6,
                "spatial": {
                    "action_cam": {
                        "rank0": {"x": 0.5, "y": 0.5},
                        "rank1": {"x": 0.5, "y": 0.5},
                    }
                },
            }
        ]
    )

    if corrupt_key == "content":
        data["content"] = corrupt_val
    elif corrupt_key == "segment_info":
        data["content"]["segment_info"] = corrupt_val
    elif corrupt_key == "sampled_at_start":
        data["content"]["segment_info"]["sampled_at_start"] = corrupt_val
    elif corrupt_key in ("action_cam", "action_cam_type"):
        data["content"]["action_cam"] = corrupt_val
    elif corrupt_key == "labels":
        data["content"]["labels"] = corrupt_val
    elif corrupt_key == "label":
        data["content"]["labels"]["action"][0]["label"] = corrupt_val
    elif corrupt_key == "start":
        data["content"]["labels"]["action"][0]["start"] = corrupt_val
    elif corrupt_key == "x":
        data["content"]["labels"]["action"][0]["spatial"]["action_cam"]["rank0"]["x"] = corrupt_val

    with pytest.raises(RetailActionConversionError, match=expected_match):
        convert_sample_metadata(
            data,
            sample_id="failing-sample",
            split=Split.VALIDATION,
            source_ref="test_source.json",
        )


# ---------------------------------------------------------------------------
# Hardening 1: partial vs complete manifests
# ---------------------------------------------------------------------------


def test_max_samples_writes_inspection_subset_not_evaluation_manifest(tmp_path: Path) -> None:
    """--max-samples must produce inspection_subset.json, not evaluation_manifest.json."""
    tar_path = tmp_path / "validation.tar"
    samples_data = {
        "validation/000001/metadata.json": _make_synthetic_sample_metadata(
            [
                {
                    "label": "take",
                    "start": 0.1,
                    "end": 0.5,
                    "spatial": {
                        "action_cam": {
                            "rank0": {"x": 0.2, "y": 0.3},
                            "rank1": {"x": 0.4, "y": 0.5},
                        }
                    },
                }
            ]
        ),
        "validation/000002/metadata.json": _make_synthetic_sample_metadata([]),
        "validation/000003/metadata.json": _make_synthetic_sample_metadata([]),
    }
    _create_synthetic_tar(tar_path, samples_data)

    out_dir = tmp_path / "partial_out"
    summary = convert_retail_action_split(tar_path, output_dir=out_dir, max_samples=1)

    assert summary.complete is False
    assert summary.total_samples == 1
    assert (out_dir / "inspection_subset.json").exists(), (
        "partial run must write inspection_subset.json"
    )
    assert not (out_dir / "evaluation_manifest.json").exists(), (
        "partial run must NOT write evaluation_manifest.json"
    )

    subset = json.loads((out_dir / "inspection_subset.json").read_text(encoding="utf-8"))
    assert subset["complete"] is False


def test_complete_conversion_writes_evaluation_manifest(tmp_path: Path) -> None:
    tar_path = tmp_path / "validation.tar"
    samples_data = {
        "validation/000001/metadata.json": _make_synthetic_sample_metadata(
            [
                {
                    "label": "take",
                    "start": 0.1,
                    "end": 0.5,
                    "spatial": {
                        "action_cam": {
                            "rank0": {"x": 0.2, "y": 0.3},
                            "rank1": {"x": 0.4, "y": 0.5},
                        }
                    },
                }
            ]
        ),
    }
    _create_synthetic_tar(tar_path, samples_data)

    out_dir = tmp_path / "complete_out"
    digest = _trust_synthetic_archive(tar_path)
    summary = convert_retail_action_split(tar_path, output_dir=out_dir, expected_sha256=digest)

    assert summary.complete is True
    assert (out_dir / "evaluation_manifest.json").exists()
    assert not (out_dir / "inspection_subset.json").exists()

    manifest = json.loads((out_dir / "evaluation_manifest.json").read_text(encoding="utf-8"))
    assert manifest["complete"] is True


def test_evaluate_actions_rejects_incomplete_manifest() -> None:
    """evaluate_actions must raise ValueError when manifest.complete is False."""
    from retailgraph.evaluation.action import evaluate_actions
    from retailgraph.schema.records import ActionEvaluationManifest

    partial_manifest = ActionEvaluationManifest(
        dataset_version="49cb590723db921a4bd5a38adea92f6abd3f7a00",
        split=Split.VALIDATION,
        sample_ids=["s1"],
        complete=False,
    )
    with pytest.raises(ValueError, match=r"manifest\.complete is False"):
        evaluate_actions([], [], partial_manifest)


# ---------------------------------------------------------------------------
# Hardening 2: minimum camera views and missing labels.action key
# ---------------------------------------------------------------------------


def test_single_camera_view_rejected() -> None:
    """Samples with fewer than 2 camera views must be rejected."""
    data: dict[str, Any] = {
        "content": {
            "segment_info": {
                "sampled_at_start": "1970-01-01T00:00:00",
                "sampled_at_end": "1970-01-01T00:00:05",
            },
            "action_cam": {
                "rank0": {
                    "face_positions": [],
                    "frame_timestamps": [],
                    "sampling_scores": [],
                    "poses": [],
                },
                # Only one view — must be rejected
            },
            "labels": {"action": []},
        }
    }
    with pytest.raises(RetailActionConversionError, match="must contain exactly rank0 and rank1"):
        convert_sample_metadata(data, sample_id="single-view", split=Split.VALIDATION)


def test_missing_labels_action_key_rejected() -> None:
    """A 'labels' dict without the 'action' key must be rejected explicitly."""
    data: dict[str, Any] = {
        "content": {
            "segment_info": {
                "sampled_at_start": "1970-01-01T00:00:00",
                "sampled_at_end": "1970-01-01T00:00:05",
            },
            "action_cam": {
                "rank0": {
                    "face_positions": [],
                    "frame_timestamps": [],
                    "sampling_scores": [],
                    "poses": [],
                },
                "rank1": {
                    "face_positions": [],
                    "frame_timestamps": [],
                    "sampling_scores": [],
                    "poses": [],
                },
            },
            "labels": {
                # 'action' key intentionally absent — should NOT silently become []
                "other_annotation": [],
            },
        }
    }
    with pytest.raises(RetailActionConversionError, match="missing required 'action' key"):
        convert_sample_metadata(data, sample_id="no-action-key", split=Split.VALIDATION)


# ---------------------------------------------------------------------------
# Hardening 3: archive checksum enforcement
# ---------------------------------------------------------------------------


def test_checksum_verification_passes_with_correct_digest(tmp_path: Path) -> None:
    import hashlib

    tar_path = tmp_path / "validation.tar"
    samples_data = {
        "validation/000001/metadata.json": _make_synthetic_sample_metadata(
            [
                {
                    "label": "take",
                    "start": 0.1,
                    "end": 0.5,
                    "spatial": {
                        "action_cam": {
                            "rank0": {"x": 0.2, "y": 0.3},
                            "rank1": {"x": 0.4, "y": 0.5},
                        }
                    },
                }
            ]
        ),
    }
    _create_synthetic_tar(tar_path, samples_data)

    correct_digest = hashlib.sha256(tar_path.read_bytes()).hexdigest()
    PINNED_ARCHIVE_SHA256[Split.VALIDATION] = correct_digest
    out_dir = tmp_path / "verified_out"
    summary = convert_retail_action_split(
        tar_path, output_dir=out_dir, expected_sha256=correct_digest
    )
    assert summary.complete is True


def test_checksum_verification_fails_with_wrong_digest(tmp_path: Path) -> None:
    tar_path = tmp_path / "validation.tar"
    samples_data = {
        "validation/000001/metadata.json": _make_synthetic_sample_metadata(
            [
                {
                    "label": "take",
                    "start": 0.1,
                    "end": 0.5,
                    "spatial": {
                        "action_cam": {
                            "rank0": {"x": 0.2, "y": 0.3},
                            "rank1": {"x": 0.4, "y": 0.5},
                        }
                    },
                }
            ]
        ),
    }
    _create_synthetic_tar(tar_path, samples_data)

    wrong_digest = "a" * 64
    PINNED_ARCHIVE_SHA256[Split.VALIDATION] = wrong_digest
    with pytest.raises(RetailActionConversionError, match="archive checksum mismatch"):
        convert_retail_action_split(tar_path, expected_sha256=wrong_digest)


def test_complete_archive_requires_pinned_digest(tmp_path: Path) -> None:
    tar_path = tmp_path / "validation.tar"
    _create_synthetic_tar(
        tar_path,
        {"validation/sample/metadata.json": _make_synthetic_sample_metadata([])},
    )
    PINNED_ARCHIVE_SHA256[Split.VALIDATION] = "b" * 64
    with pytest.raises(RetailActionConversionError, match="trusted pinned digest"):
        convert_retail_action_split(tar_path)


def test_complete_archive_rejects_unpinned_revision(tmp_path: Path) -> None:
    tar_path = tmp_path / "validation.tar"
    _create_synthetic_tar(
        tar_path,
        {"validation/sample/metadata.json": _make_synthetic_sample_metadata([])},
    )
    digest = _trust_synthetic_archive(tar_path)
    with pytest.raises(RetailActionConversionError, match="revision must match"):
        convert_retail_action_split(
            tar_path,
            revision="arbitrary-revision",
            expected_sha256=digest,
        )


def test_archive_requires_video_members_and_names_context(tmp_path: Path) -> None:
    tar_path = tmp_path / "validation.tar"
    payload = json.dumps(_make_synthetic_sample_metadata([])).encode()
    with tarfile.open(tar_path, "w") as tar:
        member = tarfile.TarInfo("validation/missing-video/metadata.json")
        member.size = len(payload)
        tar.addfile(member, io.BytesIO(payload))
    with pytest.raises(RetailActionConversionError) as exc_info:
        convert_retail_action_split(tar_path, max_samples=1)
    message = str(exc_info.value)
    assert "missing-video" in message
    assert "validation/missing-video/metadata.json" in message
    assert "rank0_video.mp4" in message


def test_duplicate_sample_ids_rejected_with_member_context(tmp_path: Path) -> None:
    tar_path = tmp_path / "validation.tar"
    _create_synthetic_tar(
        tar_path,
        {
            "validation/duplicate/metadata.json": _make_synthetic_sample_metadata([]),
            "nested/validation/duplicate/metadata.json": _make_synthetic_sample_metadata([]),
        },
    )
    with pytest.raises(RetailActionConversionError) as exc_info:
        convert_retail_action_split(tar_path, max_samples=3)
    assert "duplicate sample ID" in str(exc_info.value)
    assert "nested/validation/duplicate/metadata.json" in str(exc_info.value)


def test_failed_rerun_removes_stale_complete_manifest(tmp_path: Path) -> None:
    out_dir = tmp_path / "output"
    out_dir.mkdir()
    (out_dir / "evaluation_manifest.json").write_text("stale", encoding="utf-8")
    malformed = tmp_path / "validation.tar"
    _create_synthetic_tar(
        malformed,
        {"train/sample/metadata.json": _make_synthetic_sample_metadata([])},
    )
    with pytest.raises(RetailActionConversionError, match="disagrees"):
        convert_retail_action_split(malformed, output_dir=out_dir, max_samples=1)
    assert not (out_dir / "evaluation_manifest.json").exists()
