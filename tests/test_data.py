from pathlib import Path

import pytest

from retailgraph.data.adapters import (
    load_action_evaluation_manifest,
    load_retail_action_labels,
    load_samples,
)
from retailgraph.data.integrity import assert_no_group_leakage

FIXTURES = Path(__file__).parent / "fixtures"


def test_fixture_adapter_parses_all_classes() -> None:
    labels = load_retail_action_labels(FIXTURES / "retail_action" / "labels.jsonl")
    assert {label.action.value for label in labels} == {"take", "put", "touch"}


def test_fixture_splits_are_isolated() -> None:
    assert_no_group_leakage(load_samples(FIXTURES / "samples.jsonl"))


def test_subject_leakage_is_rejected(tmp_path: Path) -> None:
    original = (FIXTURES / "samples.jsonl").read_text()
    leaking = original.replace('"subject_id":"subject-2"', '"subject_id":"shared"')
    first = (
        original.splitlines()[1]
        .replace('"split":"test"', '"split":"train"')
        .replace('"subject_id":"subject-2"', '"subject_id":"shared"')
    )
    path = tmp_path / "leak.jsonl"
    path.write_text(first + "\n" + leaking.splitlines()[1] + "\n")
    with pytest.raises(ValueError, match="subject:shared"):
        assert_no_group_leakage(load_samples(path))


def test_malformed_record_reports_line(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text('{"action":"invented"}\n')
    with pytest.raises(ValueError, match=r"bad.jsonl:1"):
        load_retail_action_labels(path)


def test_action_evaluation_manifest_includes_empty_video() -> None:
    manifest = load_action_evaluation_manifest(
        FIXTURES / "retail_action" / "evaluation_manifest.json"
    )
    assert manifest.sample_ids == [
        "synthetic-video-1",
        "synthetic-video-2",
        "synthetic-empty-video",
    ]
