import hashlib
import json
from pathlib import Path

import pytest

from retailgraph.data.integrity import DatasetManifest, ManifestFile, sha256_file
from retailgraph.data.retail_action_source import (
    RetailActionPreflightError,
    preflight_retail_action_archive,
)


def _manifest(digest: str | None) -> DatasetManifest:
    return DatasetManifest(
        dataset="retail_action",
        version="test-revision",
        files=[ManifestFile(path="data/validation.tar", sha256=digest, required=False)],
    )


def _report(output: Path) -> dict[str, object]:
    return json.loads((output / "report.json").read_text())


def test_sha256_file_uses_chunked_reads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    archive = tmp_path / "large.tar"
    content = b"0123456789" * 1000
    archive.write_bytes(content)

    def reject_read_bytes(self: Path) -> bytes:
        raise AssertionError("read_bytes must not be used for archive hashing")

    monkeypatch.setattr(Path, "read_bytes", reject_read_bytes)
    assert sha256_file(archive, chunk_size=7) == hashlib.sha256(content).hexdigest()


def test_benchmark_preflight_fails_without_pinned_digest(tmp_path: Path) -> None:
    archive = tmp_path / "validation.tar"
    archive.write_bytes(b"synthetic archive bytes")
    output = tmp_path / "output"
    with pytest.raises(RetailActionPreflightError, match="pin this SHA-256"):
        preflight_retail_action_archive(archive, output, _manifest(None), "validation")
    report = _report(output)
    assert report["verification_status"] == "missing_expected_digest"
    assert report["actual_sha256"] == hashlib.sha256(archive.read_bytes()).hexdigest()
    assert not (output / "evaluation_manifest.json").exists()


def test_inspection_mode_records_untrusted_digest_but_is_not_benchmark_output(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "validation.tar"
    archive.write_bytes(b"synthetic archive bytes")
    output = tmp_path / "output"
    report = preflight_retail_action_archive(
        archive, output, _manifest(None), "validation", inspection_only=True
    )
    assert report.verification_status == "missing_expected_digest"
    assert report.conversion_status == "inspection_complete"
    assert report.inspection_only is True
    assert list(output.iterdir()) == [output / "report.json"]


def test_checksum_mismatch_records_expected_and_actual(tmp_path: Path) -> None:
    archive = tmp_path / "validation.tar"
    archive.write_bytes(b"actual")
    output = tmp_path / "output"
    with pytest.raises(RetailActionPreflightError, match="does not match"):
        preflight_retail_action_archive(archive, output, _manifest("0" * 64), "validation")
    report = _report(output)
    assert report["expected_sha256"] == "0" * 64
    assert report["actual_sha256"] == hashlib.sha256(b"actual").hexdigest()
    assert report["verification_status"] == "checksum_mismatch"


def test_split_override_must_match_archive_filename(tmp_path: Path) -> None:
    archive = tmp_path / "validation.tar"
    archive.write_bytes(b"synthetic")
    output = tmp_path / "output"
    with pytest.raises(RetailActionPreflightError, match="does not match"):
        preflight_retail_action_archive(archive, output, _manifest(None), "test")
    assert _report(output)["verification_status"] == "not_checked"


def test_failure_atomically_removes_stale_benchmark_artifacts(tmp_path: Path) -> None:
    archive = tmp_path / "validation.tar"
    archive.write_bytes(b"actual")
    output = tmp_path / "output"
    output.mkdir()
    (output / "evaluation_manifest.json").write_text("stale")
    (output / "labels.jsonl").write_text("stale")
    (output / "report.json").write_text('{"conversion_status":"complete"}')

    with pytest.raises(RetailActionPreflightError):
        preflight_retail_action_archive(archive, output, _manifest("0" * 64), "validation")

    assert [path.name for path in output.iterdir()] == ["report.json"]
    assert _report(output)["conversion_status"] == "failed"


def test_matching_digest_still_stops_at_unverified_schema_gate(tmp_path: Path) -> None:
    archive = tmp_path / "validation.tar"
    archive.write_bytes(b"actual")
    digest = hashlib.sha256(b"actual").hexdigest()
    output = tmp_path / "output"
    with pytest.raises(RetailActionPreflightError, match="schema is directly verified"):
        preflight_retail_action_archive(archive, output, _manifest(digest), "validation")
    report = _report(output)
    assert report["expected_sha256"] == digest
    assert report["actual_sha256"] == digest
    assert report["verification_status"] == "verified"
    assert report["conversion_status"] == "blocked_unverified_source_schema"


def test_manifest_rejects_malformed_digest() -> None:
    with pytest.raises(ValueError, match="sha256"):
        ManifestFile(path="data/validation.tar", sha256="not-a-sha256")
