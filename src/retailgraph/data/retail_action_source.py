"""Fail-closed RetailAction archive preflight.

No source converter is implemented because the upstream schema and revision have not been
verified. This module provides integrity and transactional-output guarantees that do not depend on
guessing that schema.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

from retailgraph.data.integrity import DatasetManifest, ManifestFile, sha256_file


class RetailActionPreflightError(ValueError):
    """Raised when an archive is ineligible for benchmark conversion."""


@dataclass(frozen=True)
class ArchiveVerification:
    archive: str
    split: str
    expected_sha256: str | None
    actual_sha256: str | None
    verification_status: str
    conversion_status: str
    inspection_only: bool
    message: str

    def to_dict(self) -> dict[str, str | bool | None]:
        return asdict(self)


def _manifest_entry(archive: Path, manifest: DatasetManifest) -> ManifestFile | None:
    matches = [
        item
        for item in manifest.files
        if item.path == archive.name or item.path.endswith(f"/{archive.name}")
    ]
    if len(matches) > 1:
        raise RetailActionPreflightError(
            f"manifest has multiple entries for archive {archive.name!r}"
        )
    return matches[0] if matches else None


def _atomic_report_directory(output_dir: Path, report: ArchiveVerification) -> None:
    """Replace output atomically with only report.json, removing stale artifacts."""
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    backup = output_dir.with_name(f".{output_dir.name}.previous")
    try:
        (temporary / "report.json").write_text(
            json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if backup.exists():
            shutil.rmtree(backup)
        if output_dir.exists():
            os.replace(output_dir, backup)
        os.replace(temporary, output_dir)
        if backup.exists():
            shutil.rmtree(backup)
    except Exception:
        if not output_dir.exists() and backup.exists():
            os.replace(backup, output_dir)
        raise
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def preflight_retail_action_archive(
    archive: Path,
    output_dir: Path,
    manifest: DatasetManifest,
    split: str,
    *,
    inspection_only: bool = False,
    chunk_size: int = 1024 * 1024,
) -> ArchiveVerification:
    """Verify split agreement and archive integrity, then stop at the unverified schema gate.

    The function always atomically replaces ``output_dir`` with a single ``report.json``. It never
    creates normalized labels, an evaluation manifest, or benchmark-eligible partial output.
    """
    filename_split = archive.stem
    if filename_split != split:
        report = ArchiveVerification(
            archive=str(archive),
            split=split,
            expected_sha256=None,
            actual_sha256=None,
            verification_status="not_checked",
            conversion_status="failed",
            inspection_only=inspection_only,
            message=f"archive filename split {filename_split!r} does not match --split {split!r}",
        )
        _atomic_report_directory(output_dir, report)
        raise RetailActionPreflightError(report.message)
    if not archive.is_file():
        report = ArchiveVerification(
            archive=str(archive),
            split=split,
            expected_sha256=None,
            actual_sha256=None,
            verification_status="archive_missing",
            conversion_status="failed",
            inspection_only=inspection_only,
            message=f"archive does not exist: {archive}",
        )
        _atomic_report_directory(output_dir, report)
        raise RetailActionPreflightError(report.message)

    entry = _manifest_entry(archive, manifest)
    expected = entry.sha256 if entry else None
    actual = sha256_file(archive, chunk_size=chunk_size)
    if expected is None:
        report = ArchiveVerification(
            archive=str(archive),
            split=split,
            expected_sha256=None,
            actual_sha256=actual,
            verification_status="missing_expected_digest",
            conversion_status="inspection_complete" if inspection_only else "failed",
            inspection_only=inspection_only,
            message=(
                "inspection only; pin this SHA-256 in configs/retail_action_manifest.yaml before "
                "benchmark conversion"
            ),
        )
        _atomic_report_directory(output_dir, report)
        if not inspection_only:
            raise RetailActionPreflightError(report.message)
        return report
    if actual != expected:
        report = ArchiveVerification(
            archive=str(archive),
            split=split,
            expected_sha256=expected,
            actual_sha256=actual,
            verification_status="checksum_mismatch",
            conversion_status="failed",
            inspection_only=inspection_only,
            message="archive SHA-256 does not match the trusted manifest digest",
        )
        _atomic_report_directory(output_dir, report)
        raise RetailActionPreflightError(report.message)

    report = ArchiveVerification(
        archive=str(archive),
        split=split,
        expected_sha256=expected,
        actual_sha256=actual,
        verification_status="verified",
        conversion_status="blocked_unverified_source_schema",
        inspection_only=inspection_only,
        message=(
            "archive integrity verified, but conversion is disabled until the pinned upstream "
            "metadata schema is directly verified"
        ),
    )
    _atomic_report_directory(output_dir, report)
    raise RetailActionPreflightError(report.message)
