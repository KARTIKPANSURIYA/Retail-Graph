"""Dataset manifests and split-leakage checks."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from retailgraph.schema.records import Split, VideoSample


class ManifestFile(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str
    sha256: str | None = None
    required: bool = True


class DatasetManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataset: str
    version: str
    files: list[ManifestFile]


def check_manifest(root: Path, manifest: DatasetManifest) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {"missing": [], "checksum_mismatch": [], "ok": []}
    for item in manifest.files:
        path = root / item.path
        if not path.is_file():
            if item.required:
                result["missing"].append(item.path)
            continue
        if item.sha256:
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != item.sha256:
                result["checksum_mismatch"].append(item.path)
                continue
        result["ok"].append(item.path)
    return result


def find_group_leakage(samples: list[VideoSample]) -> dict[str, list[str]]:
    """Return stable group keys occurring in more than one split.

    Priority is session, then subject, then synchronization group. Adjacent views/frames
    in the same synchronization group therefore cannot cross splits when IDs are present.
    """
    groups: dict[str, set[Split]] = defaultdict(set)
    for sample in samples:
        p = sample.provenance
        keys = []
        if p.session_id:
            keys.append(f"session:{p.session_id}")
        if p.subject_id:
            keys.append(f"subject:{p.subject_id}")
        keys.extend(
            f"sync:{view.synchronization_group}"
            for view in sample.views
            if view.synchronization_group
        )
        for key in keys:
            groups[key].add(p.split)
    return {
        key: sorted(split.value for split in splits)
        for key, splits in groups.items()
        if len(splits) > 1
    }


def assert_no_group_leakage(samples: list[VideoSample]) -> None:
    leakage = find_group_leakage(samples)
    if leakage:
        raise ValueError(f"split leakage detected: {leakage}")
