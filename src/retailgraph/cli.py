"""RetailGraph command-line tools."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Annotated

import typer
import yaml

from retailgraph.baselines.deterministic import predict_actions, predict_gaze
from retailgraph.data.adapters import (
    load_action_evaluation_manifest,
    load_retail_action_labels,
    load_retail_gaze_labels,
    load_samples,
)
from retailgraph.data.integrity import DatasetManifest, assert_no_group_leakage, check_manifest
from retailgraph.data.retail_action_source import (
    PINNED_DATASET_REVISION,
    convert_retail_action_split,
)
from retailgraph.evaluation import evaluate_actions, evaluate_gaze
from retailgraph.schema.records import Split

app = typer.Typer(no_args_is_help=True, help="RetailGraph research data and smoke-test tools.")


@app.command("check-dataset")
def check_dataset(
    manifest_path: Annotated[Path, typer.Option("--manifest", exists=True, dir_okay=False)],
    root: Annotated[Path | None, typer.Option("--root")] = None,
) -> None:
    """Validate a local dataset against a small versioned manifest."""
    payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    manifest = DatasetManifest.model_validate(payload)
    data_root = (
        root or Path(os.environ.get("RETAILGRAPH_DATA_ROOT", "data")) / "raw" / manifest.dataset
    )
    result = check_manifest(data_root, manifest)
    typer.echo(
        json.dumps({"dataset": manifest.dataset, "root": str(data_root), **result}, sort_keys=True)
    )
    if result["missing"] or result["checksum_mismatch"]:
        raise typer.Exit(code=1)


@app.command("check-splits")
def check_splits(samples_path: Annotated[Path, typer.Option("--samples", exists=True)]) -> None:
    """Fail when subject/session/synchronized groups cross splits."""
    samples = load_samples(samples_path)
    assert_no_group_leakage(samples)
    typer.echo(json.dumps({"samples": len(samples), "leakage": False}, sort_keys=True))


@app.command("smoke-test")
def smoke_test(
    fixtures: Annotated[Path, typer.Option("--fixtures", exists=True, file_okay=False)] = Path(
        "tests/fixtures"
    ),
) -> None:
    """Run synthetic plumbing only; output is not a research benchmark."""
    action_labels = load_retail_action_labels(fixtures / "retail_action" / "labels.jsonl")
    action_manifest = load_action_evaluation_manifest(
        fixtures / "retail_action" / "evaluation_manifest.json"
    )
    gaze_labels = load_retail_gaze_labels(fixtures / "retail_gaze" / "labels.jsonl")
    result = {
        "benchmark": False,
        "warning": "SYNTHETIC CONTRACT SMOKE TEST; NOT A RESEARCH RESULT",
        "retail_action": evaluate_actions(
            action_labels, predict_actions(action_labels), action_manifest
        ),
        "retail_gaze": evaluate_gaze(gaze_labels, predict_gaze(gaze_labels)).to_dict(),
    }
    typer.echo(json.dumps(result, sort_keys=True))


@app.command("convert-retail-action")
def convert_retail_action_cmd(
    source: Annotated[
        Path,
        typer.Option(
            "--source",
            exists=True,
            help="Path to split .tar archive or directory",
        ),
    ],
    output_dir: Annotated[
        Path | None,
        typer.Option(
            "--output-dir",
            help="Directory to write normalized JSONL/manifest/report files",
        ),
    ] = None,
    split: Annotated[
        str | None,
        typer.Option(
            "--split",
            help="Split name override ('train', 'validation', 'test')",
        ),
    ] = None,
    revision: Annotated[
        str,
        typer.Option(
            "--revision",
            help="Dataset commit revision",
        ),
    ] = PINNED_DATASET_REVISION,
    max_samples: Annotated[
        int | None,
        typer.Option(
            "--max-samples",
            help=(
                "Limit conversion to N samples for inspection. "
                "Writes inspection_subset.json instead of evaluation_manifest.json; "
                "benchmark evaluation will reject this partial artifact."
            ),
        ),
    ] = None,
    expected_sha256: Annotated[
        str | None,
        typer.Option(
            "--expected-sha256",
            help=(
                "Expected SHA-256 hex digest of the source .tar archive. "
                "Conversion aborts with an error if the actual digest does not match. "
                "Only valid when --source is a .tar file."
            ),
        ),
    ] = None,
    inspect_only: Annotated[
        bool,
        typer.Option(
            "--inspect-only",
            help="Only inspect and report summary without writing output files",
        ),
    ] = False,
) -> None:
    """Inspect or convert a RetailAction split without full archive extraction or video loading."""
    target_output_dir = None if inspect_only else output_dir
    split_enum = Split(split) if split else None
    summary = convert_retail_action_split(
        source,
        output_dir=target_output_dir,
        split_override=split_enum,
        revision=revision,
        max_samples=max_samples,
        expected_sha256=expected_sha256,
    )
    typer.echo(json.dumps(summary.to_dict(), indent=2, sort_keys=True))
