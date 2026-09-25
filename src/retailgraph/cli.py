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
    RetailActionPreflightError,
    preflight_retail_action_archive,
)
from retailgraph.evaluation import evaluate_actions, evaluate_gaze

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


@app.command("retail-action-preflight")
def retail_action_preflight(
    archive: Annotated[Path, typer.Option("--archive", exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option("--output")],
    split: Annotated[str, typer.Option("--split")],
    manifest_path: Annotated[Path, typer.Option("--manifest", exists=True, dir_okay=False)] = Path(
        "configs/retail_action_manifest.yaml"
    ),
    inspection_only: Annotated[bool, typer.Option("--inspection-only")] = False,
) -> None:
    """Verify an archive; conversion stays disabled until the source schema is verified."""
    manifest = DatasetManifest.model_validate(
        yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    )
    try:
        report = preflight_retail_action_archive(
            archive, output, manifest, split, inspection_only=inspection_only
        )
    except RetailActionPreflightError as error:
        typer.echo(json.dumps({"error": str(error), "report": str(output / "report.json")}))
        raise typer.Exit(code=1) from error
    typer.echo(json.dumps(report.to_dict(), sort_keys=True))


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
