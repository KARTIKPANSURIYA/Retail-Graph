# RetailGraph

Research foundations for **anonymous** shopper attention and product-interaction inference from
retail video. The repository currently provides contracts, integrity checks, deterministic
smoke plumbing, and initial metrics—not a trained model, reproduced paper result, field
measurement, or complete product.

## Status and boundaries

RetailGraph has two independent benchmark tracks:

- **RetailAction:** multi-view localization of `take`, `put`, and `touch` in space and time.
- **Retail Gaze:** third-person gaze targets and shelf/product-region attention.

They do **not** identify the same shoppers and cannot form a gaze-to-purchase pipeline. Neither
contains purchase/POS ground truth for this project. Estimated attention is not measured eye
gaze. Placement optimization and conversion claims require future intervention and POS data.

## Quickstart

Requires Python 3.11+.

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
ruff check . && ruff format --check . && mypy && pytest
retailgraph smoke-test --fixtures tests/fixtures
```

The final command emits JSON bearing `"benchmark": false`; its deliberately trivial predictions
only verify plumbing. Optional `video` (OpenCV) and `ml` (PyTorch) extras avoid burdening CPU CI:
`pip install -e '.[video,ml]'`. FFmpeg may be installed separately by the operator.

## Data acquisition without automatic large downloads

Review terms first, then use the official Hugging Face CLI explicitly:

```bash
pip install 'huggingface_hub[cli]'
hf download standard-cognition/RetailAction --repo-type dataset --local-dir data/raw/retail_action
hf download Voxel51/retail_gaze --repo-type dataset --local-dir data/raw/retail_gaze
retailgraph check-dataset --manifest configs/retail_action_manifest.yaml
retailgraph check-dataset --manifest configs/retail_gaze_manifest.yaml
```

The starter manifests intentionally require only an upstream README until exact released file
lists/checksums are pinned. A successful manifest check means only that declared local files
exist and match any declared checksums—not that acquisition, licensing, or upstream schema was
validated. See [dataset notes](docs/datasets.md) before writing converters.

## Layout

- `src/retailgraph/schema`: strict, versioned normalized contracts and coordinate conversions.
- `src/retailgraph/data`: normalized JSONL adapters, authoritative evaluation manifests, and group leakage checks.
- `src/retailgraph/evaluation`: separate gaze and simple action-event evaluation.
- `src/retailgraph/baselines`: deterministic contract smoke predictors (not scientific baselines).
- `configs`: reviewable examples and initial manifests.
- `tests/fixtures`: synthetic metadata only; no dataset imagery/video.
- `docs`: architecture, dataset caveats, source verification, experiments, and roadmap.

## Next work

1. Unblock official-source access, pin the immutable RetailAction revision, inspect the smallest
   metadata artifact, and implement a strict converter only for directly verified fields.
2. Reproduce each published/reference baseline and the confirmed RetailAction official protocol
   before evaluating new models.
3. Implement leakage-safe dataset indices and real single-track baselines with per-class/grouped
   reporting, seeds, runtime, hardware, dataset revision, and uncertainty outputs.

## Licensing

No software license has been selected; `LICENSE` reserves all rights pending owner review. This
does not grant rights to either dataset. RetailAction's custom terms require review for the
intended use; the Retail Gaze listing/repository showed no explicit license at foundation time.
Public-dataset research artifacts must remain distinct from future commercially deployable data
and weights. Never assume commercial usability without verified permission.
