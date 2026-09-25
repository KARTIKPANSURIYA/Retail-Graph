# RetailGraph

Research foundations for **anonymous** shopper attention and product-interaction inference from
retail video. The repository currently provides contracts, integrity checks, deterministic
smoke plumbing, verified source converters, and initial metrics—not a trained model, reproduced paper result, field
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

## Data acquisition and verified conversion

Review terms first (see [licensing notes](docs/datasets.md)), then acquire data explicitly:

```bash
# Pin and verify RetailAction at immutable commit 49cb590723db921a4bd5a38adea92f6abd3f7a00
mkdir -p data/raw/retail_action/data
curl -s "https://huggingface.co/datasets/standard-cognition/RetailAction/raw/49cb590723db921a4bd5a38adea92f6abd3f7a00/README.md" -o data/raw/retail_action/README.md
curl -s "https://huggingface.co/datasets/standard-cognition/RetailAction/raw/49cb590723db921a4bd5a38adea92f6abd3f7a00/LICENSE" -o data/raw/retail_action/LICENSE

# Optional: download validation split archive (1.94 GB)
curl -L "https://huggingface.co/datasets/standard-cognition/RetailAction/resolve/49cb590723db921a4bd5a38adea92f6abd3f7a00/data/validation.tar" -o data/raw/retail_action/data/validation.tar

# Verify local files against declared manifest
retailgraph check-dataset --manifest configs/retail_action_manifest.yaml

# Inspect split metadata without extracting all videos
retailgraph convert-retail-action --source data/raw/retail_action/data/validation.tar --inspect-only \
  --expected-sha256 1287dc0b491f3eab5807d216bb96db2436da845ed0ea6e4ec69daf6182873836

# Convert split into normalized contracts (samples.jsonl, labels.jsonl, evaluation_manifest.json, report.json)
retailgraph convert-retail-action --source data/raw/retail_action/data/validation.tar \
  --expected-sha256 1287dc0b491f3eab5807d216bb96db2436da845ed0ea6e4ec69daf6182873836 \
  --output-dir data/processed/retail_action/validation
```

Complete archive conversion is fail-closed: the requested digest must be the trusted digest pinned
for that split and must match the archive's chunked SHA-256. The report records the expected and
actual digests and verification status. `--max-samples` remains an inspection-only partial run and
never creates an evaluation manifest eligible for benchmark evaluation.

See [RetailAction source verification log](docs/retail_action_source_verification.md) and [dataset registry](docs/datasets.md) for detailed schema inspection notes.

## Layout

- `src/retailgraph/schema`: strict, versioned normalized contracts and coordinate conversions.
- `src/retailgraph/data`: normalized JSONL adapters, verified source converters, evaluation manifests, and leakage checks.
- `src/retailgraph/evaluation`: separate gaze and simple action-event evaluation with strict validation.
- `src/retailgraph/baselines`: deterministic contract smoke predictors (not scientific baselines).
- `configs`: reviewable examples and pinned dataset manifests.
- `tests/fixtures`: synthetic metadata only; no dataset imagery/video.
- `docs`: architecture, dataset caveats, source verification, experiments, and roadmap.

## Next work

1. Implement official spatio-temporal mAP evaluation for RetailAction matching published protocol (spatial meter-normalization via pose bone lengths or fallback factor).
2. Reproduce each published/reference baseline on RetailAction and Retail Gaze before evaluating new models.
3. Implement leakage-safe dataset indices and real single-track baselines with per-class/grouped reporting, seeds, runtime, hardware, dataset revision, and uncertainty outputs.

## Licensing

No software license has been selected; `LICENSE` reserves all rights pending owner review. This
does not grant rights to either dataset. RetailAction's custom terms require review for the
intended use; commercial deployment generating >$10k revenue requires an express license from
Standard Cognition Corp. The Retail Gaze listing/repository showed no explicit license at foundation time.
Public-dataset research artifacts must remain distinct from future commercially deployable data
and weights. Never assume commercial usability without verified permission.
