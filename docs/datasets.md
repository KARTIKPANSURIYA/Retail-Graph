# Dataset registry and assumptions

## Verification status (2026-09-24)

The official pages below were identified, but this setup environment received HTTP 401/403 from
its web gateway and direct official requests. Consequently, no upstream files, media, schemas,
checksums, exact split cardinalities, or annotation keys were verified locally. The normalized
JSONL fixture format is **our contract**, not a claim about upstream APIs. Before real use, pin a
revision, save its dataset card/README, inspect actual metadata, update the manifests, and build a
source-specific converter with golden tests. Never infer a missing field.

## RetailAction

- Dataset: <https://huggingface.co/datasets/standard-cognition/RetailAction>
- Paper: <https://openaccess.thecvf.com/content/ICCV2025W/RetailVision/html/Mazzini_RetailAction_Dataset_for_Multi-View_Spatio-Temporal_Localization_of_Human-Object_Interactions_in_ICCVW_2025_paper.html>
- Scope supplied by the project brief: multi-view spatial/temporal `take`, `put`, `touch`, with
  pose metadata where supplied. Classes are imbalanced.

**Split/evaluation rule:** ingest the released/published split mapping verbatim—never randomize
frames, clips, or views—and record its revision. Synchronized views and adjacent frames remain
in one split. Report each class plus macro aggregates and counts. The exact published split
cardinalities and official spatio-temporal matching/mAP conventions remain a tracked verification
task because source access failed; the repository refuses to impersonate them. The implemented
metric is only greedy same-class temporal-IoU event precision/recall and ignores spatial quality.

RetailAction uses custom terms; review them for research, redistribution, trained weights, and
commercial intent before acquisition/use. The terms are not replaced by this repository license.

## Retail Gaze

- Dataset: <https://huggingface.co/datasets/Voxel51/retail_gaze>
- Original repository: <https://github.com/PrimeshShamilka/RetailGazeDataset>
- Scope supplied by the project brief: third-person imagery, head box, gaze target, and product or
  shelf-region/mask information when present.

Evaluation must group by subject **and** session, never by random image/frame. Scripted gaze,
the small subject count, and capture-domain constraints limit external validity and generalization.
The exact upstream subject/session field names and official splits remain unverified: a converter
must fail rather than substitute filename guesses. The listing/repository showed no explicit
license at foundation time; treat reuse, redistribution, weights, and commercial use as unresolved.

## Local normalized metadata

`VideoSample` records provenance, split, optional store/subject/session, dimensions, FPS, time
offset, and synchronization group. RetailAction labels contain half-open time intervals and a
normalized point per available view. Retail Gaze labels contain a pixel head box, normalized gaze
target, and optional mask reference/region. Coordinates use top-left origin, x rightward/y
downward, normalized endpoints `[0,1]`; pixel boxes are half-open. Paths may be relative to a
configured data root. Mask loading/point-in-mask semantics await verified upstream formats.

## Normalized action schema migration: 1.x to 2.0

Action schema 1.x omitted `sample_id`, so it cannot be safely evaluated and is rejected rather than
silently reinterpreted. Version 2.0 requires explicit `schema_version: "2.0"` and `sample_id` on
every `RetailActionLabel`. Prediction schema 2.0 likewise requires explicit `schema_version: "2.0"`; action predictions
additionally require a unique `prediction_id`.
Migration requires joining each old event to the authoritative video/sample index; there is no
safe default. If that mapping is unavailable, discard and regenerate the normalized metadata.
Retail Gaze records remain on schema 1.x.

See [the source verification log](retail_action_source_verification.md) for the blocked revision pin
and the exact ingestion gate. No upstream converter exists until actual source structure is verified.


## Action evaluation manifests

A normalized `ActionEvaluationManifest` records the dataset revision, split, and authoritative
list of every evaluated sample. It must include zero-action videos; deriving this universe from
action labels is prohibited. Duplicate sample IDs are rejected. Labels or predictions outside the
manifest are evaluation errors, while action predictions on an in-scope zero-action sample count
as false positives.

## RetailAction archive preflight

Until the upstream revision and source schema are verified, only fail-closed archive preflight is
available:

```bash
retailgraph retail-action-preflight \
  --archive data/raw/retail_action/data/validation.tar \
  --manifest configs/retail_action_manifest.yaml \
  --split validation \
  --output outputs/retail_action_validation
```

Benchmark preflight requires the expected SHA-256 from the versioned manifest; a missing or wrong
digest fails. `--inspection-only` may calculate and report an untrusted local digest but never
creates an evaluation manifest or benchmark-eligible artifacts. `report.json` records `archive`,
`split`, `expected_sha256`, `actual_sha256`, `verification_status`, `conversion_status`,
`inspection_only`, and `message`. Output is replaced transactionally, so failure removes stale
`evaluation_manifest.json` or normalized labels from a prior run.

Once a converter exists, a complete official validation report is expected to be checked—never
forced—against 1,277 samples, 126 zero-action samples, and 1,246 actions: 1,215 `take`, 26 `put`, and
5 `touch`. These expectations came from the task specification and have not been observed locally.
