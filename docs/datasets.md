# Dataset registry and assumptions

## Verification status (2026-09-24)

- **RetailAction:** Verified against official upstream Hugging Face repository at pinned immutable commit SHA `49cb590723db921a4bd5a38adea92f6abd3f7a00`. Dataset card, LICENSE, repository tree, and `data/validation.tar` (1,277 samples, 1,246 action instances, 126 zero-action samples) were downloaded and inspected. A strict verified converter (`src/retailgraph/data/retail_action_source.py`) and CLI command (`retailgraph convert-retail-action`) are implemented. See [RetailAction source verification log](retail_action_source_verification.md) for full inspection details.
- **Retail Gaze:** The official pages were identified (<https://huggingface.co/datasets/Voxel51/retail_gaze>), but upstream raw data remains unverified in this repository. Gaze evaluation validation was tightened to reject duplicate prediction IDs, duplicate ground-truth samples, extra out-of-scope predictions, and non-gaze prediction types, while reporting missing predictions and abstentions transparently.

---

## RetailAction

- Dataset: <https://huggingface.co/datasets/standard-cognition/RetailAction>
- Pinned commit SHA: `49cb590723db921a4bd5a38adea92f6abd3f7a00`
- Paper: <https://openaccess.thecvf.com/content/ICCV2025W/RetailVision/html/Mazzini_RetailAction_Dataset_for_Multi-View_Spatio-Temporal_Localization_of_Human-Object_Interactions_in_ICCVW_2025_paper.html>
- License: Standard.AI Dataset License (custom terms: attribution, deidentification, commercial revenue threshold >$10k requires express license from Standard Cognition Corp.).
- Scope: Multi-view spatial/temporal `take`, `put`, `touch`. Imbalanced classes (`take` dominates at >97%).
- Cameras: Two ceiling-mounted cameras (`rank0` and `rank1`) per sample.
- Sample format: Each sample directory contains `rank0_video.mp4`, `rank1_video.mp4`, and `metadata.json`.

**Split/evaluation rule:** Ingest the released split mapping verbatim (`train`: 17,222 samples, `validation`: 1,277 samples, `test`: 2,501 samples). Synchronized views and adjacent frames remain together in one split. The complete evaluation sample manifest must be derived from archive directory entries to include all zero-action samples (126 zero-action samples in `validation.tar`).

---

## Retail Gaze

- Dataset: <https://huggingface.co/datasets/Voxel51/retail_gaze>
- Original repository: <https://github.com/PrimeshShamilka/RetailGazeDataset>
- Scope: Third-person imagery, head box, gaze target, and optional shelf-region/mask annotations.
- Evaluation policy: Must group by subject and session, never by random image/frame.
- Evaluation validation: `evaluate_gaze` strictly rejects duplicate ground-truth sample IDs, duplicate prediction records, duplicate prediction IDs, and predictions outside the evaluated sample set. Non-gaze prediction types raise `ValueError`. Missing predictions and explicit abstentions (`AttentionType.UNKNOWN`) are counted and reported transparently.

---

## Schema adjustments and migrations

### CameraView: Optional video dimension and FPS fields (Schema 1.0 -> 1.1 backward-compatible)
Upstream RetailAction `metadata.json` does not contain video dimensions (`width`, `height`) or `fps`. In order to support ingestion directly from TAR archives without forcing full video extraction or guessing unverified values, `CameraView.width`, `CameraView.height`, and `CameraView.fps` were updated to allow `None` with `default=None`. Existing records specifying integer dimensions remain 100% valid under schema 1.x.

### Normalized action schema migration: 1.x to 2.0
Action schema 1.x omitted `sample_id`, so it cannot be safely evaluated and is rejected rather than silently reinterpreted. Version 2.0 requires explicit `schema_version: "2.0"` and `sample_id` on every `RetailActionLabel`. Prediction schema 2.0 likewise requires explicit `schema_version: "2.0"`; action predictions additionally require a unique `prediction_id`.

---

## Action evaluation manifests

A normalized `ActionEvaluationManifest` records the dataset revision, split, and authoritative list of every evaluated sample. It must include zero-action videos; deriving this universe from action labels alone is prohibited. Duplicate sample IDs are rejected. Labels or predictions outside the manifest are evaluation errors, while action predictions on an in-scope zero-action sample count as false positives.
