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
