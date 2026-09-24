# RetailAction upstream verification log

## Current status (2026-09-24)

**Pinned upstream revision: none. Verification is blocked, not complete.** This repository does not
record `main` as a revision because a moving branch is not a reproducible pin. The environment's
HTTPS proxy returned `403 CONNECT tunnel failed` for both `git ls-remote` and direct requests to
the official Hugging Face repository. The web gateway also returned 401/403. Therefore the dataset
card, repository commit hash, license file, archive listing, and actual metadata could not be read.
No dataset archive or media was downloaded.

Attempted source endpoints:

- `https://huggingface.co/datasets/standard-cognition/RetailAction`
- `https://huggingface.co/datasets/standard-cognition/RetailAction/raw/main/README.md`
- Git remote `https://huggingface.co/datasets/standard-cognition/RetailAction`

## Directly verified from upstream

Nothing yet. The dataset name, high-level task, classes, approximate size, two-view requirement,
and URLs currently come from the project brief, **not** from locally inspected upstream files.
They must not be treated as schema evidence.

## Ingestion gate

Do not implement a source converter until an agent can complete all of these steps:

1. Resolve the immutable Hugging Face commit SHA (`git ls-remote` or API) and record it here and in
   `configs/retail_action_manifest.yaml`.
2. Download only the dataset card, license/custom-terms file, repository file tree, and archive
   central-directory listing first. Record hashes and sizes; do not fetch approximately 32 GB of
   payload merely to inspect an archive.
3. Inspect the smallest official metadata/annotation artifact and cite its path at the pinned SHA.
   Record exact field names, types, units, coordinate origin/range, interval endpoint convention,
   sample identity, official split representation, camera synchronization, and missing-value rules.
4. Add a metadata-only golden fixture copied or minimally derived only when its terms allow
   redistribution; otherwise construct a synthetic fixture matching the verified structure and
   label it as synthetic.
5. Implement a strict `retail_action_source` converter that requires the pinned source revision,
   rejects unknown/missing required fields, preserves raw provenance, and emits action schema v2.
6. Check that both camera views, timestamps, action class, interval, per-view points, sample ID, and
   official split can be preserved. If any are absent upstream, stop and document that mismatch
   rather than inventing values.

Official spatio-temporal evaluation code, matching details, thresholds, interpolation, and baseline
configuration also remain unverified. The local temporal-only smoke metric is not a substitute.
