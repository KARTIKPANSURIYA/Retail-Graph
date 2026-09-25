# Contributor and agent instructions

- Use Python 3.11+, the `src/` layout, strict Pydantic records, and explicit schema versions.
- Run `ruff check .`, `ruff format --check .`, `mypy`, `pytest`, and the synthetic smoke CLI.
- Never commit media, raw/processed datasets, faces, secrets, weights, checkpoints, or outputs.
- Never silently map guessed upstream fields. Pin a dataset revision, cite the source schema, and
  preserve source fields/provenance before adding a converter.
- Preserve official RetailAction splits. Preserve Retail Gaze subject/session grouping. Keep
  synchronized views and adjacent frames together; `check-splits` must pass.
- Coordinates are normalized `[0,1]`, top-left origin, x right/y down; boxes and intervals are
  half-open. Do not alter conventions without a schema-version change and migration.
- Keep tracks separate. There is no linked gaze-to-action or purchase/POS truth.
- Call the current action metric only `simple_temporal_event_pr_not_official_map`.
- Do not claim paper reproduction, model training, real-store performance, measured eye gaze,
  SKU attention, conversion impact, or commercial dataset/weight rights without evidence.
- Fixtures must be clearly synthetic and contain no copyrighted dataset media.
