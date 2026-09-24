# Architecture

## Research pipeline now

1. **Acquire manually:** official source to ignored `data/raw/`, only after terms review.
2. **Validate:** versioned manifests report absent files/checksum failures; strict source-specific
   converters (future work) fail on unknown or malformed fields.
3. **Normalize:** provenance and video/view records point to distinct RetailAction and Retail Gaze
   annotations. Dataset-specific meaning is retained rather than forced into one label.
4. **Split safely:** published splits take precedence; subject, session, synchronization group,
   and adjacent frames never cross partitions.
5. **Predict/evaluate:** predictions express confidence and abstention. Track-specific evaluators
   report counts/skips; artifacts record config, code/data revisions, seed, runtime, and hardware.

PyTorch and OpenCV are optional boundaries. No framework abstraction is added until a real
baseline needs it. Video paths are references; media is never serialized into metadata or Git.

## Future product boundary (not implemented)

A possible deployment has authenticated RTSP ingest → local inference → anonymous, short-lived
track IDs → versioned shelf map → privacy-filtered event store → aggregate reporting → optional
POS joins → dashboard. Retention limits, access controls, observability, consent/signage, bias
validation, and deletion workflows are design gates. Raw video should remain local by default.

No facial recognition or persistent identity is proposed. A ceiling view supplies **estimated
visual attention**, not true eye gaze, and individual-SKU attention must not be reported until a
representative field validation supports it. POS/intervention data and causal design are needed
for placement or conversion conclusions.
