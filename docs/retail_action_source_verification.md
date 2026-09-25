# RetailAction upstream verification log

## Pinned upstream revision

**Pinned immutable commit SHA:** `49cb590723db921a4bd5a38adea92f6abd3f7a00`  
**Endpoint:** `https://huggingface.co/datasets/standard-cognition/RetailAction`  
**Resolved via:** `git ls-remote https://huggingface.co/datasets/standard-cognition/RetailAction HEAD` and Hugging Face Hub Dataset API (`https://huggingface.co/api/datasets/standard-cognition/RetailAction`).

This SHA is recorded in `configs/retail_action_manifest.yaml` and `src/retailgraph/data/retail_action_source.py`. The moving `main` branch is never used as a version identifier.

---

## License and access limitations

- **License Name:** Standard.AI Dataset License
- **Copyright:** Copyright © Standard Corp. All Rights Reserved. Licensor: Standard Cognition Corp.
- **Classification:** Proprietary custom license (`license: other`). It is **not** an open-source or unrestricted license.
- **Redistribution & Attribution:** Section 1.b.i requires retaining the attribution notice:
  > "This Dataset is licensed under the Standard.AI Dataset License, Copyright © Standard Corp. All Rights Reserved."
  and identifying Standard Cognition as the author in documentation and user interfaces.
- **Privacy & Deidentification:** Section 1.b.iii explicitly mandates:
  > "You will maintain and use the information in the Dataset Materials in deidentified form and will not attempt to reidentify any individuals in the Dataset Materials."
- **Commercial Restrictions:** Section 2 imposes an express revenue gate:
  > "If, at any time, the revenue obtained from products or services made available by You that use or incorporate the Dataset Materials exceeds $10,000 (USD), you must request a license to the Dataset Materials from Standard, which Standard may grant to you in its sole discretion."
- **Evaluation Status:** The dataset is **not** automatically cleared for commercial deployment. Raw/processed videos, archives, and extracted source metadata must not be committed to Git. Public test fixtures must use synthetic metadata only.

---

## Repository files and sizes

Retrieved from the Hugging Face Tree API at commit `49cb590723db921a4bd5a38adea92f6abd3f7a00`:

| Path | Size (Bytes) | Git OID / LFS SHA256 | Description |
|---|---|---|---|
| `.gitattributes` | 2,461 | `1ef325f1b111266a6b26e0196871bd78baa8c2f3` | Git LFS tracking configuration |
| `.gitignore` | 68 | `d5732f68aca3280cc87ff0b16f83adc750f57143` | Git ignore rules |
| `LICENSE` | 8,249 | `c40dfbaff80e29cb081ca9490c392a524eceecbe` (SHA256: `89f78a96b61776eb17584edd1ea7451d90feb0fe1901fe94265335c46bb4ab40`) | Standard.AI Dataset License |
| `README.md` | 8,463 | `d5ff3847e19ec78c0551700d51537b395ee8d2fb` (SHA256: `a30ee7ee7ab1c5e4dde7cc350b3a1eea433e58fa066a86d050db774eb0431519`) | Dataset card and documentation |
| `data/validation.tar` | 1,937,612,800 (~1.80 GiB / 1.94 GB) | LFS SHA256: `1287dc0b491f3eab5807d216bb96db2436da845ed0ea6e4ec69daf6182873836` | Validation split archive |
| `data/test.tar` | 3,838,385,152 (~3.57 GiB / 3.84 GB) | LFS SHA256: `26a376627f22df0702d1bb536dd073843d3720a6b4a1788d4d226513a3ac8a25` | Test split archive |
| `data/train.tar` | 26,082,080,256 (~24.29 GiB / 26.08 GB) | LFS SHA256: `272dcd90f2a12020dfe40542b54196dc36309a0e42298ff50f3c384956369ad0` | Training split archive |

---

## Directly verified upstream metadata structure

Inspection was conducted directly on `data/validation.tar` (downloaded to ignored local storage `data/raw/retail_action/data/validation.tar`).

### Archive layout and splits
- The archive contains 1,277 sample directories: `validation/000000` through `validation/001276`.
- Every sample directory contains exactly three files:
  1. `rank0_video.mp4`: Ceiling camera view rank 0
  2. `rank1_video.mp4`: Ceiling camera view rank 1
  3. `metadata.json`: Comprehensive annotations and metadata
- Official splits are partitioned by unique shopper identity:
  - `validation`: 1,277 samples (verified)
  - `train`: 17,222 samples (documented)
  - `test`: 2,501 samples (documented)
- Sample IDs inside `validation.tar` are represented as 6-digit zero-padded strings (`000000`–`001276`).

### Actual `metadata.json` schema and types
Every sample's `metadata.json` contains a single root key `"content"`:

```json
{
  "content": {
    "segment_info": {
      "sampled_at_start": "1970-01-01T00:00:00",
      "sampled_at_end": "1970-01-01T00:00:04.600000"
    },
    "action_cam": {
      "rank0": { ... },
      "rank1": { ... }
    },
    "labels": {
      "action": [ ... ]
    }
  }
}
```

1. **`segment_info`:**
   - `sampled_at_start` (str, ISO 8601): Starts at `1970-01-01T00:00:00` for anonymization. Present in 100% of samples.
   - `sampled_at_end` (str, ISO 8601): End timestamp. Duration ranges from 0.85s to 48.4s across validation.
   - Total duration in seconds: `duration_s = (t_end - t_start).total_seconds()`.

2. **`labels.action`:**
   - List of interaction events.
   - In validation, 126 samples have 0 actions (`labels.action == []`).
   - 1,063 samples have 1 action, 81 have 2 actions, 7 have 3 actions. Total actions: 1,246 (`take`: 1,215, `put`: 26, `touch`: 5).
   - Each action item contains:
     - `label` (str): One of `"take"`, `"put"`, `"touch"`.
     - `start` (float): Normalized fraction in `[0.0, 1.0]` relative to segment duration.
     - `end` (float): Normalized fraction in `[0.0, 1.0]` relative to segment duration, satisfying `start < end`. Minimum duration in seconds is 0.3s.
     - `spatial.action_cam`: Dict with keys `"rank0"` and `"rank1"`, each mapping to `{"x": float, "y": float}`. Coordinates are normalized `[0, 1]`, top-left origin, x right, y down. Observed bounds across all validation actions: `x in [0.029, 0.976]`, `y in [0.123, 0.992]`.

3. **`action_cam` (Camera views):**
   - Keys: `"rank0"`, `"rank1"`.
   - `face_positions`: List of dicts with `col`, `row`, `sampled_at` for the subject of interest.
   - `frame_timestamps`: List of ISO 8601 strings (up to 32 frames), or `null` in 157 samples.
   - `sampling_scores`: List of `[timestamp_iso, float_score]`, or `null` in 157 samples.
   - `poses`: List of 32 frame poses for the subject (present in 2,545 of 2,554 views; absent in 9 views), containing 18 body keypoints and confidence scores.

4. **Missing fields and non-guesses:**
   - `width`, `height`, and `fps` are **not** present in `metadata.json`. MP4 video streams have resolution 600x600. `CameraView` schema was updated to make `width`, `height`, `fps` optional `None` defaults so metadata can be ingested without guessing.
   - Store identifiers, shopper identities, and cashier/POS truth are deliberately omitted by the creators for privacy.

---

## Commands to reproduce verification on another machine

```bash
# 1. Verify remote commit SHA without cloning full repository
git ls-remote https://huggingface.co/datasets/standard-cognition/RetailAction HEAD

# 2. Download README and LICENSE at the pinned SHA
mkdir -p data/raw/retail_action/data
curl -s "https://huggingface.co/datasets/standard-cognition/RetailAction/raw/49cb590723db921a4bd5a38adea92f6abd3f7a00/README.md" -o data/raw/retail_action/README.md
curl -s "https://huggingface.co/datasets/standard-cognition/RetailAction/raw/49cb590723db921a4bd5a38adea92f6abd3f7a00/LICENSE" -o data/raw/retail_action/LICENSE

# 3. Download validation split archive (1.94 GB) to ignored path
curl -L "https://huggingface.co/datasets/standard-cognition/RetailAction/resolve/49cb590723db921a4bd5a38adea92f6abd3f7a00/data/validation.tar" -o data/raw/retail_action/data/validation.tar

# 4. Verify checksums against repository manifest
retailgraph check-dataset --manifest configs/retail_action_manifest.yaml

# 5. Inspect archive metadata without extracting videos
retailgraph convert-retail-action --source data/raw/retail_action/data/validation.tar --inspect-only

# 6. Convert split into normalized contracts
retailgraph convert-retail-action --source data/raw/retail_action/data/validation.tar --output-dir data/processed/retail_action/validation
```
