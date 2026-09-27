# Dataset

This project targets **UBFC-rPPG**, not one of the three datasets the
MaskFusionNet paper itself evaluates on (VIPL-HR, COHFACE, PURE). See the
main README's "Dataset" section for why that distinction matters and what
it does (and does not) mean for how results here compare to the paper's
reported numbers.

## Expected layout

Download UBFC-rPPG yourself (it is not redistributed in this repo) and
arrange it as:

```
UBFC-rPPG/
  subject1/
    vid.avi
    ground_truth.txt
  subject3/
    vid.avi
    ground_truth.txt
  ...
```

Then either:

```bash
export MASKFUSIONNET_UBFC_ROOT=/absolute/path/to/UBFC-rPPG
python scripts/prepare_ubfc.py
```

or pass `--root /absolute/path/to/UBFC-rPPG` to `scripts/prepare_ubfc.py`
directly, and set `data.root` in `configs/*.yaml` (or leave it as
`${MASKFUSIONNET_UBFC_ROOT}` and export the environment variable instead
-- never hard-code a local path into a committed config).

## Ground truth format

`ground_truth.txt` is whitespace-separated; row 0 is the BVP waveform. If
your copy of the dataset has the 3-row layout (BVP / HR trace /
timestamps), only row 0 is read directly -- see
`src/maskfusionnet/data/preprocessing.py:load_bvp_ground_truth`. Per-clip
scalar ground-truth HR used by the frequency-domain loss term is estimated
spectrally from that BVP row rather than trusted from a separately-aligned
HR-trace row -- see the comment in `src/maskfusionnet/data/ubfc.py`.

This directory intentionally contains no video/checkpoint files -- see
`.gitignore`.
