# MaskFusionNet

**Masked pretraining + dual-stream transformer for contactless heart-rate estimation from facial video (rPPG): an audited, tested, from-scratch PyTorch reimplementation.**

[![CI](https://github.com/ShaheryarKhanZai/MaskFusionNet/actions/workflows/ci.yml/badge.svg)](https://github.com/ShaheryarKhanZai/MaskFusionNet/actions)
[![Tests](https://img.shields.io/badge/tests-34%20passing-brightgreen)](tests/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-red)]()
[![License](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)

Paper: Y. Zhang, J. Shi, J. Wang, Y. Zong, W. Zheng, G. Zhao, *"MaskFusionNet: A Dual-Stream Fusion Model With Masked Pre-Training Mechanism for rPPG Measurement,"* IEEE TCSVT, vol. 34, no. 11, pp. 11521–11534, 2024. [DOI: 10.1109/TCSVT.2024.3422849](https://doi.org/10.1109/TCSVT.2024.3422849)

This repo started as an audit of an earlier implementation that had real architectural bugs (a per-position gate instead of attention, aliased branch weights, no genuine masked-pretraining stage). [`AUDIT.md`](AUDIT.md) documents every bug and the fix, and each architectural claim below is backed by a unit test.

---

## 1. At a glance

| | |
|---|---|
| Task | Heart-rate / BVP-waveform estimation from RGB face video |
| Model | 3D-conv stem, tube tokens, ViT-style encoders, interactive-attention fusion (MFB) |
| Training | Stage 1: masked reconstruction + prediction loss. Stage 2: dual-stream fine-tuning |
| Dataset used here | UBFC-rPPG, subject-independent 70/15/15 split |
| Tests | 34 unit tests, no dataset or GPU needed |
| Result | Masked pretraining lowers MAE from 4.80 to 4.51 bpm and raises Pearson r from 0.73 to 0.79 (single run) |

## 2. Results

All numbers below are produced by `scripts/evaluate.py` on the held-out, subject-independent test split and written to `results/tables/eval_results.json`.

**This repo, UBFC-rPPG test split**

| Model | MAE (bpm) ↓ | RMSE (bpm) ↓ | SD (bpm) ↓ | Pearson r ↑ |
|---|---|---|---|---|
| MaskFusionNet (pretrain + fine-tune) | **4.51** | **7.09** | **7.01** | **0.79** |
| MaskFusionNet w/o pretraining (`--no_pretrain`) | 4.80 | 7.81 | 7.78 | 0.73 |

Run details: a single run (no seed repeats), NVIDIA T4 (16 GB) on Google Colab, 200 pretraining epochs and 200 fine-tuning epochs, batch size 2 in both stages, evaluated on the held-out subjects of the subject-level 70/15/15 split.

**Reference: the paper's reported numbers (different datasets, not comparable)**

Taken from Tables I–III of the paper. They are listed so you can see the scale of the original claims, not as a target this repo's UBFC numbers should be read against.

| Paper setting | Variant | MAE | RMSE | SD | r |
|---|---|---|---|---|---|
| VIPL-HR, 5-fold (Table I) | pretrained | 4.37 | 6.95 | 6.84 | 0.82 |
| VIPL-HR, 5-fold (Table I) | w/o pretraining | 4.67 | 7.61 | 7.57 | 0.77 |
| COHFACE (Table II) | pretrained | 1.27 | 2.22 | n/a | 0.98 |
| COHFACE (Table II) | w/o pretraining | 1.29 | 2.30 | n/a | 0.98 |
| PURE (Table III) | pretrained | 1.11 | 1.39 | n/a | 0.99 |
| PURE (Table III) | w/o pretraining | 1.09 | 1.41 | n/a | 0.99 |

> **Read this before comparing.** The paper evaluates on VIPL-HR, COHFACE and PURE. It reports nothing on UBFC-rPPG. UBFC-rPPG is a small, well-lit, near-frontal dataset, so errors on it are not interchangeable with errors on VIPL-HR (which has heavy motion and lighting variation). The evaluation protocol also differs (see [Section 7](#7-dataset)). A like-for-like check against the paper requires one of the paper's own datasets (PURE is the most accessible; see [Roadmap](#13-limitations-and-roadmap)).

### How this compares with the paper

**Overall accuracy.** On the UBFC-rPPG test split the full model reaches MAE 4.51 bpm, RMSE 7.09 bpm, SD 7.01 bpm and r 0.79. For scale, the paper's VIPL-HR 5-fold result (Table I) is MAE 4.37, RMSE 6.95, SD 6.84 and r 0.82, so this run is 0.14 bpm higher in MAE and RMSE, 0.17 bpm higher in SD and 0.03 lower in r, with the caveat above that the evaluation setup differs from the paper's.

**Does pretraining help?** The paper's central claim is that masked pretraining improves on training the dual-stream network alone. Table I reports MAE 4.67 → 4.37 bpm and r 0.77 → 0.82 on VIPL-HR (5-fold), and Tables VIII and IX report MAE 5.69 → 4.85 bpm and r 0.77 → 0.86 on fold 2. The same comparison in this repo gives MAE 4.80 → 4.51 bpm, RMSE 7.81 → 7.09 bpm, SD 7.78 → 7.01 bpm and r 0.73 → 0.79, so the direction of the benefit is reproduced, and its size (about 0.3 bpm MAE, about 0.05 to 0.06 r) is close to the paper's 5-fold gap. This is a single run, and a 0.29 bpm MAE difference is small enough that seed-to-seed variation could account for part of it; repeated seeds are on the [roadmap](#13-limitations-and-roadmap).

**Likely sources of any gap.** Different dataset (UBFC-rPPG instead of VIPL-HR/COHFACE/PURE), Haar-cascade instead of FAN face detection, a subject-level split on a small dataset, 200 pretraining epochs against the paper's 400, batch size 2 against the paper's 8 (pretraining) and 4 (fine-tuning), and a single seed.

### What is verified independent of training

All reproducible with the commands shown:

- **Forward/backward correctness** for both stages at paper-scale input size, with no NaNs and gradients reaching every parameter.
- **34/34 unit tests** (`pytest tests/ -v`): tube-mask consistency, attention actually mixing tokens, MFB's residual connection, the paper-exact 8/4/0 pretrain-to-fine-tune weight split, and BPM recovery from synthetic sinusoids across 45–160 bpm within 3 bpm.
- **The training loop** (optimizer step, checkpointing, CSV logging, resume) works end to end, on a small synthetic dataset in the tests and on UBFC-rPPG for the results above.

## 3. Overview

Remote photoplethysmography (rPPG) estimates pulse information from the faint, periodic skin-colour changes caused by blood-volume changes, using ordinary RGB video and no contact sensor.

The paper proposes a two-stage model:

1. **Masked pretraining.** Tube masking hides the same spatial regions in every frame, so the encoder must recover pulse information from whichever facial regions stay visible rather than relying only on the easiest high-SNR areas (forehead, cheek centres).
2. **Dual-stream fine-tuning.** A fine-grained *Adjacent Branch* (ADB) and a coarse *Segmented Branch* (SEB) are fused by a *Multi-Scale Fusion Block* (MFB) using cross-branch interactive attention.

## 4. Key features

- 3D-conv stem and tube-token embedding (paper Eq. 1–2)
- Real multi-head self-attention over the spatio-temporal token sequence (Eq. 4–7), not a per-position gate ([`AUDIT.md`](AUDIT.md), bug #1)
- Tube random masking with a unit test that checks the mask is identical across frames (Eq. 3, Fig. 3)
- Independent ADB and SEB encoders with no accidental weight sharing ([`AUDIT.md`](AUDIT.md), bug #2)
- MFB with the paper's Q/K/V assignment (SEB queries, ADB keys/values) and residual connection (Eq. 8–10)
- Two-stage pipeline with the paper's pretrain-to-fine-tune weight mapping (first 8 encoder layers into SEB, last 4 into the fusion encoder, ADB from scratch)
- Reconstruction loss (spatial MSE + temporal-smoothness L1, Eq. 15–17) and prediction loss (negative Pearson + frequency-domain CE/KL over 140 HR classes, Eq. 18–19), both unit-tested
- Subject-independent UBFC-rPPG pipeline, FFT-based HR estimation, real-time webcam demo
- YAML-config training, checkpointing that reports missing/unexpected keys, Docker, 34 tests

## 5. Architecture

<p align="center">
  <img src="data/architecture.PNG" alt="MaskFusionNet architecture">
</p>

<sub>Architecture diagram based on Fig. 2 of Zhang et al. (2024).</sub>

**Why two branches?** ADB uses a fine temporal tube (2 frames per token) to catch subtle frame-to-frame variation. SEB uses a coarser tube (4 frames per token) for stable, interference-resistant long-range patterns. MFB lets the coarse stream pull fine detail from the fast stream (Eq. 8–10).

**Masked pretraining flow**

```
Video clip
    │
ConvBlock + Tube Embedding      [B, 96, T/4, H/32, W/32]
    │
Tube masking (ρ = 0.75, one spatial mask replicated over all T)
    │
Encoder (12 transformer layers) ──▶ Predictor ──▶ prediction loss
    │
Decoder (6 lightweight layers)
    │
Reconstruction loss vs. pre-masking tokens, masked positions only
```

**Fine-tuning weight transfer.** SEB gets the first 8 pretrained encoder layers and the fusion encoder gets the last 4. ADB trains from scratch because its tube size gives a different token layout. This is implemented in [`MaskFusionNet.load_pretrained_encoder()`](src/maskfusionnet/models/maskfusionnet.py) and verified by `tests/test_shapes.py::test_pretrained_encoder_loading_report`.

**Tensor shapes** (from `python scripts/inspect_shapes.py --T 160 --H 128 --W 128`, saved to [`results/tables/shape_trace.txt`](results/tables/shape_trace.txt)):

| Stage | Tensor | Shape `[B, C, T, H, W]` |
|---|---|---|
| Input | raw clip | `(1, 3, 160, 128, 128)` |
| 1 | after ConvBlock | `(1, 96, 160, 16, 16)` |
| 1 | tube tokens (tube 4,4,4) | `(1, 96, 40, 4, 4)` |
| 1 | decoder output | `(1, 96, 40, 4, 4)` |
| 1 | predictor output | `(1, 160)` |
| 2 | ADB tokens (tube 2,4,4) | `(1, 96, 80, 4, 4)` |
| 2 | SEB tokens (tube 4,4,4) | `(1, 96, 40, 4, 4)` |
| 2 | ADB aligned to SEB (AvgPool ÷2) | `(1, 96, 40, 4, 4)` |
| 2 | after MFB #1 / MFB #2 / fusion encoder | `(1, 96, 40, 4, 4)` |
| 2 | predicted rPPG signal | `(1, 160)` |

## 6. Quickstart

```bash
git clone https://github.com/ShaheryarKhanZai/MaskFusionNet.git
cd MaskFusionNet
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
pytest tests/ -v          # 34 tests, no dataset needed
```

Docker smoke test:

```bash
docker build -t maskfusionnet .
docker run maskfusionnet   # runs scripts/inspect_shapes.py
```

Inference on your own video with a checkpoint you trained (see [Section 8](#8-training-and-evaluation)); run `python scripts/inference.py --help` for the available options. Passing `--ground_truth` produces a predicted-vs-ground-truth BVP plot.

Real-time webcam demo:

```bash
python scripts/realtime_demo.py --checkpoint checkpoints/finetune/best.pt
```

Pipeline: webcam → face detection → rolling buffer → MaskFusionNet → rPPG → bandpass → FFT → BPM. The displayed BPM is either the fresh estimate or a documented moving average. The original demo injected random jitter into the number ([`AUDIT.md`](AUDIT.md), bug #8).

## 7. Dataset

This reproduction trains and evaluates on **[UBFC-rPPG](https://sites.google.com/view/ybenezeth/ubfcrppg)**. This is a deliberate scope decision, not a claim of exact reproduction.

| | Paper | This repo |
|---|---|---|
| Datasets | VIPL-HR, COHFACE, PURE | UBFC-rPPG |
| Face detector | FAN | Haar cascade (OpenCV) |
| Split | 5-fold on VIPL-HR, dataset-specific otherwise | subject-level 70/15/15 |
| Clip length / size | 160 frames, 128×128, 30 fps | same |
| Test protocol | 30 s videos cut into three 10 s segments, HR averaged | HR estimated from each 160-frame clip's predicted waveform (bandpass + FFT) and compared with ground truth per clip |

Data layout and ground-truth format: [`data/README.md`](data/README.md).

```bash
export MASKFUSIONNET_UBFC_ROOT=/absolute/path/to/UBFC-rPPG
python scripts/prepare_ubfc.py     # lists subjects and prints the train/val/test split
```

## 8. Training and evaluation

```bash
# Stage 1: masked pretraining
python scripts/train_pretrain.py --config configs/pretrain.yaml

# Stage 2: fine-tuning (loads the Stage 1 checkpoint per configs/finetune.yaml)
python scripts/train_finetune.py --config configs/finetune.yaml

# Ablation: fine-tune from scratch, no pretraining
python scripts/train_finetune.py --config configs/finetune.yaml --no_pretrain

# Evaluate on the held-out subject-independent test split
python scripts/evaluate.py --config configs/finetune.yaml \
    --checkpoint checkpoints/finetune/best.pt
```

Both training scripts support `--resume`, read the dataset root from the `MASKFUSIONNET_UBFC_ROOT` environment variable (no hard-coded paths), and write per-epoch checkpoints plus a CSV metrics log. Evaluation reports MAE, RMSE, SD and Pearson r on per-clip HR estimates.

## 9. Training configuration

| Setting | Paper | This run |
|---|---|---|
| Mask ratio ρ | 0.75 | 0.75 |
| Loss weights α, β, γ | 1.0, 1.0, 1.0 | 1.0, 1.0, 1.0 (paper configuration) |
| Prediction-loss weights | λ = 0.1; µ follows the dynamic schedule in Eq. 19 (µ₀ = 1.0, η = 5.0, schedule applies for the first 25 epochs, then µ = µ₀) | paper configuration |
| Pretraining | Adam, lr 2e-6, wd 5e-5, batch 8, 400 epochs | 200 epochs, batch size 2; optimizer, learning rate and weight decay as in [`configs/pretrain.yaml`](configs/pretrain.yaml) |
| Fine-tuning | Adam, wd 5e-5, batch 4, lr 1e-4 (VIPL-HR, COHFACE) or 3.5e-3 (PURE) | 200 epochs, batch size 2; optimizer, learning rate and weight decay as in [`configs/finetune.yaml`](configs/finetune.yaml) |
| Data loading | n/a | 2 DataLoader workers |
| Hardware | not stated in the paper's setup | NVIDIA T4, 16 GB (Google Colab) |
| Runs | n/a | single run |

Every hyperparameter with a paper-specified value is annotated `(paper)` in `configs/*.yaml`; every UBFC-specific choice is annotated `(ours)`.

## 10. Pipeline verification figures

These figures use a synthetic sinusoid and a synthetic clip. They verify the masking and signal-processing code and are **not** model predictions.

**Tube masking** (identical spatial mask at every frame, the property illustrated in the paper's Fig. 3):

![Tube masking](data/masking.PNG)

**Signal processing** (78 bpm synthetic sinusoid; the FFT stage recovers 78.0 bpm):

![Spectrum verification](results/figures/signal_processing_verification_spectrum.png)

## 11. Project structure

```
src/maskfusionnet/
  models/       ConvBlock, EmbeddingLayer, attention, transformer, masking,
                MFB, Stage-1 and Stage-2 models
  data/         UBFC-rPPG dataset, subject-level splits, preprocessing
  losses/       reconstruction loss (Eq 15-17), prediction loss (Eq 18-19)
  signal/       bandpass filtering, FFT, waveform -> BPM
  training/     generic Trainer, per-stage step functions
  utils/        metrics, checkpoints, config loading, logging, plotting
scripts/        train, evaluate, inference, demo, shape inspection
tests/          34 unit tests, no dataset required
configs/        YAML configs for both stages
results/        figures and tables (synthetic-verification files are labelled)
AUDIT.md        line-by-line bug report against the original repository
```

## 12. Engineering notes

- Mixed precision (`use_amp`) and gradient accumulation are supported in `Trainer` through the config.
- Checkpoints bundle model, optimizer, scheduler, epoch, config and metrics, and always report missing/unexpected keys. There is no silent `strict=False` ([`AUDIT.md`](AUDIT.md), bug #9).
- All 34 tests use synthetic tensors only, so CI runs without a GPU or dataset.
- No extra infrastructure (queues, orchestration, tracking services): the stack is sized to the problem.

## 13. Limitations and roadmap

**Limitations**

- The reported numbers come from a **single run** on a small dataset. With a subject-level 70/15/15 split of UBFC-rPPG, the test set contains only a few subjects, so the metrics are high-variance and the 0.29 bpm MAE gain from pretraining is not statistically established.
- Results are not comparable to the paper's tables (different dataset, detector, split and per-clip evaluation protocol; see Section 2).
- Training was shorter and used smaller batches than the paper's protocol (200 pretraining epochs against 400, batch size 2 against 8 and 4), because of a single 16 GB T4 GPU.
- Haar-cascade face detection is less robust than the paper's FAN detector under extreme pose or occlusion.
- Several paper details are underspecified (pooling stride, whether masking is applied at pixel or token resolution, MFB weight sharing, the HR-binning procedure). Each gap is resolved with an explicit, documented assumption in [`AUDIT.md`](AUDIT.md), "Ambiguities".
- UBFC-rPPG is single-scenario (static, well-lit, near-frontal). Cross-dataset generalisation, the paper's strongest claim, has not been tested here.
- Predicted-vs-ground-truth plots from a trained checkpoint are not yet included; Section 10 shows synthetic verification figures only.

**Roadmap**

- Repeat training over several seeds and report mean ± std.
- Evaluate on PURE for a like-for-like comparison with Table III of the paper.
- Add cross-dataset evaluation (train on one dataset, test on another).
- Add real predicted-vs-ground-truth BVP, scatter and Bland–Altman plots, including a failure case.
- Swap the Haar cascade for a FAN-style detector.
- Ablate the mask ratio (paper Table V) and loss components (Table VI).

## 14. Citation

```bibtex
@article{zhang2024maskfusionnet,
  title   = {MaskFusionNet: A Dual-Stream Fusion Model With Masked Pre-Training Mechanism for rPPG Measurement},
  author  = {Zhang, Yizhu and Shi, Jingang and Wang, Jiayin and Zong, Yuan and Zheng, Wenming and Zhao, Guoying},
  journal = {IEEE Transactions on Circuits and Systems for Video Technology},
  volume  = {34},
  number  = {11},
  pages   = {11521--11534},
  year    = {2024},
  doi     = {10.1109/TCSVT.2024.3422849}
}
```

This is an independent reimplementation for research and education, and is not affiliated with the paper's authors. Code is released under the MIT License ([`LICENSE`](LICENSE)). If you use this repository, please cite the original paper.
