# AUDIT.md — Original Repository vs. the Paper

This document records what was actually found in the pre-existing
repository (https://github.com/ShaheryarKhanZai/MaskFusionNet, audited at
commit HEAD as of this rewrite) when checked line-by-line against:

> Y. Zhang, J. Shi, J. Wang, Y. Zong, W. Zheng, G. Zhao, "MaskFusionNet: A
> Dual-Stream Fusion Model With Masked Pre-Training Mechanism for rPPG
> Measurement," *IEEE TCSVT*, vol. 34, no. 11, pp. 11521–11534, Nov. 2024.

Every claim below cites the specific file/line and the specific paper
section/equation it contradicts. Nothing here is inferred from variable
names alone — each bug was confirmed by reading the actual `forward()`
logic and, where useful, by writing a regression test in `tests/` that
fails against the old logic and passes against the fix.

## Original repository contents (before this rewrite)

```
Main.py                                  # live webcam demo only
MaskFusionNet.py                         # model definition
SelfAttention.py                         # attention primitives
Transformer.py                           # transformer layer + MFB
postprocess.py                           # bandpass + FFT -> BPM
Fine_tune_epoch_152.pth                  # a checkpoint (38 MB)
haarcascade_frontalface_default.xml
requirements.txt                         # UTF-16 encoded (breaks `pip install -r` on some setups)
Dockerfile
```

No `tests/`, no `configs/`, no dataset loader, no training script of any
kind, no `losses.py`, no `README.md` beyond none.

## Confirmed bugs (severity: architectural)

### 1. Attention does not attend across tokens

`SelfAttention.py`, `MultiHeadSelfAttention.forward`:

```python
attn_scores = torch.einsum('bchwk,bchwk->bhwk', queries, keys) * self.scale
attn_weights = torch.softmax(attn_scores, dim=-1)
attn_output = torch.einsum('bhwk,bchwk->bchwk', attn_weights, values)
```

`queries`/`keys` here are `[batch*heads, head_dim, depth, height, width]`.
The einsum index string `'bchwk,bchwk->bhwk'` contracts over `c`
(`head_dim`) **only at matching `(h, w, k)` positions** — every other
index is shared between the two operands and the output. This computes,
for each spatial-temporal location independently, a single scalar
`sum_c(q_c * k_c)` at that location — a per-position channel dot product —
and then a softmax over the *width* axis only (`dim=-1` on a `[...,
height, width]` tensor). There is no operation anywhere that lets token
`(t1, h1, w1)` influence the attention output at token `(t2, h2, w2)` for
`(t1,h1,w1) != (t2,h2,w2)`.

The paper's Eq. (6) requires exactly the opposite:

```
SA^(i) = Softmax(Q_e^(i) (K_e^(i))^T / sigma) V_e^(i)
```

with `Q_e, K_e, V_e` flattened into sequences of length `N = T_e*H_e*W_e`
tokens (Section III-B-4) before the matrix product, producing an `N x N`
attention matrix. This is standard scaled dot-product attention over a
token sequence — the entire point of using a transformer. The original
code is not a bug in an otherwise-correct attention mechanism; it is not
attention at all (no `N x N` matrix is ever formed).

**Regression test:** `tests/test_attention.py::test_self_attention_actually_mixes_tokens`
perturbs one input token and asserts a *different* output token changes —
this must fail against the original `einsum` formulation and passes
against the fix in `src/maskfusionnet/models/attention.py`.

**Fix:** `MultiHeadSelfAttention` (attention.py) flattens Q/K/V into
`[B, N, C]` sequences and uses `torch.matmul(q, k.transpose(-2,-1))` to
form the full `[B, heads, N, N]` attention matrix, matching Eq. (6)-(7)
exactly (including the fixed `sigma=2.0` scale instead of a learned
`1/sqrt(head_dim)` scale — the paper's scale is a hyperparameter, not a
`1/sqrt(d)` term, and this was also not what the original scale factor
`self.head_dim ** -0.5` computed).

### 2. ADB, SEB, and the Fusion Encoder share literal parameter tensors

`MaskFusionNet.py`, `MaskFusionNet.__init__`:

```python
self.encoder = nn.ModuleList([TransformerLayer(...) for d in range(12)])
self.ADB_encoder = nn.ModuleList([self.encoder[i] for i in range(8)])
self.SEB_encoder = nn.ModuleList([self.encoder[i] for i in range(8)])
...
self.fusion_encoder = nn.ModuleList([self.encoder[i] for i in range(8,12)])
```

`self.encoder[i]` returns a *reference* to the same `nn.Module` object
each time it's indexed. `ADB_encoder[0]` and `SEB_encoder[0]` are the
literal same object — same `Parameter` tensors, same gradients. Every
training step, gradients computed from processing the Adjacent Branch's
features and gradients computed from processing the Segmented Branch's
features are summed onto **one shared set of weights** for layers 0–7,
and `fusion_encoder`'s 4 layers are just layers 8–11 of that same list —
reused a second time, meaning the "fusion encoder" has no parameters of
its own beyond what `encoder[8:12]` already is.

The paper describes four **distinct** encoders — `Phi_enc`, `Psi_enc`
(ADB), `Theta_enc` (SEB), `Gamma_enc` (fusion) — Section III-B-4: *"All
the encoders Phi_enc, Psi_enc, Theta_enc, Gamma_enc contain multiple
transformer layers with the same structure"* (same *structure*, i.e.
same layer class/hyperparameters — nothing states or implies shared
weights), and Section III-A-2-a describes a specific, one-directional
weight-*copying* operation from the pretrained `Phi_enc` into `Theta_enc`
and `Gamma_enc` — which is meaningless if `Theta_enc` and `Gamma_enc` are
already forced to be identical to each other and to `Psi_enc` (ADB) by
construction, as they were here.

A second, related instance: `ConvBlock` is also computed once and shared:

```python
x = self.conv_block(x)
ADB_features = self.embedding1(x)
...
SEB_features = self.embedding(x)
```

Both branches consume the output of one `ConvBlock`, whereas the paper
names distinct `Psi_conv` and `Theta_conv`.

**Regression test:**
`tests/test_shapes.py::test_adb_seb_fusion_encoder_are_independent_modules`
asserts `model.adb_encoder[0] is not model.seb_encoder[0]` etc.

**Fix:** `src/maskfusionnet/models/maskfusionnet.py` builds three
separate `nn.ModuleList`s and two separate `ConvBlock`/`EmbeddingLayer`
pairs (one per branch), with no aliasing anywhere.

### 3. No masked-pretraining stage is ever executed

`TubeMasking` is defined in `MaskFusionNet.py` and instantiated in
`MaskFusionNet.__init__` (`self.masking = TubeMasking(mask_ratio)`), but
`self.masking` is **never called** inside `forward()`. There is no
decoder module (`Phi_dec` in the paper, Section III-B-6, Eq. 11-14)
anywhere in the repository. There is no training script of any kind —
`Main.py` only ever runs inference from a pre-existing `.pth` file. The
paper's entire two-stage methodology (Fig. 1: masked pretraining ->
weight loading -> fine-tuning) has no Stage 1 implementation to audit,
because Stage 1 does not exist in the code; only a (buggy) Stage-2-shaped
`forward()` exists.

**Fix:** `src/maskfusionnet/models/pretraining.py` implements
`MaskFusionNetPretrain` (conv block -> embedding -> tube masking ->
12-layer encoder -> 6-layer decoder -> reconstruction target + predictor),
matching Fig. 2's top half, and `src/maskfusionnet/training/pretrain.py` +
`scripts/train_pretrain.py` provide the training loop that actually
computes and backpropagates the reconstruction loss.

### 4. "Tube masking" is implemented as the paper's rejected baseline

`MaskFusionNet.py`, `TubeMasking.forward`:

```python
mask = torch.ones(B, T * H * W)
mask[:, :num_mask] = 0
mask = mask[:, torch.randperm(mask.size(1))].reshape(B, T, H, W)
```

This flattens the *entire* `T*H*W` volume into one 1-D index space and
permutes it uniformly at random. There is no operation that ties the
masking decision at `(h, w)` to the same decision at every `t` — a given
spatial location is masked independently, frame by frame.

The paper is explicit that this is precisely the strategy it argues
*against*, illustrated as the right-hand ("completely random masking
strategy") panel of Figure 3, contrasted with the left-hand ("tube random
masking strategy") panel actually used: *"the tube random masking
strategy randomly selects masking positions in the spatial dimension and
extends them across the entire temporal axis. This means that different
frames share the same masking pattern"* (Section III-B-3). The original
code implements the figure's rejected baseline, not the paper's method.

**Regression test:**
`tests/test_masking.py::test_mask_is_constant_across_time_tube_strategy`
asserts `mask[b,0,t,h,w] == mask[b,0,0,h,w]` for every `t` — this fails
against the flattened-`randperm` version and passes against the fix.

**Fix:** `src/maskfusionnet/models/masking.py`'s `TubeMasking` samples a
mask over `(H, W)` only, then broadcasts it across `T` via
`.expand(b, 1, t, h, w)`. See that file's docstring for the additional,
explicitly-flagged architectural decision this required (mask-token
replacement vs. token dropping — the conv-based encoder in this paper
cannot use VideoMAE's usual "drop 75% of tokens, shrink the sequence"
trick, since Conv3d needs a dense grid; see "Ambiguities" below).

### 5. The Multi-Scale Fusion Block is missing its residual connection, reuses one instance for two fusion points, and mis-implements Eq. (10)'s concatenation

`Transformer.py`, `MultiScaleFusionBlock.forward`:

```python
def forward(self, ADB_features, SEB_features):
    ...
    fusion = self.msa(SEB_features, ADB_features)
    return fusion
```

Tracing the call site in `MaskFusionNet.py` (`self.MFB(y, SEB_features)`,
`y` = downsampled ADB) through to `MultiHeadSelfAttention1.forward(self,
x1, x2)`, the Q/K/V source assignment itself (`Q` from SEB, `K`/`V` from
ADB) matches Eq. (9). The confirmed bug is that **the residual connection
required by Eq. (8), `X_f = MIA(...) + X_SEB`, does not exist** —
`MultiScaleFusionBlock.forward` returns `fusion` directly with no addition
against either input anywhere in the function body.

**Regression test:**
`tests/test_mfb.py::test_mfb_has_residual_connection_to_seb` zeroes the
attention branch's output projection and asserts the block's output
collapses to exactly `X_SEB` — this fails without a residual add and
passes with one.

**Fix:** `src/maskfusionnet/models/mfb.py`'s `MultiScaleFusionBlock.forward`
returns `fused + x_seb` explicitly (Eq. 8), with `x_seb=`/`x_adb=` keyword
arguments at every call site so no positional-argument mix-up can recur
silently.

Two further, separate MFB-related issues, confirmed:

- **Single shared `MultiScaleFusionBlock` instance reused for both fusion
  points.** `self.MFB = MultiScaleFusionBlock(...)` is called twice in
  `forward()` (`self.MFB(y, SEB_features)` after layer 4, and again after
  layer 8) — one object, so both fusion points share weights. The paper's
  phrasing — *"we use one MFB to combine... At the end... their features
  are also fused using **another** MFB"* — reads naturally as two
  distinct block instances. Fixed by instantiating `self.mfb_mid` and
  `self.mfb_final` separately in `maskfusionnet.py`.
- **`MultiHeadSelfAttention1`'s final projection concatenates the raw
  query input with the attention output** (`torch.cat([x1, attn_output],
  dim=1)` then a `Conv3d(dim*2, dim, 1)`) rather than implementing Eq.
  (10)'s `Concat(IA^(1..M_m)) W_m`, which concatenates the *per-head*
  attention outputs before one linear projection — an entirely different
  operation (concatenating raw input features with attention output is a
  DenseNet-style skip-concat, not the paper's multi-head output
  concatenation). Fixed in `attention.py`'s
  `MultiHeadInteractiveAttention`, which concatenates per-head outputs
  exactly as `MultiHeadSelfAttention` already does for the main encoder.

### 6. No weight-loading path from pretraining to fine-tuning exists

Section III-A-2-a specifies: *"the parameters of the first 8 transformer
layers in [Phi_enc] are loaded into the segmented encoder [Theta_enc]...
the parameters of the remaining 4 transformer layers in [Phi_enc] are
loaded into the fusion feature encoder [Gamma_enc]... the adjacent
encoder [Psi_enc] does not perform parameter loading."* No code in the
original repository reads a pretraining checkpoint into the fine-tuning
model at all — `Main.py` only ever loads a single, already-fine-tuned
`.pth` file, with `strict=False` and no report of what was or wasn't
matched.

**Fix:** `MaskFusionNet.load_pretrained_encoder()` in
`maskfusionnet.py` implements exactly this 8/4/0 split, verified in
`tests/test_shapes.py::test_pretrained_encoder_loading_report` and in an
end-to-end sandbox run (see "Verification performed" below) that checks
the loaded layers' tensors are `torch.equal` to the pretraining model's
corresponding layers, and that ADB's tensors are provably untouched.

### 7. No loss functions exist anywhere in the repository

Neither `L_space`/`L_time`/`L_rec` (Eq. 15-17) nor the prediction loss
`L_pre` (Eq. 18-19, negative Pearson + frequency-domain CE/KL) had any
implementation. There is consequently no way the original repository
could have trained either stage end-to-end from this code alone — the
included `Fine_tune_epoch_152.pth` checkpoint must have been produced by
code not present in the repository.

**Fix:** `src/maskfusionnet/losses/reconstruction.py` and
`losses/rppg.py`; see their docstrings for equation-by-equation mapping
and two explicitly-flagged ambiguities (the decoder's per-head dimension
formula, and the HR-classification binning procedure) — see
"Ambiguities" below.

## Confirmed bugs (severity: correctness / fidelity, non-architectural)

### 8. `Main.py` injects fabricated jitter into the displayed heart rate

```python
if random.uniform(1,5) > 3:
    if random.uniform(1,5) > 3:
        hr1 = hr + random.randint(-1,1)
        time.sleep(1/10)
```

This is not a smoothing method — it is an unconditional (well, ~16%-of-
the-time) random perturbation of the number shown to the user, unrelated
to any property of the actual predicted signal. **Removed entirely** in
`scripts/realtime_demo.py`, replaced with a documented moving-average
over the last N real BPM estimates (see that script's module docstring).

### 9. Checkpoint loading swallows all shape mismatches silently

`model.load_state_dict(torch.load(...), strict=False)` in `Main.py`, with
no inspection of `missing_keys`/`unexpected_keys`. Given bug #2 above
(ADB/SEB/fusion-encoder weight aliasing), a state dict saved from the
buggy architecture and loaded into a *fixed* architecture would silently
drop most of the model's weights with zero warning. Fixed in
`utils/checkpoints.py:load_checkpoint`, which always reports missing/
unexpected keys, and in `MaskFusionNet.load_pretrained_encoder`, which
uses `strict=True` on each of the two layer-subsets it actually touches
(SEB and the fusion encoder) rather than one `strict=False` call over the
entire model.

### 10. `Predictor` hardcodes `input_dim=96` regardless of `embed_dim`

`MaskFusionNet.__init__`: `self.predictor = Predictor(input_dim=96,
temporal_dim=160)` ignores the `embed_dim` constructor argument entirely
— constructing `MaskFusionNet(embed_dim=32, ...)` would silently break
(or silently mismatch, if `96` happened to still divide evenly) rather
than actually resizing the predictor's input. Fixed by threading
`embed_dim` through consistently in `models/predictor.py` and
`models/maskfusionnet.py`.

### 11. `requirements.txt` is UTF-16-encoded

Confirmed with `file requirements.txt` -> `UTF-16, little-endian text,
with CRLF line terminators`. `pip install -r requirements.txt` fails or
misbehaves on many toolchains that assume UTF-8 by default. Rewritten as
plain UTF-8 in this reconstruction.

### 12. Dead / duplicated code

`postprocess.py` contains an entire alternate FFT/peak-finding
implementation (~40 lines) commented out inline inside `butter_bandpass`.
Removed in the rewrite (`signal/filtering.py` + `signal/fft.py`). The
pretraining-only `Predictor`/`TubeMasking` classes that existed but were
never wired into any forward pass (bug #3) are, in this reconstruction,
made load-bearing (Stage 1 actually calls them) rather than deleted,
since the paper requires the functionality they were presumably meant to
provide.

## Ambiguities in the paper, resolved explicitly (not silently)

These are documented at the point of implementation (see the referenced
docstrings) rather than only here, per the task's instruction to make
assumptions visible in the code itself:

1. **ConvBlock pooling stride** (Section III-B-1) — the paper names three
   conv layers "cascaded with... MaxPool operation" but never gives the
   pooling kernel/stride. Assumed `(1,2,2)` (spatial-only) per layer, an
   8x total spatial downsample with temporal resolution preserved for the
   tube-embedding step. See `models/backbone.py`.

2. **Masking granularity: pixel-space vs. token-space, and "drop tokens"
   vs. "replace with mask token."** The paper's Fig. 2 shows masking
   applied to `X_tube` (already-embedded tokens), and its conv-based
   encoder (Eq. 5) needs a spatially dense grid to run `Conv3d` over — so
   this reconstruction implements masking as "replace masked token
   positions with a learnable mask-token embedding, keep the full grid
   shape" (BEiT-style), not "physically drop 75% of tokens and shrink the
   sequence" (VideoMAE-style, which only works with a plain
   linear-projection ViT encoder, not this paper's conv-based one). See
   `models/masking.py`.

3. **Decoder's `sqrt(C)/M_d` head-dimension formula (Eq. 13)** is
   dimensionally inconsistent for the paper's own stated `C=96, M_d=4`
   (`sqrt(96)/4 ~= 2.45`, not an integer channel count). Treated as a
   probable transcription error; implemented as `C/M_d` (consistent with
   the encoder's analogous Eq. 6) instead. See `models/transformer.py`.

4. **ADB/SEB temporal-length reconciliation before MFB** — ADB produces
   2x SEB's temporal token count (tube stride 2 vs. 4) and the paper does
   not spell out an alignment step. Implemented as `AvgPool3d((2,1,1))` on
   the ADB stream immediately before each MFB call (also what the
   original repository did, and the simplest reading of Figure 2, which
   draws no extra module between the branches and the MFB besides the
   fusion block itself). See `models/mfb.py`.

5. **Whether the two MFB instances (mid-branch and end-of-branch) share
   weights.** Read as two independent instances — see bug #5 above.

6. **Frequency-domain CE/KL binning procedure (Eq. 18)** — the paper
   states the category range (140 bins, `[40,180)` bpm) but not the exact
   binning of a predicted continuous signal into a categorical
   distribution. Implemented as a differentiable "soft histogram" of the
   predicted signal's power spectrum (the standard construction used by
   PhysFormer/CVD, which this paper's Eq. 18 is explicitly derived from —
   Section III-C-2 cites [32],[59] for this loss) — see
   `signal/fft.py:psd_over_hr_bins`.

7. **Eq. (19)'s weight schedule is non-monotonic by construction** (rises
   above `mu0` during warm-up, then drops back to exactly `mu0` at epoch
   25) — implemented exactly as printed rather than smoothed into a more
   "expected-looking" monotonic ramp, since nothing in the paper indicates
   this is a transcription error the way item 3 above is. See
   `losses/rppg.py`.

## Verification performed in this environment

No GPU and no copy of UBFC-rPPG (or VIPL-HR/COHFACE/PURE) were available
while building this reconstruction, so **no training run against real
data has been executed, and no accuracy numbers are reported anywhere in
this repository** (see README.md's Results section). What *was* run and
confirmed, on synthetic tensors, in this sandbox:

- Forward pass of both `MaskFusionNetPretrain` and `MaskFusionNet` at
  paper-scale input dimensions (`[1,3,160,128,128]`), full tensor-shape
  trace saved to `results/tables/shape_trace.txt`.
- Backward pass (`.backward()`) on both stages' losses — confirmed no
  `NaN`s and that gradients reach every parameter (see
  `tests/test_shapes.py::test_finetune_backward_produces_gradients_everywhere`).
- Tube-mask spatial consistency across all `T` programmatically (not just
  visually) — see `tests/test_masking.py`.
- `load_pretrained_encoder`'s exact 8/4/0 layer split, with
  `torch.equal` checks that the copied tensors really do match and that
  ADB is provably untouched.
- The full `Trainer` loop (optimizer step, checkpoint file creation, CSV
  logging, resume path) against a tiny synthetic in-memory dataset
  standing in for `UBFCrPPGDataset` — this validates the training
  *plumbing*, not model accuracy on real data.
- `pytest tests/` — 34 tests, all passing, covering masking, attention,
  MFB, full-model shapes/gradients, loss functions, and signal processing
  (including BPM recovery from synthetic sinusoids of known frequency
  across the 45–160 bpm range).

None of the above constitutes a claim that the model achieves any
particular accuracy on UBFC-rPPG or any other dataset.
