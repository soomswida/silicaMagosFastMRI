# SDUM: Dual Acceleration-Routed Scalable Deep Unrolled Model

This branch (`sdum_net`) replaces the VarNet reconstruction model with **SDUM** — the
Scalable Deep Unrolled Model architecture from NVIDIA's
[NV-Raw2insights-MRI](https://github.com/NVIDIA-Medtech/NV-Raw2insights-MRI) repository.
Only the **structure** is taken from that repo; **no pretrained weights are used**
(forbidden by the challenge rules). All weights are trained from scratch on the
challenge data.

Everything around the model is unchanged from the original project, so the trained
model is used exactly like before and results remain submission-valid:

| Unchanged | Detail |
|---|---|
| Inputs/outputs | `forward(masked_kspace (B,15,640,W,2), mask (B,1,1,W,1) uint8) → (B,384,384)` |
| Acc classifier | deterministic mask-sampling-ratio rule (threshold 0.25), no filenames/metadata |
| Training loss | `SSIMLoss` (utils/common/loss_function.py), unchanged |
| Val metric | per-volume SSIM (`ssim_loss` in utils/common/utils.py), unchanged |
| Checkpoints | same dict keys and paths: `../result/<net_name>/checkpoints/{model,best_model}.pt` |
| Entry points | `train.py`, `reconstruct.py`, `recon_eval.py` (untouched harness), `leaderboard_eval.py` |

## 1. Architecture

### Dual-network routing

`AccSDUM` (utils/model/acc_sdum.py) holds two structurally identical SDUMs:
`sdum4` (specialized for 4× acceleration) and `sdum8` (8×). Each input batch is
classified by `classify_acc(mask)`: the undersampling mask's sampling ratio
(`mask.mean()`) is ~0.31 for acc4 and ~0.20 for acc8 (measured over all training
volumes), so a 0.25 threshold separates them with a >25% relative margin. The mask
is part of the model input — no filename or metadata is consulted. A batch mixing
accelerations raises `ValueError` (impossible at the required batch size 1).

### One SDUM (utils/model/sdum.py)

SDUM is an unrolled image-domain reconstruction: `num_cascades` repetitions of
"denoise then enforce data consistency", starting from the zero-filled image
`x0 = ifft2c(masked_kspace)`.

Per cascade (`SDUMCascade`):

```
x ──────────────────────────────────────────────┐ (residual skip)
│                                               │
├─ CSM: CoilSensitivityModel(coil images)       │   cascade 0 sees only the
│       → sensitivity maps S (RSS-normalized)   │   ACS region (get_acs_image);
│                                               │   later cascades see full x
├─ sens_reduce(x, S): 15 coils → 1 virtual      │
│       coil complex image (B,1,H,W,2)          │
├─ Restormer(2-channel image, cas_skips)        │
│       → refined virtual-coil image            │
├─ sens_expand(·, S): back to 15 coils          │
└─ x = skip + expand ───────────────────────────┘
   x = soft_dc(x, ref_image, mask)     # weighted data consistency in k-space
```

- **Restormer** is a U-shaped transformer: `embed_conv` → L levels of
  `TransformerBlock`s (pre-LayerNorm **MDTA** channel attention + **GDFN** gated
  depth-wise-conv FFN, optional DropPath), `PixelUnshuffle`/`PixelShuffle`
  down/upsampling, skip-concatenation with 1×1 reduction, refinement blocks, and a
  3×3 output conv. MDTA attends **across channels** (attention matrix is
  C/heads × C/heads), so compute/memory are linear in image size — this is what
  makes a transformer feasible at 640×W.
- **Cross-cascade skips** (`cas_skips`): each cascade's deeper decoder/encoder
  features are handed to the next cascade and added to its encoder inputs, so later
  cascades don't start from scratch (the NV "skip-connected cascade" design).
- **Sampling-aware weighted DC** (`soft_dc`): at sampled k-space locations,
  `k = (1-w)·k_pred + w·k_ref`; unsampled locations keep `k_pred`. The weight `w`
  is a **learned per-pixel map** (`dc_weight_map`, one 640×768 parameter per
  cascade, center-cropped/replicate-padded to the actual k-space size at runtime),
  letting the model trust the measurement more in some regions than others.
  `--no-dc-weight-map` falls back to a single learned scalar per cascade.
- Final output: `rss(complex_abs(x))`, center-cropped to 384×384 — identical to the
  original VarNet head.

### Deviations from the NV reference (and why)

| Change | Reason |
|---|---|
| No pretrained weights | Forbidden by challenge rules. |
| Removed mask-type / acceleration / acquisition **label conditioning** and timestep embeddings | Challenge forbids metadata use; the dual-network routing already specializes per acceleration, making conditioning redundant. |
| Removed MoE, OCAB / HybridRestormer, flow variants | Not part of the released SDUM configs (all default off); keeps the model minimal. |
| Per-slice scalar normalization added in `SDUM.forward` | Raw fastMRI k-space magnitudes are ~1e-5 and this repo's `DataTransform` does not normalize (NV normalizes upstream in its data pipeline). The zero-filled image is divided by its std and the final magnitude output is multiplied back. Every cascade op (FFT, DC, sens reduce/expand) is linear and the CSM output is RSS-normalized, so this rescaling is mathematically exact. |
| DC uses `mask.bool()` + `torch.where` | Our mask arrives as uint8; NV's `~mask` arithmetic on uint8 would produce 254, not logical negation. |
| CSM network is this repo's `NormUnet` (copied into sdum.py) instead of MONAI's `BasicUNet` | Keeps the project dependency-free (no monai/einops/timm anywhere) and reuses a component already proven on this data. |
| Single `dc_weight_map` per cascade (not mask-type-specific) | Each sub-net sees exactly one mask type (equispaced acc4 or acc8). |

## 2. Code map

| File | Contents |
|---|---|
| `utils/model/sdum.py` | Everything model-side: vendored ops (`DropPath`, divisible padding, complex↔channel reshapes, `sens_reduce/expand`, `acs_bounds`/`get_acs_image`), Restormer blocks (`MDTA`, `GDFN`, `TransformerBlock`, `DownSample`, `UpSample`, `Restormer`), `NormUnet` + `CoilSensitivityModel`, `SDUMCascade`, `SDUM`. |
| `utils/model/acc_sdum.py` | `classify_acc`, `AccSDUM`, and `build_acc_sdum(args)` — the single args→model factory used by both training and inference. |
| `utils/learning/train_part.py` | `adjust_learning_rate` (warmup + cosine), `train_epoch`, `validate` (with per-acc SSIM reporting), `save_model`, `train`. |
| `utils/learning/test_part.py` | `load_model` (builds the model from `checkpoint['args']`), `prep_volume`, `recon_slice` — the recon_eval contract — plus the DataLoader-based `forward` used by reconstruct.py. |
| `utils/data/load_data.py` | Dataset/DataLoader; now with a seeded shuffle generator and worker seeding. |
| `tests/test_sdum.py` | Unit checks (see §6). |

Key tensor shapes through one forward pass (B=1, W = per-volume width 320–480):

```
masked_kspace (1,15,640,W,2) ── ifft2c ──> x (1,15,640,W,2)      [image domain]
CSM input                       (1,15,640,W,2) → maps (1,15,640,W,2)
sens_reduce                     → (1,1,640,W,2) → 2-chan (1,2,640,W)
Restormer level i               (1, channels[i], 640/2^i, W/2^i)  [after pad to 2^(L-1)]
output                          rss → (1,640,W) → center_crop → (1,384,384)
```

## 3. Training

```bash
python train.py \
  -n sdum_base -e 60 -t /Data/train/ -v /Data/val/ \
  --lr 3e-4 --min-lr 1e-6 --warmup-epochs 2 --weight-decay 1e-4 --lr-schedule cosine \
  --grad-checkpointing nonreentrant --accumulation_steps 4 --no-resume
```

One joint run trains both sub-networks: each batch is routed by its mask, gradients
flow only into the routed sub-net (the other's grads stay `None` and are skipped by
the optimizer), and a single AdamW/schedule/checkpoint covers both.

### CLI reference (model size — per sub-network)

| Flag | Default | Meaning |
|---|---|---|
| `--cascade` | 4 | Unrolled iterations (SDUMCascades) |
| `--channels` | `24 48 96` | Restormer channels per level; **must double per level** (PixelShuffle transitions); list length = number of levels L |
| `--num-blocks` | `2 3 3` | TransformerBlocks per level (same length as `--channels`) |
| `--num-heads` | `1 2 4` | MDTA heads per level (must divide the level's channels) |
| `--num-refinement` | 2 | Refinement blocks at the top level |
| `--mlp-ratio` | 2.0 | GDFN hidden expansion |
| `--drop-path` | 0.0 | Stochastic depth (useful ≥0.05 for big/long runs) |
| `--csm-chans` / `--csm-pools` | 8 / 4 | Coil-sensitivity U-Net size |
| `--no-dc-weight-map` | off | Scalar DC weight instead of the per-pixel map |

### CLI reference (optimization)

| Flag | Default | Meaning |
|---|---|---|
| `-l/--lr` | 3e-4 | Peak LR (AdamW; transformer-typical, replaces the old 1e-3 Adam value) |
| `--min-lr` | 1e-6 | Cosine floor |
| `--warmup-epochs` | 2.0 | Linear warmup length (fractional allowed; LR is stepped **every iteration**) |
| `--weight-decay` | 1e-4 | AdamW weight decay |
| `--lr-schedule` | cosine | `cosine` (half-cycle to min-lr at the last epoch) or `constant` after warmup |
| `--accumulation_steps` | 1 | Gradient accumulation (batch stays 1 due to varying W; this is the effective-batch knob) |
| `--grad-checkpointing` | nonreentrant | `off` / `reentrant` / `nonreentrant` (see below) |
| `--seed` | 430 | Global seed (see Reproducibility) |
| `--init-from` | – | Warm start: an AccSDUM checkpoint loads as-is; a single-SDUM state_dict initializes **both** sub-nets |

**Optimizer/schedule rationale.** Transformer blocks (LayerNorm + attention) train
poorly under the old Adam@1e-3 + OneCycleLR recipe; AdamW with decoupled weight
decay plus linear warmup and half-cycle cosine decay is the standard recipe and is
what NV uses for SDUM (their Muon experiments excluded). The LR is a pure function
of the fractional epoch `epoch + iter/len(loader)` — there is **no scheduler
state**, so resuming from a checkpoint at epoch k automatically continues at the
mathematically correct LR (the old pipeline never saved OneCycleLR state and would
silently restart the ramp; that bug class is gone by construction).

### Gradient checkpointing modes

Cascades are the unit of checkpointing (the NV design): activations inside a
cascade are recomputed during backward, so peak memory ≈ one cascade's graph
instead of all of them.

| Mode | Behavior |
|---|---|
| `off` | No checkpointing; fastest backward, highest memory. |
| `reentrant` | Legacy `torch.utils.checkpoint(use_reentrant=True)`. Requires a grad-requiring input (handled internally) and flat tensor outputs (the cascade call is wrapped/flattened internally). |
| `nonreentrant` | Modern recommended mode (`use_reentrant=False`). Additionally checkpoints the CSM sub-graph *inside* each cascade (nested checkpointing is only safe non-reentrantly), reducing peak memory further to ≈ max(CSM graph, Restormer graph). |

All three modes produce identical losses and full gradients (verified by
`tests/test_sdum.py`); they differ only in memory/compute trade-off. Checkpointing
is training-only — inference always runs plain forwards.

### Reproducibility

- `--seed` (default 430) drives `seed_fix`: torch/CUDA/numpy/random seeding plus
  `cudnn.deterministic = True`, `cudnn.benchmark = False`.
- The training DataLoader now gets a seeded `torch.Generator` (shuffle order is
  reproducible, including with persistent workers) and a `worker_init_fn` that
  seeds numpy/random inside each worker from the loader's base seed.
- Verified: two identical seeded runs produce **bit-identical** per-iteration
  losses, LRs, and gradient norms.

### Resuming and warm starts

- Auto-resume (default) loads `best_model.pt` under the same `--net-name`;
  `--no-resume` starts fresh. VarNet-era checkpoints fail loudly with a key
  mismatch — use a fresh `--net-name`.
- The checkpoint stores the full training `args`, the model and optimizer state,
  epoch, and best val loss (same format as before).

## 4. Scaling guide

Defaults target a **GTX 1080 (8GB)**. Parameter counts: default AccSDUM =
13.6M total (6.8M per sub-net, of which 2M are the per-cascade DC weight maps).

**Before any long run on new hardware, do a one-iteration memory probe:** run the
training command with `-e 1 -r 1` on a couple of volumes and watch
`nvidia-smi` / add `print(torch.cuda.max_memory_allocated()/2**30)` — the worst
case is the widest volume (W=480). If it exceeds VRAM:
`--channels 16 32 64` → `--num-blocks 1 2 3` → `--cascade 3`, in that order.

Scale-up levers (in order of effectiveness), for bigger GPUs or longer training
budgets (multi-day training and >1s/slice inference are acceptable per challenge
history):

1. `--channels 36 72 144` (or `48 96 192`) — width is the main capacity knob.
2. `--num-blocks 2 4 4` and `--num-refinement 4` — depth per level.
3. `--cascade 6` — more unrolled iterations (NV-small uses 6).
4. A fourth level `--channels 24 48 96 192 --num-blocks 2 3 3 4 --num-heads 1 2 4 8`.
5. `--drop-path 0.05..0.1` when the model gets big enough to overfit.

Memory scales roughly linearly in `channels²·pixels` per block for GDFN/MDTA and
linearly in blocks; `--grad-checkpointing nonreentrant` keeps the cascade count
almost free memory-wise (compute cost ~+30%).

Inference speed: MDTA is linear in H·W; the default config reconstructs a slice in
well under a second on the smoke GPU (17s for two full volumes ≈ 0.3s/slice), far
inside `recon_eval.py`'s 60s/volume budget. `--cascade` and `--channels` are the
speed levers if you scale up.

## 5. Inference / evaluation

```bash
python reconstruct.py -n sdum_base -p /Data/leaderboard   # writes reconstructions_leaderboard/{acc4,acc8}
python leaderboard_eval.py -lp /Data/leaderboard -yp ../result/sdum_base/reconstructions_leaderboard
```

**No model hyperparameters are needed (or accepted) at inference** — `load_model`
rebuilds the network from `checkpoint['args']` stored at training time and loads
the weights strictly. This is why `recon_eval.py` (which still passes the legacy
`--cascade/--chans/--sens_chans` flags) works untouched: those values are simply
never read. If you ever need to override a stored hyperparameter, reconstruct.py
exposes optional `--sdum-cascade`, `--sdum-channels`, `--sdum-num-blocks`,
`--sdum-num-heads`, `--sdum-num-refinement`, `--sdum-mlp-ratio`,
`--sdum-csm-chans`, `--sdum-csm-pools` flags.

The recon_eval contract (`load_model` / `prep_volume` / `recon_slice` in
utils/learning/test_part.py) is unchanged in signature and behavior; routing to
sdum4/sdum8 happens inside the model's forward from the mask alone.

## 6. Tests

```bash
python tests/test_sdum.py        # add --cpu to force CPU
```

Covers: output shape/finiteness for W ∈ {320, 396, 480} at both accelerations;
correct sub-net routing (forward hooks); mixed-batch rejection; hard-DC limit
(weight=1 ⇒ sampled k-space equals the reference); and, for all three
checkpointing modes, identical losses and complete gradients on the routed sub-net
with zero gradient leakage into the idle one.

Determinism check: run the same seeded training command twice under different
`--net-name`s and diff the per-iteration loss lines (strip timings) — they must be
identical.

## 7. Known caveats

- The dual checkpoint is ~2× a single net; old VarNet checkpoints cannot be
  resumed or warm-started from (structurally different — loads fail loudly).
- `dc_weight_map` is sized 640×768; any hypothetical volume wider than 768 (none
  exist in this dataset) is handled by replicate-padding the map at runtime.
- Batch size must stay 1 (per-volume k-space width varies and there is no padding
  collate); use `--accumulation_steps` for larger effective batches.
- The 8GB fit of the default config is estimated from the architecture and probed
  at tiny scale on a 4GB dev GPU — run the one-iteration memory probe on the
  actual 1080 before committing to a multi-day run.
