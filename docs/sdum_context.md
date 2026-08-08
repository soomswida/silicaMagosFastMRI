# SDUM Work Context (session of 2026-08-08)

Reference document for the SDUM replacement work on branch `sdum_net`. The model
itself is documented in `docs/sdum.md` (architecture, CLI, scaling, tests) — this
file records the session context: what was decided, why, what was verified, and
what remains.

## What happened

Branch `sdum_net` is fully devoted to replacing the dual-VarNet (AccVarNet, commit
`1e94c46`) with **AccSDUM**: two SDUMs (Scalable Deep Unrolled Model, structure
from NVIDIA's [NV-Raw2insights-MRI](https://github.com/NVIDIA-Medtech/NV-Raw2insights-MRI))
routed by the unchanged mask-ratio acc4/acc8 classifier. Committed as `63efa11`
("Replace VarNet with dual acceleration-routed SDUM").

## User decisions (fixed constraints)

- **Structure only** from the NV repo — pretrained weights are **forbidden**.
- Classifier, inputs, outputs, loss (SSIMLoss only), and metrics identical to the
  original project — required for submission validity. No filenames/metadata may
  influence the model.
- VarNet **fully replaced** (no model-selection flag); `varnet.py`/`acc_varnet.py`
  remain in-tree but orphaned.
- Must be **trainable on a GTX 1080 (8GB)** — the user's training machine. Long
  wall-clock is acceptable (prior winning teams: ~7-day trainings, >100s/instance
  inference), so the model must be **scalable via CLI** (cascades, channels,
  blocks, heads, refinement, MLP ratio, CSM size).
- Gradient checkpointing selectable per train command: `--grad-checkpointing
  {off,reentrant,nonreentrant}` (user explicitly wanted the reentrant choice).
- Optimizer: AdamW + per-iteration linear-warmup/half-cosine (NV recipe minus
  Muon). Seeds fixed for reproducibility.

## Key implementation facts

- New: `utils/model/sdum.py` (self-contained; no monai/einops/timm),
  `utils/model/acc_sdum.py` (classifier + dual wrapper + `build_acc_sdum` factory),
  `tests/test_sdum.py`, `docs/sdum.md`.
- Modified: `train.py` (new CLI), `utils/learning/train_part.py` (AdamW, stateless
  `adjust_learning_rate` — resume-exact LR, replacing the never-saved OneCycleLR),
  `utils/learning/test_part.py` (`load_model` rebuilds from `checkpoint['args']`,
  so the do-not-edit `recon_eval.py` works with its legacy flags ignored),
  `reconstruct.py` (optional `--sdum-*` overrides), `utils/data/load_data.py`
  (seeded shuffle generator + worker seeding).
- Two data-driven deviations from NV (documented in docs/sdum.md §1): exact
  per-slice scalar normalization in `SDUM.forward` (raw fastMRI k-space ~1e-5,
  DataTransform doesn't normalize) and `mask.bool()` in DC (mask is uint8; NV's
  `~mask` would corrupt it).
- Default size: cascade 4, channels [24,48,96] → **13.6M params** total AccSDUM
  (6.8M per sub-net, 2M of that in per-cascade 640×768 DC weight maps).

## Verification performed (all passed, local GTX 1650 Ti 4GB, tiny config)

1. `tests/test_sdum.py`: shapes/finiteness for W∈{320,396,480} both accs; routing
   hooks; mixed-batch ValueError; hard-DC limit (weight=1 ⇒ sampled k-space ==
   ref); all three checkpoint modes give identical losses (spread 0.0) with full
   gradients on the routed sub-net and none on the idle one.
2. Smoke train (`--cascade 2 --channels 8 16 ...`): decreasing loss, warmup→cosine
   LR visible in logs, `sdum4.*`/`sdum8.*` checkpoint keys, resume works.
3. `reconstruct.py` with **zero model flags** → `(slices,384,384)` H5s under
   `reconstructions_leaderboard/{acc4,acc8}`; recon_eval-style
   `load_model`/`prep_volume`/`recon_slice` contract exercised directly with
   legacy args present-but-ignored.
4. Determinism: two seeded (`--seed 430`) runs, 4 workers → per-iteration losses,
   LRs, grad norms **bit-identical**.

## Open items / next steps

- **Mandatory before a long run**: one-iteration memory probe of the default
  config on the actual GTX 1080 at worst-case W=480 (`docs/sdum.md` §4); fallback
  ladder `--channels 16 32 64` → `--num-blocks 1 2 3` → `--cascade 3`.
- Full training once the kspace download finishes (see environment notes below);
  no val/ or leaderboard/ splits exist locally yet.
- Old VarNet-era result dirs can't be resumed into SDUM (loud key mismatch) — use
  fresh `-n` names.
- `docs/accvarnet_context.md` (previous session's context) is intentionally left
  untracked; it describes the retired VarNet-era state.

## Environment (this machine)

- Data: `/home/kirill/fastmri-data/train/{image,kspace}`; image/ complete (170
  files), kspace/ still downloading (stray `Unconfirmed *.crdownload` partials
  crash `SliceData` — use symlinked clean subsets; image/kspace count mismatch
  desyncs positional pairing until complete).
- Python: `/home/kirill/miniconda3/envs/fastmri-repo/bin/python` (torch 2.5.1;
  scikit-image + opencv-python-headless installed 2026-08-08, numpy 2.2.6).
- Local GPU: GTX 1650 Ti 4GB (dev/smoke only); training target: GTX 1080 8GB.
- Outputs: `../result/<net_name>/` relative to repo root → `/home/kirill/result/`.
- NV reference clone lives in the session scratchpad (temporary — re-clone from
  GitHub if needed later).
