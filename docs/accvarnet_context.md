# AccVarNet Work Context (session of 2026-08-08)

Reference document for the acceleration-aware dual-VarNet work done on branch `magos_preprocess`.

## Project background

SNU 2026 FastMRI knee challenge, "baby VarNet" baseline. Undersampled multi-coil k-space (acc4 and acc8) is reconstructed to 384×384 knee images by an E2E-VarNet. Scoring is SSIM (full + bbox) averaged over acc4 and acc8 leaderboard sets, plus reconstruction wall-clock time. `recon_eval.py` is a fixed do-not-edit harness; the team-editable contract is `load_model` / `prep_volume` / `recon_slice` in `utils/learning/test_part.py`.

**Problem addressed:** the baseline trained one VarNet blindly on a mix of acc4/acc8 volumes — the model never knew which acceleration it was seeing. Competition rules forbid using filenames or other metadata to tell them apart.

## What was built

A dual-network model, `AccVarNet` (`utils/model/acc_varnet.py`):

- Two identical VarNets, `varnet4` and `varnet8`, one specializing per acceleration.
- A deterministic classifier `classify_acc(mask)` routes each batch using only the undersampling mask (legitimate model input, not metadata): sampling ratio `mask.mean()` > **0.25** → acc4, else acc8.
- Empirical basis, measured over every training kspace H5: acc4 ratios ∈ [0.309, 0.312], acc8 ∈ [0.196, 0.200] — a >25% relative margin on both sides of the threshold. Classifier was 166/166 correct on real volumes.
- A batch mixing accelerations raises `ValueError` (can't happen at batch size 1, which is forced anyway by varying k-space width W = 320–480 with no padding collate).

## Files changed (all on `magos_preprocess`)

| File | Change |
|---|---|
| `utils/model/acc_varnet.py` | **New**: `AccVarNet`, `classify_acc`, `ACC_THRESHOLD` |
| `utils/learning/train_part.py` | Builds `AccVarNet` instead of `VarNet`; `validate()` also prints per-acc `Val(acc4)`/`Val(acc8)` losses (derived from the classifier, not fnames); `--init-from` warm-start block |
| `utils/learning/test_part.py` | `load_model` builds `AccVarNet`; the rest of the recon contract untouched |
| `train.py` | New optional `--init-from <path>` flag |
| `reconstruct.py` | Fixed pre-existing bug: missing `--num-workers` arg (default 4) |

**Deliberately unchanged:** SSIMLoss, Adam + OneCycleLR (one optimizer/scheduler over both sub-nets; the idle net gets `grad=None` and is skipped), checkpoint format/path (`../result/<net_name>/checkpoints/{model,best_model}.pt`, state-dict keys now `varnet4.*`/`varnet8.*`), all CLI usage, `recon_eval.py`, data loading, `leaderboard_eval.py`.

## Usage

Identical to the original — `bash train.sh`, `bash reconstruct.sh`, `bash leaderboard_eval.sh` work verbatim and automatically use the new model. One joint training run trains both sub-nets (each batch routed by its mask).

**First-run caveat:** an old single-VarNet `best_model.pt` under the same net name fails auto-resume with a state-dict key mismatch. Resolve with one of:
- fresh `-n` name,
- `--no-resume`,
- `--init-from '../result/<old_name>/checkpoints/best_model.pt'` — initializes **both** sub-nets from the old single-VarNet weights (recommended; likely fastest route to beating the baseline). Subsequent resumes then work normally.

## Verification performed (all passed)

1. Classifier: 166/166 correct vs filename ground truth on `/home/kirill/fastmri-data/train/kspace/*.h5`; mixed-batch detection works.
2. Routing: forward hooks confirmed acc4→`varnet4`, acc8→`varnet8` in both the training forward and the `recon_slice` path; outputs `(384, 384)`.
3. Smoke train (1 epoch, 4-volume symlinked subset, tiny net): per-acc val lines printed, checkpoint saved with dual keys, resume from it worked.
4. `reconstruct.py` on a fake leaderboard layout: H5 outputs `(slices, 384, 384)` under `reconstructions_leaderboard/{acc4,acc8}`.
5. `--init-from`: both sub-nets bit-identical to the source single-VarNet weights after loading.

## Environment facts (this machine)

- Data: `/home/kirill/fastmri-data/train/{image,kspace}` (scripts' default `/root/Data/...` paths don't exist here). image/ complete (170 files); kspace/ still downloading — stray `Unconfirmed *.crdownload` partials **will crash** `SliceData` (it only filters dot-files), and an image/kspace count mismatch desyncs its positional pairing. Use a clean symlinked subset until downloads finish. No local val/ or leaderboard/ splits.
- Python: `/home/kirill/miniconda3/envs/fastmri-repo/bin/python`. `scikit-image` + `opencv-python-headless` were installed this session (were missing despite requirements.txt); numpy went 2.0.2 → 2.2.6.
- GPU: GTX 1650 Ti (small VRAM → `--grad-checkpointing`, batch size 1).
- Outputs: `../result/<net_name>/` relative to repo root, i.e. `/home/kirill/result/`.

## Status / next steps

- All changes verified but **uncommitted** as of 2026-08-08.
- Sensible next steps: commit the branch; full training run once kspace downloads finish (fresh `-n` or `--init-from`); compare per-acc val losses against the single-net baseline.
