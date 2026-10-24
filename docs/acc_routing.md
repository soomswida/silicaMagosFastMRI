# Acceleration-aware dual VarNet (`AccVarNet`)

Design note for `utils/model/acc_varnet.py`.

## Problem

The challenge mixes two undersampling factors, 4x (`acc4`) and 8x (`acc8`), in both the
training set and the leaderboard set. The baseline trains a single E2E-VarNet on the mix, so
the network never knows which regime it is reconstructing even though the two look very
different in k-space (31% vs 20% of lines sampled) and in the severity of aliasing.
Competition rules forbid using filenames or any other metadata to tell the two apart.

## Design

`AccVarNet` holds two structurally identical E2E-VarNets, `varnet4` and `varnet8`, and a
deterministic classifier that picks one per batch using only the undersampling mask. The mask
is a legitimate model input (the baseline already feeds it to the data-consistency term), so
the routing is rule-compliant.

```
masked k-space (B,15,640,W,2) ─┐
                               ├─► classify_acc(mask) ──► 4 ─► varnet4 ─┐
mask (B,1,1,W,1) ──────────────┘                      └─► 8 ─► varnet8 ─┴─► RSS ─► crop 384x384
```

Routing rule (`classify_acc`): compute the mask's sampling ratio `mask.mean()`; a ratio above
`ACC_THRESHOLD = 0.25` is acc4, otherwise acc8.

| Acceleration | Measured sampling ratio (all training volumes) | Margin to threshold |
|---|---|---|
| acc4 | 0.309 – 0.312 | > 23% |
| acc8 | 0.196 – 0.200 | > 20% |

The classifier was checked against the filename ground truth on every training k-space
volume and was correct on all of them. A batch that mixes accelerations raises `ValueError`;
this cannot happen in practice because the variable k-space width (320–480) already forces
batch size 1.

This is a hard, rule-based router in front of two acceleration-specialised networks. It is
related to a mixture-of-experts layout but has no learned gate: the gate is a fixed threshold,
and only the selected expert runs, so inference cost equals a single VarNet.

## Training and checkpoints

- One joint training run trains both sub-networks: every batch is routed to one of them, the
  idle one receives no gradient (`grad is None`) and is skipped by the optimizer step. A single
  Adam + OneCycleLR instance covers both parameter sets.
- `validate()` reports the validation SSIM loss per acceleration (`Val(acc4)`, `Val(acc8)`),
  derived from the classifier rather than filenames.
- Checkpoint layout is unchanged (`../result/<net_name>/checkpoints/{model,best_model}.pt`);
  state-dict keys are prefixed `varnet4.*` / `varnet8.*`.
- `--init-from <single_varnet_best_model.pt>` initialises **both** sub-networks from a
  previously trained single VarNet, which is the fastest route to beating the single-network
  baseline. An old single-VarNet checkpoint under the same `-n` name cannot be auto-resumed
  (key mismatch); use a fresh name, `--no-resume`, or `--init-from`.
- Inference (`utils/learning/test_part.py::recon_slice`) is untouched apart from building
  `AccVarNet`; the routing happens inside `forward`, so the organiser's `recon_eval.py`
  harness needs no changes.
