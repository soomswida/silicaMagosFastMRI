# Add gradient accumulation to VarNet training

## Context
`batch_size` is pinned to `1` in this project because VarNet is memory-hungry (README).
Gradient accumulation simulates a larger effective batch — accumulating gradients over `N`
micro-batches before a single `optimizer.step()` — without needing more GPU memory. The model
uses `InstanceNorm2d` (`utils/model/unet.py`), not `BatchNorm`, so there's no batch-statistics
issue that would make this unsafe.

It does **not** slot in as a pure addition, though — the current loop calls
`zero_grad()` → `backward()` → `clip_grad_norm_()` → `optimizer.step()` → `scheduler.step()`
on every micro-batch, and all of that needs to be gated to only fire once per accumulation
cycle:
- `zero_grad()` must move out of the per-iteration path, or accumulated gradients get wiped
  every micro-batch and nothing actually accumulates.
- `scheduler.step()` must only fire on real optimizer steps, and [[optimization-1cycle-lr|OneCycleLR's]]
  `steps_per_epoch` must change from `len(train_loader)` to the number of *optimizer* steps per
  epoch (`ceil(len(train_loader) / N)`), or the schedule finishes far earlier than training does.
- [[optimization-gradient-clipping|Gradient clipping]] must move to once per accumulated step —
  this was already flagged as a TODO in the `6408fd9` commit message.
- The loss must be divided by `N` before `.backward()` so the summed micro-batch gradients are
  equivalent to the *average* over the effective batch — otherwise accumulated gradients are
  `N`× too large, which (combined with the unchanged `max_norm=1.0`) would make clipping trigger
  far more than intended and effectively multiply the LR by `N`.
- `len(train_loader)` won't generally be divisible by `N`, so the last partial group at the end
  of each epoch needs to be flushed as its own (smaller) optimizer step, or those samples'
  gradients are silently dropped every epoch.

With `accumulation_steps=1` (the default below), the loop behaves identically to today — every
iteration is an accumulation boundary — so this is backward compatible.

## Files to edit
- `train.py` — new `--accumulation-steps` CLI flag.
- `utils/learning/train_part.py` — accumulation-aware loop + scheduler sizing.

## Changes

**1. `train.py` — add the CLI flag** (near the other hyperparameters, e.g. after `-l/--lr`):
```python
    parser.add_argument('--accumulation-steps', type=int, default=1, help='Number of micro-batches to accumulate gradients over before each optimizer step')
```

**2. `utils/learning/train_part.py` — add `import math`** (top of file, with the other imports):
```python
import shutil
import numpy as np
import torch
from torch.optim.lr_scheduler import OneCycleLR
import torch.nn as nn
import time
import math
from pathlib import Path
import copy
```

**3. Size the scheduler to real optimizer steps, not micro-batches** — in `train()`, current
code (lines 120-128):
```python
    scheduler = OneCycleLR(
        optimizer,
        max_lr = args.lr,
        epochs = args.num_epochs,
        steps_per_epoch = len(train_loader),
        pct_start = 0.3,
        div_factor = 25,
        final_div_factor = 1e4,
    )
```
becomes:
```python
    scheduler = OneCycleLR(
        optimizer,
        max_lr = args.lr,
        epochs = args.num_epochs,
        steps_per_epoch = math.ceil(len(train_loader) / args.accumulation_steps),
        pct_start = 0.3,
        div_factor = 25,
        final_div_factor = 1e4,
    )
```

**4. Gate `zero_grad`/clip/`step` in `train_epoch`** — current code (lines 18-49):
```python
def train_epoch(args, epoch, model, data_loader, optimizer, scheduler, loss_type):
    model.train()
    start_epoch = start_iter = time.perf_counter()
    len_loader = len(data_loader)
    total_loss = 0.

    for iter, data in enumerate(data_loader):
        mask, kspace, target, maximum, _, _ = data
        mask = mask.cuda(non_blocking=True)
        kspace = kspace.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        maximum = maximum.cuda(non_blocking=True)

        output = model(kspace, mask)
        loss = loss_type(output, target, maximum)
        optimizer.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

        if iter % args.report_interval == 0:
            print(
                f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
                f'Iter = [{iter:4d}/{len(data_loader):4d}] '
                f'Loss = {loss.item():.4g} '
                f'LR = {scheduler.get_last_lr()[0]:.3e} '
                f'GradNorm = {grad_norm:.3e} '
                f'Time = {time.perf_counter() - start_iter:.4f}s',
            )
            start_iter = time.perf_counter()
    total_loss = total_loss / len_loader
    return total_loss, time.perf_counter() - start_epoch
```
becomes:
```python
def train_epoch(args, epoch, model, data_loader, optimizer, scheduler, loss_type):
    model.train()
    start_epoch = start_iter = time.perf_counter()
    len_loader = len(data_loader)
    total_loss = 0.
    grad_norm = torch.tensor(0.)

    optimizer.zero_grad()
    for iter, data in enumerate(data_loader):
        mask, kspace, target, maximum, _, _ = data
        mask = mask.cuda(non_blocking=True)
        kspace = kspace.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        maximum = maximum.cuda(non_blocking=True)

        output = model(kspace, mask)
        loss = loss_type(output, target, maximum)
        (loss / args.accumulation_steps).backward()

        is_accum_boundary = (iter + 1) % args.accumulation_steps == 0
        is_last_iter = (iter + 1) == len_loader
        if is_accum_boundary or is_last_iter:
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        total_loss += loss.item()

        if iter % args.report_interval == 0:
            print(
                f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
                f'Iter = [{iter:4d}/{len(data_loader):4d}] '
                f'Loss = {loss.item():.4g} '
                f'LR = {scheduler.get_last_lr()[0]:.3e} '
                f'GradNorm = {grad_norm:.3e} '
                f'Time = {time.perf_counter() - start_iter:.4f}s',
            )
            start_iter = time.perf_counter()
    total_loss = total_loss / len_loader
    return total_loss, time.perf_counter() - start_epoch
```

Notes on this diff:
- `loss.item()` used for `total_loss`/logging is still the **unscaled** per-micro-batch loss
  (division by `args.accumulation_steps` only applies to the tensor that gets `.backward()`'d),
  so the reported `train_loss` average and the `Loss = ...` log values are unaffected by this
  change and stay comparable to previous runs.
- `grad_norm` is initialized before the loop and only updated on accumulation-boundary
  iterations, so the periodic log line always has a value to print even between boundaries
  (it'll just show the last real step's norm on non-boundary report ticks).
- `is_last_iter` guarantees the trailing partial group (when `len_loader` isn't divisible by
  `accumulation_steps`) still gets an optimizer step instead of being dropped.

## Verification
1. Sanity check with the default `--accumulation-steps 1`: behavior should be bit-for-bit
   equivalent to the current runs (every iteration is an accumulation boundary), so a short run
   should produce the same loss/LR/grad-norm trajectory as before this change.
2. Run with e.g. `--accumulation-steps 4` on a short config (`python train.py -e 1 -r 5
   --accumulation-steps 4`) and confirm:
   - No `OneCycleLR` "Tried to step X times, but total steps is Y" `ValueError` — this is the
     main indicator that `steps_per_epoch` and the accumulation-boundary condition are in sync.
   - The `LR = ...` and `GradNorm = ...` values in the log only change roughly every 4
     iterations, confirming steps are actually being batched.
3. Compare `GradNorm` magnitudes against the pre-accumulation runs — they should be in a similar
   range (not ~4x larger), confirming the loss-scaling-before-`backward()` is correctly keeping
   accumulated gradients equivalent to an averaged batch.
4. Confirm the final (possibly partial) group each epoch still triggers a step — e.g. pick a
   `--accumulation-steps` value that doesn't evenly divide the training set size and check the
   last logged iteration each epoch still shows an updated `GradNorm`/`LR`.
