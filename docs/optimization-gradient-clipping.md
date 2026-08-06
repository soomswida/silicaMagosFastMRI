# Add gradient clipping to VarNet training

## Context
VarNet cascades multiple U-Net sub-networks in sequence (RNN-like unrolling), which makes it
more prone than a plain feed-forward net to occasional gradient spikes/explosion — a risk that
goes up specifically at the peak of the [[optimization-1cycle-lr|OneCycleLR schedule]] already
in place. Clipping the gradient norm before `optimizer.step()` is a standard, low-risk pairing
with 1cycle policies: it doesn't fight the schedule, it just caps how large a single step can
be, which can let `max_lr` be pushed a bit higher later without risking divergence.

No conflicts with the current optimizer/scheduler setup — clipping just slots in between
`loss.backward()` and `optimizer.step()`. The only future interaction to keep in mind: if
gradient accumulation gets added later, clipping must happen once per *effective* (accumulated)
batch, right before the (less frequent) `optimizer.step()`/`scheduler.step()` calls — not once
per micro-batch.

## File to edit
Only `utils/learning/train_part.py`, inside `train_epoch`. No changes needed to `train.py` or
`train()`.

## Changes

**1. Clip after `loss.backward()`, before `optimizer.step()`** (currently lines 33-36):
```python
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
```
becomes:
```python
        optimizer.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
```
`torch.nn` is already imported as `nn` at the top of the file, so no new import is needed.
`max_norm=1.0` is a common, well-tested default for this kind of clipping — treat it as a
tunable hyperparameter, not a fixed constant, once you have a sense of the model's typical
gradient norms (see verification below).

**2. Surface the pre-clip norm in the periodic log line**, so you can see whether clipping is
actually engaging (currently lines 39-46):
```python
        if iter % args.report_interval == 0:
            print(
                f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
                f'Iter = [{iter:4d}/{len(data_loader):4d}] '
                f'Loss = {loss.item():.4g} '
                f'LR = {scheduler.get_last_lr()[0]:.3e} '
                f'Time = {time.perf_counter() - start_iter:.4f}s',
            )
```
becomes:
```python
        if iter % args.report_interval == 0:
            print(
                f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
                f'Iter = [{iter:4d}/{len(data_loader):4d}] '
                f'Loss = {loss.item():.4g} '
                f'LR = {scheduler.get_last_lr()[0]:.3e} '
                f'GradNorm = {grad_norm:.3e} '
                f'Time = {time.perf_counter() - start_iter:.4f}s',
            )
```
`clip_grad_norm_` returns the *pre-clip* total norm as a tensor, so this prints the actual
gradient magnitude each logged step, clipped or not.

## Verification
1. Run training (`bash train.sh` or a short `python train.py -e 2 -r 5` run) and watch the
   `GradNorm = ...` values in the log.
2. If `GradNorm` is consistently well below `max_norm=1.0`, clipping is a no-op in practice —
   either lower `max_norm` to where it starts engaging, or note that clipping isn't currently
   load-bearing for this model/data and can stay as a cheap safety net.
3. If `GradNorm` regularly exceeds `1.0` (especially near the LR peak from OneCycleLR), clipping
   is actively capping updates — worth comparing a short run with vs. without clipping (or at a
   couple of `max_norm` values) to see if `train_loss`/`val_loss` curves are smoother or allow a
   higher `max_lr` without divergence.
4. Confirm no shape/type errors — `clip_grad_norm_` returns a 0-d tensor, and the `.3e` format
   spec used above works fine on it directly (no `.item()` needed for printing).
