# Add 1cycle LR schedule to VarNet training

## Context
The repo (`2026 baby VarNet` baseline, commits `df9b04b`/`38949d1`) trains with a flat
`torch.optim.Adam(model.parameters(), args.lr)` and no LR scheduler at all — confirmed via
full-repo search (no `scheduler`, `OneCycleLR`, `CosineAnnealing`, etc. anywhere). This is
the first of a planned series of training optimizations: the 1cycle policy (Smith,
"Super-Convergence"), which ramps the LR up from a low value to a peak (`max_lr`) and back
down (with an even lower final LR) over the course of training, typically speeding
convergence and improving generalization versus a constant LR.

This is a raw PyTorch training loop (no Lightning), so `OneCycleLR.step()` must be called
manually once per **batch**, not per epoch — PyTorch's `OneCycleLR` requires this to track
its internal cycle position correctly.

Decisions:
- `args.lr` (existing `-l/--lr` flag) becomes the OneCycle peak (`max_lr`), rather than adding
  a new `--max-lr` flag. `train.sh`'s existing `-l 0.001` keeps working unchanged. Note this
  does change training dynamics: average LR over a run will now be lower than before, since
  it ramps up then anneals down instead of holding constant at 0.001.
- `pct_start=0.3`, `div_factor=25`, `final_div_factor=1e4` are hardcoded (PyTorch's own
  defaults) rather than exposed as new CLI flags, to keep this first optimization's diff small.

No checkpoint-resume path currently exists (`start_epoch` is hardcoded to `0` in `train()`,
and nothing loads a saved checkpoint back in), so scheduler state does not need to be
persisted/restored — out of scope here.

## File to edit
Only `utils/learning/train_part.py`. `train.py` needs no changes since we're reusing the
existing `--lr` flag.

## Changes

**1. Import `OneCycleLR`** (top of file, near other imports):
```python
from torch.optim.lr_scheduler import OneCycleLR
```

**2. Construct the scheduler in `train()`**, right after `train_loader` is created (so
`len(train_loader)` — steps per epoch — is available). Current code around line 106-113:
```python
    loss_type = SSIMLoss().to(device=device)
    optimizer = torch.optim.Adam(model.parameters(), args.lr)

    best_val_loss = 1.
    start_epoch = 0

    
    train_loader = create_data_loaders(data_path = args.data_path_train, args = args, shuffle=True)
    val_loader = create_data_loaders(data_path = args.data_path_val, args = args)
```
becomes:
```python
    loss_type = SSIMLoss().to(device=device)
    optimizer = torch.optim.Adam(model.parameters(), args.lr)

    best_val_loss = 1.
    start_epoch = 0

    
    train_loader = create_data_loaders(data_path = args.data_path_train, args = args, shuffle=True)
    val_loader = create_data_loaders(data_path = args.data_path_val, args = args)

    scheduler = OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        div_factor=25,
        final_div_factor=1e4,
    )
```

**3. Pass `scheduler` into `train_epoch`** — update the call site (currently line 119):
```python
        train_loss, train_time = train_epoch(args, epoch, model, train_loader, optimizer, loss_type)
```
becomes:
```python
        train_loss, train_time = train_epoch(args, epoch, model, train_loader, optimizer, scheduler, loss_type)
```

**4. Step the scheduler per batch inside `train_epoch`** (currently lines 17-46). Add the
`scheduler` param, call `scheduler.step()` right after `optimizer.step()`, and surface the
current LR in the periodic log line so the schedule's shape is visible while training:
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
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

        if iter % args.report_interval == 0:
            print(
                f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
                f'Iter = [{iter:4d}/{len(data_loader):4d}] '
                f'Loss = {loss.item():.4g} '
                f'LR = {scheduler.get_last_lr()[0]:.3e} '
                f'Time = {time.perf_counter() - start_iter:.4f}s',
            )
            start_iter = time.perf_counter()
    total_loss = total_loss / len_loader
    return total_loss, time.perf_counter() - start_epoch
```

That's the entire change — `save_model`/`validate`/`train.py` are untouched.

## Verification
1. Run the existing training entry point end-to-end on a short config, e.g. `bash train.sh`
   (or a quick invocation like `python train.py -e 2 -r 5` against the configured data paths),
   and confirm it runs to completion without `OneCycleLR`'s "Tried to step X times, but total
   steps is Y" `ValueError` (this fires if the actual number of `scheduler.step()` calls ever
   diverges from `epochs * steps_per_epoch`, e.g. if `drop_last` behavior or dataset size
   changes unexpectedly).
2. Inspect the printed `LR = ...` values across the report-interval log lines: they should
   start near `max_lr/div_factor` (≈ `args.lr/25`), climb to `args.lr` around `pct_start`
   (30%) of the way through total training steps, then anneal down toward
   `args.lr/(div_factor*final_div_factor)` (≈ `args.lr/250000`) by the final step — confirming
   the schedule is wired correctly rather than silently behaving like a constant LR.
3. Confirm `model.pt`/`best_model.pt` checkpoints under `args.exp_dir` still save correctly
   (unchanged `save_model` call/signature).
