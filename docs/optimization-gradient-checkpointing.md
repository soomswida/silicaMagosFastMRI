# Add gradient checkpointing to VarNet training

## Context
On a GTX 1080 (8GB), scaling up `--cascade`/`--chans` to a competition-relevant model size
runs out of VRAM, because without checkpointing autograd keeps every cascade's internal U-Net
activations alive at once — `VarNet.forward` (`utils/model/varnet.py:238-246`) loops over
`self.cascades` (an `nn.ModuleList` of `VarNetBlock`s, each wrapping a full `NormUnet`)
sequentially:
```python
for cascade in self.cascades:
    kspace_pred = cascade(kspace_pred, masked_kspace, mask, sens_maps)
```
Peak activation memory therefore scales with `num_cascades × (one U-Net's activations)`.
Checkpointing each cascade call drops that to roughly one cascade's worth, independent of
`num_cascades`, at the cost of recomputing each checkpointed cascade's forward pass during
backward. Standard rule of thumb: normal step cost ≈ 1 (forward) + 2 (backward) = 3
forward-units; with every cascade checkpointed ≈ 1 + 1 (recompute) + 2 = 4 → **~30% extra
wall-clock time** for a memory footprint that no longer scales with depth. Given you're
memory-blocked rather than time-blocked, checkpointing every cascade (not a fraction of them)
is the recommended starting point.

No conflicts with the other optimizations already in place: checkpointing operates *within* a
single micro-batch's forward/backward, while gradient accumulation operates *across*
micro-batches — they're orthogonal. It's gated to training only (see below), so it doesn't
touch `validate()` or `reconstruct.py`/`test_part.py`'s inference path at all.

## Files to edit
- `utils/model/varnet.py` — add an opt-in `use_checkpoint` flag to `VarNet` and checkpoint the
  per-cascade forward call.
- `train.py` — new `--grad-checkpointing` CLI flag.
- `utils/learning/train_part.py` — pass the flag through when constructing the model.

`utils/learning/test_part.py` needs no change — it also constructs `VarNet(...)` but doesn't
pass `use_checkpoint`, so it keeps the default (`False`), which is correct since inference has
no backward pass to save memory on.

## Changes

**1. `utils/model/varnet.py` — import `checkpoint`** (top of file, with the other imports):
```python
import fastmri
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from fastmri.data import transforms
```

**2. `VarNet.__init__` — accept and store the flag** (currently lines 212-236):
```python
    def __init__(
        self,
        num_cascades: int = 12,
        sens_chans: int = 8,
        sens_pools: int = 4,
        chans: int = 18,
        pools: int = 4,
    ):
        """
        Args:
            num_cascades: Number of cascades (i.e., layers) for variational
                network.
            sens_chans: Number of channels for sensitivity map U-Net.
            sens_pools Number of downsampling and upsampling layers for
                sensitivity map U-Net.
            chans: Number of channels for cascade U-Net.
            pools: Number of downsampling and upsampling layers for cascade
                U-Net.
        """
        super().__init__()

        self.sens_net = SensitivityModel(sens_chans, sens_pools)
        self.cascades = nn.ModuleList(
            [VarNetBlock(NormUnet(chans, pools)) for _ in range(num_cascades)]
        )
```
becomes:
```python
    def __init__(
        self,
        num_cascades: int = 12,
        sens_chans: int = 8,
        sens_pools: int = 4,
        chans: int = 18,
        pools: int = 4,
        use_checkpoint: bool = False,
    ):
        """
        Args:
            num_cascades: Number of cascades (i.e., layers) for variational
                network.
            sens_chans: Number of channels for sensitivity map U-Net.
            sens_pools Number of downsampling and upsampling layers for
                sensitivity map U-Net.
            chans: Number of channels for cascade U-Net.
            pools: Number of downsampling and upsampling layers for cascade
                U-Net.
            use_checkpoint: If True, recompute each cascade's activations
                during backward instead of storing them, trading compute for
                memory. Only takes effect in training mode.
        """
        super().__init__()

        self.sens_net = SensitivityModel(sens_chans, sens_pools)
        self.cascades = nn.ModuleList(
            [VarNetBlock(NormUnet(chans, pools)) for _ in range(num_cascades)]
        )
        self.use_checkpoint = use_checkpoint
```

**3. `VarNet.forward` — checkpoint the per-cascade call, training only** (currently lines
238-246):
```python
    def forward(self, masked_kspace: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        sens_maps = self.sens_net(masked_kspace, mask)
        kspace_pred = masked_kspace.clone()

        for cascade in self.cascades:
            kspace_pred = cascade(kspace_pred, masked_kspace, mask, sens_maps)
        result = fastmri.rss(fastmri.complex_abs(fastmri.ifft2c(kspace_pred)), dim=1)
        result = center_crop(result, 384, 384)
        return result
```
becomes:
```python
    def forward(self, masked_kspace: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        sens_maps = self.sens_net(masked_kspace, mask)
        kspace_pred = masked_kspace.clone()

        for cascade in self.cascades:
            if self.use_checkpoint and self.training:
                kspace_pred = checkpoint(
                    cascade, kspace_pred, masked_kspace, mask, sens_maps, use_reentrant=False
                )
            else:
                kspace_pred = cascade(kspace_pred, masked_kspace, mask, sens_maps)
        result = fastmri.rss(fastmri.complex_abs(fastmri.ifft2c(kspace_pred)), dim=1)
        result = center_crop(result, 384, 384)
        return result
```
`use_reentrant=False` is the modern, recommended checkpoint mode — it doesn't depend on the
older reentrant implementation's requirement that at least one input tensor already have
`requires_grad=True` going in. The `and self.training` guard means `model.eval()` (used in
`validate()` and inference) always takes the plain, non-checkpointed path, so there's zero
recompute overhead outside of training.

**4. `train.py` — add the CLI flag** (near the other hyperparameters):
```python
    parser.add_argument('--grad-checkpointing', action='store_true', help='Enable gradient checkpointing on VarNet cascades to trade ~30 percent more compute for lower peak memory')
```
`action='store_true'` means it defaults to `False` (off) unless you pass the flag — no
behavior change for existing invocations of `train.sh`.

**5. `utils/learning/train_part.py` — pass it through to the model** (currently lines
100-102):
```python
    model = VarNet(num_cascades=args.cascade, 
                   chans=args.chans, 
                   sens_chans=args.sens_chans)
```
becomes:
```python
    model = VarNet(num_cascades=args.cascade, 
                   chans=args.chans, 
                   sens_chans=args.sens_chans,
                   use_checkpoint=args.grad_checkpointing)
```

## How to use it once installed

- **Turning it on**: pass `--grad-checkpointing` on the command line, e.g.
  `python train.py -e 5 -l 0.001 --cascade 8 --chans 18 --grad-checkpointing`, or add the flag
  to `train.sh`. Leaving it off (the default) reproduces the exact current training behavior —
  nothing changes unless you opt in.
- **When to turn it on**: only when you're actually memory-constrained — i.e. when a
  `--cascade`/`--chans` combination you want to try currently OOMs on the GTX 1080. It costs
  ~30% extra training wall-clock time for no benefit if you're not memory-bound, so don't leave
  it on for configurations that already fit comfortably.
- **Tuning model scale with it on**: with checkpointing enabled, you can push `--cascade` (and
  to a lesser extent `--chans`, which affects per-cascade activation size more than count) well
  past what fit before, since peak activation memory no longer scales with `--cascade`. Increase
  `--cascade` incrementally and watch for OOM again — the new ceiling is set by other memory
  users (model parameters/optimizer state, the single largest cascade's activations, batch
  data) rather than by depth.
- **It only affects training**: `validate()` and `reconstruct.py`/`leaderboard_eval.py` run the
  model in `eval()` mode, so checkpointing never activates there even if the model was trained
  with `--grad-checkpointing` — no extra inference latency, and no change needed on the
  evaluation side.
- **Combining with gradient accumulation**: no interaction to manage — checkpointing changes
  how one micro-batch's forward/backward computes internally, while
  [[optimization-gradient-accumulation|accumulation]] changes how often `optimizer.step()` is
  called across micro-batches. Both can be on at once with no extra wiring.
- **If it's still not enough memory**: the next lever (not implemented here) would be
  checkpointing *inside* each cascade's `NormUnet`/`Unet` too (e.g. per resolution level)
  instead of just at the cascade boundary — finer-grained checkpointing recovers more memory
  at a higher recompute cost. Only worth doing if checkpointing every cascade plus a
  `--cascade`/`--chans` you actually need still doesn't fit in 8GB.

## Verification
1. Confirm default behavior is unchanged: run without `--grad-checkpointing` and check
   `--cascade`/`--chans` values that previously worked still train identically (no accuracy or
   speed regression, since the flag defaults to off).
2. With a `--cascade`/`--chans` combination that currently OOMs on the 1080, rerun with
   `--grad-checkpointing` added and confirm it now fits. You can quantify the memory savings by
   wrapping a training step with `torch.cuda.reset_peak_memory_stats()` before and
   `torch.cuda.max_memory_allocated()` after, comparing with vs. without the flag on a
   configuration that fits both ways.
3. Confirm gradients still flow correctly with checkpointing on: compare `train_loss` for the
   first few iterations at a small, fixed `--cascade`/`--chans`/`--seed` with and without
   `--grad-checkpointing` — they should match closely (checkpointing changes *when* activations
   are computed, not the actual forward/backward math, so losses should track each other).
4. Time a fixed number of iterations with and without the flag on a configuration that fits
   both ways, and confirm the overhead is in the expected ~20-40% range rather than something
   much larger (which would suggest checkpointing granularity needs revisiting).
5. Confirm `validate()` and a `reconstruct.py` run are unaffected — same reconstructions/timing
   whether or not the model was trained with `--grad-checkpointing`, since `model.eval()`
   always takes the non-checkpointed path.
