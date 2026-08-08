"""Acceleration-aware wrapper: routes each batch to an acc4- or acc8-specialized
SDUM based on the undersampling mask's sampling ratio. No filename/metadata used."""
import torch
import torch.nn as nn

from utils.model.sdum import SDUM

# Measured on training data: acc4 ratio in [0.309, 0.312], acc8 in [0.196, 0.200].
ACC_THRESHOLD = 0.25


def classify_acc(mask: torch.Tensor) -> int:
    """Return 4 or 8 from the mask sampling ratio. Accepts any mask shape with
    batch dim 0. Raises ValueError if a batch mixes accelerations."""
    ratios = mask.detach().reshape(mask.shape[0], -1).float().mean(dim=1)
    is_acc4 = ratios > ACC_THRESHOLD
    if not (bool(is_acc4.all()) or bool((~is_acc4).all())):
        raise ValueError(f"Mixed accelerations in one batch: ratios={ratios.tolist()}")
    return 4 if bool(is_acc4[0]) else 8


class AccSDUM(nn.Module):
    """Two identical SDUMs; a deterministic mask classifier picks one per batch."""

    def __init__(self, **sdum_kwargs):
        super().__init__()
        self.sdum4 = SDUM(**sdum_kwargs)
        self.sdum8 = SDUM(**sdum_kwargs)

    def forward(self, masked_kspace: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        net = self.sdum4 if classify_acc(mask) == 4 else self.sdum8
        return net(masked_kspace, mask)


def build_acc_sdum(args, checkpoint_mode: str = None) -> AccSDUM:
    """Single source of truth for args -> AccSDUM, shared by training and inference.

    `args` may be the training CLI namespace or `checkpoint['args']`; missing
    attributes fall back to the defaults below.
    """
    return AccSDUM(
        num_cascades=getattr(args, "cascade", 4),
        channels=list(getattr(args, "channels", [24, 48, 96])),
        num_blocks=list(getattr(args, "num_blocks", [2, 3, 3])),
        num_heads=list(getattr(args, "num_heads", [1, 2, 4])),
        num_refinement=getattr(args, "num_refinement", 2),
        mlp_ratio=getattr(args, "mlp_ratio", 2.0),
        drop_path=getattr(args, "drop_path", 0.0),
        csm_chans=getattr(args, "csm_chans", 8),
        csm_pools=getattr(args, "csm_pools", 4),
        use_dc_weight_map=not getattr(args, "no_dc_weight_map", False),
        checkpoint_mode=checkpoint_mode
        if checkpoint_mode is not None
        else getattr(args, "grad_checkpointing", "off"),
    )
