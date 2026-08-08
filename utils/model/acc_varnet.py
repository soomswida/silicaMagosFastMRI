"""Acceleration-aware wrapper: routes each batch to an acc4- or acc8-specialized
VarNet based on the undersampling mask's sampling ratio. No filename/metadata used."""
import torch
import torch.nn as nn

from utils.model.varnet import VarNet

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


class AccVarNet(nn.Module):
    """Two identical VarNets; a deterministic mask classifier picks one per batch."""

    def __init__(
        self,
        num_cascades: int = 12,
        sens_chans: int = 8,
        sens_pools: int = 4,
        chans: int = 18,
        pools: int = 4,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        kwargs = dict(
            num_cascades=num_cascades,
            sens_chans=sens_chans,
            sens_pools=sens_pools,
            chans=chans,
            pools=pools,
            use_checkpoint=use_checkpoint,
        )
        self.varnet4 = VarNet(**kwargs)
        self.varnet8 = VarNet(**kwargs)

    def forward(self, masked_kspace: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        net = self.varnet4 if classify_acc(mask) == 4 else self.varnet8
        return net(masked_kspace, mask)
