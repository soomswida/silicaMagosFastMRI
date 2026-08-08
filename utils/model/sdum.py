"""
SDUM: Scalable Deep Unrolled Model for accelerated MRI reconstruction.

Architecture reimplemented (structure only, no pretrained weights) from
NVIDIA's NV-Raw2insights-MRI repository:
    https://github.com/NVIDIA-Medtech/NV-Raw2insights-MRI

Each cascade works in the image domain:
    coil-sensitivity estimation (ACS-region input on the first cascade)
    -> sensitivity-reduce to one virtual coil
    -> Restormer regularizer (MDTA + GDFN transformer blocks) with
       cross-cascade skip features (cas_skips)
    -> sensitivity-expand back to multi-coil
    -> residual skip
    -> sampling-aware weighted soft data consistency in k-space with a
       learned per-pixel dc_weight_map (scalar dc_weight fallback).

Deviations from the NV reference (see docs/sdum.md for rationale):
    - all metadata conditioning (mask type / acceleration / acquisition
      labels, timestep embeddings), MoE, OCAB/HybridRestormer removed;
    - per-slice scalar input normalization added at the wrapper level;
    - data-consistency mask handled as bool (input mask arrives as uint8);
    - CSM network is this repo's NormUnet instead of MONAI's BasicUNet.
"""

import math
from typing import List, Optional, Sequence, Tuple

import fastmri
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from fastmri.data import transforms

from unet import Unet
from utils.common.utils import center_crop


# ---------------------------------------------------------------------------
# Vendored ops (einops/timm/monai-free equivalents of the NV helpers)
# ---------------------------------------------------------------------------


class DropPath(nn.Module):
    """Per-sample stochastic depth (timm-equivalent)."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        mask = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(keep_prob)
        return x * mask / keep_prob


def divisible_pad(x: torch.Tensor, k: int) -> Tuple[torch.Tensor, Tuple[List[int], List[int], int, int]]:
    """Pad the last two dims of (B,C,H,W) to multiples of k."""
    _, _, h, w = x.shape
    h_mult = ((h - 1) // k + 1) * k
    w_mult = ((w - 1) // k + 1) * k
    w_pad = [math.floor((w_mult - w) / 2), math.ceil((w_mult - w) / 2)]
    h_pad = [math.floor((h_mult - h) / 2), math.ceil((h_mult - h) / 2)]
    x = F.pad(x, w_pad + h_pad)
    return x, (h_pad, w_pad, h_mult, w_mult)


def inverse_divisible_pad(
    x: torch.Tensor, pad_sizes: Tuple[List[int], List[int], int, int]
) -> torch.Tensor:
    h_pad, w_pad, h_mult, w_mult = pad_sizes
    return x[..., h_pad[0] : h_mult - h_pad[1], w_pad[0] : w_mult - w_pad[1]]


def complex_to_chan_dim(x: torch.Tensor) -> torch.Tensor:
    b, c, h, w, two = x.shape
    assert two == 2
    return x.permute(0, 4, 1, 2, 3).reshape(b, 2 * c, h, w)


def chan_complex_to_last_dim(x: torch.Tensor) -> torch.Tensor:
    b, c2, h, w = x.shape
    assert c2 % 2 == 0
    c = c2 // 2
    return x.view(b, 2, c, h, w).permute(0, 2, 3, 4, 1).contiguous()


def sens_reduce(x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
    """Combine coil images (B,C,H,W,2) into one virtual coil (B,1,H,W,2)."""
    return fastmri.complex_mul(x, fastmri.complex_conj(sens_maps)).sum(dim=1, keepdim=True)


def sens_expand(x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
    """Expand a virtual-coil image (B,1,H,W,2) back to coils (B,C,H,W,2)."""
    return fastmri.complex_mul(x, sens_maps)


def acs_bounds(mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Locate the fully-sampled center of a (B,1,1,W,1) mask.

    Same argmin-from-center logic as the original VarNet SensitivityModel.
    Returns (pad, num_low_freqs) per batch element.
    """
    squeezed_mask = mask[:, 0, 0, :, 0]
    cent = squeezed_mask.shape[1] // 2
    # running argmin returns the first non-zero
    left = torch.argmin(squeezed_mask[:, :cent].flip(1), dim=1)
    right = torch.argmin(squeezed_mask[:, cent:], dim=1)
    num_low_freqs = torch.max(
        2 * torch.min(left, right), torch.ones_like(left)
    )  # force a symmetric center unless 1
    pad = (mask.shape[-2] - num_low_freqs + 1) // 2
    return pad, num_low_freqs


def get_acs_image(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Zero out everything but the fully-sampled k-space center of coil images x."""
    kspace = fastmri.fft2c(x)
    pad, num_low_freqs = acs_bounds(mask)
    kspace = transforms.batched_mask_center(kspace, pad, pad + num_low_freqs)
    return fastmri.ifft2c(kspace)


# ---------------------------------------------------------------------------
# Restormer building blocks (NV structure, einops-free)
# ---------------------------------------------------------------------------


class MDTA(nn.Module):
    """Multi-DConv-head Transposed Attention: attention across channels, linear in H*W."""

    def __init__(self, channels: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(1, num_heads, 1, 1))

        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.qkv_conv = nn.Conv2d(
            channels * 3, channels * 3, kernel_size=3, padding=1, groups=channels * 3, bias=False
        )
        self.project_out = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        q, k, v = self.qkv_conv(self.qkv(x)).chunk(3, dim=1)

        q = q.reshape(b, self.num_heads, -1, h * w)
        k = k.reshape(b, self.num_heads, -1, h * w)
        v = v.reshape(b, self.num_heads, -1, h * w)
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)

        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1).contiguous()) * self.temperature, dim=-1)
        out = self.project_out(torch.matmul(attn, v).reshape(b, -1, h, w))
        return out


class GDFN(nn.Module):
    """Gated-DConv Feed-forward Network."""

    def __init__(self, channels: int, expansion_factor: float):
        super().__init__()
        hidden_channels = int(channels * expansion_factor)
        self.project_in = nn.Conv2d(channels, hidden_channels * 2, kernel_size=1, bias=False)
        self.conv = nn.Conv2d(
            hidden_channels * 2,
            hidden_channels * 2,
            kernel_size=3,
            padding=1,
            groups=hidden_channels * 2,
            bias=False,
        )
        self.project_out = nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = self.conv(self.project_in(x)).chunk(2, dim=1)
        return self.project_out(F.gelu(x1) * x2)


class TransformerBlock(nn.Module):
    """Pre-LayerNorm MDTA + GDFN block with DropPath (NV 'ln' norm_type path)."""

    def __init__(self, channels: int, num_heads: int, expansion_factor: float, drop_path: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.attn = MDTA(channels, num_heads)
        self.ffn = GDFN(channels, expansion_factor)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    @staticmethod
    def _ln(norm: nn.LayerNorm, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x = norm(x.reshape(b, c, -1).transpose(-2, -1).contiguous())
        return x.transpose(-2, -1).contiguous().reshape(b, c, h, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self._ln(self.norm1, x)))
        x = x + self.drop_path(self.ffn(self._ln(self.norm2, x)))
        return x


class DownSample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1, bias=False),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class UpSample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels * 2, kernel_size=3, padding=1, bias=False),
            nn.PixelShuffle(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class Restormer(nn.Module):
    """Restormer-style U-shaped transformer with cross-cascade skip features.

    cas_skips contract (NV): cas_skips is a tuple of length L-1;
    cas_skips[i-1] is added to the encoder input at level i (i = 1..L-1).
    Returns (out, cas_outs) with
    cas_outs = (dec_out(level 1), ..., dec_out(level L-2), enc_out(level L-1)).
    """

    def __init__(
        self,
        in_channel: int = 2,
        out_channel: int = 2,
        channels: Sequence[int] = (24, 48, 96),
        num_blocks: Sequence[int] = (2, 3, 3),
        num_heads: Sequence[int] = (1, 2, 4),
        num_refinement: int = 2,
        expansion_factor: float = 2.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        L = len(channels)
        assert L >= 2, "channels must have at least 2 levels"
        assert len(num_blocks) == len(num_heads) == L, (
            "channels, num_blocks and num_heads must have the same length"
        )
        for i in range(L - 1):
            assert channels[i + 1] == 2 * channels[i], (
                f"channels must double per level (PixelShuffle transitions), got {list(channels)}"
            )
        for c, h in zip(channels, num_heads):
            assert c % h == 0, f"channels ({c}) must be divisible by num_heads ({h})"
        assert channels[0] % 2 == 0, "channels[0] must be even (PixelUnshuffle transition)"
        self.L = L
        self.pad_factor = 2 ** (L - 1)

        self.embed_conv = nn.Conv2d(in_channel, channels[0], kernel_size=3, padding=1, bias=False)

        def blocks(i: int, n: int) -> nn.Sequential:
            return nn.Sequential(
                *[
                    TransformerBlock(channels[i], num_heads[i], expansion_factor, drop_path)
                    for _ in range(n)
                ]
            )

        self.encoders = nn.ModuleList([blocks(i, num_blocks[i]) for i in range(L)])
        self.downs = nn.ModuleList([DownSample(channels[i]) for i in range(L - 1)])
        self.ups = nn.ModuleList([UpSample(channels[i + 1]) for i in range(L - 1)])
        self.reduces = nn.ModuleList(
            [nn.Conv2d(2 * channels[i], channels[i], kernel_size=1, bias=False) for i in range(L - 1)]
        )
        self.decoders = nn.ModuleList([blocks(i, num_blocks[i]) for i in range(L - 1)])
        self.refinement = blocks(0, num_refinement)
        self.output = nn.Conv2d(channels[0], out_channel, kernel_size=3, padding=1, bias=False)

    def forward(
        self, x: torch.Tensor, cas_skips: Optional[Tuple[torch.Tensor, ...]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        L = self.L
        if cas_skips is not None:
            assert len(cas_skips) == L - 1, f"cas_skips must have length {L - 1}, got {len(cas_skips)}"

        x, pad_sizes = divisible_pad(x, self.pad_factor)

        # encoder
        feats: List[torch.Tensor] = [self.encoders[0](self.embed_conv(x))]
        for i in range(1, L):
            down_in = self.downs[i - 1](feats[i - 1])
            if cas_skips is not None:
                down_in = down_in + cas_skips[i - 1]
            feats.append(self.encoders[i](down_in))

        # decoder
        x_dec = feats[-1]
        dec_outs: List[Optional[torch.Tensor]] = [None] * (L - 1)
        for i in reversed(range(L - 1)):
            up = self.ups[i](x_dec)
            red = self.reduces[i](torch.cat([up, feats[i]], dim=1))
            x_dec = self.decoders[i](red)
            dec_outs[i] = x_dec

        out = self.output(self.refinement(dec_outs[0]))
        out = inverse_divisible_pad(out, pad_sizes)

        cas_outs = tuple(dec_outs[1 : L - 1]) + (feats[-1],)
        return out, cas_outs


# ---------------------------------------------------------------------------
# Coil sensitivity estimation (NormUnet copied from utils/model/varnet.py so
# this module stays independent of the retired VarNet file)
# ---------------------------------------------------------------------------


class NormUnet(nn.Module):
    """U-Net with group normalization of the complex input (as in E2E-VarNet)."""

    def __init__(
        self,
        chans: int,
        num_pools: int,
        in_chans: int = 2,
        out_chans: int = 2,
        drop_prob: float = 0.0,
    ):
        super().__init__()
        self.unet = Unet(
            in_chans=in_chans,
            out_chans=out_chans,
            chans=chans,
            num_pool_layers=num_pools,
            drop_prob=drop_prob,
        )

    def norm(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # group norm
        b, c, h, w = x.shape
        x = x.view(b, 2, c // 2 * h * w)
        mean = x.mean(dim=2).view(b, c, 1, 1)
        std = x.std(dim=2).view(b, c, 1, 1)
        x = x.view(b, c, h, w)
        return (x - mean) / std, mean, std

    def unnorm(self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        return x * std + mean

    def pad(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[List[int], List[int], int, int]]:
        _, _, h, w = x.shape
        w_mult = ((w - 1) | 15) + 1
        h_mult = ((h - 1) | 15) + 1
        w_pad = [math.floor((w_mult - w) / 2), math.ceil((w_mult - w) / 2)]
        h_pad = [math.floor((h_mult - h) / 2), math.ceil((h_mult - h) / 2)]
        x = F.pad(x, w_pad + h_pad)
        return x, (h_pad, w_pad, h_mult, w_mult)

    def unpad(
        self, x: torch.Tensor, h_pad: List[int], w_pad: List[int], h_mult: int, w_mult: int
    ) -> torch.Tensor:
        return x[..., h_pad[0] : h_mult - h_pad[1], w_pad[0] : w_mult - w_pad[1]]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.shape[-1] == 2:
            raise ValueError("Last dimension must be 2 for complex.")
        x = complex_to_chan_dim(x)
        x, mean, std = self.norm(x)
        x, pad_sizes = self.pad(x)
        x = self.unet(x)
        x = self.unpad(x, *pad_sizes)
        x = self.unnorm(x, mean, std)
        return chan_complex_to_last_dim(x)


class CoilSensitivityModel(nn.Module):
    """Estimate coil sensitivity maps from coil images (image-domain input)."""

    def __init__(self, chans: int, num_pools: int):
        super().__init__()
        self.norm_unet = NormUnet(chans, num_pools)

    def forward(self, coil_images: torch.Tensor) -> torch.Tensor:
        b, c, h, w, comp = coil_images.shape
        x = coil_images.view(b * c, 1, h, w, comp)
        x = self.norm_unet(x)
        x = x.view(b, c, h, w, comp)
        return x / fastmri.rss_complex(x, dim=1).unsqueeze(-1).unsqueeze(1)


# ---------------------------------------------------------------------------
# SDUM cascade and full model
# ---------------------------------------------------------------------------


class SDUMCascade(nn.Module):
    """One unrolled iteration: CSM -> reduce -> Restormer -> expand -> skip -> weighted DC."""

    def __init__(
        self,
        channels: Sequence[int],
        num_blocks: Sequence[int],
        num_heads: Sequence[int],
        num_refinement: int,
        mlp_ratio: float,
        drop_path: float,
        csm_chans: int,
        csm_pools: int,
        use_dc_weight_map: bool = True,
        first_cascade: bool = False,
        dc_map_size: Tuple[int, int] = (640, 768),
    ):
        super().__init__()
        self.first_cascade = first_cascade
        self.use_dc_weight_map = use_dc_weight_map
        # set by SDUM: checkpoint the CSM sub-graph separately in training
        self.csm_checkpoint = False

        self.csm = CoilSensitivityModel(csm_chans, csm_pools)
        self.recon = Restormer(
            in_channel=2,
            out_channel=2,
            channels=channels,
            num_blocks=num_blocks,
            num_heads=num_heads,
            num_refinement=num_refinement,
            expansion_factor=mlp_ratio,
            drop_path=drop_path,
        )
        if use_dc_weight_map:
            self.dc_weight_map = nn.Parameter(torch.ones(1, 1, *dc_map_size, 1))
        else:
            self.dc_weight = nn.Parameter(torch.ones(1))

    def _dc_weight(self, k_shape: torch.Size) -> torch.Tensor:
        if not self.use_dc_weight_map:
            return self.dc_weight
        w = self.dc_weight_map  # (1,1,Hm,Wm,1)
        h, width = k_shape[-3], k_shape[-2]
        h_map, w_map = w.shape[2], w.shape[3]
        if h_map < h or w_map < width:
            pad_h, pad_w = max(h - h_map, 0), max(width - w_map, 0)
            perm = w.permute(0, 1, 4, 2, 3)  # (1,1,1,Hm,Wm)
            perm = F.pad(
                perm,
                (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2, 0, 0),
                mode="replicate",
            )
            w = perm.permute(0, 1, 3, 4, 2)
            h_map, w_map = w.shape[2], w.shape[3]
        top, left = (h_map - h) // 2, (w_map - width) // 2
        return w[:, :, top : top + h, left : left + width, :]

    def soft_dc(self, x: torch.Tensor, ref_image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        kspace_pred = fastmri.fft2c(x)
        ref_kspace = fastmri.fft2c(ref_image)
        weight = self._dc_weight(kspace_pred.shape)
        # mask arrives as uint8: convert to bool, never use bitwise ~ on it
        kspace_pred = torch.where(
            mask.bool(), (1 - weight) * kspace_pred + weight * ref_kspace, kspace_pred
        )
        return fastmri.ifft2c(kspace_pred)

    def forward(
        self,
        x: torch.Tensor,
        ref_image: torch.Tensor,
        mask: torch.Tensor,
        cas_skips: Optional[Tuple[torch.Tensor, ...]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        skip = x
        csm_in = get_acs_image(x, mask) if self.first_cascade else x
        if self.training and self.csm_checkpoint:
            sens_maps = checkpoint(self.csm, csm_in, use_reentrant=False)
        else:
            sens_maps = self.csm(csm_in)

        v = complex_to_chan_dim(sens_reduce(x, sens_maps))  # (B,2,H,W)
        out, cas_skips = self.recon(v, cas_skips)
        v = chan_complex_to_last_dim(out)  # (B,1,H,W,2)

        x = skip + sens_expand(v, sens_maps)
        x = self.soft_dc(x, ref_image, mask)
        return x, cas_skips


class SDUM(nn.Module):
    """Full unrolled SDUM: forward(masked_kspace (B,C,H,W,2), mask (B,1,1,W,1)) -> (B,384,384)."""

    CHECKPOINT_MODES = ("off", "reentrant", "nonreentrant")

    def __init__(
        self,
        num_cascades: int = 4,
        channels: Sequence[int] = (24, 48, 96),
        num_blocks: Sequence[int] = (2, 3, 3),
        num_heads: Sequence[int] = (1, 2, 4),
        num_refinement: int = 2,
        mlp_ratio: float = 2.0,
        drop_path: float = 0.0,
        csm_chans: int = 8,
        csm_pools: int = 4,
        use_dc_weight_map: bool = True,
        checkpoint_mode: str = "off",
    ):
        super().__init__()
        assert checkpoint_mode in self.CHECKPOINT_MODES, (
            f"checkpoint_mode must be one of {self.CHECKPOINT_MODES}, got {checkpoint_mode!r}"
        )
        self.checkpoint_mode = checkpoint_mode
        self.cascades = nn.ModuleList(
            [
                SDUMCascade(
                    channels=channels,
                    num_blocks=num_blocks,
                    num_heads=num_heads,
                    num_refinement=num_refinement,
                    mlp_ratio=mlp_ratio,
                    drop_path=drop_path,
                    csm_chans=csm_chans,
                    csm_pools=csm_pools,
                    use_dc_weight_map=use_dc_weight_map,
                    first_cascade=(i == 0),
                )
                for i in range(num_cascades)
            ]
        )
        # nested (CSM-level) checkpointing is only safe under non-reentrant mode
        for cascade in self.cascades:
            cascade.csm_checkpoint = checkpoint_mode == "nonreentrant"

    def _run_cascade(self, cascade, x, ref_image, mask, cas_skips):
        if not self.training or self.checkpoint_mode == "off":
            return cascade(x, ref_image, mask, cas_skips)

        skips = cas_skips if cas_skips is not None else ()

        # reentrant checkpoint cannot return nested tuples: flatten in/out
        def fn(x, ref_image, mask, *skips):
            out, new_skips = cascade(x, ref_image, mask, tuple(skips) if skips else None)
            return (out, *new_skips)

        flat = checkpoint(
            fn, x, ref_image, mask, *skips, use_reentrant=self.checkpoint_mode == "reentrant"
        )
        return flat[0], tuple(flat[1:])

    def forward(self, masked_kspace: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = fastmri.ifft2c(masked_kspace)

        # Per-slice scalar normalization: raw fastMRI magnitudes are ~1e-5,
        # which the un-normalized Restormer path can't handle. All cascade ops
        # (FFT, DC, sens reduce/expand) are linear and the CSM is
        # RSS-normalized, so dividing here and re-scaling the output is exact.
        scale = x.std() + 1e-12
        x = x / scale
        if self.training and self.checkpoint_mode == "reentrant":
            # reentrant checkpointing needs at least one grad-requiring input
            x = x.requires_grad_(True)

        ref_image = x.clone()
        cas_skips = None
        for cascade in self.cascades:
            x, cas_skips = self._run_cascade(cascade, x, ref_image, mask, cas_skips)

        out = fastmri.rss(fastmri.complex_abs(x), dim=1) * scale
        return center_crop(out, 384, 384)
