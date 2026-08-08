"""Unit checks for the SDUM model. Run from the repo root:

    python tests/test_sdum.py [--cpu]
"""
import argparse
import os
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)
sys.path.insert(0, REPO)
sys.path.insert(1, os.path.join(REPO, "utils/model"))

from utils.common.loss_function import SSIMLoss
from utils.model.acc_sdum import AccSDUM, classify_acc

TINY = dict(
    num_cascades=2,
    channels=[8, 16],
    num_blocks=[1, 1],
    num_heads=[1, 2],
    num_refinement=1,
    mlp_ratio=1.5,
    drop_path=0.0,
    csm_chans=4,
    csm_pools=3,
    use_dc_weight_map=True,
)


def make_mask(width, stride, device):
    m = np.zeros(width, dtype=np.float32)
    m[::stride] = 1
    c = int(0.08 * width)
    m[width // 2 - c // 2 : width // 2 + c // 2] = 1
    return torch.from_numpy(m.reshape(1, 1, 1, width, 1)).byte().to(device)


def make_kspace(width, device, coils=15, height=640):
    return torch.randn(1, coils, height, width, 2, device=device) * 1e-5


def test_shapes_and_routing(device):
    model = AccSDUM(**TINY, checkpoint_mode="off").to(device).eval()
    fired = []
    model.sdum4.register_forward_hook(lambda m, i, o: fired.append(4))
    model.sdum8.register_forward_hook(lambda m, i, o: fired.append(8))

    for width in (320, 396, 480):
        for stride, expect in ((4, 4), (8, 8)):
            fired.clear()
            with torch.no_grad():
                out = model(make_kspace(width, device), make_mask(width, stride, device))
            assert out.shape == (1, 384, 384), out.shape
            assert torch.isfinite(out).all(), f"non-finite output (W={width}, acc{expect})"
            assert fired == [expect], f"W={width} stride {stride}: routed {fired}"
    print("shapes/finiteness/routing OK for W in {320, 396, 480}, acc4 and acc8")

    mixed = torch.cat([make_mask(320, 4, device), make_mask(320, 8, device)], dim=0)
    try:
        classify_acc(mixed)
        raise AssertionError("mixed batch did not raise")
    except ValueError:
        print("mixed-batch ValueError OK")


def test_dc_sanity(device):
    model = AccSDUM(**TINY, checkpoint_mode="off").to(device).eval()
    cascade = model.sdum4.cascades[0]
    cascade.dc_weight_map.data.fill_(1.0)
    import fastmri

    x = torch.randn(1, 15, 640, 320, 2, device=device)
    ref = torch.randn(1, 15, 640, 320, 2, device=device)
    mask = make_mask(320, 4, device)
    out = cascade.soft_dc(x, ref, mask)
    k_out, k_ref = fastmri.fft2c(out), fastmri.fft2c(ref)
    m = mask.bool().expand_as(k_out)
    err = (k_out[m] - k_ref[m]).abs().max().item()
    assert err < 1e-4, f"hard-DC limit violated: max err {err}"
    print(f"DC sanity OK (weight=1 => sampled k-space == ref, max err {err:.2e})")


def test_checkpoint_modes(device):
    # smaller-than-real spatial size: grad plumbing doesn't need full 640xW
    width, height = 320, 448
    kspace = make_kspace(width, device, height=height)
    target = torch.rand(1, 384, 384, device=device)
    mask4 = make_mask(width, 4, device)
    loss_fn = SSIMLoss().to(device)
    maximum = torch.ones(1, device=device)

    torch.manual_seed(0)
    ref_state = AccSDUM(**TINY, checkpoint_mode="off").state_dict()

    losses = {}
    for mode in ("off", "reentrant", "nonreentrant"):
        model = AccSDUM(**TINY, checkpoint_mode=mode).to(device)
        model.load_state_dict(ref_state)
        model.train()
        model.zero_grad()
        out = model(kspace, mask4)
        loss = loss_fn(out, target, maximum)
        loss.backward()
        losses[mode] = loss.item()

        missing = [n for n, p in model.sdum4.named_parameters() if p.grad is None]
        assert not missing, f"mode {mode}: sdum4 params without grad: {missing[:5]}"
        idle = [n for n, p in model.sdum8.named_parameters() if p.grad is not None]
        assert not idle, f"mode {mode}: sdum8 (idle) params unexpectedly got grads: {idle[:5]}"
        print(f"checkpoint mode {mode}: loss {losses[mode]:.6f}, all routed grads present")
        del model, out, loss
        if device.type == "cuda":
            torch.cuda.empty_cache()

    spread = max(losses.values()) - min(losses.values())
    assert spread <= 1e-6, f"loss differs across checkpoint modes: {losses}"
    print(f"loss identical across checkpoint modes (spread {spread:.2e})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda:0")
    print(f"device: {device}")

    model = AccSDUM(**TINY, checkpoint_mode="off")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"tiny AccSDUM parameters: {n_params:,}")
    del model

    test_shapes_and_routing(device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    test_dc_sanity(device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    test_checkpoint_modes(device)

    if device.type == "cuda":
        print(f"max CUDA memory allocated: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
    print("ALL SDUM TESTS PASSED")


if __name__ == "__main__":
    main()
