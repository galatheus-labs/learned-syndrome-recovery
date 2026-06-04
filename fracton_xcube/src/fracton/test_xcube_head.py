"""Unit tests for the AxisLineonHead spatial readout of the X-cube decoder.

The single most important test is the PLANTED-SIGNAL test: place a single 1 in
the syndrome grid at a known (x, y, z) and confirm that, with an identity-ish
backbone+tower, the correct sector logits (and only those) light up. This is the
test that would have immediately caught the reduce_axis=(4,3,2) transpose bug
(which reintroduced the pooling defect on 2 of 3 axes).

We also regression-test the GroupNorm divisibility guard across ablation widths.

Run: cd fracton_xcube/src/fracton && python3 test_xcube_head.py
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from xcube_code import make_xcube, logical_operators
from xcube_decoder import (
    AxisLineonHead, XCubeDecoder, build_lineon_index, _gn_groups,
)


def _planted_test(L):
    """Feed an IDENTITY backbone (so feat == one-hot syndrome in channel 0) and a
    hand-built sum-tower (logit_map = sum over collapsed axis of channel 0) into
    AxisLineonHead. A planted error at one edge flips its 2 incident cubes; we
    instead plant directly in the syndrome grid and check which sectors fire.

    Simplest decisive check: plant a single syndrome bit at (x,y,z). After
    reduction along each axis, the transverse cell that contains (x,y,z) on that
    axis carries the only nonzero mass. The gathered sector for that (axis, uv)
    is the one whose transverse cell matches. We assert the lit set equals
    EXACTLY the set of logical lines whose transverse cell matches the planted
    cell -- axis-correct iff reduce_axis and the (u,v) gather convention agree.
    The transposed (4,3,2) bug fires the wrong cells on 2 of 3 axes and this
    assertion catches it immediately.

    Note: the 6L-3 logical basis is NOT a full 3L^2 grid of lines, so a given
    cell may have 0..3 lines through it; `expected` is computed from the basis
    rather than hard-coded to 3.
    """
    code = make_xcube(L)
    lz = logical_operators(code)["logical_Z"]
    k = lz.shape[0]
    sectors_d, uv_d, _ = build_lineon_index(code, lz)

    width = 1
    head = AxisLineonHead(width, sectors_d, uv_d, k, hidden=1)
    # Build a deterministic "sum along axis, read channel 0" tower per axis:
    #   red = cat([mean(ax), amax(ax)], dim=1) has 2 channels here (width=1).
    #   We want logit_map = amax channel (channel index 1) so any planted 1 in the
    #   collapsed column yields exactly 1.0 at its transverse cell, 0 elsewhere.
    with torch.no_grad():
        for d in range(3):
            tower = head.towers[d]
            # tower[0]: Conv2d(2*width=2, hidden=1, 1)  -> pick channel 1 (amax)
            tower[0].weight.zero_(); tower[0].weight[0, 1, 0, 0] = 1.0
            tower[0].bias.zero_()
            # tower[1]: GroupNorm -> make it identity (weight=1, bias=0); with a
            # single value GN would normalize to ~0, so bypass by setting the conv
            # AFTER GN (tower[3]) to read a fresh path. Simpler: replace GN/SiLU
            # with identity to keep the planted signal linear.
            tower[1] = nn.Identity()
            tower[2] = nn.Identity()
            # tower[3]: Conv2d(hidden=1, hidden=1, 3, circular) -> identity kernel
            tower[3].weight.zero_(); tower[3].weight[0, 0, 1, 1] = 1.0
            tower[3].bias.zero_()
            tower[4] = nn.Identity()
            tower[5] = nn.Identity()
            # tower[6]: Conv2d(hidden=1, 1, 1) -> identity, bias -0.5 so the planted
            # cell (1.0 -> +0.5) is positive and empty cells (0 -> -0.5) are negative.
            tower[6].weight.zero_(); tower[6].weight[0, 0, 0, 0] = 1.0
            tower[6].bias.fill_(-0.5)

    # plant a single syndrome bit at (x,y,z)
    xyz = (1 % L, 2 % L, 0 % L)
    feat = torch.zeros(1, width, L, L, L)
    feat[0, 0, xyz[0], xyz[1], xyz[2]] = 1.0
    head.eval()
    with torch.no_grad():
        out = head(feat).numpy()[0]  # (k,)

    # which sectors SHOULD be positive: the three lines through (x,y,z), one per axis
    edge = code["edge_coords"]
    expected = set()
    for r in range(k):
        supp = np.nonzero(lz[r])[0]
        coords = [edge[int(qi)] for qi in supp]
        d = coords[0][3]
        others = [a for a in range(3) if a != d]
        uv = (coords[0][others[0]], coords[0][others[1]])
        plant_uv = (xyz[others[0]], xyz[others[1]])
        if uv == plant_uv:
            expected.add(r)

    pos = set(np.nonzero(out > 0)[0].tolist())
    assert pos == expected, (
        f"L={L}: planted-signal lit sectors {sorted(pos)} != expected "
        f"{sorted(expected)} (out={np.round(out,3)}) -- reduce_axis / (u,v) "
        "gather convention mismatch (this is the (4,3,2) transpose bug signature)")
    assert len(expected) >= 1, f"L={L}: planted cell hits no logical line; pick another"
    print(f"  [planted-signal] L={L}: OK -- sectors {sorted(expected)} lit at "
          f"cell {xyz}, {k - len(expected)} others off")


def _transpose_negative_control(L):
    """Sanity: if we deliberately gather with a TRANSPOSED (u,v) convention, the
    lit set must DIFFER from the correct set on at least one axis. Proves the
    planted-signal test actually discriminates the transpose bug (rather than
    passing vacuously)."""
    code = make_xcube(L)
    lz = logical_operators(code)["logical_Z"]
    k = lz.shape[0]
    sectors_d, uv_d, _ = build_lineon_index(code, lz)
    # swap the (u,v) order in every gather -> the (4,3,2)-style defect
    uv_bad = {d: torch.flip(uv_d[d], dims=[1]) for d in (0, 1, 2)}
    head = AxisLineonHead(1, sectors_d, uv_bad, k, hidden=1)
    with torch.no_grad():
        for d in range(3):
            t = head.towers[d]
            t[0].weight.zero_(); t[0].weight[0, 1, 0, 0] = 1.0; t[0].bias.zero_()
            t[1] = nn.Identity(); t[2] = nn.Identity()
            t[3].weight.zero_(); t[3].weight[0, 0, 1, 1] = 1.0; t[3].bias.zero_()
            t[4] = nn.Identity(); t[5] = nn.Identity()
            t[6].weight.zero_(); t[6].weight[0, 0, 0, 0] = 1.0; t[6].bias.fill_(-0.5)
    xyz = (1 % L, 2 % L, 0 % L)
    feat = torch.zeros(1, 1, L, L, L); feat[0, 0, xyz[0], xyz[1], xyz[2]] = 1.0
    head.eval()
    with torch.no_grad():
        out_bad = set(np.nonzero(head(feat).numpy()[0] > 0)[0].tolist())
    edge = code["edge_coords"]
    correct = set()
    for r in range(k):
        supp = np.nonzero(lz[r])[0]
        coords = [edge[int(qi)] for qi in supp]
        d = coords[0][3]
        others = [a for a in range(3) if a != d]
        if (coords[0][others[0]], coords[0][others[1]]) == (xyz[others[0]], xyz[others[1]]):
            correct.add(r)
    assert out_bad != correct, (
        f"L={L}: transposed gather gave the SAME lit set as correct -- the "
        "planted-signal test is not discriminating the transpose bug")


def test_planted_signal():
    for L in (3, 4):
        _planted_test(L)
        _transpose_negative_control(L)


def test_groupnorm_divisibility():
    """Ablation widths not divisible by 8 must not crash GroupNorm anywhere."""
    for w in (20, 17, 13, 64, 96):
        assert w % _gn_groups(w) == 0, f"_gn_groups({w}) not a divisor"
    L = 3
    code = make_xcube(L)
    lz = logical_operators(code)["logical_Z"]
    for w in (20, 17, 13):
        for arch in ("a1", "globalpool", "coordconv"):
            m = XCubeDecoder(code, lz, width=w, depth=2, arch=arch)
            x = torch.zeros(2, 1, L, L, L)
            y = m(x)
            assert y.shape == (2, lz.shape[0]), f"w={w} arch={arch} bad out {y.shape}"
    print("  [groupnorm-divisibility] OK -- widths {20,17,13} x {a1,globalpool,coordconv}")


def test_construction_asserts():
    """build_lineon_index must accept the real basis and recover L."""
    for L in (2, 3, 4):
        code = make_xcube(L)
        lz = logical_operators(code)["logical_Z"]
        sectors_d, uv_d, Lr = build_lineon_index(code, lz)
        assert Lr == L
        total = sum(sectors_d[d].numel() for d in (0, 1, 2))
        assert total == 6 * L - 3, f"L={L}: indexed {total} sectors != {6*L-3}"
    print("  [construction-asserts] OK -- L in {2,3,4} basis accepted, sectors complete")


def test_forward_shapes():
    """Shape/plumbing test at L=2 (zero-headroom, never a performance test)."""
    L = 2
    code = make_xcube(L)
    lz = logical_operators(code)["logical_Z"]
    for arch in ("a1", "globalpool", "coordconv"):
        m = XCubeDecoder(code, lz, width=8, depth=2, arch=arch)
        out = m(torch.zeros(3, 1, L, L, L))
        assert out.shape == (3, 6 * L - 3)
    print("  [forward-shapes] OK -- L=2 plumbing, all archs")


if __name__ == "__main__":
    print("test_xcube_head:")
    test_forward_shapes()
    test_construction_asserts()
    test_groupnorm_divisibility()
    test_planted_signal()
    print("ALL HEAD TESTS PASSED")
