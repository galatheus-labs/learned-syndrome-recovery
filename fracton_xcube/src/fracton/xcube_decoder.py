"""Learned decoder harness for the X-cube fracton code (code-capacity).

Decodes independent X (bit-flip) errors from the cube (Z-stabilizer) syndrome,
which is a natural L*L*L scalar field -- so a 3D residual CNN drops straight in,
the same architecture family used for the surface code but without a temporal
axis. The decoder predicts the k = 6L-3 logical-Z parities of the error (the
logical-X-error class); see xcube_code.logical_operators / sample_xerror_decoding_data.

Geometry (verified against logical_operators()): each logical_Z operator is a
STRAIGHT WRAPPING LINE of length L running along one of the three axes at a fixed
transverse (u, v) cell. It is *translation-invariant along its axis* and
*localized in the transverse plane*. The X-cube excitations themselves are
immobile 4-at-a-time fractons (a cube flip creates a cluster of point-like
fracton charges with no free 1D string connecting them), so matching-based
decoding (MWPM) does not straightforwardly apply -- there is no 1D worldline to
match. This is the structural reason a learned decoder has headroom here.

Why a spatial head, not global average pooling: the 6L-3 logical sectors live at
spatially-distinct transverse cells. AdaptiveAvgPool3d(1) collapses the whole
volume to a single vector, discarding *where* the error support sits, so it
cannot distinguish sectors that differ only by transverse position -- empirically
it only TIES the trivial baseline. The AxisLineonHead instead, for each axis d,
collapses the feature volume ALONG d (the operator's translation-invariant
direction) and runs a small circular-padded 2D conv tower over the transverse
torus, emitting one logit per transverse cell, then gathers the per-sector logit
at each line's (axis, transverse-(u,v)). This is necessary (removes the
translation-invariance-collapse failure mode) but, per arXiv:2409.04300, not
guaranteed sufficient.

Baselines: trivial "predict no logical flip" decoder (per-sector LER == the
marginal flip rate), and BP(+OSD-0) in xcube_baseline_bposd.py (a wrong-coset
diagnostic -- on the wrapping-line sector it is actually WORSE than trivial).

Run on a GPU via bin/gpurun; CPU-smoke-testable at small L. Example:
    python xcube_decoder.py --L 3 --p 0.05 --arch a1 --epochs 20 \
        --n-train 200000 --n-test 50000
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch
import torch.nn as nn

from xcube_code import make_xcube, logical_operators, sample_xerror_decoding_data


def _gn_groups(width: int) -> int:
    """Largest divisor of width that is <= 8 (so GroupNorm never crashes on
    ablation widths like 20 that are not divisible by 8). Falls back to 1."""
    for g in range(min(8, width), 0, -1):
        if width % g == 0:
            return g
    return 1


class ResBlock3d(nn.Module):
    def __init__(self, width):
        super().__init__()
        g = _gn_groups(width)
        self.c1 = nn.Conv3d(width, width, 3, padding=1, padding_mode="circular")
        self.n1 = nn.GroupNorm(g, width)
        self.c2 = nn.Conv3d(width, width, 3, padding=1, padding_mode="circular")
        self.n2 = nn.GroupNorm(g, width)
        self.act = nn.SiLU()

    def forward(self, x):
        h = self.act(self.n1(self.c1(x)))
        h = self.n2(self.c2(h))
        return self.act(x + h)


def build_lineon_index(code: dict, logical_Z: np.ndarray):
    """Construction-time geometry extraction + asserts for the AxisLineonHead.

    Returns, for each axis d in {0,1,2}:
      sectors_d[d] : LongTensor of sector indices (rows of logical_Z) on axis d
      uv_d[d]      : LongTensor (len, 2) of transverse (u, v) cells for those rows

    Convention (verified against make_xcube / sample_xerror_decoding_data):
      syn_grid is filled syn_grid[:, 0, x, y, z], so conv3d spatial dims are
      (X=dim2, Y=dim3, Z=dim4). A line along axis d is translation-invariant
      along spatial dim 2+d; its transverse plane is the OTHER two axes taken in
      increasing order. So for axis 0 the transverse (u,v)=(y,z); axis 1 (x,z);
      axis 2 (x,y). The head collapses dim 2+d and gathers logit_map[u, v].

    Asserts (logical_operators() is deterministic across calls -- verified -- so
    we freeze this basis and fail loudly if it ever changes):
      * each logical_Z row is a single axis-aligned line of weight L,
      * each row touches exactly one transverse (u,v) cell,
      * per-axis transverse (u,v) cells are unique (well-defined scatter).
    """
    L = round(code["N"] ** (1.0 / 3) / 3 ** (1.0 / 3))
    # robust L recovery: N = 3 L^3
    L = round((code["N"] / 3.0) ** (1.0 / 3))
    assert 3 * L ** 3 == code["N"], f"could not recover L from N={code['N']}"
    edge = code["edge_coords"]
    k = logical_Z.shape[0]
    assert k == 6 * L - 3, f"expected {6*L-3} logical_Z rows, got {k}"

    sectors_d = {0: [], 1: [], 2: []}
    uv_d = {0: [], 1: [], 2: []}
    for r in range(k):
        supp = np.nonzero(logical_Z[r])[0]
        assert len(supp) == L, (
            f"logical_Z row {r} has weight {len(supp)} != L={L}; "
            "basis is not a set of axis-aligned wrapping lines -- "
            "logical_operators() returned an unexpected basis")
        coords = [edge[int(q)] for q in supp]
        axes = set(c[3] for c in coords)
        assert len(axes) == 1, (
            f"logical_Z row {r} spans multiple edge axes {axes}; not a single line")
        d = coords[0][3]
        others = [a for a in range(3) if a != d]  # increasing transverse axes
        uvs = set((c[others[0]], c[others[1]]) for c in coords)
        assert len(uvs) == 1, (
            f"logical_Z row {r} touches multiple transverse cells {uvs}; "
            "not a straight line")
        alongs = set(c[d] for c in coords)
        assert alongs == set(range(L)), (
            f"logical_Z row {r} does not wrap fully along axis {d}: {sorted(alongs)}")
        uv = uvs.pop()
        sectors_d[d].append(r)
        uv_d[d].append(uv)

    for d in (0, 1, 2):
        uvs = uv_d[d]
        assert len(set(uvs)) == len(uvs), (
            f"axis {d} has duplicate transverse (u,v) cells {uvs}; "
            "out[:, sectors] scatter would be ill-defined")

    sectors_t = {d: torch.tensor(sectors_d[d], dtype=torch.long) for d in (0, 1, 2)}
    uv_t = {d: torch.tensor(uv_d[d], dtype=torch.long).reshape(-1, 2) for d in (0, 1, 2)}
    return sectors_t, uv_t, L


class AxisLineonHead(nn.Module):
    """Spatial-structure-preserving readout for axis-aligned wrapping-line logicals.

    For each axis d: collapse the feature volume along axis d (mean AND amax over
    that dim) -> (B, 2*width, A, B) transverse plane; run a weight-shared,
    circular-padded 2D conv tower -> one logit per transverse cell; gather the
    sector logits at each line's transverse (u,v). Weights are shared across
    transverse positions, so the parameter count is L-independent: it trains at
    L=3 on CPU and scales to L=6/8 on GPU with no new parameters.
    """

    def __init__(self, width, sectors_d, uv_d, k, hidden=None):
        super().__init__()
        hidden = hidden or width
        g = _gn_groups(hidden)
        self.k = k
        self.towers = nn.ModuleList()
        for _ in range(3):
            self.towers.append(nn.Sequential(
                nn.Conv2d(2 * width, hidden, 1),
                nn.GroupNorm(g, hidden),
                nn.SiLU(),
                nn.Conv2d(hidden, hidden, 3, padding=1, padding_mode="circular"),
                nn.GroupNorm(g, hidden),
                nn.SiLU(),
                nn.Conv2d(hidden, 1, 1),
            ))
        # register index tables as buffers so .to(device) moves them
        for d in (0, 1, 2):
            self.register_buffer(f"sectors_{d}", sectors_d[d], persistent=False)
            self.register_buffer(f"uv_{d}", uv_d[d], persistent=False)

    def forward(self, feat):
        # feat: (B, width, X=dim2, Y=dim3, Z=dim4)
        B = feat.shape[0]
        out = feat.new_zeros((B, self.k))
        for d in (0, 1, 2):
            ax = 2 + d  # spatial dim to collapse (line's translation-invariant axis)
            red = torch.cat([feat.mean(ax), feat.amax(ax)], dim=1)  # (B, 2*width, A, Bp)
            logit_map = self.towers[d](red).squeeze(1)              # (B, A, Bp)
            sectors = getattr(self, f"sectors_{d}")
            uv = getattr(self, f"uv_{d}")
            if sectors.numel() == 0:
                continue
            vals = logit_map[:, uv[:, 0], uv[:, 1]]                 # (B, len)
            out[:, sectors] = vals
        return out


class GlobalPoolHead(nn.Module):
    """The original AdaptiveAvgPool3d(1) head (kept as the ablation baseline)."""

    def __init__(self, width, k):
        super().__init__()
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1), nn.Flatten(),
            nn.Linear(width, width), nn.SiLU(), nn.Linear(width, k),
        )

    def forward(self, feat):
        return self.head(feat)


class CoordConvLineonHead(nn.Module):
    """Ablation variant: AxisLineonHead but the transverse tower is also fed
    normalized transverse coordinate channels (coord-conditioned). Reserved as
    an ablation column, NOT the headline head."""

    def __init__(self, width, sectors_d, uv_d, k, hidden=None):
        super().__init__()
        hidden = hidden or width
        g = _gn_groups(hidden)
        self.k = k
        self.towers = nn.ModuleList()
        for _ in range(3):
            self.towers.append(nn.Sequential(
                nn.Conv2d(2 * width + 2, hidden, 1),
                nn.GroupNorm(g, hidden),
                nn.SiLU(),
                nn.Conv2d(hidden, hidden, 3, padding=1, padding_mode="circular"),
                nn.GroupNorm(g, hidden),
                nn.SiLU(),
                nn.Conv2d(hidden, 1, 1),
            ))
        for d in (0, 1, 2):
            self.register_buffer(f"sectors_{d}", sectors_d[d], persistent=False)
            self.register_buffer(f"uv_{d}", uv_d[d], persistent=False)

    def _coord(self, A, Bp, device, dtype):
        u = torch.linspace(0, 1, A, device=device, dtype=dtype).view(1, 1, A, 1).expand(1, 1, A, Bp)
        v = torch.linspace(0, 1, Bp, device=device, dtype=dtype).view(1, 1, 1, Bp).expand(1, 1, A, Bp)
        return torch.cat([u, v], dim=1)

    def forward(self, feat):
        B = feat.shape[0]
        out = feat.new_zeros((B, self.k))
        for d in (0, 1, 2):
            ax = 2 + d
            red = torch.cat([feat.mean(ax), feat.amax(ax)], dim=1)
            A, Bp = red.shape[2], red.shape[3]
            coord = self._coord(A, Bp, red.device, red.dtype).expand(B, 2, A, Bp)
            red = torch.cat([red, coord], dim=1)
            logit_map = self.towers[d](red).squeeze(1)
            sectors = getattr(self, f"sectors_{d}")
            uv = getattr(self, f"uv_{d}")
            if sectors.numel() == 0:
                continue
            vals = logit_map[:, uv[:, 0], uv[:, 1]]
            out[:, sectors] = vals
        return out


class XCubeDecoder(nn.Module):
    """3D ResNet over the L^3 cube-syndrome -> k logical-Z parity logits.

    Circular padding matches the periodic (3-torus) boundary conditions of the code.
    The readout head is selectable via `arch`:
      'a1'         -> AxisLineonHead (spatial-structure-preserving, the contribution)
      'globalpool' -> GlobalPoolHead (the original tying baseline / ablation)
      'coordconv'  -> CoordConvLineonHead (coord-conditioned ablation column)
    """

    def __init__(self, code, logical_Z, width=64, depth=8, arch="a1"):
        super().__init__()
        self.arch = arch
        k = logical_Z.shape[0]
        self.k = k
        self.stem = nn.Conv3d(1, width, 3, padding=1, padding_mode="circular")
        self.blocks = nn.Sequential(*[ResBlock3d(width) for _ in range(depth)])
        if arch == "globalpool":
            self.head = GlobalPoolHead(width, k)
        elif arch == "a1":
            sectors_d, uv_d, _ = build_lineon_index(code, logical_Z)
            self.head = AxisLineonHead(width, sectors_d, uv_d, k)
        elif arch == "coordconv":
            sectors_d, uv_d, _ = build_lineon_index(code, logical_Z)
            self.head = CoordConvLineonHead(width, sectors_d, uv_d, k)
        else:
            raise ValueError(f"unknown arch {arch!r}")

    def forward(self, x):
        return self.head(self.blocks(self.stem(x)))


def evaluate(model, syn, labels, device, batch=4096, return_per_shot=False):
    """Per-sector LER + (optionally) per-shot correctness arrays for paired tests.

    Returns a dict with:
      per_sector_ler_mean : mean over the 6L-3 sectors of P(sector predicted wrong)
                            (DIAGNOSTIC, basis-relative to the frozen logical_Z basis)
      per_sector_ler_max  : worst sector (basis-fragile; do not headline)
      block_failure       : P(any of the 6L-3 sectors wrong) -- BASIS-INVARIANT, headline
    If return_per_shot: also
      per_shot_correct    : (n, k) bool, per-sector correctness
      block_correct       : (n,)  bool, all-sectors-correct per shot (for McNemar)
    """
    model.eval()
    n, k = labels.shape
    wrong = np.zeros(k)
    block_correct = np.zeros(n, dtype=bool)
    per_shot_correct = np.zeros((n, k), dtype=bool) if return_per_shot else None
    with torch.no_grad():
        for i in range(0, n, batch):
            xb = torch.from_numpy(syn[i:i + batch]).to(device)
            pred = (model(xb) > 0).cpu().numpy().astype(np.float32)
            corr = (pred == labels[i:i + batch])
            wrong += (~corr.astype(bool)).sum(0)
            block_correct[i:i + batch] = corr.all(1)
            if return_per_shot:
                per_shot_correct[i:i + batch] = corr.astype(bool)
    per_sector = wrong / n
    res = {
        "per_sector_ler_mean": float(per_sector.mean()),
        "per_sector_ler_max": float(per_sector.max()),
        "block_failure": float(1.0 - block_correct.mean()),
    }
    if return_per_shot:
        res["per_shot_correct"] = per_shot_correct
        res["block_correct"] = block_correct
    return res


def main():
    ap = argparse.ArgumentParser(description="X-cube fracton learned decoder (code-capacity)")
    ap.add_argument("--L", type=int, default=4)
    ap.add_argument("--p", type=float, default=0.02, help="X-error rate per qubit")
    ap.add_argument("--arch", type=str, default="a1",
                    choices=["a1", "globalpool", "coordconv"],
                    help="readout head (a1=AxisLineonHead, the spatial contribution)")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--n-train", type=int, default=200000)
    ap.add_argument("--n-val", type=int, default=20000)
    ap.add_argument("--n-test", type=int, default=50000)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=str, default=None, help="write metrics JSON here")
    ap.add_argument("--per-shot-out", type=str, default=None,
                    help="write per-shot (n,k) correctness arrays here (.npz) for paired tests")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    code = make_xcube(args.L)
    logical_Z = logical_operators(code)["logical_Z"]
    k = logical_Z.shape[0]
    assert k == 6 * args.L - 3, f"expected {6*args.L-3} logicals, got {k}"

    # disjoint sampler seeds: train=seed; val=seed*1000+500; test=seed*1000+800.
    # val is used ONLY for predeclared checkpoint selection (one epoch, one rule);
    # test is evaluated exactly once on the chosen epoch.
    syn_tr, y_tr = sample_xerror_decoding_data(args.L, args.p, args.n_train, args.seed, logical_Z)
    syn_va, y_va = sample_xerror_decoding_data(args.L, args.p, args.n_val, args.seed * 1000 + 500, logical_Z)
    syn_te, y_te = sample_xerror_decoding_data(args.L, args.p, args.n_test, args.seed * 1000 + 800, logical_Z)

    # trivial baseline: predict the most-likely fixed class per sector (no syndrome use)
    base_rate = y_tr.mean(0)
    base_per_sector = np.minimum(base_rate, 1 - base_rate)
    # block failure of trivial: predict the per-sector majority class on every shot
    triv_pred = (base_rate > 0.5).astype(np.float32)
    triv_block_correct = (y_te == triv_pred).all(1)
    baseline = {
        "per_sector_ler_mean": float(base_per_sector.mean()),
        "per_sector_ler_max": float(base_per_sector.max()),
        "block_failure": float(1.0 - triv_block_correct.mean()),
    }

    model = XCubeDecoder(code, logical_Z, args.width, args.depth, arch=args.arch).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    lossfn = nn.BCEWithLogitsLoss()
    nparams = sum(p.numel() for p in model.parameters())

    # predeclared checkpoint selection: epoch minimizing mean per-sector VAL LER.
    best_val = float("inf")
    best_state = None
    best_epoch = -1
    idx = np.arange(args.n_train)
    for ep in range(args.epochs):
        model.train()
        np.random.shuffle(idx)
        tot = 0.0
        for i in range(0, args.n_train, args.batch_size):
            b = idx[i:i + args.batch_size]
            xb = torch.from_numpy(syn_tr[b]).to(device)
            yb = torch.from_numpy(y_tr[b]).to(device)
            opt.zero_grad()
            loss = lossfn(model(xb), yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
        sched.step()
        vm = evaluate(model, syn_va, y_va, device)
        if vm["per_sector_ler_mean"] < best_val:
            best_val = vm["per_sector_ler_mean"]
            best_state = {k_: v.detach().cpu().clone() for k_, v in model.state_dict().items()}
            best_epoch = ep
        if ep == 0 or (ep + 1) % 5 == 0 or ep == args.epochs - 1:
            print(f"epoch {ep+1:>3}  loss {tot/args.n_train:.4f}  "
                  f"val per-sector LER {vm['per_sector_ler_mean']:.4f}  "
                  f"(trivial {baseline['per_sector_ler_mean']:.4f})")

    # evaluate the predeclared-selected epoch ONCE on test
    if best_state is not None:
        model.load_state_dict(best_state)
    final = evaluate(model, syn_te, y_te, device, return_per_shot=True)
    per_shot_correct = final.pop("per_shot_correct")
    block_correct = final.pop("block_correct")

    result = {
        "L": args.L, "p": args.p, "k": k, "params": nparams,
        "arch": args.arch, "width": args.width, "depth": args.depth,
        "epochs": args.epochs, "selected_epoch": best_epoch + 1,
        "n_train": args.n_train, "n_val": args.n_val, "n_test": args.n_test,
        "seed": args.seed,
        "decoder": final, "trivial_baseline": baseline,
        "improvement_vs_baseline_pct": (
            100.0 * (baseline["per_sector_ler_mean"] - final["per_sector_ler_mean"])
            / baseline["per_sector_ler_mean"] if baseline["per_sector_ler_mean"] > 0 else 0.0),
    }
    print("RESULT", json.dumps(result))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
    if args.per_shot_out:
        np.savez_compressed(
            args.per_shot_out,
            per_shot_correct=per_shot_correct,
            block_correct=block_correct,
            triv_block_correct=triv_block_correct,
            labels=y_te,
        )


if __name__ == "__main__":
    main()
