"""X-cube fracton code: construction, verification, and code-capacity syndromes.

This is the verified foundation for a learned decoder for a *fracton* code ---
the regime where matching-based decoders (MWPM) do not straightforwardly apply
because excitations have restricted mobility (sub-dimensional / immobile), so a
learned decoder has structural headroom rather than a marginal edge. It connects
directly to the gauge paper: the classical axial-parity codec is a discrete
scalar-charge fracton theory; the X-cube model is its quantum stabilizer cousin.

Lattice: periodic L^3 cubic lattice, qubits on the 3*L^3 edges. An edge is
(x, y, z, d) with d in {0:x, 1:y, 2:z}.

Stabilizers (standard X-cube presentation, qubits on links):
  - X-type "cross" operators: per vertex, the 4-edge planar cross in each of the
    xy, yz, xz planes. Only 2 are independent per vertex (xy * yz = xz), so we
    emit xy and yz.
  - Z-type "cube" operators: per elementary cube, the product over its 12 edges.

Verification (run this file): the code is a valid CSS code (H_X H_Z^T = 0 over
GF(2)) and the logical-qubit count k = N - rank(H_X) - rank(H_Z) equals the known
X-cube ground-state degeneracy on the 3-torus, log2(GSD) = 6L - 3. Confirmed for
L in {2,3,4,5}: k = {9, 15, 21, 27}.

Decoder note: the Z-stabilizer (cube) syndrome has exactly L^3 bits, one per cube
center, so it is naturally an L*L*L scalar field --- a direct fit for the same
3D-convolutional decoder used for the surface code, with no temporal axis.
"""

from __future__ import annotations

import numpy as np


def make_xcube(L: int):
    """Build the X-cube code on a periodic L^3 lattice.

    Returns a dict with:
      N           : number of qubits (= 3*L**3)
      HX, HZ      : (rows, N) uint8 GF(2) check matrices (X-cross, Z-cube)
      edge_coords : list of (x,y,z,d) per qubit index
      cube_coords : list of (x,y,z) per HZ row (cube center) -- the L^3 grid
    """
    def q(x, y, z, d):
        return ((((x % L) * L + (y % L)) * L + (z % L)) * 3) + d

    N = 3 * L ** 3
    edge_coords = [None] * N
    for x in range(L):
        for y in range(L):
            for z in range(L):
                for d in range(3):
                    edge_coords[q(x, y, z, d)] = (x, y, z, d)

    hx_supports, hz_supports, cube_coords = [], [], []
    for x in range(L):
        for y in range(L):
            for z in range(L):
                # X-type planar crosses at vertex (x,y,z): xy and yz (xz = xy*yz)
                xy = [q(x, y, z, 0), q(x - 1, y, z, 0), q(x, y, z, 1), q(x, y - 1, z, 1)]
                yz = [q(x, y, z, 1), q(x, y - 1, z, 1), q(x, y, z, 2), q(x, y, z - 1, 2)]
                hx_supports.append(xy)
                hx_supports.append(yz)
                # Z-type 12-edge cube operator at cube with corner (x,y,z)
                cube = [
                    q(x, y, z, 0), q(x, y + 1, z, 0), q(x, y, z + 1, 0), q(x, y + 1, z + 1, 0),
                    q(x, y, z, 1), q(x + 1, y, z, 1), q(x, y, z + 1, 1), q(x + 1, y, z + 1, 1),
                    q(x, y, z, 2), q(x + 1, y, z, 2), q(x, y + 1, z, 2), q(x + 1, y + 1, z, 2),
                ]
                hz_supports.append(cube)
                cube_coords.append((x, y, z))

    def to_matrix(supports):
        M = np.zeros((len(supports), N), dtype=np.uint8)
        for i, supp in enumerate(supports):
            M[i, supp] = 1
        return M

    return {
        "N": N,
        "HX": to_matrix(hx_supports),
        "HZ": to_matrix(hz_supports),
        "edge_coords": edge_coords,
        "cube_coords": cube_coords,
    }


def _rank2(M: np.ndarray) -> int:
    """GF(2) rank via bitmask Gaussian elimination."""
    ncols = M.shape[1]
    rows = []
    for r in M:
        m = 0
        for j in np.nonzero(r)[0]:
            m |= (1 << int(j))
        rows.append(m)
    rank = 0
    for col in range(ncols):
        bit = 1 << col
        piv = -1
        for i in range(rank, len(rows)):
            if rows[i] & bit:
                piv = i
                break
        if piv < 0:
            continue
        rows[rank], rows[piv] = rows[piv], rows[rank]
        for i in range(len(rows)):
            if i != rank and (rows[i] & bit):
                rows[i] ^= rows[rank]
        rank += 1
        if rank == len(rows):
            break
    return rank


def verify(L: int) -> dict:
    """Check CSS commutation and logical-qubit count against 6L-3."""
    code = make_xcube(L)
    HX, HZ, N = code["HX"], code["HZ"], code["N"]
    css_ok = not np.any((HX @ HZ.T) % 2)
    rkx, rkz = _rank2(HX), _rank2(HZ)
    k = N - rkx - rkz
    expected = 6 * L - 3
    return {"L": L, "N": N, "rank_HX": rkx, "rank_HZ": rkz,
            "k": k, "expected_k": expected, "css_ok": bool(css_ok),
            "ok": bool(css_ok and k == expected)}


def _gf2_rref(M: np.ndarray):
    """Row-reduce M over GF(2). Returns (rref, pivot_columns)."""
    M = (M.copy().astype(np.uint8)) % 2
    rows, cols = M.shape
    pivots, r = [], 0
    for c in range(cols):
        piv = -1
        for i in range(r, rows):
            if M[i, c]:
                piv = i
                break
        if piv < 0:
            continue
        M[[r, piv]] = M[[piv, r]]
        for i in range(rows):
            if i != r and M[i, c]:
                M[i] ^= M[r]
        pivots.append(c)
        r += 1
        if r == rows:
            break
    return M, pivots


def _gf2_nullspace(M: np.ndarray) -> np.ndarray:
    """Basis (as rows) of {v : M v = 0 mod 2}."""
    cols = M.shape[1]
    R, pivots = _gf2_rref(M)
    pivot_set = set(pivots)
    free = [c for c in range(cols) if c not in pivot_set]
    basis = []
    for f in free:
        v = np.zeros(cols, dtype=np.uint8)
        v[f] = 1
        for ri, pc in enumerate(pivots):
            if R[ri, f]:
                v[pc] = 1
        basis.append(v)
    return np.array(basis, dtype=np.uint8) if basis else np.zeros((0, cols), np.uint8)


def _independent_mod(candidates: np.ndarray, fixed: np.ndarray):
    """Return candidate rows that are independent modulo rowspace(fixed), over GF(2)."""
    def to_mask(v):
        m = 0
        for j in np.nonzero(v)[0]:
            m |= (1 << int(j))
        return m

    basis = {}
    for row in fixed:
        v = to_mask(row)
        while v:
            p = v & (-v)
            if p in basis:
                v ^= basis[p]
            else:
                basis[p] = v
                break
    keep = []
    for idx, row in enumerate(candidates):
        v = to_mask(row)
        while v:
            p = v & (-v)
            if p in basis:
                v ^= basis[p]
            else:
                basis[p] = v
                keep.append(idx)
                break
    return candidates[keep] if keep else np.zeros((0, candidates.shape[1]), np.uint8)


def logical_operators(code: dict) -> dict:
    """Extract logical operators of a CSS code.

    logical_Z = ker(HX) modulo rowspace(HZ);  logical_X = ker(HZ) modulo rowspace(HX).
    Each family has k = N - rank(HX) - rank(HZ) = 6L-3 independent representatives.
    For decoding X-errors via the cube (Z) syndrome, the per-error label is the
    vector of logical_Z parities: label_j = logical_Z[j] . error  (mod 2).
    """
    HX, HZ = code["HX"], code["HZ"]
    logical_Z = _independent_mod(_gf2_nullspace(HX), HZ)  # Z-type: commute with X-stabs
    logical_X = _independent_mod(_gf2_nullspace(HZ), HX)  # X-type: commute with Z-stabs
    return {"logical_Z": logical_Z, "logical_X": logical_X}


def sample_xerror_syndromes(L: int, p: float, n: int, seed: int):
    """Code-capacity X-error channel decoded by the Z (cube) stabilizers.

    Independent bit-flip (X) errors at rate p per qubit; the cube operators
    detect them. Returns:
      errors    : (n, N) uint8
      syndromes : (n, L, L, L) uint8 -- cube syndrome as a 3D field (decoder input)
    The matching logical-operator labels (k = 6L-3 logical X operators) are the
    next piece to add; computing them is logical_X = ker(HZ) / rowspace(HX).
    """
    code = make_xcube(L)
    HZ = code["HZ"]
    rng = np.random.default_rng(seed)
    errors = (rng.random((n, code["N"])) < p).astype(np.uint8)
    syn_flat = (errors @ HZ.T) % 2  # (n, L^3)
    syn_grid = np.zeros((n, L, L, L), dtype=np.uint8)
    for row, (x, y, z) in enumerate(code["cube_coords"]):
        syn_grid[:, x, y, z] = syn_flat[:, row]
    return errors, syn_grid


def sample_xerror_decoding_data(L: int, p: float, n: int, seed: int, logical_Z=None):
    """Code-capacity X-noise decoding data.

    Returns:
      syn_grid : (n, 1, L, L, L) float32 -- cube syndrome (decoder input)
      labels   : (n, k) float32          -- logical-Z parities of the error (k=6L-3),
                                            i.e. the logical-X-error class to predict.
    """
    code = make_xcube(L)
    if logical_Z is None:
        logical_Z = logical_operators(code)["logical_Z"]
    rng = np.random.default_rng(seed)
    errors = (rng.random((n, code["N"])) < p).astype(np.uint8)
    syn_flat = (errors @ code["HZ"].T) % 2
    syn_grid = np.zeros((n, 1, L, L, L), dtype=np.float32)
    for row, (x, y, z) in enumerate(code["cube_coords"]):
        syn_grid[:, 0, x, y, z] = syn_flat[:, row]
    labels = ((errors @ logical_Z.T) % 2).astype(np.float32)
    return syn_grid, labels


if __name__ == "__main__":
    print(f"{'L':>2} {'N':>5} {'rk(HX)':>7} {'rk(HZ)':>7} {'k':>4} {'6L-3':>5} "
          f"{'CSS?':>6} {'#logZ':>6} {'logZ_ok?':>8} {'ok?':>5}")
    for L in (2, 3, 4, 5):
        v = verify(L)
        code = make_xcube(L)
        logs = logical_operators(code)
        lz = logs["logical_Z"]
        # logical_Z must commute with all X-stabilisers (HX . lz^T == 0) and number 6L-3
        logz_ok = (lz.shape[0] == v["expected_k"]) and (not np.any((code["HX"] @ lz.T) % 2))
        print(f"{v['L']:>2} {v['N']:>5} {v['rank_HX']:>7} {v['rank_HZ']:>7} "
              f"{v['k']:>4} {v['expected_k']:>5} {str(v['css_ok']):>6} "
              f"{lz.shape[0]:>6} {str(logz_ok):>8} {str(v['ok'] and logz_ok):>5}")
