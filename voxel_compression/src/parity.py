import hashlib
import json
from dataclasses import dataclass, field, asdict

import torch
from tqdm import tqdm


@dataclass
class ConstraintConfig:
    axis_parity: bool = True
    diagonal_parity: bool = False
    subcube_2x2x2: bool = False
    plane_parity: bool = False
    global_parity: bool = True
    random_rows: int = 0  # number of random binary projection rows


CONSTRAINT_CACHE_VERSION = "v2_true_axes_preserve_rows"
LEGACY_PARITY_CACHE_VERSION = "v2_true_axes"


# ------------------------------------------------------------------
# Individual constraint builders
# ------------------------------------------------------------------

def _build_axis_parity(n: int) -> torch.Tensor:
    """Axis-aligned line parity checks (without global row).

    Returns shape ``(3n(n-1), n^3)`` with ``n`` nonzeros per row.
    """
    rows = []
    # rows along x, skip last y plane to remove redundancy
    for y in range(n - 1):
        for z in range(n):
            row = torch.zeros(n ** 3, dtype=torch.uint8)
            for x in range(n):
                row[x + y * n + z * n * n] = 1
            rows.append(row)
    # columns along y, skip last x plane
    for x in range(n - 1):
        for z in range(n):
            row = torch.zeros(n ** 3, dtype=torch.uint8)
            for y in range(n):
                row[x + y * n + z * n * n] = 1
            rows.append(row)
    # depth along z, skip last y plane
    for y in range(n - 1):
        for x in range(n):
            row = torch.zeros(n ** 3, dtype=torch.uint8)
            for z in range(n):
                row[x + y * n + z * n * n] = 1
            rows.append(row)
    return torch.stack(rows)


def _build_diagonal_parity(n: int) -> torch.Tensor:
    """Main and anti-diagonal parity per 2D slice, for each axis family.

    For each of the 3 axis families and each of the ``n`` slices, two
    diagonal checks are produced: one along the main diagonal and one
    along the anti-diagonal.  Each row has ``n`` nonzeros.

    Returns shape ``(6n, n^3)``.
    """
    rows = []
    for z in range(n):
        # XY-plane main diagonal
        row = torch.zeros(n ** 3, dtype=torch.uint8)
        for i in range(n):
            row[i + i * n + z * n * n] = 1
        rows.append(row)
        # XY-plane anti-diagonal
        row = torch.zeros(n ** 3, dtype=torch.uint8)
        for i in range(n):
            row[(n - 1 - i) + i * n + z * n * n] = 1
        rows.append(row)
    for y in range(n):
        # XZ-plane main diagonal
        row = torch.zeros(n ** 3, dtype=torch.uint8)
        for i in range(n):
            row[i + y * n + i * n * n] = 1
        rows.append(row)
        # XZ-plane anti-diagonal
        row = torch.zeros(n ** 3, dtype=torch.uint8)
        for i in range(n):
            row[(n - 1 - i) + y * n + i * n * n] = 1
        rows.append(row)
    for x in range(n):
        # YZ-plane main diagonal
        row = torch.zeros(n ** 3, dtype=torch.uint8)
        for i in range(n):
            row[x + i * n + i * n * n] = 1
        rows.append(row)
        # YZ-plane anti-diagonal
        row = torch.zeros(n ** 3, dtype=torch.uint8)
        for i in range(n):
            row[x + (n - 1 - i) * n + i * n * n] = 1
        rows.append(row)
    return torch.stack(rows)


def _build_subcube_2x2x2(n: int) -> torch.Tensor:
    """Parity of each overlapping 2x2x2 block.

    Returns shape ``((n-1)^3, n^3)`` with 8 nonzeros per row.
    """
    rows = []
    for z in range(n - 1):
        for y in range(n - 1):
            for x in range(n - 1):
                row = torch.zeros(n ** 3, dtype=torch.uint8)
                for dz in range(2):
                    for dy in range(2):
                        for dx in range(2):
                            idx = (x + dx) + (y + dy) * n + (z + dz) * n * n
                            row[idx] = 1
                rows.append(row)
    return torch.stack(rows)


def _build_plane_parity(n: int) -> torch.Tensor:
    """Parity of each full plane (3 axis families, ``n`` planes each).

    Returns shape ``(3n, n^3)`` with ``n^2`` nonzeros per row.
    """
    rows = []
    # XY planes (fixed z)
    for z in range(n):
        row = torch.zeros(n ** 3, dtype=torch.uint8)
        for y in range(n):
            for x in range(n):
                row[x + y * n + z * n * n] = 1
        rows.append(row)
    # XZ planes (fixed y)
    for y in range(n):
        row = torch.zeros(n ** 3, dtype=torch.uint8)
        for z in range(n):
            for x in range(n):
                row[x + y * n + z * n * n] = 1
        rows.append(row)
    # YZ planes (fixed x)
    for x in range(n):
        row = torch.zeros(n ** 3, dtype=torch.uint8)
        for z in range(n):
            for y in range(n):
                row[x + y * n + z * n * n] = 1
        rows.append(row)
    return torch.stack(rows)


def _build_global_parity(n: int) -> torch.Tensor:
    """Single all-ones row.  Returns shape ``(1, n^3)``."""
    return torch.ones((1, n ** 3), dtype=torch.uint8)


def _gf2_row_reduce(A: torch.Tensor) -> torch.Tensor:
    """Remove linearly dependent rows from ``A`` over GF(2).

    Returns original rows that form an independent basis for the row space.
    The elimination scratch matrix is transformed internally, but returned
    checks preserve the locality/weight of the input constraint rows.
    """
    original = A.clone()
    work = A.clone()
    m, n = work.shape
    row_ids = torch.arange(m)
    pivot_original_rows: list[int] = []
    col = 0
    for row_idx in range(m):
        # find next pivot column
        while col < n:
            # vectorized pivot search
            candidates = work[row_idx:, col].nonzero(as_tuple=False)
            if len(candidates) == 0:
                col += 1
                continue
            found = candidates[0].item() + row_idx
            # swap
            if found != row_idx:
                work[[row_idx, found]] = work[[found, row_idx]]
                row_ids[[row_idx, found]] = row_ids[[found, row_idx]]
            pivot_original_rows.append(int(row_ids[row_idx].item()))
            # vectorized elimination: XOR pivot row into all rows that have a 1 in this column
            mask = work[:, col].bool()
            mask[row_idx] = False
            if mask.any():
                work[mask] ^= work[row_idx]
            col += 1
            break
        else:
            break  # no more columns
    if not pivot_original_rows:
        return original[:0]
    return original[pivot_original_rows]


def _constraint_config_hash(n: int, cfg: ConstraintConfig) -> str:
    """Deterministic hash for caching constraint ops."""
    data = {"n": n, "version": CONSTRAINT_CACHE_VERSION, **asdict(cfg)}
    return hashlib.md5(json.dumps(data, sort_keys=True).encode()).hexdigest()[:12]


def build_constraint_matrix(n: int, cfg: ConstraintConfig | None = None) -> torch.Tensor:
    """Build composite constraint matrix from config, then row-reduce.

    If ``cfg`` is ``None``, uses the default ``ConstraintConfig()`` which
    gives axis parity + global parity (matching the legacy behaviour).
    """
    if cfg is None:
        cfg = ConstraintConfig()

    parts: list[torch.Tensor] = []
    if cfg.axis_parity:
        parts.append(_build_axis_parity(n))
    if cfg.diagonal_parity:
        parts.append(_build_diagonal_parity(n))
    if cfg.subcube_2x2x2:
        parts.append(_build_subcube_2x2x2(n))
    if cfg.plane_parity:
        parts.append(_build_plane_parity(n))
    if cfg.global_parity:
        parts.append(_build_global_parity(n))
    if cfg.random_rows > 0:
        # Generate random binary rows with fixed seed for reproducibility
        rng = torch.Generator()
        rng.manual_seed(12345)
        rand_part = (torch.rand(cfg.random_rows, n ** 3, generator=rng) > 0.5).to(torch.uint8)
        parts.append(rand_part)

    if not parts:
        return torch.zeros((0, n ** 3), dtype=torch.uint8)
    stacked = torch.cat(parts, dim=0)
    return _gf2_row_reduce(stacked)


def build_random_constraint_matrix(N: int, M: int, seed: int = 42,
                                    max_attempts: int = 20) -> torch.Tensor:
    """Build a random binary constraint matrix with exactly M independent rows.

    Generates random binary rows over N columns and row-reduces over GF(2).
    Retries with increasing oversampling until exactly M independent rows
    are obtained. Raises RuntimeError if it cannot achieve the target rank.

    This is used as a structural control: same rank/bit-budget as the
    algebraic parity matrix, but without spatial structure.
    """
    rng = torch.Generator()
    rng.manual_seed(seed)
    for attempt in range(max_attempts):
        # Oversample more aggressively with each attempt
        n_rows = int(M * (1.2 + 0.1 * attempt)) + 50
        H = (torch.rand(n_rows, N, generator=rng) > 0.5).to(torch.uint8)
        H_reduced = _gf2_row_reduce(H)
        if H_reduced.size(0) >= M:
            return H_reduced[:M]
    raise RuntimeError(
        f"Could not generate random constraint matrix with rank {M} "
        f"over {N} columns after {max_attempts} attempts "
        f"(best rank: {H_reduced.size(0)})"
    )


def build_parity_matrix(n: int) -> torch.Tensor:
    """Return parity-check matrix for an n^3 cube.

    The matrix has shape (m, n^3) where m = 3n^2 - 3n + 1.
    Each row represents the parity of a line of voxels or the global parity.
    """
    m = 3 * n * n - 3 * n + 1
    A = torch.zeros((m, n ** 3), dtype=torch.uint8)
    r = 0
    # rows along x, skip last y plane to remove redundancy
    for y in range(n - 1):
        for z in range(n):
            for x in range(n):
                idx = x + y * n + z * n * n
                A[r, idx] = 1
            r += 1
    # columns along y, skip last x plane
    for x in range(n - 1):
        for z in range(n):
            for y in range(n):
                idx = x + y * n + z * n * n
                A[r, idx] = 1
            r += 1
    # depth along z, skip last y plane
    for y in range(n - 1):
        for x in range(n):
            for z in range(n):
                idx = x + y * n + z * n * n
                A[r, idx] = 1
            r += 1
    # global parity line
    A[r].fill_(1)
    return A


def _gf2_gauss_jordan(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Solve A x = b over GF(2) with free variables set to zero."""
    A = A.clone()
    b = b.clone()
    m, n = A.shape
    row = 0
    pivot_cols = []
    for col in tqdm(range(n)):
        if row >= m:
            break
        # ``torch.argmax`` returns a scalar tensor, which would cause
        # ``pivot`` to be a tensor as well after the addition below.
        # We need a Python ``int`` for subsequent advanced indexing
        # operations on 1-dimensional tensors (``b[[row, pivot]]``).
        # Keeping it as a tensor leads to an "too many indices" error
        # because PyTorch interprets the mix of list and tensor indices
        # as multi-dimensional indexing.
        pivot = int(torch.argmax(A[row:, col]).item()) + row
        if A[pivot, col] == 0:
            continue
        if pivot != row:
            A[[row, pivot]] = A[[pivot, row]]
            b[[row, pivot]] = b[[pivot, row]]
        pivot_cols.append(col)
        for r2 in range(m):
            if r2 != row and A[r2, col]:
                A[r2] ^= A[row]
                b[r2] ^= b[row]
        row += 1
    x = torch.zeros(n, dtype=torch.uint8, device=b.device)
    for i, col in enumerate(pivot_cols):
        x[col] = b[i]
    return x


def project_to_parity(x: torch.Tensor, A: torch.Tensor, target: torch.Tensor | None = None) -> torch.Tensor:
    """Project ``x`` so that ``A @ x`` equals ``target`` over GF(2).

    Parameters
    ----------
    x : ``torch.Tensor``
        Batch of flattened cubes with shape ``(b, n^3)``.
    A : ``torch.Tensor``
        Parity matrix with shape ``(m, n^3)``.
    target : ``torch.Tensor`` or ``None``
        Desired parity syndrome for each sample with shape ``(b, m)``.  If
        ``None`` (default), the syndrome is assumed to be all zeros.
    """
    b, d = x.shape
    if target is None:
        target = torch.zeros(b, A.size(0), dtype=torch.uint8, device=x.device)

    out = x.clone().to(torch.uint8)
    for i in tqdm(range(b)):
        r = ((A.float() @ out[i].float()) % 2).to(torch.uint8) ^ target[i]
        # print how many bits are different
        print(torch.sum(r))
        if torch.any(r):
            delta = _gf2_gauss_jordan(A, r)
            out[i] ^= delta
    return out


def _gf2_null_space(A: torch.Tensor) -> torch.Tensor:
    """Return a basis for the null space of ``A`` over GF(2)."""
    A = A.clone()
    m, n = A.shape
    row = 0
    pivot_cols: list[int] = []
    for col in tqdm(range(n), desc="Creating null space"):
        if row >= m:
            break
        pivot = int(torch.argmax(A[row:, col]).item()) + row
        if A[pivot, col] == 0:
            continue
        if pivot != row:
            A[[row, pivot]] = A[[pivot, row]]
        pivot_cols.append(col)
        for r2 in range(m):
            if r2 != row and A[r2, col]:
                A[r2] ^= A[row]
        row += 1

    free_cols = [c for c in range(n) if c not in pivot_cols]
    basis = []
    for fc in tqdm(free_cols, desc="Creating null space basis"):
        vec = torch.zeros(n, dtype=torch.uint8)
        vec[fc] = 1
        for i, col in enumerate(pivot_cols):
            if A[i, fc]:
                vec[col] = 1
        basis.append(vec)
    if not basis:
        return torch.zeros(n, 0, dtype=torch.uint8)
    return torch.stack(basis, dim=1)


def _gf2_left_inverse(A: torch.Tensor) -> torch.Tensor:
    """Return a left inverse ``B`` such that ``A @ B == I`` over GF(2).
    This vectorized implementation performs Gauss–Jordan elimination once on an
    augmented matrix whose right-hand side is the identity matrix. It is
    drastically faster than solving *m* independent systems in a Python loop
    and leverages PyTorch's efficient tensor operations for parallelism.
    """
    # Work on clones so the original matrix is left intact.
    A = A.clone()
    m, n = A.shape

    # The right-hand side is the identity matrix; each column corresponds to one
    # unit vector e_i.
    B_rhs = torch.eye(m, dtype=torch.uint8, device=A.device)

    row = 0
    pivot_cols: list[int] = []
    for col in range(n):
        if row >= m:
            break
        # Find pivot row.
        pivot = int(torch.argmax(A[row:, col]).item()) + row
        if A[pivot, col] == 0:
            continue
        # Swap current row with the pivot row in both A and B_rhs.
        if pivot != row:
            A[[row, pivot]] = A[[pivot, row]]
            B_rhs[[row, pivot]] = B_rhs[[pivot, row]]
        pivot_cols.append(col)

        # Eliminate the current column from all other rows.
        for r2 in range(m):
            if r2 != row and A[r2, col]:
                A[r2] ^= A[row]
                B_rhs[r2] ^= B_rhs[row]
        row += 1

    # Reconstruct the solution matrix (n × m).
    left_inv = torch.zeros(n, m, dtype=torch.uint8, device=A.device)
    for i, col in enumerate(pivot_cols):
        left_inv[col] = B_rhs[i]

    return left_inv


def load_parity_ops(n: int):
    """Load or compute null space and left inverse for ``n``.

    Backward-compatible: uses the original ``build_parity_matrix`` (axis +
    global) so cached files remain valid.

    Returns ``(null_space, left_inverse, parity_matrix)``.
    """
    from pathlib import Path

    base = Path(__file__).resolve().parent
    ns_path = base / f"nullspace_{LEGACY_PARITY_CACHE_VERSION}_{n}.pt"
    li_path = base / f"leftinv_{LEGACY_PARITY_CACHE_VERSION}_{n}.pt"

    A = build_parity_matrix(n)

    if ns_path.exists() and li_path.exists():
        null = torch.load(ns_path)
        left = torch.load(li_path)
        return null, left, A

    null = _gf2_null_space(A)
    left = _gf2_left_inverse(A)
    torch.save(null, ns_path)
    torch.save(left, li_path)
    return null, left, A


def load_constraint_ops(n: int, cfg: ConstraintConfig | None = None):
    """Load or compute null space and left inverse for a constraint config.

    Returns ``(null_space, left_inverse, constraint_matrix)``.

    Always uses the row-reduced ``build_constraint_matrix`` for consistency
    with training.
    """
    from pathlib import Path

    if cfg is None:
        cfg = ConstraintConfig()

    A = build_constraint_matrix(n, cfg)

    h = _constraint_config_hash(n, cfg)
    base = Path(__file__).resolve().parent
    ns_path = base / f"nullspace_{h}.pt"
    li_path = base / f"leftinv_{h}.pt"

    if ns_path.exists() and li_path.exists():
        null = torch.load(ns_path)
        left = torch.load(li_path)
        return null, left, A

    null = _gf2_null_space(A)
    left = _gf2_left_inverse(A)
    torch.save(null, ns_path)
    torch.save(left, li_path)
    return null, left, A


def gf2_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Matrix-multiply `A` and `B` over GF(2).

    Works for any shapes that `torch.matmul` accepts.
    Both inputs must be uint8 / bool and contain only 0/1.
    """
    if A.device.type == "cpu":
        return (torch.matmul(A.to(torch.int64), B.to(torch.int64)) & 1).to(torch.uint8)
    return (torch.matmul(A.float(), B.float()).to(torch.int64) & 1).to(torch.uint8)

def make_canonical_base(
    syndrome_uint8: torch.Tensor,   # (B, M) uint8
    parity: torch.Tensor,           # (M, N) uint8
    left_inv: torch.Tensor,         # (N, M) uint8
    device: torch.device,
) -> torch.Tensor:
    """Build the deterministic canonical base solution from a syndrome.

    Returns x_base such that parity @ x_base == syndrome over GF(2),
    starting from zeros so the result is fully deterministic.
    """
    B, M = syndrome_uint8.shape
    N = parity.size(1)
    zeros = torch.zeros(B, N, dtype=torch.uint8, device=device)
    x_base, _ = project_with_left_inverse(zeros, parity, left_inv, syndrome_uint8)
    return x_base


def project_with_left_inverse(
    x: torch.Tensor, A: torch.Tensor, left_inv: torch.Tensor, target: torch.Tensor | None = None
) -> torch.Tensor:
    """Project ``x`` to satisfy the parity constraint using a precomputed left inverse."""
    b, _ = x.shape
    if target is None:
        target = torch.zeros(b, A.size(0), dtype=torch.uint8, device=x.device)
    # r = ((x.float() @ A.t().float()) % 2).to(torch.uint8) ^ target
    # delta = (r.to(torch.float) @ left_inv.t().to(torch.float)) % 2
    r     = gf2_matmul(A, x.t()).t() ^ target        # (b, m)
    delta = gf2_matmul(r, left_inv.t())              # (b, n)
    x = (x.int() ^ delta.int()).to(torch.uint8)
    return x, r
