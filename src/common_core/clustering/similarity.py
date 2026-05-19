"""Composite-Dice similarity on the two binary signature matrices.

Mirrors the portal's ``services.clustering.bgc_similarity.compute_composite_similarity``
but kept pure-numpy/scipy and Django-free. The math is:

    sim = w_d · Dice(M_domains) + w_a · Dice(M_pairs)
        Dice(M) = 2 · (M @ M.T) / (rowsum_i + rowsum_j)

Diagonal is zeroed and the result is symmetrised. The matmul is
single-threaded; row-chunked parallelism is opt-in via ``matmul_workers``
and only really pays off at N ≥ ~100k.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import scipy.sparse as sp

log = logging.getLogger(__name__)


def dice_similarity(M: "sp.csr_matrix", *, matmul_workers: int = 1) -> "sp.csr_matrix":
    """Sørensen–Dice on a binary feature matrix. CPU."""
    import numpy as np
    import scipy.sparse as sp

    if M.shape[0] == 0:
        return sp.csr_matrix((0, 0), dtype=np.float32)

    Mf = M.astype(np.float32)
    if matmul_workers <= 1:
        inter = (Mf @ Mf.T).tocoo(copy=False)
    else:
        inter = _matmul_chunked(Mf, matmul_workers).tocoo(copy=False)

    if inter.nnz == 0:
        return sp.csr_matrix(inter.shape, dtype=np.float32)

    sizes = np.asarray(M.sum(axis=1), dtype=np.float32).ravel()
    denom = sizes[inter.row] + sizes[inter.col]
    safe = denom > 0
    values = np.zeros_like(inter.data, dtype=np.float32)
    values[safe] = 2.0 * inter.data[safe] / denom[safe]

    sim = sp.csr_matrix(
        (values, (inter.row, inter.col)), shape=inter.shape, dtype=np.float32,
    )
    sim.eliminate_zeros()
    return sim


def compute_composite_similarity(
    M_domains: "sp.csr_matrix",
    M_pairs: "sp.csr_matrix",
    *,
    weights: tuple[float, float] = (0.5, 0.5),
    prune_below: float = 0.05,
    matmul_workers: int = 1,
) -> "sp.csr_matrix":
    """Weighted-mean Sørensen-Dice over both signature matrices.

    Identical numerics to the portal's in-process pipeline so HPC and dev
    results stay comparable. Pruning below ``prune_below`` mirrors the
    existing pipeline.
    """
    if M_domains.shape[0] != M_pairs.shape[0]:
        raise ValueError(
            f"row mismatch: M_domains={M_domains.shape}, M_pairs={M_pairs.shape}"
        )

    w_d, w_a = weights
    total = float(w_d) + float(w_a)
    if total <= 0:
        raise ValueError(f"weights must sum > 0, got {weights}")
    w_d, w_a = w_d / total, w_a / total

    sim_d = dice_similarity(M_domains, matmul_workers=matmul_workers) if w_d > 0 else None
    sim_a = dice_similarity(M_pairs, matmul_workers=matmul_workers) if w_a > 0 else None

    if sim_d is not None and sim_a is not None:
        sim = (w_d * sim_d) + (w_a * sim_a)
    elif sim_d is not None:
        sim = w_d * sim_d
    else:
        sim = w_a * sim_a  # type: ignore[operator]

    sim = sim.tocsr()

    if prune_below > 0.0 and sim.nnz:
        import scipy.sparse as sp

        coo = sim.tocoo(copy=False)
        keep = coo.data >= prune_below
        if int(keep.sum()) != coo.nnz:
            sim = sp.csr_matrix(
                (coo.data[keep], (coo.row[keep], coo.col[keep])),
                shape=coo.shape,
            )

    sim.setdiag(0)
    sim.eliminate_zeros()
    sim = sim.maximum(sim.T).tocsr()
    log.info(
        "compute_composite_similarity: shape=%s nnz=%d w=(%.3f,%.3f)",
        sim.shape, sim.nnz, w_d, w_a,
    )
    return sim


def _matmul_chunked(Mf: "sp.csr_matrix", workers: int) -> "sp.csr_matrix":
    """Row-chunked sparse matmul. Only useful at N ≥ ~100k.

    Each worker computes ``M[start:end] @ M.T`` and we vstack the partial
    results. Uses ``joblib`` with the loky backend; the cost is one pickle
    of ``M`` per worker, so the break-even depends on the matrix size.
    """
    import math

    import scipy.sparse as sp
    from joblib import Parallel, delayed

    n = Mf.shape[0]
    chunk = max(1, math.ceil(n / max(workers, 1)))
    ranges = [(i, min(n, i + chunk)) for i in range(0, n, chunk)]

    def _block(rng):
        s, e = rng
        return (Mf[s:e] @ Mf.T).tocsr()

    blocks = Parallel(n_jobs=workers, backend="loky")(
        delayed(_block)(r) for r in ranges
    )
    return sp.vstack(blocks, format="csr")
