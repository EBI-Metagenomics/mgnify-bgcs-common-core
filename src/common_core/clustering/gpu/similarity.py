"""GPU composite-Dice via cuSPARSE.

Mirrors :mod:`common_core.clustering.similarity` numerics exactly; only the
matmul backend changes. The Dice formula is computed on the device and the
result returned as a SciPy CSR for downstream use (KNN top-k, Leiden,
partial projection). UMAP on GPU takes its KNN from this same matrix, also
on device.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import scipy.sparse as sp

log = logging.getLogger(__name__)


def dice_similarity_gpu(M: "sp.csr_matrix"):
    """Sørensen-Dice on device, returns a cupy CSR."""
    import cupy as cp
    import cupyx.scipy.sparse as cusp

    if M.shape[0] == 0:
        return cusp.csr_matrix((0, 0), dtype=cp.float32)

    Mf_gpu = cusp.csr_matrix(M.astype("float32"))
    inter = (Mf_gpu @ Mf_gpu.T).tocoo()
    if inter.nnz == 0:
        return cusp.csr_matrix(inter.shape, dtype=cp.float32)

    sizes = Mf_gpu.sum(axis=1).ravel().astype(cp.float32)
    denom = sizes[inter.row] + sizes[inter.col]
    safe = denom > 0
    values = cp.zeros_like(inter.data, dtype=cp.float32)
    values[safe] = 2.0 * inter.data[safe] / denom[safe]
    sim = cusp.csr_matrix(
        (values, (inter.row, inter.col)), shape=inter.shape, dtype=cp.float32,
    )
    sim.eliminate_zeros()
    return sim


def compute_composite_similarity_gpu(
    M_domains: "sp.csr_matrix",
    M_pairs: "sp.csr_matrix",
    *,
    weights: tuple[float, float] = (0.5, 0.5),
    prune_below: float = 0.05,
) -> "sp.csr_matrix":
    """GPU composite similarity. Returns a host-side scipy CSR."""
    import cupy as cp
    import cupyx.scipy.sparse as cusp
    import scipy.sparse as sp

    if M_domains.shape[0] != M_pairs.shape[0]:
        raise ValueError(
            f"row mismatch: M_domains={M_domains.shape}, M_pairs={M_pairs.shape}"
        )
    w_d, w_a = weights
    total = float(w_d) + float(w_a)
    if total <= 0:
        raise ValueError(f"weights must sum > 0, got {weights}")
    w_d, w_a = w_d / total, w_a / total

    sim_d = dice_similarity_gpu(M_domains) if w_d > 0 else None
    sim_a = dice_similarity_gpu(M_pairs) if w_a > 0 else None

    if sim_d is not None and sim_a is not None:
        sim_gpu = (w_d * sim_d) + (w_a * sim_a)
    elif sim_d is not None:
        sim_gpu = w_d * sim_d
    else:
        sim_gpu = w_a * sim_a  # type: ignore[operator]

    sim_gpu = sim_gpu.tocsr()
    if prune_below > 0.0 and sim_gpu.nnz:
        coo = sim_gpu.tocoo()
        keep = coo.data >= prune_below
        if int(keep.sum().item()) != coo.nnz:
            sim_gpu = cusp.csr_matrix(
                (coo.data[keep], (coo.row[keep], coo.col[keep])),
                shape=coo.shape,
            )

    sim_gpu.setdiag(0)
    sim_gpu.eliminate_zeros()
    sim_gpu_t = sim_gpu.T
    sim_gpu = cusp.csr_matrix(cp.maximum(sim_gpu.todense(), sim_gpu_t.todense()))
    # Convert back to host as scipy CSR — downstream KNN / Leiden / scoring
    # work on host, and the size is fine for a single transfer.
    sim_host = sp.csr_matrix(sim_gpu.get())
    log.info(
        "compute_composite_similarity_gpu: shape=%s nnz=%d w=(%.3f,%.3f)",
        sim_host.shape, sim_host.nnz, w_d, w_a,
    )
    return sim_host
