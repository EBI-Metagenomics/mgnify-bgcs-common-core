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


def dice_similarity_gpu(M: "sp.csr_matrix", *, block: int = 4096):
    """Sørensen-Dice on device, returns a cupy CSR.

    The intersection ``Mf @ Mf.T`` is computed in row-blocks of ``block``
    rows. cupyx's CSR SpGEMM silently returns an empty result when the
    full-shape output exceeds cuSPARSE's legacy nnz workspace limits
    (observed at n=129,777 with ~2.2M input nnz: the full call returned
    nnz=0). Chunking keeps each per-call output well under the limit;
    block size trades device memory for the number of SpGEMM launches.
    """
    import cupy as cp
    import cupyx.scipy.sparse as cusp

    n = M.shape[0]
    if n == 0:
        return cusp.csr_matrix((0, 0), dtype=cp.float32)

    Mf_gpu = cusp.csr_matrix(M.astype("float32"))
    MfT_gpu = Mf_gpu.T.tocsr()
    sizes = Mf_gpu.sum(axis=1).ravel().astype(cp.float32)

    rows_chunks: list = []
    cols_chunks: list = []
    data_chunks: list = []
    for start in range(0, n, block):
        end = min(start + block, n)
        block_inter = (Mf_gpu[start:end] @ MfT_gpu).tocoo()
        if block_inter.nnz == 0:
            continue
        rr = block_inter.row + start
        cc = block_inter.col
        denom = sizes[rr] + sizes[cc]
        safe = denom > 0
        if not bool(safe.any()):
            continue
        rows_chunks.append(rr[safe])
        cols_chunks.append(cc[safe])
        data_chunks.append(2.0 * block_inter.data[safe] / denom[safe])

    if not rows_chunks:
        return cusp.csr_matrix((n, n), dtype=cp.float32)

    rows = cp.concatenate(rows_chunks)
    cols = cp.concatenate(cols_chunks)
    data = cp.concatenate(data_chunks)
    sim = cusp.csr_matrix((data, (rows, cols)), shape=(n, n), dtype=cp.float32)
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

    sim_gpu.setdiag(cp.asarray(0, dtype=sim_gpu.dtype))
    sim_gpu.eliminate_zeros()
    # Symmetrise on host. Densifying on device allocated two n*n*4 byte
    # buffers (≈63 GiB at n=130k float32) and OOMed an 80 GiB A100; cupyx's
    # sparse .maximum() is NotImplementedError. scipy's sparse .maximum()
    # works fine, and the host transfer happens here anyway.
    sim_host = sp.csr_matrix(sim_gpu.get())
    sim_host = sim_host.maximum(sim_host.T).tocsr()
    log.info(
        "compute_composite_similarity_gpu: shape=%s nnz=%d w=(%.3f,%.3f)",
        sim_host.shape, sim_host.nnz, w_d, w_a,
    )
    return sim_host
