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
    block: int = 4096,
) -> "sp.csr_matrix":
    """GPU composite similarity, fused per row-block.

    Returns a host-side scipy CSR. Computes both intersection matmuls
    (M_domains and M_pairs) over the same row block, combines weighted,
    drops entries below ``prune_below`` before accumulating. This keeps
    the working set bounded by per-block nnz rather than the global
    unpruned intersection count, which can exceed 10^9 entries at
    production scale (observed: ~4×10^9 on n≈130k) and OOM the device
    even before reduction.
    """
    import cupy as cp
    import cupyx.scipy.sparse as cusp
    import scipy.sparse as sp

    if M_domains.shape[0] != M_pairs.shape[0]:
        raise ValueError(
            f"row mismatch: M_domains={M_domains.shape}, M_pairs={M_pairs.shape}"
        )
    n = M_domains.shape[0]
    w_d, w_a = weights
    total = float(w_d) + float(w_a)
    if total <= 0:
        raise ValueError(f"weights must sum > 0, got {weights}")
    w_d, w_a = w_d / total, w_a / total

    Mfd = cusp.csr_matrix(M_domains.astype("float32")) if w_d > 0 else None
    Mfa = cusp.csr_matrix(M_pairs.astype("float32")) if w_a > 0 else None
    MfdT = Mfd.T.tocsr() if Mfd is not None else None
    MfaT = Mfa.T.tocsr() if Mfa is not None else None
    sizes_d = Mfd.sum(axis=1).ravel().astype(cp.float32) if Mfd is not None else None
    sizes_a = Mfa.sum(axis=1).ravel().astype(cp.float32) if Mfa is not None else None

    rows_chunks: list = []
    cols_chunks: list = []
    data_chunks: list = []
    total_inter = 0
    total_kept = 0

    def _block_csr(Mf, MfT, sizes, weight, start, end):
        inter = (Mf[start:end] @ MfT).tocoo()
        nnz = int(inter.nnz)
        if nnz == 0:
            return None, 0
        rr = inter.row + start
        cc = inter.col
        denom = sizes[rr] + sizes[cc]
        safe = denom > 0
        vals = cp.zeros(nnz, dtype=cp.float32)
        vals[safe] = weight * (2.0 * inter.data[safe] / denom[safe])
        return (
            cusp.csr_matrix(
                (vals, (inter.row, inter.col)),
                shape=(end - start, n),
                dtype=cp.float32,
            ),
            nnz,
        )

    for start in range(0, n, block):
        end = min(start + block, n)
        parts = []
        if Mfd is not None:
            d_csr, d_nnz = _block_csr(Mfd, MfdT, sizes_d, w_d, start, end)
            total_inter += d_nnz
            if d_csr is not None:
                parts.append(d_csr)
        if Mfa is not None:
            a_csr, a_nnz = _block_csr(Mfa, MfaT, sizes_a, w_a, start, end)
            total_inter += a_nnz
            if a_csr is not None:
                parts.append(a_csr)
        if not parts:
            continue

        combo = parts[0]
        for part in parts[1:]:
            combo = combo + part
        coo = combo.tocoo()
        keep = coo.data >= prune_below
        n_keep = int(keep.sum().item())
        if n_keep == 0:
            continue
        rows_chunks.append(coo.row[keep] + start)
        cols_chunks.append(coo.col[keep])
        data_chunks.append(coo.data[keep])
        total_kept += n_keep

    log.info(
        "compute_composite_similarity_gpu: shape=(%d, %d) inter_nnz=%d kept(>=%.3f)=%d "
        "w=(%.3f,%.3f) block=%d",
        n, n, total_inter, prune_below, total_kept, w_d, w_a, block,
    )

    if not rows_chunks:
        return sp.csr_matrix((n, n), dtype="float32")

    rows = cp.concatenate(rows_chunks)
    cols = cp.concatenate(cols_chunks)
    data = cp.concatenate(data_chunks)
    # Drop the diagonal — clustering treats self-similarity as 0.
    off_diag = rows != cols
    rows = rows[off_diag]
    cols = cols[off_diag]
    data = data[off_diag]

    sim_gpu = cusp.csr_matrix((data, (rows, cols)), shape=(n, n), dtype=cp.float32)
    # Symmetrise on host. cupyx's sparse .maximum() raises NotImplementedError;
    # densifying on device allocates n*n*4 byte buffers and OOMs. scipy's sparse
    # .maximum() is O(nnz) and well-tested.
    sim_host = sp.csr_matrix(sim_gpu.get())
    sim_host = sim_host.maximum(sim_host.T).tocsr()
    log.info(
        "compute_composite_similarity_gpu: final nnz=%d",
        sim_host.nnz,
    )
    return sim_host
