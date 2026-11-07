#!/usr/bin/env python3

# Copyright 2026 EMBL - European Bioinformatics Institute
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Composite similarity on the two binary signature matrices.

Mirrors the portal's ``services.clustering.bgc_similarity.compute_composite_similarity``
but kept pure-numpy/scipy and Django-free. The composite is

    sim = w_d · S(M_domains) + w_a · S(M_pairs)

with ``S`` chosen by ``metric`` (ADR 0003):

* ``"dice"`` — ``Dice(M) = 2 · (M @ M.T) / (rowsum_i + rowsum_j)``; the
  original metric, kept so older runs stay reproducible.
* ``"cosine_idf_enrich"`` — weighted cosine, ``X @ X.T`` with
  ``X = rownorm(M · diag(w))`` and ``w = idf × enrich`` per feature (see
  :mod:`common_core.clustering.weights`).

Diagonal is zeroed and the result is symmetrised. The matmul is
single-threaded; row-chunked parallelism is opt-in via ``matmul_workers``
and only really pays off at N ≥ ~100k.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

    import numpy as np
    import scipy.sparse as sp

log = logging.getLogger(__name__)

METRIC_DICE = "dice"
METRIC_COSINE_IDF_ENRICH = "cosine_idf_enrich"
SIMILARITY_METRICS: tuple[str, ...] = (METRIC_DICE, METRIC_COSINE_IDF_ENRICH)


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


def weighted_rows(M: "sp.csr_matrix", w: "np.ndarray") -> "sp.csr_matrix":
    """``rownorm(M · diag(w))`` as float32 CSR; all-zero rows stay zero."""
    import numpy as np
    import scipy.sparse as sp

    w = np.asarray(w, dtype=np.float32).ravel()
    if w.shape[0] != M.shape[1]:
        raise ValueError(f"weight length {w.shape[0]} ≠ feature count {M.shape[1]}")
    X = (M.astype(np.float32) @ sp.diags(w)).tocsr()
    norms = np.sqrt(np.asarray(X.multiply(X).sum(axis=1), dtype=np.float64).ravel())
    inv = np.zeros_like(norms)
    np.divide(1.0, norms, out=inv, where=norms > 0)
    X = (sp.diags(inv.astype(np.float32)) @ X).tocsr()
    X.eliminate_zeros()
    return X


def weighted_cosine_similarity(
    M: "sp.csr_matrix", w: "np.ndarray", *, matmul_workers: int = 1,
) -> "sp.csr_matrix":
    """Weighted cosine ``Σ_{A∩B} w² / sqrt(Σ_A w² · Σ_B w²)``. CPU.

    Values are clipped to 1.0 so float32 round-off on self-matches can't
    push ``1 − sim`` (novelty) below zero.
    """
    import numpy as np
    import scipy.sparse as sp

    if M.shape[0] == 0:
        return sp.csr_matrix((0, 0), dtype=np.float32)

    X = weighted_rows(M, w)
    if matmul_workers <= 1:
        sim = (X @ X.T).tocsr()
    else:
        sim = _matmul_chunked(X, matmul_workers)
    sim = sim.astype(np.float32)
    np.minimum(sim.data, 1.0, out=sim.data)
    sim.eliminate_zeros()
    return sim


def check_metric(
    metric: str,
    M_domains: "sp.csr_matrix",
    M_pairs: "sp.csr_matrix",
    domain_weights,
    pair_weights,
) -> None:
    """Raise ``ValueError`` on an unknown metric or missing / misaligned weights."""
    if metric not in SIMILARITY_METRICS:
        raise ValueError(f"unknown similarity metric {metric!r}; expected one of {SIMILARITY_METRICS}")
    if metric == METRIC_DICE:
        return
    for name, M, w in (("domain", M_domains, domain_weights), ("pair", M_pairs, pair_weights)):
        if w is None:
            raise ValueError(f"metric {metric!r} needs {name}_weights")
        if len(w) != M.shape[1]:
            raise ValueError(
                f"{name}_weights has {len(w)} entries but the matrix has {M.shape[1]} columns"
            )


def family_similarity(
    M: "sp.csr_matrix", w, *, metric: str, matmul_workers: int = 1,
) -> "sp.csr_matrix":
    """One feature family (domains or pairs) under ``metric``."""
    if metric == METRIC_DICE:
        return dice_similarity(M, matmul_workers=matmul_workers)
    return weighted_cosine_similarity(M, w, matmul_workers=matmul_workers)


def compute_composite_similarity(
    M_domains: "sp.csr_matrix",
    M_pairs: "sp.csr_matrix",
    *,
    weights: tuple[float, float] = (0.5, 0.5),
    prune_below: float = 0.05,
    matmul_workers: int = 1,
    metric: str = METRIC_DICE,
    domain_weights: "np.ndarray | None" = None,
    pair_weights: "np.ndarray | None" = None,
) -> "sp.csr_matrix":
    """Weighted mean of the per-family similarities over both signature matrices.

    ``metric="dice"`` (default) has identical numerics to the portal's
    in-process pipeline so HPC and dev results stay comparable.
    ``metric="cosine_idf_enrich"`` needs ``domain_weights`` / ``pair_weights``
    aligned to the matrix columns. Pruning below ``prune_below`` mirrors the
    existing pipeline.
    """
    if M_domains.shape[0] != M_pairs.shape[0]:
        raise ValueError(
            f"row mismatch: M_domains={M_domains.shape}, M_pairs={M_pairs.shape}"
        )
    check_metric(metric, M_domains, M_pairs, domain_weights, pair_weights)

    w_d, w_a = weights
    total = float(w_d) + float(w_a)
    if total <= 0:
        raise ValueError(f"weights must sum > 0, got {weights}")
    w_d, w_a = w_d / total, w_a / total

    sim_d = (
        family_similarity(M_domains, domain_weights, metric=metric, matmul_workers=matmul_workers)
        if w_d > 0
        else None
    )
    sim_a = (
        family_similarity(M_pairs, pair_weights, metric=metric, matmul_workers=matmul_workers)
        if w_a > 0
        else None
    )

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

    _zero_diagonal(sim)
    sim.eliminate_zeros()
    sim = sim.maximum(sim.T).tocsr()
    log.info(
        "compute_composite_similarity: metric=%s shape=%s nnz=%d w=(%.3f,%.3f)",
        metric, sim.shape, sim.nnz, w_d, w_a,
    )
    return sim


def iter_cross_similarity(
    Q_domains: "sp.csr_matrix",
    Q_pairs: "sp.csr_matrix",
    P_domains: "sp.csr_matrix",
    P_pairs: "sp.csr_matrix",
    *,
    weights: tuple[float, float] = (0.5, 0.5),
    metric: str = METRIC_DICE,
    domain_weights: "np.ndarray | None" = None,
    pair_weights: "np.ndarray | None" = None,
    chunk_rows: int = 2048,
) -> "Iterator[tuple[int, sp.csr_matrix]]":
    """Yield ``(start, block)`` row chunks of the query × reference composite.

    ``block`` is ``composite(Q[start:start+len], P)`` as float32 CSR with
    sorted indices: unpruned, no diagonal handling, not symmetrised. Each
    entry equals the matching entry of :func:`compute_composite_similarity`
    on ``vstack([P, Q])`` with ``prune_below=0`` — per-row Dice sizes and
    cosine row norms don't depend on the other rows, and the matmul sums
    shared columns in the same order — but the reference × reference and
    query × query blocks are never built, and only ``chunk_rows`` query rows
    are in memory at a time.
    """
    if Q_domains.shape[0] != Q_pairs.shape[0] or P_domains.shape[0] != P_pairs.shape[0]:
        raise ValueError(
            f"row mismatch: Q={Q_domains.shape}/{Q_pairs.shape}, "
            f"P={P_domains.shape}/{P_pairs.shape}"
        )
    if Q_domains.shape[1] != P_domains.shape[1] or Q_pairs.shape[1] != P_pairs.shape[1]:
        raise ValueError("query and reference matrices must share their vocabularies")
    check_metric(metric, P_domains, P_pairs, domain_weights, pair_weights)
    if chunk_rows < 1:
        raise ValueError(f"chunk_rows must be ≥ 1, got {chunk_rows}")

    w_d, w_a = weights
    total = float(w_d) + float(w_a)
    if total <= 0:
        raise ValueError(f"weights must sum > 0, got {weights}")
    w_d, w_a = w_d / total, w_a / total

    families = []
    if w_d > 0:
        families.append((w_d, _CrossFamily(Q_domains, P_domains, domain_weights, metric)))
    if w_a > 0:
        families.append((w_a, _CrossFamily(Q_pairs, P_pairs, pair_weights, metric)))

    n_q = Q_domains.shape[0]
    for start in range(0, n_q, chunk_rows):
        end = min(n_q, start + chunk_rows)
        block = None
        for w, fam in families:
            part = w * fam.block(start, end)
            block = part if block is None else block + part
        block = block.tocsr()
        block.eliminate_zeros()
        block.sort_indices()
        yield start, block


class _CrossFamily:
    """One feature family's query × reference similarity, built per row chunk.

    The reference side (``P.T`` and, for Dice, its row sizes) is prepared
    once; the query side is weighted/normalised once and sliced per chunk.
    """

    def __init__(self, Q, P, w, metric: str) -> None:
        import numpy as np

        self.metric = metric
        if metric == METRIC_DICE:
            self.Q = Q.astype(np.float32).tocsr()
            P_f = P.astype(np.float32).tocsr()
            self.q_sizes = np.asarray(Q.sum(axis=1), dtype=np.float32).ravel()
            self.p_sizes = np.asarray(P.sum(axis=1), dtype=np.float32).ravel()
        else:
            self.Q = weighted_rows(Q, w)
            P_f = weighted_rows(P, w)
        self.PT = P_f.T.tocsr()

    def block(self, start: int, end: int) -> "sp.csr_matrix":
        import numpy as np
        import scipy.sparse as sp

        prod = self.Q[start:end] @ self.PT
        if self.metric != METRIC_DICE:
            prod = prod.astype(np.float32).tocsr()
            np.minimum(prod.data, 1.0, out=prod.data)
            prod.eliminate_zeros()
            return prod

        inter = prod.tocoo(copy=False)
        if inter.nnz == 0:
            return sp.csr_matrix(inter.shape, dtype=np.float32)
        denom = self.q_sizes[start:end][inter.row] + self.p_sizes[inter.col]
        safe = denom > 0
        values = np.zeros_like(inter.data, dtype=np.float32)
        values[safe] = 2.0 * inter.data[safe] / denom[safe]
        sim = sp.csr_matrix(
            (values, (inter.row, inter.col)), shape=inter.shape, dtype=np.float32,
        )
        sim.eliminate_zeros()
        return sim


def _zero_diagonal(sim: "sp.csr_matrix") -> None:
    """Zero stored diagonal entries in place.

    ``csr_matrix.setdiag(0)`` is a no-op on scipy 1.14 (the portal's pin) when
    some diagonal entries are absent from the structure, which is the usual
    case here (empty signatures have no self-match).
    """
    import numpy as np

    rows = np.repeat(np.arange(sim.shape[0], dtype=sim.indices.dtype), np.diff(sim.indptr))
    sim.data[sim.indices == rows] = 0


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
