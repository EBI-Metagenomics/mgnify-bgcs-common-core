"""GPU UMAP via cuml.

This is the biggest single GPU win in the pipeline. Operates on the
precomputed-KNN derived from the host-side sim matrix (the matrix is small
enough that the device transfer overhead is negligible).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import igraph as ig
    import numpy as np
    import scipy.sparse as sp

log = logging.getLogger(__name__)

_LAYOUT_RANGE = 10.0


def compute_2d_layout_gpu(
    graph: "ig.Graph",
    sim: "sp.csr_matrix",
    *,
    seed: int = 42,
    n_neighbors: int | None = None,
) -> "np.ndarray":
    """Run cuml.UMAP on the precomputed KNN from ``sim``. Returns host coords."""
    import cuml
    import numpy as np

    n = sim.shape[0]
    if n == 0:
        return np.zeros((0, 2), dtype=np.float64)

    sim_csr = sim.tocsr(copy=False)
    k_default = max(5, min(15, max(n - 1, 1)))
    k = max(2, min(n, n_neighbors or k_default))

    knn_idx = np.zeros((n, k), dtype=np.int32)
    knn_dist = np.zeros((n, k), dtype=np.float32)
    for row in range(n):
        start = sim_csr.indptr[row]
        end = sim_csr.indptr[row + 1]
        cols = sim_csr.indices[start:end]
        vals = sim_csr.data[start:end]
        keep = cols != row
        cols = cols[keep]
        vals = vals[keep]
        order = np.argsort(-vals)[: k - 1]
        cols = cols[order]
        dists = np.clip(1.0 - vals[order].astype(np.float32), 0.0, 1.0)
        knn_idx[row, 0] = row
        m = cols.size
        if m:
            knn_idx[row, 1 : 1 + m] = cols
            knn_dist[row, 1 : 1 + m] = dists
        if 1 + m < k:
            knn_idx[row, 1 + m :] = row
            knn_dist[row, 1 + m :] = 0.0

    reducer = cuml.UMAP(
        n_components=2,
        random_state=seed,
        n_neighbors=k,
        precomputed_knn=(knn_idx, knn_dist),
    )
    coords = reducer.fit_transform(np.zeros((n, 1), dtype=np.float32))
    coords = np.asarray(coords, dtype=np.float64)

    finite = np.isfinite(coords).all(axis=1)
    if (~finite).any():
        centre = coords[finite].mean(axis=0) if finite.any() else np.zeros(2)
        coords[~finite] = centre

    centred = coords - coords.mean(axis=0, keepdims=True)
    extent = float(np.max(np.abs(centred))) or 1.0
    return (centred / extent) * _LAYOUT_RANGE
