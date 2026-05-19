"""2D layout via UMAP fed with a precomputed KNN derived from ``sim``.

CPU implementation only — the GPU variant lives in ``gpu/layout.py``.
Numba parallelism is the only thing that helps here; we skip it for N<5k
where the overhead would dominate.

Coordinates are normalized to ``[-10, 10]`` on both axes so the dashboard
scatter plot scales stay stable across re-runs.
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
_UMAP_MIN_VERTICES = 50


def compute_2d_layout(
    graph: "ig.Graph",
    sim: "sp.csr_matrix",
    *,
    seed: int = 42,
    n_neighbors: int | None = None,
) -> "np.ndarray":
    """Return a normalized ``(n_rows, 2)`` array.

    Falls back to igraph DRL for very small graphs / when umap-learn isn't
    importable.
    """
    import numpy as np

    n = graph.vcount()
    if n == 0:
        return np.zeros((0, 2), dtype=np.float64)

    coords: np.ndarray | None = None
    if n >= _UMAP_MIN_VERTICES:
        coords = _umap_layout(sim, seed=seed, n_neighbors=n_neighbors)
    if coords is None:
        coords = _igraph_layout(graph)

    return _normalize(coords)


def _umap_layout(
    sim: "sp.csr_matrix",
    *,
    seed: int,
    n_neighbors: int | None,
) -> "np.ndarray | None":
    try:
        import numpy as np
        import umap as umap_lib
    except ImportError:
        log.warning("umap-learn not available; falling back to igraph layout")
        return None

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
        knn_dist[row, 0] = 0.0
        m = cols.size
        if m:
            knn_idx[row, 1 : 1 + m] = cols
            knn_dist[row, 1 : 1 + m] = dists
        if 1 + m < k:
            knn_idx[row, 1 + m :] = row
            knn_dist[row, 1 + m :] = 0.0

    try:
        reducer = umap_lib.UMAP(
            n_components=2,
            random_state=seed,
            n_neighbors=k,
            precomputed_knn=(knn_idx, knn_dist, None),
        )
        coords = reducer.fit_transform(
            np.zeros((n, 1), dtype=np.float32),
            ensure_all_finite=False,
        )
    except Exception:  # pragma: no cover
        log.exception("UMAP layout failed; falling back to igraph layout")
        return None

    coords = np.asarray(coords, dtype=np.float64)
    finite_mask = np.isfinite(coords).all(axis=1)
    n_bad = int((~finite_mask).sum())
    if n_bad:
        if finite_mask.sum() < max(2, n // 2):
            log.warning(
                "UMAP produced %d/%d non-finite rows; falling back to igraph",
                n_bad, n,
            )
            return None
        centre = coords[finite_mask].mean(axis=0)
        coords[~finite_mask] = centre
    return coords


def _igraph_layout(graph: "ig.Graph") -> "np.ndarray":
    import numpy as np

    n = graph.vcount()
    if n == 0:
        return np.zeros((0, 2), dtype=np.float64)
    try:
        layout = graph.layout_drl(seed=None)
    except Exception:
        try:
            layout = graph.layout_fruchterman_reingold(seed=None)
        except Exception:
            layout = graph.layout_random()
    return np.asarray(layout.coords, dtype=np.float64)


def _normalize(coords: "np.ndarray") -> "np.ndarray":
    import numpy as np

    if coords.shape[0] == 0:
        return coords
    centred = coords - coords.mean(axis=0, keepdims=True)
    extent = float(np.max(np.abs(centred))) or 1.0
    return (centred / extent) * _LAYOUT_RANGE
