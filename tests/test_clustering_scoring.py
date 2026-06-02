"""Pure-array tests for ``compute_novelty_against_validated``.

Locks the corrected novelty math: a validated NRB matches itself (Dice 1.0)
and is therefore not novel, and near-threshold similarities are not pruned
away (the block is built fresh, unpruned, with the diagonal intact — unlike
the clustering similarity matrix that feeds KNN / Leiden).
"""

from __future__ import annotations

import numpy as np
import pytest

scipy_sparse = pytest.importorskip("scipy.sparse")

from common_core.clustering.scoring import (  # noqa: E402
    compute_novelty_against_validated,
)


def _coo(rows: list[list[int]], n_cols: int) -> "scipy_sparse.csr_matrix":
    coords_r: list[int] = []
    coords_c: list[int] = []
    for r, cols in enumerate(rows):
        for c in cols:
            coords_r.append(r)
            coords_c.append(c)
    data = np.ones(len(coords_r), dtype=np.uint8)
    return scipy_sparse.csr_matrix(
        (data, (coords_r, coords_c)), shape=(len(rows), n_cols), dtype=np.uint8,
    )


def _zeros(n: int) -> "scipy_sparse.csr_matrix":
    return scipy_sparse.csr_matrix((n, 1), dtype=np.uint8)


def test_no_validated_returns_nan():
    M = _coo([[0, 1], [1, 2]], n_cols=3)
    out = compute_novelty_against_validated(M, _zeros(2), [], weights=(1.0, 0.0))
    assert np.all(np.isnan(out))


def test_validated_row_is_zero_without_a_similar_validated_neighbour():
    # Regression: the bug returned 1.0 here (zeroed diagonal dropped the
    # self-match). Now the validated row's self-Dice of 1.0 → novelty 0.
    M = _coo([[0, 1], [2, 3]], n_cols=4)  # row 0 validated, row 1 disjoint
    out = compute_novelty_against_validated(M, _zeros(2), [0], weights=(1.0, 0.0))
    assert out[0] == pytest.approx(0.0, abs=1e-6)
    assert out[1] == pytest.approx(1.0, abs=1e-6)


def test_near_threshold_similarity_survives():
    # Dice = 2/60 ≈ 0.033 — below the old prune cutoff of 0.05, so the old
    # path would have read novelty 1.0 for row 1.
    M = _coo([list(range(0, 30)), list(range(29, 59))], n_cols=59)
    out = compute_novelty_against_validated(M, _zeros(2), [0], weights=(1.0, 0.0))
    assert out[1] == pytest.approx(1.0 - (2.0 / 60.0), abs=1e-4)
    assert out[1] < 1.0
