"""Regression tests for ``compute_composite_similarity_gpu``.

Locks two properties:

1. **No dense n*n allocation.** The symmetrisation step previously did
   ``cp.maximum(sim.todense(), sim.T.todense())`` which OOMed an 80 GiB A100
   at n≈130k (two ~63 GiB float32 buffers). The source must use the sparse
   ``sim.maximum(sim.T)`` form. This is a static check on the file so it runs
   without a GPU.
2. **GPU output equals CPU output.** When cupy / cuml are importable, the
   GPU and CPU composite-Dice paths must produce element-wise identical CSR
   matrices on a small fixture. Otherwise the test is skipped.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

scipy_sparse = pytest.importorskip("scipy.sparse")


def test_gpu_similarity_source_does_not_densify():
    """Guard the source against re-introducing the dense n*n allocation."""
    src = Path(__file__).resolve().parents[1] / (
        "src/common_core/clustering/gpu/similarity.py"
    )
    text = src.read_text()
    assert ".todense()" not in text, (
        "compute_composite_similarity_gpu must symmetrise sparsely "
        "(sim.maximum(sim.T)); .todense() allocates an n*n*4 byte buffer "
        "and OOMs at production scale."
    )


def _binary_csr(rows: list[list[int]], n_cols: int) -> "scipy_sparse.csr_matrix":
    rs: list[int] = []
    cs: list[int] = []
    for r, cols in enumerate(rows):
        for c in cols:
            rs.append(r)
            cs.append(c)
    data = np.ones(len(rs), dtype=np.uint8)
    return scipy_sparse.csr_matrix(
        (data, (rs, cs)), shape=(len(rows), n_cols), dtype=np.uint8,
    )


def test_gpu_matches_cpu_on_small_fixture():
    """GPU composite-Dice must match the CPU reference exactly."""
    cp = pytest.importorskip("cupy")
    pytest.importorskip("cupyx.scipy.sparse")

    try:
        cp.zeros(1, dtype=cp.float32)
    except Exception as exc:  # pragma: no cover — host-dependent
        pytest.skip(f"no usable CUDA device: {exc}")

    from common_core.clustering.gpu.similarity import (
        compute_composite_similarity_gpu,
    )
    from common_core.clustering.similarity import compute_composite_similarity

    rng = np.random.default_rng(0)
    n, d_dom, d_pair = 64, 24, 40
    M_domains = scipy_sparse.csr_matrix(
        (rng.random((n, d_dom)) > 0.7).astype(np.uint8)
    )
    M_pairs = scipy_sparse.csr_matrix(
        (rng.random((n, d_pair)) > 0.85).astype(np.uint8)
    )

    sim_cpu = compute_composite_similarity(M_domains, M_pairs)
    sim_gpu = compute_composite_similarity_gpu(M_domains, M_pairs)

    assert sim_cpu.shape == sim_gpu.shape
    a = sim_cpu.toarray()
    b = sim_gpu.toarray()
    np.testing.assert_allclose(a, b, rtol=0, atol=1e-6)
    # Symmetry — what the buggy densification was trying to enforce.
    np.testing.assert_allclose(b, b.T, rtol=0, atol=1e-6)
