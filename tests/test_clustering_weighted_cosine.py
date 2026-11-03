"""Tests for the ``cosine_idf_enrich`` similarity metric (ADR 0003).

Locks:

1. **Kernel maths.** ``weighted_cosine_similarity`` equals the dense
   ``Σ_{A∩B} w² / sqrt(Σ_A w² · Σ_B w²)`` formula; the composite applies the
   same prune / zero-diagonal / symmetrise steps as Dice.
2. **Parity with the benchmark.** When the experiment package ``bgcsim``
   (``sanntis_dev/experiments/2026-10-23_bgc_similarity_robustness/src``) is
   importable, the composite and the weights match ``bgcsim.scoring.Term`` /
   ``bgcsim.features`` to ≤ 1e-5. Otherwise those tests are skipped.
3. **Dice is untouched.** ``metric="dice"`` (the default) still returns the
   exact Dice values, and the run sha of a Dice run is the v1 payload.
4. Novelty and partial projection honour the metric.
5. Partial projection's chunked partials × primaries block equals the
   matching block of the stacked composite exactly, so moving off the
   stacked matmul changes no similarity value.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

scipy_sparse = pytest.importorskip("scipy.sparse")

from common_core.clustering import weights as W
from common_core.clustering.cli import _compute_run_sha
from common_core.clustering.partial import project_partials
from common_core.clustering.scoring import (
    compute_novelty_against_validated,
)
from common_core.clustering.similarity import (
    compute_composite_similarity,
    iter_cross_similarity,
    weighted_cosine_similarity,
)

COSINE = "cosine_idf_enrich"


def _fixture(seed: int = 0, n: int = 60, d_dom: int = 30, d_pair: int = 50):
    """Random binary matrices with a few empty rows and some zero weights."""
    rng = np.random.default_rng(seed)
    M_dom = (rng.random((n, d_dom)) > 0.75).astype(np.uint8)
    M_pair = (rng.random((n, d_pair)) > 0.85).astype(np.uint8)
    M_dom[3] = 0  # empty domain row
    M_pair[3] = 0
    M_pair[7] = 0  # domains but no pairs
    w_dom = rng.uniform(0.0, 3.0, d_dom).astype(np.float32)
    w_pair = rng.uniform(0.0, 3.0, d_pair).astype(np.float32)
    w_dom[:2] = 0.0  # idf = 0 for a feature present in every group
    return (
        scipy_sparse.csr_matrix(M_dom),
        scipy_sparse.csr_matrix(M_pair),
        w_dom,
        w_pair,
    )


def _dense_cosine(M: np.ndarray, w: np.ndarray) -> np.ndarray:
    M = M.astype(np.float64)
    w2 = w.astype(np.float64) ** 2
    inter = (M * w2) @ M.T
    a2 = M @ w2
    den = np.sqrt(np.outer(a2, a2))
    out = np.zeros_like(inter)
    np.divide(inter, den, out=out, where=den > 0)
    return out


def _dense_dice(M: np.ndarray) -> np.ndarray:
    M = M.astype(np.float64)
    inter = M @ M.T
    s = M.sum(axis=1)
    den = s[:, None] + s[None, :]
    out = np.zeros_like(inter)
    np.divide(2.0 * inter, den, out=out, where=den > 0)
    return out


# ── Kernel ──────────────────────────────────────────────────────────────────


def test_weighted_cosine_matches_dense_formula():
    M_dom, _, w_dom, _ = _fixture()
    got = weighted_cosine_similarity(M_dom, w_dom).toarray()
    np.testing.assert_allclose(got, _dense_cosine(M_dom.toarray(), w_dom), atol=1e-6)


def test_weighted_cosine_chunked_matches_single():
    M_dom, _, w_dom, _ = _fixture()
    a = weighted_cosine_similarity(M_dom, w_dom).toarray()
    b = weighted_cosine_similarity(M_dom, w_dom, matmul_workers=2).toarray()
    np.testing.assert_allclose(a, b, atol=1e-7)


def test_weighted_cosine_self_match_is_one_and_never_above():
    M_dom, _, w_dom, _ = _fixture()
    sim = weighted_cosine_similarity(M_dom, w_dom).toarray()
    assert sim.max() <= 1.0
    nonempty = np.asarray((M_dom.multiply(w_dom[None, :])).sum(axis=1)).ravel() > 0
    np.testing.assert_allclose(np.diag(sim)[nonempty], 1.0, atol=1e-6)
    assert np.diag(sim)[~nonempty].max(initial=0.0) == 0.0


def test_composite_cosine_matches_dense_and_post_processing():
    M_dom, M_pair, w_dom, w_pair = _fixture()
    sim = compute_composite_similarity(
        M_dom,
        M_pair,
        prune_below=0.0,
        metric=COSINE,
        domain_weights=w_dom,
        pair_weights=w_pair,
    ).toarray()
    want = 0.5 * _dense_cosine(M_dom.toarray(), w_dom) + 0.5 * _dense_cosine(
        M_pair.toarray(), w_pair
    )
    np.fill_diagonal(want, 0.0)
    np.testing.assert_allclose(sim, want, atol=1e-6)
    np.testing.assert_allclose(sim, sim.T, atol=0)

    pruned = compute_composite_similarity(
        M_dom,
        M_pair,
        prune_below=0.3,
        metric=COSINE,
        domain_weights=w_dom,
        pair_weights=w_pair,
    ).toarray()
    np.testing.assert_allclose(pruned, np.where(want >= 0.3, want, 0.0), atol=1e-6)


def test_composite_term_weights_apply_to_cosine():
    M_dom, M_pair, w_dom, w_pair = _fixture()
    kw = {
        "prune_below": 0.0,
        "metric": COSINE,
        "domain_weights": w_dom,
        "pair_weights": w_pair,
    }
    dom_only = compute_composite_similarity(
        M_dom, M_pair, weights=(1.0, 0.0), **kw
    ).toarray()
    want = _dense_cosine(M_dom.toarray(), w_dom)
    np.fill_diagonal(want, 0.0)
    np.testing.assert_allclose(dom_only, want, atol=1e-6)


def test_dice_default_is_unchanged():
    M_dom, M_pair, w_dom, w_pair = _fixture()
    default = compute_composite_similarity(M_dom, M_pair, prune_below=0.0).toarray()
    # Weights passed alongside metric="dice" are ignored.
    explicit = compute_composite_similarity(
        M_dom,
        M_pair,
        prune_below=0.0,
        metric="dice",
        domain_weights=w_dom,
        pair_weights=w_pair,
    ).toarray()
    want = 0.5 * _dense_dice(M_dom.toarray()) + 0.5 * _dense_dice(M_pair.toarray())
    np.fill_diagonal(want, 0.0)
    np.testing.assert_allclose(default, want, atol=1e-6)
    np.testing.assert_array_equal(default, explicit)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"metric": "jaccard"}, "unknown similarity metric"),
        ({"metric": COSINE}, "needs domain_weights"),
        ({"metric": COSINE, "domain_weights": np.ones(30)}, "needs pair_weights"),
        (
            {
                "metric": COSINE,
                "domain_weights": np.ones(29),
                "pair_weights": np.ones(50),
            },
            "29 entries",
        ),
    ],
)
def test_composite_rejects_bad_metric_or_weights(kwargs, match):
    M_dom, M_pair, _, _ = _fixture()
    with pytest.raises(ValueError, match=match):
        compute_composite_similarity(M_dom, M_pair, **kwargs)


# ── Weights ─────────────────────────────────────────────────────────────────


def test_idf_counts_each_group_once():
    # Domain 0: in every group; domain 1: three rows but one group;
    # domain 2: one row.
    M = scipy_sparse.csr_matrix(
        np.array([[1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 0, 1]], dtype=np.uint8)
    )
    idf = W.idf_weights(M, np.array(["sp1", "sp1", "sp1", "sp2"], dtype=object))
    n = 2
    np.testing.assert_allclose(
        idf,
        [
            np.log((1 + n) / (1 + 2)),
            np.log((1 + n) / (1 + 1)),
            np.log((1 + n) / (1 + 1)),
        ],
        rtol=1e-6,
    )


def test_enrichment_shrinks_unseen_to_half_and_floors():
    core = scipy_sparse.csr_matrix(np.array([[1, 0, 0]] * 20, dtype=np.uint8))
    flank = scipy_sparse.csr_matrix(np.array([[0, 1, 0]] * 80, dtype=np.uint8))
    e = W.enrichment_weights(core, flank)
    assert e[0] > 0.95  # core-only
    assert e[1] == pytest.approx(W.ENRICH_FLOOR)  # flank-only, floored
    # Unseen → close to 0.5, not exactly: the proportional pseudocounts are
    # divided by n + 2, so p_core / p_flank ≠ 1 when n_c ≠ n_f.
    # This is the reference's behaviour (parity test below), kept verbatim.
    n_c, n_f = 20.0, 80.0
    ratio = (2 * n_c / (n_c + n_f) / (n_c + 2)) / (2 * n_f / (n_c + n_f) / (n_f + 2))
    assert e[2] == pytest.approx(1 / (1 + np.exp(-np.log2(ratio))), rel=1e-5)
    assert abs(e[2] - 0.5) < 0.05


def test_enrichment_from_counts_matches_matrix_form():
    rng = np.random.default_rng(3)
    core = scipy_sparse.csr_matrix((rng.random((50, 12)) > 0.7).astype(np.uint8))
    flank = scipy_sparse.csr_matrix((rng.random((140, 12)) > 0.85).astype(np.uint8))
    got = W.enrichment_from_counts(
        np.asarray(core.sum(axis=0)).ravel(),
        np.asarray(flank.sum(axis=0)).ravel(),
        50,
        140,
    )
    np.testing.assert_array_equal(got, W.enrichment_weights(core, flank))


def test_pair_enrichment_is_geometric_mean_with_unknown_fallback():
    accs = np.array(["IPR1", "IPR2", "IPR3"], dtype=object)
    e = np.array([0.9, 0.4, 0.1], dtype=np.float32)
    pairs = [("IPR1", "IPR2"), ("IPR2", "IPR3"), ("IPR1", "PF999")]
    got = W.pair_enrichment(pairs, accs, e)
    np.testing.assert_allclose(
        got,
        [np.sqrt(0.9 * 0.4), np.sqrt(0.4 * 0.1), np.sqrt(0.9 * W.UNKNOWN_ENRICH)],
        rtol=1e-6,
    )


def test_enrichment_vector_defaults_unknown_labels():
    got = W.enrichment_vector(["a", "b"], {"a": 0.8})
    np.testing.assert_allclose(got, [0.8, W.UNKNOWN_ENRICH])


def test_combine_and_hash():
    idf = np.array([1.0, 2.0], dtype=np.float32)
    enr = np.array([0.5, 0.25], dtype=np.float32)
    np.testing.assert_allclose(W.combine(idf, enr), [0.5, 0.5])
    with pytest.raises(ValueError):
        W.combine(idf, enr[:1])
    h = W.weights_sha256(idf, enr)
    assert h == W.weights_sha256(idf.astype(np.float64), enr)  # dtype-stable
    assert h != W.weights_sha256(enr, idf)
    assert h != W.weights_sha256(idf, np.array([0.5, 0.26], dtype=np.float32))


def test_combine_idf_exponent():
    idf = np.array([0.5, 1.0, 4.0], dtype=np.float32)
    enr = np.array([0.5, 0.25, 1.0], dtype=np.float32)
    # p = 1 is the benchmark reference, bit for bit.
    np.testing.assert_array_equal(W.combine(idf, enr, idf_exponent=1.0), idf * enr)
    np.testing.assert_allclose(
        W.combine(idf, enr, idf_exponent=W.COSINE_IDF_EXPONENT), np.sqrt(idf) * enr, rtol=1e-6
    )
    assert W.combine(idf, enr, idf_exponent=0.5).dtype == np.float32
    for bad in (0.0, -1.0):
        with pytest.raises(ValueError, match="idf_exponent"):
            W.combine(idf, enr, idf_exponent=bad)


# ── Parity with the experiment's reference implementation ───────────────────


def _bgcsim():
    scoring = pytest.importorskip("bgcsim.scoring")
    features = pytest.importorskip("bgcsim.features")
    return scoring, features


def test_parity_with_bgcsim_term_cosine():
    scoring, _ = _bgcsim()
    M_dom, M_pair, w_dom, w_pair = _fixture(seed=1, n=80)
    rows = np.arange(M_dom.shape[0])
    M_dom_f = M_dom.astype(np.float32)
    M_pair_f = M_pair.astype(np.float32)
    ref = scoring.composite(
        scoring.Term(M_dom_f, w_dom, rows), scoring.Term(M_pair_f, w_pair, rows), rows
    )["cosine"].astype(np.float64)
    np.fill_diagonal(ref, 0.0)

    got = compute_composite_similarity(
        M_dom,
        M_pair,
        prune_below=0.0,
        metric=COSINE,
        domain_weights=w_dom,
        pair_weights=w_pair,
    ).toarray()
    np.testing.assert_allclose(got, ref, rtol=0, atol=1e-5)


def test_parity_with_bgcsim_features():
    _, features = _bgcsim()
    rng = np.random.default_rng(2)
    M = scipy_sparse.csr_matrix((rng.random((120, 40)) > 0.8).astype(np.float32))
    groups = rng.choice(np.array([f"d__B;s__sp{i}" for i in range(17)] + ["NA"]), 120)
    np.testing.assert_allclose(
        W.idf_weights(M, groups), features.idf_weights(M, groups), rtol=0, atol=1e-6
    )

    core = scipy_sparse.csr_matrix((rng.random((300, 40)) > 0.7).astype(np.float32))
    flank = scipy_sparse.csr_matrix((rng.random((900, 40)) > 0.9).astype(np.float32))
    e_ref = features.enrichment_weights(core, flank)
    np.testing.assert_allclose(
        W.enrichment_weights(core, flank), e_ref, rtol=0, atol=1e-7
    )

    accs = [f"IPR{i:06d}" for i in range(40)]
    dvocab = {a: i for i, a in enumerate(accs)}
    pairs = [
        (accs[i], accs[j])
        for i in range(40)
        for j in range(i + 1, 40)
        if (i * j) % 7 == 1
    ]
    pvocab = {p: k for k, p in enumerate(pairs)}
    np.testing.assert_allclose(
        W.pair_enrichment(pairs, accs, e_ref),
        features.pair_enrichment_from_domains(pvocab, dvocab, e_ref),
        rtol=0,
        atol=1e-7,
    )


# ── Novelty + partial projection ────────────────────────────────────────────


def test_novelty_cosine_validated_zero_and_matches_dense():
    M_dom, M_pair, w_dom, w_pair = _fixture()
    validated = [0, 5, 11]
    nov = compute_novelty_against_validated(
        M_dom,
        M_pair,
        validated,
        metric=COSINE,
        domain_weights=w_dom,
        pair_weights=w_pair,
    )
    full = 0.5 * _dense_cosine(M_dom.toarray(), w_dom) + 0.5 * _dense_cosine(
        M_pair.toarray(), w_pair
    )
    want = 1.0 - full[:, validated].max(axis=1)
    want[validated] = 0.0
    np.testing.assert_allclose(nov, want, atol=1e-6)
    assert nov.min() >= 0.0


def test_novelty_cosine_requires_weights():
    M_dom, M_pair, _, _ = _fixture()
    with pytest.raises(ValueError, match="needs domain_weights"):
        compute_novelty_against_validated(M_dom, M_pair, [0], metric=COSINE)


def test_partial_projection_uses_metric_and_threshold():
    # Two primary families; partial 0 is a fragment of family B with a heavy
    # flank domain (col 9, weight ~0) that Dice counts in full.
    M = np.zeros((6, 10), dtype=np.uint8)
    M[0:3, 0:3] = 1  # family A: domains 0-2
    M[3:6, 4:7] = 1  # family B: domains 4-6
    M_pri = scipy_sparse.csr_matrix(M)
    M_q = scipy_sparse.csr_matrix(
        np.array([[0, 0, 0, 0, 1, 0, 0, 0, 1, 1]], dtype=np.uint8)
    )
    w = np.ones(10, dtype=np.float32)
    w[8:] = 0.01
    empty_pairs = scipy_sparse.csr_matrix((6, 1), dtype=np.uint8)
    empty_pairs_q = scipy_sparse.csr_matrix((1, 1), dtype=np.uint8)
    common = {
        "M_dom_pri": M_pri,
        "M_pair_pri": empty_pairs,
        "pri_ibgc_ids": np.arange(6),
        "pri_leaf_paths": ["0.0"] * 3 + ["1.0"] * 3,
        "pri_validated_rows": [3],
        "M_dom_q": M_q,
        "M_pair_q": empty_pairs_q,
        "partial_ibgc_ids": np.array([100]),
        "weights": (1.0, 0.0),
        "knn_k": 3,
    }
    # Dice: 2·1/(3+3) = 0.333 per neighbour → summed 1.0
    dice, _ = project_partials(**common, min_total_similarity=0.1)
    # Cosine: 1 / sqrt(3 · (1 + 2·0.01²)) ≈ 0.577 per neighbour
    cos, _ = project_partials(
        **common,
        min_total_similarity=0.1,
        metric=COSINE,
        domain_weights=w,
        pair_weights=np.ones(1, dtype=np.float32),
    )
    assert dice[0]["leaf_path"] == cos[0]["leaf_path"] == "1.0"
    assert dice[0]["novelty_score"] == pytest.approx(1 - 1 / 3, abs=1e-5)
    assert cos[0]["novelty_score"] == pytest.approx(
        1 - 1 / np.sqrt(3 * 1.0002), abs=1e-5
    )

    # The threshold is honoured: summed cosine ≈ 1.73 < 2.0 → skipped.
    skipped, n_skipped = project_partials(
        **common,
        min_total_similarity=2.0,
        metric=COSINE,
        domain_weights=w,
        pair_weights=np.ones(1, dtype=np.float32),
    )
    assert skipped == [] and n_skipped == 1


@pytest.mark.parametrize("metric", ["dice", COSINE])
@pytest.mark.parametrize("chunk_rows", [1, 7, 2048])
def test_cross_similarity_equals_stacked_composite_block(metric, chunk_rows):
    M_dom, M_pair, w_dom, w_pair = _fixture(seed=3)
    n_pri = 40
    kw = {"weights": (0.6, 0.4), "metric": metric}
    if metric == COSINE:
        kw |= {"domain_weights": w_dom, "pair_weights": w_pair}
    stacked = compute_composite_similarity(M_dom, M_pair, prune_below=0.0, **kw)
    expected = stacked[n_pri:, :n_pri].toarray()

    blocks = list(iter_cross_similarity(
        M_dom[n_pri:], M_pair[n_pri:], M_dom[:n_pri], M_pair[:n_pri],
        chunk_rows=chunk_rows, **kw,
    ))
    assert [s for s, _ in blocks] == list(range(0, M_dom.shape[0] - n_pri, chunk_rows))
    got = scipy_sparse.vstack([b for _, b in blocks]).toarray()
    assert got.dtype == np.float32
    np.testing.assert_array_equal(got, expected)  # bit-identical, not approx


def test_partial_projection_does_not_depend_on_chunk_rows():
    M_dom, M_pair, w_dom, w_pair = _fixture(seed=5, n=80)
    n_pri = 50
    common = {
        "M_dom_pri": M_dom[:n_pri],
        "M_pair_pri": M_pair[:n_pri],
        "pri_ibgc_ids": np.arange(n_pri),
        "pri_leaf_paths": [f"{i % 4}.{i % 7}" for i in range(n_pri)],
        "pri_validated_rows": [1, 9, 20],
        "M_dom_q": M_dom[n_pri:],
        "M_pair_q": M_pair[n_pri:],
        "partial_ibgc_ids": np.arange(100, 100 + M_dom.shape[0] - n_pri),
        "weights": (0.6, 0.4),
        "knn_k": 5,
        "min_total_similarity": 0.1,
        "metric": COSINE,
        "domain_weights": w_dom,
        "pair_weights": w_pair,
    }
    ref = project_partials(**common, chunk_rows=2048)
    assert ref[0]  # something was projected
    for chunk_rows in (1, 4, 29):
        assert project_partials(**common, chunk_rows=chunk_rows) == ref


# ── Run sha ─────────────────────────────────────────────────────────────────

_SHA_BASE = {
    "sources": ("PFAM", "NCBIFAM"),
    "weights": (0.5, 0.5),
    "knn_k": 12,
    "leiden_resolutions": (0.03, 0.08, 0.15, 0.25),
    "seed": 42,
    "ibgc_etag": "100:999",
    "domain_etag": "5000:321",
}


def test_dice_run_sha_is_the_v1_payload():
    v1_payload = "sources=NCBIFAM,PFAM|weights=0.500000,0.500000|k=12|res=0.030000,0.080000,0.150000,0.250000|seed=42|ibgc_etag=100:999|domain_etag=5000:321"
    assert (
        _compute_run_sha(**_SHA_BASE) == hashlib.sha256(v1_payload.encode()).hexdigest()
    )
    assert _compute_run_sha(**_SHA_BASE, metric="dice", domain_vocab="X") == (
        _compute_run_sha(**_SHA_BASE)
    )
    # Non-default thresholds change a Dice run's sha too.
    assert _compute_run_sha(**_SHA_BASE, prune_below=0.1) != _compute_run_sha(
        **_SHA_BASE
    )


def test_cosine_run_sha_keys_metric_vocab_enrichment_and_weights():
    cos = dict(
        _SHA_BASE,
        metric=COSINE,
        domain_vocab="IPR_GAPFILLED",
        enrichment_version="v1",
        weights_sha="abc",
    )
    base = _compute_run_sha(**cos)
    assert base != _compute_run_sha(**_SHA_BASE)
    for key, value in (
        ("domain_vocab", "IPR_PROJECTED"),
        ("enrichment_version", "v2"),
        ("weights_sha", "abd"),
        ("prune_below", 0.1),
        ("min_projection_similarity", 0.2),
        ("idf_exponent", 0.5),
    ):
        assert _compute_run_sha(**dict(cos, **{key: value})) != base, key


def test_idf_exponent_one_keeps_the_pre_amendment_sha():
    cos = dict(
        _SHA_BASE, metric=COSINE, domain_vocab="IPR_GAPFILLED", enrichment_version="v1", weights_sha="abc"
    )
    assert _compute_run_sha(**cos, idf_exponent=1.0) == _compute_run_sha(**cos)
    # Dice ignores it entirely.
    assert _compute_run_sha(**_SHA_BASE, idf_exponent=0.5) == _compute_run_sha(**_SHA_BASE)

