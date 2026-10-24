"""Input / output tarball contract tests (schema v1 + v2).

Locks:

1. A **v1** input tarball — exactly the file set and ``params.json`` the
   v1 exporter wrote — still reads, as composite Dice with the v1 thresholds,
   and ``bgc-cluster run`` reproduces the Dice run (same sha as before v2).
2. A **v2** tarball round-trips the metric, vocab, enrichment version,
   thresholds and both weight vectors; the output echoes the weights with
   their vocabularies.
3. Tarballs a given common-core can't score faithfully are rejected on read.
"""

from __future__ import annotations

import json
import tarfile
from pathlib import Path

import numpy as np
import pytest

scipy_sparse = pytest.importorskip("scipy.sparse")
pytest.importorskip("pyarrow")

from common_core.clustering.io import (
    read_inputs_tarball,
    read_outputs_tarball,
    write_inputs_tarball,
)
from common_core.clustering.schema import (
    SCHEMA_VERSION,
    ClusteringInputs,
    RunParams,
)

COSINE = "cosine_idf_enrich"
V1_PARAMS = {
    "domain_sources": ["PFAM", "NCBIFAM"],
    "score_weights": [0.5, 0.5],
    "leiden_resolutions": [0.03, 0.08, 0.15, 0.25],
    "knn_k": None,
    "seed": 42,
    "run_tag": "v1-fixture",
    "exporter_versions": {"common-core": "1.9.1"},
}


def _matrices(seed: int = 0, n: int = 40, n_partial: int = 5):
    """Three planted families (shared core domains + noise) and some partials."""
    rng = np.random.default_rng(seed)
    d_dom, d_pair = 30, 45
    dom = (rng.random((n + n_partial, d_dom)) > 0.9).astype(np.uint8)
    pair = (rng.random((n + n_partial, d_pair)) > 0.93).astype(np.uint8)
    for i in range(n + n_partial):
        fam = i % 3
        dom[i, fam * 6 : fam * 6 + 5] = 1
        pair[i, fam * 8 : fam * 8 + 4] = 1
    M_dom = scipy_sparse.csr_matrix(dom)
    M_pair = scipy_sparse.csr_matrix(pair)
    accs = np.array([f"IPR{i:06d}" for i in range(d_dom)], dtype=object)
    pvocab = np.empty(d_pair, dtype=object)
    for j in range(d_pair):
        pvocab[j] = (accs[j % d_dom], accs[(j * 7 + 1) % d_dom])
    return M_dom[:n], M_pair[:n], M_dom[n:], M_pair[n:], accs, pvocab


def _inputs(params: RunParams, *, with_weights: bool) -> ClusteringInputs:
    M_dom, M_pair, P_dom, P_pair, accs, pvocab = _matrices()
    n = M_dom.shape[0]
    w_dom = w_pair = None
    if with_weights:
        rng = np.random.default_rng(1)
        w_dom = rng.uniform(0.05, 2.0, M_dom.shape[1]).astype(np.float32)
        w_pair = rng.uniform(0.05, 2.0, M_pair.shape[1]).astype(np.float32)
    return ClusteringInputs(
        M_domains=M_dom,
        M_pairs=M_pair,
        domain_accs=accs,
        pair_vocab=pvocab,
        ibgc_ids=np.arange(1000, 1000 + n, dtype=np.int64),
        partials_M_domains=P_dom,
        partials_M_pairs=P_pair,
        partials_ibgc_ids=np.arange(5000, 5000 + P_dom.shape[0], dtype=np.int64),
        validated_ibgc_ids=np.array([1000, 1004], dtype=np.int64),
        params=params,
        domain_weights=w_dom,
        pair_weights=w_pair,
    )


def _write_v1_tarball(path: Path) -> Path:
    """Write the v1 layout by hand: no weights, no v2 keys in params.json."""
    inputs = _inputs(RunParams(**V1_PARAMS), with_weights=False)
    write_inputs_tarball(path, inputs)
    # Rewrite params.json to the exact v1 payload.
    scratch = path.parent / "v1_unpacked"
    with tarfile.open(path, "r:gz") as tf:
        tf.extractall(scratch)
    (scratch / "params.json").write_text(json.dumps(V1_PARAMS))
    with tarfile.open(path, "w:gz") as tf:
        for f in sorted(scratch.iterdir()):
            tf.add(f, arcname=f.name)
    return path


def _v2_params(**over) -> RunParams:
    kw = dict(
        V1_PARAMS,
        similarity_metric=COSINE,
        domain_vocab="IPR_GAPFILLED",
        enrichment_version="v1",
        prune_below=0.08,
        min_projection_similarity=0.2,
    )
    kw.update(over)
    kw["score_weights"] = tuple(kw["score_weights"])
    kw["leiden_resolutions"] = tuple(kw["leiden_resolutions"])
    return RunParams(**kw)


# ── Input tarball ───────────────────────────────────────────────────────────


def test_v1_tarball_reads_as_dice(tmp_path):
    path = _write_v1_tarball(tmp_path / "v1.tar.gz")
    with tarfile.open(path, "r:gz") as tf:
        names = set(tf.getnames())
    assert "domain_weights.npy" not in names

    inputs = read_inputs_tarball(path)
    p = inputs.params
    assert p.schema_version == 1
    assert p.similarity_metric == "dice"
    assert (p.prune_below, p.min_projection_similarity) == (0.05, 0.1)
    assert p.domain_vocab == "" and p.enrichment_version == ""
    assert inputs.domain_weights is None and inputs.pair_weights is None


def test_v2_round_trip(tmp_path):
    src = _inputs(_v2_params(), with_weights=True)
    path = write_inputs_tarball(tmp_path / "v2.tar.gz", src)
    got = read_inputs_tarball(path)

    assert got.params == src.params
    assert got.params.schema_version == SCHEMA_VERSION == 2
    np.testing.assert_array_equal(got.domain_weights, src.domain_weights)
    np.testing.assert_array_equal(got.pair_weights, src.pair_weights)
    assert got.domain_weights.dtype == np.float32
    assert (got.M_domains != src.M_domains).nnz == 0
    assert list(got.pair_vocab) == list(src.pair_vocab)


def test_v2_dice_tarball_has_no_weight_files(tmp_path):
    src = _inputs(_v2_params(similarity_metric="dice"), with_weights=False)
    path = write_inputs_tarball(tmp_path / "v2dice.tar.gz", src)
    with tarfile.open(path, "r:gz") as tf:
        assert not {"domain_weights.npy", "pair_weights.npy"} & set(tf.getnames())
    assert read_inputs_tarball(path).params.similarity_metric == "dice"


def test_cosine_without_weights_is_rejected(tmp_path):
    src = _inputs(_v2_params(), with_weights=False)
    path = write_inputs_tarball(tmp_path / "bad.tar.gz", src)
    with pytest.raises(ValueError, match="needs domain_weights"):
        read_inputs_tarball(path)


def test_misaligned_weights_are_rejected(tmp_path):
    src = _inputs(_v2_params(), with_weights=True)
    src.pair_weights = src.pair_weights[:-1]
    path = write_inputs_tarball(tmp_path / "bad.tar.gz", src)
    with pytest.raises(ValueError, match="pair_weights has"):
        read_inputs_tarball(path)


def test_newer_schema_is_rejected(tmp_path):
    src = _inputs(_v2_params(schema_version=SCHEMA_VERSION + 1), with_weights=True)
    path = write_inputs_tarball(tmp_path / "future.tar.gz", src)
    with pytest.raises(ValueError, match="newer than this"):
        read_inputs_tarball(path)


# ── bgc-cluster run end-to-end ──────────────────────────────────────────────


def _run_cli(inp: Path, out: Path) -> dict:
    pytest.importorskip("igraph")
    pytest.importorskip("leidenalg")
    from common_core.clustering.cli import main

    assert main(["run", "--input", str(inp), "--output", str(out)]) == 0
    return read_outputs_tarball(out)


def test_cli_v1_tarball_reproduces_dice_run(tmp_path):
    from common_core.clustering.cli import _compute_run_sha
    from common_core.clustering.scoring import compute_novelty_against_validated

    inp = _write_v1_tarball(tmp_path / "v1.tar.gz")
    out = _run_cli(inp, tmp_path / "out_v1.tar.gz")
    inputs = read_inputs_tarball(inp)

    # sha is the v1 payload (no metric / threshold fields).
    n = len(inputs.ibgc_ids)
    assert out["run"]["sha256"] == _compute_run_sha(
        sources=("PFAM", "NCBIFAM"),
        weights=(0.5, 0.5),
        knn_k=5,
        leiden_resolutions=(0.03, 0.08, 0.15, 0.25),
        seed=42,
        ibgc_etag=f"{n}:{int(inputs.ibgc_ids[-1])}",
        domain_etag=f"{inputs.M_domains.nnz}:{inputs.M_domains.shape[1]}",
    )
    assert out["run"]["params"]["similarity_metric"] == "dice"
    assert out["domain_weights"] is None and out["domain_accs"] is None

    rows = [int(i) - 1000 for i in out["scores"]["ibgc_id"]]
    want = compute_novelty_against_validated(
        inputs.M_domains, inputs.M_pairs, [0, 4], weights=(0.5, 0.5)
    )
    np.testing.assert_allclose(out["scores"]["novelty_score"], want[rows], atol=1e-6)


def test_cli_v2_cosine_run_echoes_weights(tmp_path):
    from common_core.clustering.scoring import compute_novelty_against_validated

    src = _inputs(_v2_params(), with_weights=True)
    inp = write_inputs_tarball(tmp_path / "v2.tar.gz", src)
    out = _run_cli(inp, tmp_path / "out_v2.tar.gz")

    params = out["run"]["params"]
    assert params["similarity_metric"] == COSINE
    assert params["enrichment_version"] == "v1"
    assert params["prune_below"] == pytest.approx(0.08)
    np.testing.assert_array_equal(out["domain_weights"], src.domain_weights)
    np.testing.assert_array_equal(out["pair_weights"], src.pair_weights)
    assert list(out["domain_accs"]) == list(src.domain_accs)
    assert list(out["pair_vocab"]) == list(src.pair_vocab)

    want = compute_novelty_against_validated(
        src.M_domains,
        src.M_pairs,
        [0, 4],
        weights=(0.5, 0.5),
        metric=COSINE,
        domain_weights=src.domain_weights,
        pair_weights=src.pair_weights,
    )
    np.testing.assert_allclose(out["scores"]["novelty_score"], want, atol=1e-6)
    # Validated iBGCs aren't novel under cosine either.
    assert out["scores"]["novelty_score"][0] == pytest.approx(0.0, abs=1e-6)

    # Weights feed the sha: same inputs, different weights → different run.
    src.domain_weights = src.domain_weights * 1.5
    inp2 = write_inputs_tarball(tmp_path / "v2b.tar.gz", src)
    out2 = _run_cli(inp2, tmp_path / "out_v2b.tar.gz")
    assert out2["run"]["sha256"] != out["run"]["sha256"]

    # Partials are projected (planted families → every partial has neighbours).
    assert len(out["partial_assignments"]["ibgc_id"]) == src.partials_ibgc_ids.size
