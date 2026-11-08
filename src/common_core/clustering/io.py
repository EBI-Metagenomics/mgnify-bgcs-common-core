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

"""Tarball <-> in-memory packing for the HPC clustering inputs / outputs.

The input tarball is produced by the portal's ``export_clustering_inputs``
management command. The output tarball is consumed by ``import_clustering_results``.

File layout (input):
    M_domains.npz                — sparse CSR uint8, primary iBGCs × domain vocab
    M_pairs.npz                  — sparse CSR uint8, primary iBGCs × pair vocab
    domain_accs.npy              — object array of accession strings
    pair_vocab.npy               — object array of (acc_a, acc_b) tuples
    ibgc_ids.npy                  — int64, primary iBGC pks (row-aligned)
    partials_M_domains.npz       — partial iBGCs × same domain vocab (may be empty)
    partials_M_pairs.npz         — partial iBGCs × same pair vocab (may be empty)
    partials_ibgc_ids.npy         — int64, partial iBGC pks
    validated.npy                — int64, subset of ibgc_ids (validated iBGCs)
    params.json                  — RunParams as JSON
    domain_weights.npy           — v2, float32 per domain_accs column
                                   (only when similarity_metric needs weights)
    pair_weights.npy             — v2, float32 per pair_vocab column (ditto)

A v1 tarball (no ``schema_version`` in params.json) reads as composite Dice.

File layout (output):
    clustering_run.json          — sha256, params, counts, library versions, device
    hierarchy.parquet            — ibgc_id, level_0, level_1, …, leaf_path
    gcf_nodes.parquet            — family_path, parent_path, level,
                                   member_count, descendant_count,
                                   representative_ibgc_id
    scores.parquet               — ibgc_id, novelty_score, domain_novelty
    partial_assignments.parquet  — ibgc_id, leaf_path, novelty_score,
                                   domain_novelty
    domain_weights.npy,          — v2, echoed from the input together with the
    pair_weights.npy,              vocabularies they are aligned to, so the
    domain_accs.npy,               portal can persist the run's exact weights
    pair_vocab.npy                 (only when the input carried weights)

Schema v3 dropped the UMAP layout. v1/v2 output tarballs still read: their
``coords.parquet`` and the ``umap_x`` / ``umap_y`` partial columns are
ignored, so the result has the same shape for every schema version.
"""

from __future__ import annotations

import json
import logging
import tarfile
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

from common_core.clustering.schema import (
    LEGACY_OUTPUT_MEMBERS,
    LEGACY_PARTIAL_COLUMNS,
    PARTIAL_ASSIGNMENT_COLUMNS,
    SCHEMA_VERSION,
    ClusteringInputs,
    ClusteringOutputs,
    RunParams,
)

log = logging.getLogger(__name__)

_WEIGHT_FILES = ("domain_weights.npy", "pair_weights.npy")


# ── Input read/write ─────────────────────────────────────────────────────────


def write_inputs_tarball(out_path: Path, inputs: ClusteringInputs) -> Path:
    """Write a clustering-input tarball to ``out_path``.

    Used by the portal exporter and by tests. The HPC CLI doesn't call this.
    """
    import numpy as np
    import scipy.sparse as sp

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as scratch:
        scratch = Path(scratch)
        sp.save_npz(scratch / "M_domains.npz", inputs.M_domains)
        sp.save_npz(scratch / "M_pairs.npz", inputs.M_pairs)
        np.save(scratch / "domain_accs.npy", np.asarray(inputs.domain_accs, dtype=object))
        np.save(scratch / "pair_vocab.npy", np.asarray(inputs.pair_vocab, dtype=object))
        np.save(scratch / "ibgc_ids.npy", np.asarray(inputs.ibgc_ids, dtype=np.int64))
        sp.save_npz(scratch / "partials_M_domains.npz", inputs.partials_M_domains)
        sp.save_npz(scratch / "partials_M_pairs.npz", inputs.partials_M_pairs)
        np.save(
            scratch / "partials_ibgc_ids.npy",
            np.asarray(inputs.partials_ibgc_ids, dtype=np.int64),
        )
        np.save(scratch / "validated.npy", np.asarray(inputs.validated_ibgc_ids, dtype=np.int64))
        (scratch / "params.json").write_text(json.dumps(_params_as_jsonable(inputs.params)))
        weight_files = _save_weights(scratch, inputs)

        with tarfile.open(out_path, "w:gz") as tf:
            for name in (
                "M_domains.npz",
                "M_pairs.npz",
                "domain_accs.npy",
                "pair_vocab.npy",
                "ibgc_ids.npy",
                "partials_M_domains.npz",
                "partials_M_pairs.npz",
                "partials_ibgc_ids.npy",
                "validated.npy",
                "params.json",
                *weight_files,
            ):
                tf.add(scratch / name, arcname=name)
    log.info("Wrote clustering input tarball: %s", out_path)
    return out_path


def read_inputs_tarball(path: Path) -> ClusteringInputs:
    """Open a clustering-input tarball and load its contents into memory."""
    import numpy as np
    import scipy.sparse as sp

    path = Path(path)
    with tempfile.TemporaryDirectory() as scratch:
        scratch = Path(scratch)
        with tarfile.open(path, "r:gz") as tf:
            tf.extractall(scratch, filter="data")

        params = _params_from_jsonable(json.loads((scratch / "params.json").read_text()))
        domain_weights, pair_weights = _load_weights(scratch)
        inputs = ClusteringInputs(
            M_domains=sp.load_npz(scratch / "M_domains.npz"),
            M_pairs=sp.load_npz(scratch / "M_pairs.npz"),
            domain_accs=np.load(scratch / "domain_accs.npy", allow_pickle=True),
            pair_vocab=np.load(scratch / "pair_vocab.npy", allow_pickle=True),
            ibgc_ids=np.load(scratch / "ibgc_ids.npy"),
            partials_M_domains=sp.load_npz(scratch / "partials_M_domains.npz"),
            partials_M_pairs=sp.load_npz(scratch / "partials_M_pairs.npz"),
            partials_ibgc_ids=np.load(scratch / "partials_ibgc_ids.npy"),
            validated_ibgc_ids=np.load(scratch / "validated.npy"),
            params=params,
            domain_weights=domain_weights,
            pair_weights=pair_weights,
        )
    validate_inputs(inputs)
    return inputs


def validate_inputs(inputs: ClusteringInputs) -> None:
    """Reject a tarball this version can't score faithfully.

    Raises ``ValueError`` for a newer schema, an unknown metric, or weights
    missing / misaligned for a weighted metric.
    """
    from common_core.clustering.similarity import check_metric

    p = inputs.params
    if p.schema_version > SCHEMA_VERSION:
        raise ValueError(
            f"input tarball schema v{p.schema_version} is newer than this "
            f"common-core (v{SCHEMA_VERSION}); upgrade mgnify-bgcs-common-core"
        )
    check_metric(
        p.similarity_metric,
        inputs.M_domains,
        inputs.M_pairs,
        inputs.domain_weights,
        inputs.pair_weights,
    )


# ── Output read/write ────────────────────────────────────────────────────────


def write_outputs_tarball(
    out_path: Path,
    inputs: ClusteringInputs,
    outputs: ClusteringOutputs,
) -> Path:
    """Write a clustering-output tarball to ``out_path``."""
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as scratch:
        scratch = Path(scratch)

        # clustering_run.json — everything the importer needs about the run
        run_json = {
            "sha256": outputs.sha256,
            "n_ibgcs": int(inputs.M_domains.shape[0]),
            "n_levels": len(outputs.levels),
            "n_root_communities": outputs.n_root_communities,
            "n_leaf_communities": outputs.n_leaf_communities,
            "library_versions": outputs.library_versions,
            "device": outputs.device,
            "params": _params_as_jsonable(inputs.params),
        }
        (scratch / "clustering_run.json").write_text(json.dumps(run_json, indent=2))

        # hierarchy.parquet — one row per primary iBGC
        n_levels = len(outputs.levels)
        hierarchy_cols: dict[str, list] = {
            "ibgc_id": [int(x) for x in inputs.ibgc_ids.tolist()],
            "leaf_path": list(outputs.leaf_paths),
        }
        for d in range(n_levels):
            hierarchy_cols[f"level_{d}"] = [int(x) for x in outputs.levels[d]]
        pq.write_table(pa.table(hierarchy_cols), scratch / "hierarchy.parquet")

        # gcf_nodes.parquet — one row per node in the tree
        gcf_keys = [
            "family_path",
            "parent_path",
            "level",
            "member_count",
            "descendant_count",
            "representative_ibgc_id",
        ]
        gcf_cols = {key: [node[key] for node in outputs.gcf_nodes] for key in gcf_keys}
        pq.write_table(pa.table(gcf_cols), scratch / "gcf_nodes.parquet")

        # scores.parquet — novelty + domain_novelty; NaN → null in parquet
        nv = np.asarray(outputs.novelty_score, dtype=np.float64)
        dn = np.asarray(outputs.domain_novelty, dtype=np.float64)
        scores_table = pa.table(
            {
                "ibgc_id": pa.array(
                    [int(x) for x in inputs.ibgc_ids.tolist()], type=pa.int64()
                ),
                "novelty_score": pa.array(_nan_to_none(nv), type=pa.float64()),
                "domain_novelty": pa.array(_nan_to_none(dn), type=pa.float64()),
            }
        )
        pq.write_table(scores_table, scratch / "scores.parquet")

        # partial_assignments.parquet — variable length
        partial_keys = list(PARTIAL_ASSIGNMENT_COLUMNS)
        if outputs.partial_assignments:
            partial_cols = {
                key: [row.get(key) for row in outputs.partial_assignments]
                for key in partial_keys
            }
        else:
            partial_cols = {key: [] for key in partial_keys}
        pq.write_table(pa.table(partial_cols), scratch / "partial_assignments.parquet")

        # Weights + the vocabularies they're aligned to (v2, weighted metrics).
        weight_files = _save_weights(scratch, inputs)
        if weight_files:
            np.save(scratch / "domain_accs.npy", np.asarray(inputs.domain_accs, dtype=object))
            np.save(scratch / "pair_vocab.npy", np.asarray(inputs.pair_vocab, dtype=object))
            weight_files += ["domain_accs.npy", "pair_vocab.npy"]

        members = [
            "clustering_run.json",
            "hierarchy.parquet",
            "gcf_nodes.parquet",
            "scores.parquet",
            "partial_assignments.parquet",
            *weight_files,
        ]
        with tarfile.open(out_path, "w:gz") as tf:
            for name in members:
                tf.add(scratch / name, arcname=name)

    log.info("Wrote clustering output tarball: %s", out_path)
    return out_path


def read_outputs_tarball(path: Path) -> dict:
    """Open a clustering-output tarball.

    Returns a dict with raw parquet tables and the clustering_run.json payload.
    The Django importer turns these into DB writes. ``domain_weights`` /
    ``pair_weights`` / ``domain_accs`` / ``pair_vocab`` are ``None`` unless
    the run used a weighted metric.

    Accepts v1/v2 tarballs too: a legacy ``coords.parquet`` is never read and
    legacy ``umap_x`` / ``umap_y`` partial columns are dropped.
    """
    import numpy as np
    import pyarrow.parquet as pq

    path = Path(path)
    with tempfile.TemporaryDirectory() as scratch:
        scratch = Path(scratch)
        with tarfile.open(path, "r:gz") as tf:
            tf.extractall(
                scratch,
                members=[m for m in tf.getmembers() if m.name not in LEGACY_OUTPUT_MEMBERS],
                filter="data",
            )
        run = json.loads((scratch / "clustering_run.json").read_text())
        domain_weights, pair_weights = _load_weights(scratch)
        vocab: dict[str, object] = {}
        for name in ("domain_accs", "pair_vocab"):
            f = scratch / f"{name}.npy"
            vocab[name] = np.load(f, allow_pickle=True) if f.exists() else None
        return {
            "run": run,
            "hierarchy": pq.read_table(scratch / "hierarchy.parquet").to_pydict(),
            "gcf_nodes": pq.read_table(scratch / "gcf_nodes.parquet").to_pydict(),
            "scores": pq.read_table(scratch / "scores.parquet").to_pydict(),
            "partial_assignments": _read_partials(scratch / "partial_assignments.parquet"),
            "domain_weights": domain_weights,
            "pair_weights": pair_weights,
            **vocab,
        }


# ── Helpers ─────────────────────────────────────────────────────────────────


def _read_partials(path: Path) -> dict:
    """Read partial_assignments.parquet without the legacy layout columns."""
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    keep = [c for c in table.column_names if c not in LEGACY_PARTIAL_COLUMNS]
    return table.select(keep).to_pydict()


def _params_as_jsonable(p: RunParams) -> dict:
    d = asdict(p)
    d["score_weights"] = list(p.score_weights)
    d["leiden_resolutions"] = list(p.leiden_resolutions)
    return d


def _params_from_jsonable(d: dict) -> RunParams:
    # A v1 params.json has none of the v2 keys; the RunParams defaults for
    # them are the v1 behaviour (Dice, prune 0.05, projection floor 0.1).
    return RunParams(
        domain_sources=list(d.get("domain_sources", [])),
        score_weights=tuple(d.get("score_weights", (0.5, 0.5))),
        leiden_resolutions=tuple(d.get("leiden_resolutions", (0.03, 0.08, 0.15, 0.25))),
        knn_k=d.get("knn_k"),
        seed=int(d.get("seed", 42)),
        run_tag=str(d.get("run_tag", "")),
        exporter_versions=dict(d.get("exporter_versions", {})),
        similarity_metric=str(d.get("similarity_metric", "dice")),
        domain_vocab=str(d.get("domain_vocab", "")),
        enrichment_version=str(d.get("enrichment_version", "")),
        prune_below=float(d.get("prune_below", 0.05)),
        min_projection_similarity=float(d.get("min_projection_similarity", 0.1)),
        idf_exponent=float(d.get("idf_exponent", 1.0)),
        schema_version=int(d.get("schema_version", 1)),
    )


def _save_weights(scratch: Path, inputs: ClusteringInputs) -> list[str]:
    """Write whichever weight vectors are set; return the file names written."""
    import numpy as np

    written: list[str] = []
    for name, w in zip(_WEIGHT_FILES, (inputs.domain_weights, inputs.pair_weights)):
        if w is not None:
            np.save(scratch / name, np.asarray(w, dtype=np.float32))
            written.append(name)
    return written


def _load_weights(scratch: Path):
    import numpy as np

    return tuple(
        np.load(scratch / name) if (scratch / name).exists() else None
        for name in _WEIGHT_FILES
    )


def _nan_to_none(arr) -> list:
    """Convert a NumPy float array to a Python list with NaN → None.

    Parquet uses null for missing values; the importer writes NULL on null.
    """
    import math

    return [None if math.isnan(float(x)) else float(x) for x in arr.tolist()]
