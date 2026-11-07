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

"""In-memory dataclasses for the input / output tarball contracts.

These mirror the on-disk layout described in the README and plan. Keep
field names stable — the Django importer reads them by name.

Schema versions:
  1 — composite Dice only; ``params.json`` has no ``schema_version``.
  2 — adds the similarity metric, its per-feature weight vectors
      (``domain_weights.npy`` / ``pair_weights.npy``) and the prune /
      projection thresholds. A v1 tarball reads as ``similarity_metric="dice"``
      with the v1 thresholds, so older runs reproduce unchanged.
      ``idf_exponent`` (ADR 0003 amendment) is a later, optional v2 key; it
      reads as 1.0 when absent. The weight vectors already carry it, so it is
      provenance (and part of the run sha), not an input to the similarity.
  3 — drops the 2D layout: no ``coords.parquet`` and no ``umap_x`` /
      ``umap_y`` in ``partial_assignments.parquet``; the run JSON no longer
      carries GPU fields. Inputs are unchanged from v2. v1/v2 *output*
      tarballs still read: their coordinates are ignored.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import scipy.sparse as sp

SCHEMA_VERSION = 3

# Columns of partial_assignments.parquet (v3). v1/v2 tarballs also carry
# umap_x / umap_y, which readers drop.
PARTIAL_ASSIGNMENT_COLUMNS: tuple[str, ...] = (
    "ibgc_id",
    "leaf_path",
    "novelty_score",
    "domain_novelty",
)
# Output-tarball members written by schema <= 2 and ignored on read.
LEGACY_OUTPUT_MEMBERS: tuple[str, ...] = ("coords.parquet",)
LEGACY_PARTIAL_COLUMNS: tuple[str, ...] = ("umap_x", "umap_y")


@dataclass
class RunParams:
    """Pipeline parameters carried through the input/output tarballs."""

    domain_sources: list[str]
    score_weights: tuple[float, float]
    leiden_resolutions: tuple[float, ...]
    knn_k: int | None  # None → auto
    seed: int
    run_tag: str
    exporter_versions: dict[str, str] = field(default_factory=dict)
    # v2 — see ADR 0003. Defaults are the v1 behaviour.
    similarity_metric: str = "dice"  # "dice" | "cosine_idf_enrich"
    domain_vocab: str = ""  # portal ClusteringRun.domain_vocab, e.g. "IPR_GAPFILLED"
    enrichment_version: str = ""  # DomainEnrichment table version; "" for dice
    prune_below: float = 0.05  # similarity-graph edge prune before KNN
    min_projection_similarity: float = 0.1  # partials: min summed top-k similarity
    idf_exponent: float = 1.0  # w = idf ** idf_exponent × enrich; 1.0 for dice
    schema_version: int = SCHEMA_VERSION


@dataclass
class ClusteringInputs:
    """The contents of the input tarball after unpacking."""

    M_domains: "sp.csr_matrix"
    M_pairs: "sp.csr_matrix"
    domain_accs: "np.ndarray"
    pair_vocab: "np.ndarray"
    ibgc_ids: "np.ndarray"

    # Partials share the column vocab with the primaries (same domain_accs /
    # pair_vocab). Row count may be zero if there are no partials.
    partials_M_domains: "sp.csr_matrix"
    partials_M_pairs: "sp.csr_matrix"
    partials_ibgc_ids: "np.ndarray"

    # Subset of ibgc_ids whose source DashboardBgcs include is_validated=True.
    validated_ibgc_ids: "np.ndarray"

    params: RunParams

    # v2: per-feature weights for ``similarity_metric="cosine_idf_enrich"``,
    # aligned to domain_accs / pair_vocab. None for Dice runs.
    domain_weights: "np.ndarray | None" = None
    pair_weights: "np.ndarray | None" = None


@dataclass
class ClusteringOutputs:
    """What the HPC job emits in the output tarball."""

    # Per-row leaf path; aligned to ibgc_ids.
    leaf_paths: list[str]
    # Per-level integer labels: levels[d][v] is the label for vertex v at depth d.
    levels: list[list[int]]
    # Per-row novelty_score / domain_novelty; NaN means "leave NULL in DB".
    novelty_score: "np.ndarray"
    domain_novelty: "np.ndarray"

    # GCF tree nodes (roots, internals, leaves) — one record per node.
    gcf_nodes: list[dict]

    # Partial iBGC projections; one record per successfully projected partial.
    partial_assignments: list[dict]

    # Run metadata.
    sha256: str
    n_root_communities: int
    n_leaf_communities: int
    library_versions: dict[str, str]
    device: str = "cpu"
