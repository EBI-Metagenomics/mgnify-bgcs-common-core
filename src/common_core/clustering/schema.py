"""In-memory dataclasses for the input / output tarball contracts.

These mirror the on-disk layout described in the README and plan. Keep
field names stable — the Django importer reads them by name.

Schema versions:
  1 — composite Dice only; ``params.json`` has no ``schema_version``.
  2 — adds the similarity metric, its per-feature weight vectors
      (``domain_weights.npy`` / ``pair_weights.npy``) and the prune /
      projection thresholds. A v1 tarball reads as ``similarity_metric="dice"``
      with the v1 thresholds, so older runs reproduce unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import scipy.sparse as sp

SCHEMA_VERSION = 2


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
    # (n_ibgcs, 2) float64; aligned to ibgc_ids; normalised to [-10, 10].
    coords: "np.ndarray"
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
    device: str
    cuda_version: str = ""
    gpu_model: str = ""
