"""HPC-side BGC clustering: composite-Dice, KNN, hierarchical Leiden, UMAP.

The portal exports per-iBGC signature matrices and asks an HPC job to produce
the hierarchy, coordinates, and novelty scores. The ``bgc-cluster`` CLI is
the entry point; everything in this package is pure compute, no Django.

Parallelism is opt-in and bounded — most stages are single-threaded because
the libraries they call don't benefit from worker pools. See the plan in
``/Users/fragoso/.claude/plans/`` (or the README) for the rationale.
"""

from common_core.clustering.schema import (
    ClusteringInputs,
    ClusteringOutputs,
    RunParams,
)

__all__ = ["ClusteringInputs", "ClusteringOutputs", "RunParams"]
