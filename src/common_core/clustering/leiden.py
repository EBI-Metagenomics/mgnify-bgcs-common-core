"""Strictly nested hierarchical Leiden partitioning.

Same algorithm as the portal's ``services.clustering.leiden`` — copied so
the HPC package has no Django imports. ``leidenalg`` is single-threaded;
within-level subgraph parallelism doesn't pay off because one or two big
communities dominate the wall clock.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import igraph as ig

log = logging.getLogger(__name__)


def run_hierarchical_leiden(
    graph: "ig.Graph",
    resolutions: tuple[float, ...] = (0.03, 0.08, 0.15, 0.25),
    *,
    seed: int = 42,
    min_community_size: int = 2,
) -> list[list[int]]:
    """Partition ``graph`` at each resolution top-down within each parent.

    Returns ``levels`` where ``levels[d][v]`` is the integer label for
    vertex ``v`` at depth ``d``. Strict nesting is guaranteed: two vertices
    sharing a label at depth d also share their label at every depth < d.
    """
    import leidenalg as la

    n = graph.vcount()
    if n == 0:
        return [[] for _ in resolutions]

    n_levels = len(resolutions)
    levels: list[list[int]] = [[0] * n for _ in range(n_levels)]
    next_label_at = [0] * n_levels

    def _emit_singleton_subtree(vertices: list[int], depth: int) -> None:
        for sublevel in range(depth, n_levels):
            label = next_label_at[sublevel]
            next_label_at[sublevel] += 1
            for v in vertices:
                levels[sublevel][v] = label

    def _partition(vertices: list[int], depth: int) -> None:
        if depth >= n_levels or not vertices:
            return
        sub = graph.subgraph(vertices)
        if sub.vcount() < min_community_size:
            _emit_singleton_subtree(vertices, depth)
            return
        weights = sub.es["weight"] if "weight" in sub.es.attributes() else None
        partition = la.find_partition(
            sub,
            la.CPMVertexPartition,
            weights=weights,
            resolution_parameter=resolutions[depth],
            seed=seed + depth,
        )
        sorted_communities = sorted(
            partition, key=lambda c: (-len(c), min(c) if c else 0)
        )
        for community_sub in sorted_communities:
            if not community_sub:
                continue
            members_global = [vertices[v] for v in community_sub]
            label = next_label_at[depth]
            next_label_at[depth] += 1
            for v in members_global:
                levels[depth][v] = label
            _partition(members_global, depth + 1)

    _partition(list(range(n)), depth=0)
    log.info(
        "run_hierarchical_leiden: %d vertices, %d levels (counts=%s)",
        n, n_levels, next_label_at,
    )
    return levels
