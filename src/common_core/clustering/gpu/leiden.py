"""GPU hierarchical Leiden via cugraph.

cuGraph's Leiden is non-hierarchical; we recurse on subgraphs just like the
CPU path does, calling cugraph.leiden per partition level. This pays off
at N ≥ ~50k; below that the kernel-launch overhead can wash out the gain.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import igraph as ig

log = logging.getLogger(__name__)


def run_hierarchical_leiden_gpu(
    graph: "ig.Graph",
    resolutions: tuple[float, ...] = (0.03, 0.08, 0.15, 0.25),
    *,
    seed: int = 42,
    min_community_size: int = 2,
) -> list[list[int]]:
    """GPU-accelerated hierarchical Leiden.

    Same return shape as the CPU version. The graph stays in host memory
    (igraph); we move each subgraph to cuGraph as we go.
    """
    import cudf
    import cugraph

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

    def _to_cugraph(sub):
        edges = sub.get_edgelist()
        if not edges:
            return None
        weights = sub.es["weight"] if "weight" in sub.es.attributes() else [1.0] * sub.ecount()
        df = cudf.DataFrame(
            {
                "src": [int(a) for a, _ in edges],
                "dst": [int(b) for _, b in edges],
                "weight": list(weights),
            }
        )
        G = cugraph.Graph()
        G.from_cudf_edgelist(df, source="src", destination="dst", edge_attr="weight")
        return G

    def _partition(vertices: list[int], depth: int) -> None:
        if depth >= n_levels or not vertices:
            return
        sub = graph.subgraph(vertices)
        if sub.vcount() < min_community_size:
            _emit_singleton_subtree(vertices, depth)
            return
        G = _to_cugraph(sub)
        if G is None:
            _emit_singleton_subtree(vertices, depth)
            return
        parts, _ = cugraph.leiden(
            G, resolution=resolutions[depth], random_state=seed + depth,
        )
        parts_pd = parts.to_pandas()
        # vertex -> partition label (local to this subgraph)
        local_partitions: dict[int, list[int]] = {}
        for v_local, lbl in zip(parts_pd["vertex"].tolist(), parts_pd["partition"].tolist()):
            local_partitions.setdefault(int(lbl), []).append(int(v_local))

        sorted_communities = sorted(
            local_partitions.values(), key=lambda c: (-len(c), min(c) if c else 0),
        )
        for community_sub in sorted_communities:
            members_global = [vertices[v] for v in community_sub]
            label = next_label_at[depth]
            next_label_at[depth] += 1
            for v in members_global:
                levels[depth][v] = label
            _partition(members_global, depth + 1)

    _partition(list(range(n)), depth=0)
    log.info(
        "run_hierarchical_leiden_gpu: %d vertices, %d levels (counts=%s)",
        n, n_levels, next_label_at,
    )
    return levels
