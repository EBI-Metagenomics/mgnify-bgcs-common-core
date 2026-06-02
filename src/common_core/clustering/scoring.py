"""Novelty + domain-novelty scoring, plus medoid / hierarchy path helpers.

Pure NumPy / SciPy. Same numerics as the portal's
``services.clustering.ibgc_scoring`` and ``services.clustering.paths``.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import scipy.sparse as sp

log = logging.getLogger(__name__)


CLUSTER_SEGMENT = "cluster"


# ── Novelty + domain novelty ────────────────────────────────────────────────


def compute_novelty_against_validated(
    M_domains: "sp.csr_matrix",
    M_pairs: "sp.csr_matrix",
    validated_cols: list[int],
    *,
    weights: tuple[float, float] = (0.5, 0.5),
) -> "np.ndarray":
    """Per-row ``1 − max(composite_dice_sim_to_validated)``.

    Unlike :func:`compute_novelty_array`, this does **not** reuse the
    clustering similarity matrix. That matrix has its diagonal zeroed and is
    pruned (``prune_below``) so KNN / Leiden see clean edges — both of which
    corrupt novelty:

    * the zeroed diagonal drops a validated iBGC's self-match (sim 1.0), so a
      validated row would be scored against *other* validated iBGCs and come
      out ``> 0`` instead of ``0``;
    * pruning truncates near-threshold similarities to 0, inflating the
      novelty of non-validated rows whose nearest validated neighbour sits
      just under the prune cutoff.

    Here the composite Dice block (rows × validated-columns) is recomputed
    unpruned and with the diagonal intact, so a validated iBGC's self-Dice of
    1.0 yields novelty 0 *by construction*. Validated rows are additionally
    clamped to 0.0 to cover the degenerate empty-signature case (self-Dice
    undefined → 0). Returns all-NaN when there are no validated iBGCs.
    """
    import numpy as np

    n_rows = M_domains.shape[0]
    if not validated_cols:
        return np.full(n_rows, np.nan, dtype=np.float32)

    w_d, w_a = weights
    total = float(w_d) + float(w_a)
    if total <= 0:
        raise ValueError(f"weights must sum > 0, got {weights}")
    w_d, w_a = w_d / total, w_a / total

    val = np.asarray(sorted(validated_cols), dtype=np.int64)

    block: "np.ndarray | None" = None
    for M, w in ((M_domains, w_d), (M_pairs, w_a)):
        if w <= 0:
            continue
        Mf = M.astype(np.float32)
        sizes = np.asarray(Mf.sum(axis=1), dtype=np.float32).ravel()
        # rows × |validated| intersection counts (validated set is small).
        inter = np.asarray((Mf @ Mf[val].T).todense(), dtype=np.float32)
        denom = sizes[:, None] + sizes[val][None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            dice = np.where(denom > 0.0, 2.0 * inter / denom, 0.0).astype(np.float32)
        block = w * dice if block is None else block + w * dice

    max_sim = block.max(axis=1)
    novelty = (1.0 - max_sim).astype(np.float32)
    novelty[val] = 0.0
    return novelty


def compute_novelty_array(
    sim: "sp.csr_matrix",
    validated_cols: list[int],
) -> "np.ndarray":
    """Per-row ``1 − max(sim_to_validated)``; NaN when there are no validated iBGCs.

    .. deprecated::
        Reuses the clustering similarity matrix (diagonal zeroed + pruned),
        which mis-scores validated rows and near-threshold non-validated rows.
        Use :func:`compute_novelty_against_validated`. Retained only for
        backward-compatible callers / tests.
    """
    import numpy as np

    n_rows = sim.shape[0]
    if not validated_cols:
        return np.full(n_rows, np.nan, dtype=np.float32)
    sim_to_validated = sim[:, validated_cols]
    max_sim = np.asarray(sim_to_validated.max(axis=1).todense()).reshape(-1)
    return (1.0 - max_sim).astype(np.float32)


def compute_domain_novelty_array(
    M_domains: "sp.csr_matrix",
    leaf_paths: list[str],
) -> "np.ndarray":
    """Per-row fraction of domains unique within the row's leaf GCF.

    NaN for singleton leaf GCFs, empty leaf paths, and rows with zero
    domains — caller persists those as NULL.
    """
    import numpy as np

    n_rows = M_domains.shape[0]
    if len(leaf_paths) != n_rows:
        raise ValueError(
            f"leaf_paths length {len(leaf_paths)} ≠ M_domains row count {n_rows}"
        )

    path_to_rows: dict[str, list[int]] = defaultdict(list)
    for i, p in enumerate(leaf_paths):
        path_to_rows[p].append(i)

    out = np.full(n_rows, np.nan, dtype=np.float32)
    for path, rows in path_to_rows.items():
        if not path or len(rows) < 2:
            continue
        sub = M_domains[rows].tocsr()
        col_sums = np.asarray(sub.sum(axis=0)).reshape(-1)
        for local_i, abs_row in enumerate(rows):
            start = sub.indptr[local_i]
            end = sub.indptr[local_i + 1]
            if start == end:
                continue
            domain_cols = sub.indices[start:end]
            n_domains = int(end - start)
            n_unique = int((col_sums[domain_cols] == 1).sum())
            out[abs_row] = n_unique / n_domains
    return out


# ── Medoid pick ─────────────────────────────────────────────────────────────


def pick_medoid(member_indices: list[int], sim: "sp.csr_matrix") -> int:
    """Member with the highest summed similarity to other members.

    Ties broken deterministically (smallest member index wins).
    """
    import numpy as np

    if not member_indices:
        raise ValueError("member_indices must be non-empty")
    if len(member_indices) == 1:
        return member_indices[0]

    sim_csr = sim.tocsr(copy=False)
    members_set = set(member_indices)
    sums = np.zeros(len(member_indices), dtype=np.float64)
    for i, idx in enumerate(member_indices):
        start = sim_csr.indptr[idx]
        end = sim_csr.indptr[idx + 1]
        cols = sim_csr.indices[start:end]
        vals = sim_csr.data[start:end]
        s = 0.0
        for c, v in zip(cols.tolist(), vals.tolist()):
            if c == idx or c not in members_set:
                continue
            s += float(v)
        sums[i] = s

    best_score = float(sums.max())
    candidates = [
        member_indices[i] for i in range(len(member_indices)) if sums[i] == best_score
    ]
    return min(candidates)


# ── Hierarchy paths ─────────────────────────────────────────────────────────


def build_ltree_paths(
    levels: list[list[int]],
    ibgc_ids,
) -> tuple[dict[int, str], list[dict]]:
    """Map per-level labels to ltree paths and produce GCF node records.

    Returns:
        leaf_path_per_row: ``{ibgc_id: leaf_family_path}``
        nodes: list of dicts ready for the gcf_nodes parquet table; each has
            family_path, parent_path, level, member_indices.
    """
    import numpy as np

    n_levels = len(levels)
    n = len(ibgc_ids) if hasattr(ibgc_ids, "__len__") else 0
    if n == 0 or n_levels == 0:
        return {}, []

    ibgc_ids_arr = ibgc_ids if isinstance(ibgc_ids, np.ndarray) else np.asarray(list(ibgc_ids))

    label_to_vertices: dict[tuple[int, int], list[int]] = defaultdict(list)
    for d in range(n_levels):
        for v, lbl in enumerate(levels[d]):
            label_to_vertices[(d, lbl)].append(v)

    label_parent_path: dict[tuple[int, int], str] = {}
    label_path: dict[tuple[int, int], str] = {}

    level0 = sorted(
        {lbl for lbl in levels[0]},
        key=lambda lbl: (
            -len(label_to_vertices[(0, lbl)]),
            min(label_to_vertices[(0, lbl)]),
        ),
    )
    for idx, lbl in enumerate(level0):
        path = f"{CLUSTER_SEGMENT}.{idx:04d}"
        label_parent_path[(0, lbl)] = ""
        label_path[(0, lbl)] = path

    for d in range(1, n_levels):
        parent_to_children: dict[int, set[int]] = defaultdict(set)
        for v in range(n):
            parent_to_children[levels[d - 1][v]].add(levels[d][v])
        for parent_lbl, child_lbls in parent_to_children.items():
            parent_path = label_path[(d - 1, parent_lbl)]
            ordered = sorted(
                child_lbls,
                key=lambda lbl: (
                    -len(label_to_vertices[(d, lbl)]),
                    min(label_to_vertices[(d, lbl)]),
                ),
            )
            for idx, lbl in enumerate(ordered):
                path = f"{parent_path}.{idx:04d}"
                label_parent_path[(d, lbl)] = parent_path
                label_path[(d, lbl)] = path

    nodes: list[dict] = []
    for (d, lbl), members_v in sorted(label_to_vertices.items()):
        nodes.append(
            {
                "family_path": label_path[(d, lbl)],
                "parent_path": label_parent_path[(d, lbl)],
                "level": int(d),
                "member_indices": list(members_v),
            }
        )

    leaf_level = n_levels - 1
    paths_per_row: dict[int, str] = {}
    for v in range(n):
        leaf_lbl = levels[leaf_level][v]
        paths_per_row[int(ibgc_ids_arr[v])] = label_path[(leaf_level, leaf_lbl)]

    log.info(
        "build_ltree_paths: %d nodes across %d levels", len(nodes), n_levels,
    )
    return paths_per_row, nodes


def annotate_gcf_nodes(
    nodes: list[dict],
    sim: "sp.csr_matrix",
    ibgc_ids,
) -> list[dict]:
    """Add representative_ibgc_id, member_count, descendant_count to every node.

    Mutates and returns ``nodes`` in place.
    """
    import numpy as np

    ibgc_ids_arr = ibgc_ids if isinstance(ibgc_ids, np.ndarray) else np.asarray(list(ibgc_ids))

    parent_children_count: dict[str, int] = defaultdict(int)
    for node in nodes:
        if node.get("parent_path"):
            parent_children_count[node["parent_path"]] += 1

    for node in nodes:
        members = node["member_indices"]
        node["member_count"] = len(members)
        node["descendant_count"] = parent_children_count.get(node["family_path"], 0)
        medoid_v = pick_medoid(members, sim)
        node["representative_ibgc_id"] = int(ibgc_ids_arr[medoid_v])

    # Drop the bulky per-node member list — it isn't shipped in the parquet.
    for node in nodes:
        node.pop("member_indices", None)
    return nodes
