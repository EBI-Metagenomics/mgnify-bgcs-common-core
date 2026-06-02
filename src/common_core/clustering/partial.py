"""Project partial iBGCs onto the primary clustering.

For each partial we compute composite-Dice against every primary (a single
sparse vec @ M_primaries.T), take top-K, derive:

  * umap_x / umap_y — similarity-weighted mean of the top-K primaries' coords
  * leaf_path — weighted-majority vote of the top-K primaries' leaf paths
  * novelty_score — 1 − max(sim_to_validated_primary); forced to 0 when the
    partial is itself validated (it matches itself, so it is not novel)
  * domain_novelty — fraction of the partial's domains absent from any primary
    of the assigned leaf GCF

Partials with summed top-K similarity below ``min_total_similarity`` are
left unprojected (returned as ``skipped``).
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import scipy.sparse as sp

log = logging.getLogger(__name__)


def project_partials(
    *,
    M_dom_pri: "sp.csr_matrix",
    M_pair_pri: "sp.csr_matrix",
    pri_ibgc_ids,
    pri_coords: "np.ndarray",
    pri_leaf_paths: list[str],
    pri_validated_rows: list[int],
    M_dom_q: "sp.csr_matrix",
    M_pair_q: "sp.csr_matrix",
    partial_ibgc_ids,
    validated_ibgc_ids=None,
    weights: tuple[float, float] = (0.5, 0.5),
    knn_k: int = 5,
    min_total_similarity: float = 0.1,
) -> tuple[list[dict], int]:
    """Return ``(assignments, n_skipped)``.

    ``assignments`` is a list of dicts ready for the partial_assignments
    parquet: ibgc_id, leaf_path, umap_x, umap_y, novelty_score, domain_novelty.
    """
    import numpy as np
    import scipy.sparse as sp

    from common_core.clustering.similarity import compute_composite_similarity

    n_primary = M_dom_pri.shape[0]
    n_partial = M_dom_q.shape[0]
    if n_partial == 0 or n_primary == 0:
        return [], n_partial

    partial_ibgc_ids_arr = np.asarray(partial_ibgc_ids, dtype=np.int64)
    validated_set = set(pri_validated_rows)
    # iBGC ids that are themselves validated — a validated partial is, by
    # definition, not novel (it matches itself), so its novelty is forced to
    # 0 regardless of similarity to the primary validated set.
    validated_id_set = (
        {int(x) for x in np.asarray(validated_ibgc_ids).tolist()}
        if validated_ibgc_ids is not None
        else set()
    )

    # Cache per-leaf primary column-sums on the primary domain matrix.
    leaf_to_rows: dict[str, list[int]] = defaultdict(list)
    for i, p in enumerate(pri_leaf_paths):
        if p:
            leaf_to_rows[p].append(i)
    leaf_col_sums: dict[str, np.ndarray] = {}
    for leaf, rows in leaf_to_rows.items():
        leaf_col_sums[leaf] = np.asarray(M_dom_pri[rows].sum(axis=0)).reshape(-1)

    # One fat sparse matmul: stack primaries and partials, compute the joint
    # composite-Dice, then slice out the partials-vs-primaries block.
    M_dom_full = sp.vstack([M_dom_pri, M_dom_q], format="csr")
    M_pair_full = sp.vstack([M_pair_pri, M_pair_q], format="csr")
    sim_full = compute_composite_similarity(
        M_dom_full, M_pair_full, weights=weights, prune_below=0.0,
    )
    sim_block = sim_full[n_primary:, :n_primary].tocsr()
    M_dom_q_csr = M_dom_q.tocsr()

    assignments: list[dict] = []
    skipped = 0
    for q_row in range(n_partial):
        sp_start = sim_block.indptr[q_row]
        sp_end = sim_block.indptr[q_row + 1]
        if sp_start == sp_end:
            skipped += 1
            continue
        cols = sim_block.indices[sp_start:sp_end]
        vals = sim_block.data[sp_start:sp_end]
        order = np.argsort(-vals)[:knn_k]
        top_cols = cols[order]
        top_vals = vals[order]
        top_sum = float(top_vals.sum())
        if top_sum < min_total_similarity:
            skipped += 1
            continue

        weights_norm = top_vals / top_sum
        umap_x = float((pri_coords[top_cols, 0] * weights_norm).sum())
        umap_y = float((pri_coords[top_cols, 1] * weights_norm).sum())

        votes: Counter[str] = Counter()
        for col, val in zip(top_cols.tolist(), top_vals.tolist()):
            p = pri_leaf_paths[col]
            if p:
                votes[p] += float(val)
        if not votes:
            skipped += 1
            continue
        best_leaf, _ = votes.most_common(1)[0]

        novelty: float | None = None
        if int(partial_ibgc_ids_arr[q_row]) in validated_id_set:
            novelty = 0.0
        elif validated_set:
            max_sim_validated = 0.0
            for col, val in zip(cols.tolist(), vals.tolist()):
                if col in validated_set and val > max_sim_validated:
                    max_sim_validated = float(val)
            novelty = 1.0 - max_sim_validated

        q_dom_start = M_dom_q_csr.indptr[q_row]
        q_dom_end = M_dom_q_csr.indptr[q_row + 1]
        n_dom = int(q_dom_end - q_dom_start)
        domain_novelty: float | None
        if n_dom == 0:
            domain_novelty = None
        else:
            col_sums_L = leaf_col_sums.get(best_leaf)
            if col_sums_L is None:
                domain_novelty = None
            else:
                domain_cols = M_dom_q_csr.indices[q_dom_start:q_dom_end]
                n_unique = int((col_sums_L[domain_cols] == 0).sum())
                domain_novelty = n_unique / n_dom

        assignments.append(
            {
                "ibgc_id": int(partial_ibgc_ids_arr[q_row]),
                "leaf_path": best_leaf,
                "umap_x": umap_x,
                "umap_y": umap_y,
                "novelty_score": novelty,
                "domain_novelty": domain_novelty,
            }
        )

    log.info(
        "project_partials: assigned=%d skipped=%d (of %d partials)",
        len(assignments), skipped, n_partial,
    )
    return assignments, skipped
