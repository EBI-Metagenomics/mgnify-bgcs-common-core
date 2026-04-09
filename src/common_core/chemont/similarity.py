"""IC-based semantic similarity for ChemOnt ontology terms.

Implements Resnik similarity (Most Informative Common Ancestor) and
Best Match Average (BMA) for comparing sets of ChemOnt annotations.
"""

from __future__ import annotations

import math
from collections import defaultdict

from common_core.chemont.ontology import ChemOntOntology


def compute_ic_values(
    term_counts: dict[str, int],
    total: int,
    ontology: ChemOntOntology,
) -> dict[str, float]:
    """Compute Information Content for every term in the ontology.

    IC(term) = -log2(p(term)) where p(term) is the fraction of items
    annotated with that term or any of its descendants.

    Args:
        term_counts: direct annotation counts {chemont_id: count}.
        total: total number of annotated items (e.g. natural products).
        ontology: the ChemOnt ontology instance.

    Returns:
        Mapping of chemont_id → IC value. Terms with p=0 are omitted.
    """
    if total <= 0:
        return {}

    # Propagate counts upward: each term's effective count includes
    # all of its descendants' counts.
    propagated: dict[str, int] = defaultdict(int)

    # Start with direct counts for terms that have annotations.
    for tid, count in term_counts.items():
        propagated[tid] += count

    # Propagate upward: for each annotated term, add its count to all ancestors.
    for tid, count in term_counts.items():
        term = ontology.terms.get(tid)
        if term is None:
            continue
        for ancestor in ontology.get_ancestors(tid):
            propagated[ancestor.id] += count

    ic_values: dict[str, float] = {}
    for tid, prop_count in propagated.items():
        p = min(prop_count / total, 1.0)
        if p > 0:
            ic_values[tid] = -math.log2(p)

    return ic_values


def resnik_similarity(
    t1: str,
    t2: str,
    ic_values: dict[str, float],
    ontology: ChemOntOntology,
    *,
    _mica_cache: dict[tuple[str, str], float] | None = None,
) -> float:
    """Resnik similarity: IC of the Most Informative Common Ancestor.

    Args:
        t1, t2: ChemOnt term IDs.
        ic_values: precomputed IC values from :func:`compute_ic_values`.
        ontology: the ChemOnt ontology instance.
        _mica_cache: optional mutable cache for (t1, t2) → IC(MICA).

    Returns:
        IC(MICA) or 0.0 if no common ancestor has IC > 0.
    """
    key = (min(t1, t2), max(t1, t2))
    if _mica_cache is not None and key in _mica_cache:
        return _mica_cache[key]

    if t1 == t2:
        val = ic_values.get(t1, 0.0)
        if _mica_cache is not None:
            _mica_cache[key] = val
        return val

    ancestors_1 = ontology.get_ancestor_ids(t1)
    ancestors_2 = ontology.get_ancestor_ids(t2)
    common = ancestors_1 & ancestors_2

    if not common:
        if _mica_cache is not None:
            _mica_cache[key] = 0.0
        return 0.0

    val = max(ic_values.get(cid, 0.0) for cid in common)
    if _mica_cache is not None:
        _mica_cache[key] = val
    return val


def best_match_average(
    query_terms: list[str],
    target_terms: list[str],
    ic_values: dict[str, float],
    ontology: ChemOntOntology,
) -> float:
    """Symmetric Best Match Average between two sets of ChemOnt terms.

    For each term in set A, find its best Resnik match in set B (and vice
    versa). BMA = mean of all best-match values.

    Returns 0.0 if either set is empty.
    """
    if not query_terms or not target_terms:
        return 0.0

    cache: dict[tuple[str, str], float] = {}

    # Best match for each query term against target terms.
    query_best = []
    for qt in query_terms:
        best = max(
            resnik_similarity(qt, tt, ic_values, ontology, _mica_cache=cache)
            for tt in target_terms
        )
        query_best.append(best)

    # Best match for each target term against query terms.
    target_best = []
    for tt in target_terms:
        best = max(
            resnik_similarity(tt, qt, ic_values, ontology, _mica_cache=cache)
            for qt in query_terms
        )
        target_best.append(best)

    all_scores = query_best + target_best
    return sum(all_scores) / len(all_scores)


def normalize_similarity(
    raw_bma: float,
    ic_values: dict[str, float],
) -> float:
    """Normalize a raw BMA score into [0, 1].

    Divides by the maximum IC across all terms so the result is
    compatible with a 0–1 threshold slider.
    """
    if not ic_values:
        return 0.0
    max_ic = max(ic_values.values())
    if max_ic <= 0:
        return 0.0
    return min(raw_bma / max_ic, 1.0)
