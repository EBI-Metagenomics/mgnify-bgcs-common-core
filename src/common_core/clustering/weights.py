"""Per-feature weights for the ``cosine_idf_enrich`` similarity metric.

Ported from the 2026-10-23 similarity benchmark
(``sanntis_dev/experiments/2026-10-23_bgc_similarity_robustness``,
``bgcsim.features``); see ADR 0003. Numerics must stay identical to that
reference — the benchmark numbers in the ADR were produced with it.

    w_f        = idf_f × enrich_f
    idf_f      = ln((1 + N) / (1 + df_f)), df counted once per group
                 (species, i.e. ``taxonomy_path``) so over-sampled clades
                 don't make their own domains look common
    enrich_f   = max(0.05, sigmoid(log2(p_core / p_flank))) for domains,
                 pseudocount split in proportion to the core / flank set sizes
    enrich_ab  = sqrt(enrich_a · enrich_b) for an adjacent pair (a, b)

IDF is recomputed per clustering run; the enrichment table is a frozen,
versioned artefact built by the portal. Labels missing from the table
weigh ``UNKNOWN_ENRICH`` (0.5, i.e. ``e = 0``).
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    import numpy as np
    import scipy.sparse as sp

ENRICH_FLOOR = 0.05
ENRICH_PSEUDOCOUNT = 2.0
UNKNOWN_ENRICH = 0.5


def idf_weights(M: sp.csr_matrix, groups) -> np.ndarray:
    """Smooth IDF with document frequency counted once per group.

    ``groups`` is row-aligned with ``M``; callers map a missing group key to
    a single ``"NA"`` group, as the benchmark did.
    """
    import numpy as np
    import scipy.sparse as sp

    # np.unique stands in for the reference's pd.factorize: only the number
    # of groups and their membership matter, not the code order.
    uniq, codes = np.unique(np.asarray(groups).astype(str), return_inverse=True)
    codes = codes.ravel()
    G = sp.csr_matrix(
        (np.ones(len(codes), dtype=np.float32), (codes, np.arange(len(codes)))),
        shape=(len(uniq), len(codes)),
    )
    df = np.asarray(((G @ M) > 0).sum(axis=0)).ravel()
    n = len(uniq)
    return np.log((1.0 + n) / (1.0 + df)).astype(np.float32)


def enrichment_weights(
    M_core: sp.csr_matrix,
    M_flank: sp.csr_matrix,
    pseudo: float = ENRICH_PSEUDOCOUNT,
) -> np.ndarray:
    """sigmoid(log2 p_core/p_flank) per column, smoothed, floored.

    Rows are CDS (core = inside GECCO/SanntiS iBGCs; flank = outside any
    iBGC plus antiSMASH overhang), columns the domain vocabulary.
    Pseudocounts are split in proportion to the two set sizes, so a domain
    seen in neither set (or rarely) shrinks to ``e = 0`` → ``w = 0.5``
    rather than towards the core:flank size ratio.
    """
    import numpy as np

    c = np.asarray((M_core > 0).sum(axis=0)).ravel()
    f = np.asarray((M_flank > 0).sum(axis=0)).ravel()
    return enrichment_from_counts(c, f, M_core.shape[0], M_flank.shape[0], pseudo)


def enrichment_from_counts(
    c,
    f,
    n_core: int,
    n_flank: int,
    pseudo: float = ENRICH_PSEUDOCOUNT,
) -> np.ndarray:
    """:func:`enrichment_weights` from per-label CDS counts.

    ``c`` / ``f`` are the number of core / flank CDS carrying each label;
    ``n_core`` / ``n_flank`` count *all* core / flank CDS, including those
    with no label. Lets callers aggregate in SQL instead of materialising
    CDS × label matrices.
    """
    import numpy as np

    c = np.asarray(c, dtype=np.float64)
    f = np.asarray(f, dtype=np.float64)
    n_c, n_f = float(n_core), float(n_flank)
    a_c, a_f = pseudo * n_c / (n_c + n_f), pseudo * n_f / (n_c + n_f)
    p_c = (c + a_c) / (n_c + pseudo)
    p_f = (f + a_f) / (n_f + pseudo)
    e = np.log2(p_c / p_f)
    w = 1.0 / (1.0 + np.exp(-e))
    return np.maximum(w, ENRICH_FLOOR).astype(np.float32)


def enrichment_vector(
    domain_accs: Iterable[str],
    table: Mapping[str, float],
) -> np.ndarray:
    """Align a ``label → enrich weight`` table to the domain vocabulary."""
    import numpy as np

    return np.asarray(
        [float(table.get(str(acc), UNKNOWN_ENRICH)) for acc in domain_accs],
        dtype=np.float32,
    )


def pair_enrichment(
    pair_vocab: Iterable[tuple[str, str]],
    domain_accs: Iterable[str],
    enrich_dom: np.ndarray,
) -> np.ndarray:
    """Pair weight = geometric mean of its two domain enrichment weights.

    ``enrich_dom`` is aligned to ``domain_accs``. A pair label absent from
    ``domain_accs`` (can't happen when both vocabularies come from the same
    export) falls back to ``UNKNOWN_ENRICH`` for that side.
    """
    import numpy as np

    dvocab = {str(acc): j for j, acc in enumerate(domain_accs)}
    enrich_dom = np.asarray(enrich_dom, dtype=np.float32)

    def _w(acc) -> float:
        j = dvocab.get(str(acc))
        return float(enrich_dom[j]) if j is not None else UNKNOWN_ENRICH

    pairs = list(pair_vocab)
    w = np.empty(len(pairs), dtype=np.float32)
    for j, (a, b) in enumerate(pairs):
        w[j] = np.sqrt(np.float32(_w(a)) * np.float32(_w(b)))
    return w


def combine(idf: np.ndarray, enrich: np.ndarray) -> np.ndarray:
    """Feature weight ``idf × enrich`` (float32, same length as both inputs)."""
    import numpy as np

    idf = np.asarray(idf, dtype=np.float32)
    enrich = np.asarray(enrich, dtype=np.float32)
    if idf.shape != enrich.shape:
        raise ValueError(f"idf {idf.shape} and enrich {enrich.shape} differ")
    return idf * enrich


def weights_sha256(domain_weights: np.ndarray, pair_weights: np.ndarray) -> str:
    """Stable digest of the two weight vectors (float32 little-endian bytes).

    Goes into the clustering run sha so two runs that differ only in their
    weights don't collide.
    """
    import numpy as np

    h = hashlib.sha256()
    for name, w in (("domain", domain_weights), ("pair", pair_weights)):
        arr = np.ascontiguousarray(np.asarray(w, dtype="<f4"))
        h.update(f"{name}:{arr.size}:".encode())
        h.update(arr.tobytes())
    return h.hexdigest()
