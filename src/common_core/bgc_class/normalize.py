"""Normalise per-tool ``classification_path`` strings into canonical classes.

Two layers:

1. **Per prediction** — :func:`categories_for` turns one tool's raw
   ``classification_path`` into a set of canonical *base* classes. antiSMASH
   strings are greedy longest-atom-run matched against the vendored
   ``bgc_types`` vocabulary; GECCO/SanntiS/mibig already emit base-class names
   joined by ``_`` and are split directly.

2. **Per iBGC** — :func:`classify_ibgc` unions the base classes of *all* of an
   iBGC's source predictions and reduces them to a single label via
   :func:`reduce_to_label`. The hybrid rule is "P+N wins if both present":
   ``Hybrid(P+N)`` whenever Polyketide *and* NRP both appear (even alongside a
   third real class); any other 2+ real-class combo is ``Hybrid``; a single
   real class wins (``Other`` is ignored for hybrid detection); nothing real
   left is ``Other``.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

from .bgc_types import (
    ANTISMASH_CATEGORY_TO_BASE,
    ANTISMASH_TYPE_TO_CATEGORY,
    BASE_NAME_TO_CLASS,
)

log = logging.getLogger(__name__)


def _tool_family(tool: str) -> str:
    """Map a detector ``tool`` name to a parser family."""
    t = (tool or "").strip().lower()
    if t.startswith("antismash"):
        return "antismash"
    if t.startswith("gecco"):
        return "gecco"
    if t.startswith("sanntis"):
        return "sanntis"
    if t.startswith("mibig"):
        return "mibig"
    return "unknown"


def _atoms(raw: str) -> list[str]:
    """Split a raw path into ``_``-delimited atoms (``-`` already rewritten)."""
    return [a for a in raw.lower().replace("-", "_").split("_") if a]


def _resolve_antismash_token(candidate: str) -> str | None:
    """Resolve one ``_``-joined atom-run to a base class, or None.

    antiSMASH 8 stages a mix of *granular product types* (``terpene_precursor``,
    ``nrps_like``, ``rre_containing``) and its own *category names*
    (``NRP``, ``Terpene``, ``Polyketide``, …). Try the granular vocabulary
    first (more specific), then fall back to the base-class names so a bare
    ``NRP`` — whose antiSMASH *type* is ``nrps`` — still maps to ``NRP`` rather
    than ``Other``.
    """
    category = ANTISMASH_TYPE_TO_CATEGORY.get(candidate)
    if category is not None:
        return ANTISMASH_CATEGORY_TO_BASE[category]
    return BASE_NAME_TO_CLASS.get(candidate)


def _antismash_categories(raw: str) -> set[str]:
    """Greedy longest-atom-run match of an antiSMASH path to base classes.

    Atoms are joined back with ``_`` and probed from longest run to shortest,
    so multi-atom types (``nrps_like``, ``rre_containing``, ``acyl_amino_acids``)
    and substring collisions (``fungal_cdps`` before ``cdps``) resolve
    correctly. Each atom-run is checked against both the granular antiSMASH
    vocabulary and the base-class names. An atom that matches nothing is logged
    and contributes ``Other``.
    """
    atoms = _atoms(raw)
    n = len(atoms)
    cats: set[str] = set()
    i = 0
    while i < n:
        matched = False
        for j in range(n, i, -1):
            base = _resolve_antismash_token("_".join(atoms[i:j]))
            if base is not None:
                cats.add(base)
                i = j
                matched = True
                break
        if not matched:
            log.warning("Unknown antiSMASH BGC-type atom %r in %r", atoms[i], raw)
            cats.add("Other")
            i += 1
    return cats


def _base_categories(raw: str) -> set[str]:
    """Parse a GECCO/SanntiS/mibig path (base-class names joined by ``_``)."""
    cats: set[str] = set()
    for atom in _atoms(raw):
        base = BASE_NAME_TO_CLASS.get(atom)
        if base is None:
            log.warning("Unknown base BGC-class atom %r in %r", atom, raw)
            base = "Other"
        cats.add(base)
    return cats


def categories_for(tool: str, raw: str) -> set[str]:
    """Return the set of canonical base classes for one prediction.

    Empty/blank ``raw`` yields an empty set (no class signal).
    """
    if not raw or not raw.strip():
        return set()
    if _tool_family(tool) == "antismash":
        return _antismash_categories(raw)
    # gecco / sanntis / mibig / unknown all emit base-class names directly.
    return _base_categories(raw)


def reduce_to_label(categories: set[str]) -> str:
    """Reduce a set of base classes to one canonical label (P+N wins)."""
    real = {c for c in categories if c != "Other"}
    if not real:
        return "Other"
    if "Polyketide" in real and "NRP" in real:
        return "Hybrid(P+N)"
    if len(real) == 1:
        return next(iter(real))
    return "Hybrid"


def normalize_single(tool: str, raw: str) -> str:
    """Final label for a single prediction (convenience wrapper)."""
    cats = categories_for(tool, raw)
    return reduce_to_label(cats) if cats else ""


def classify_ibgc(predictions: Iterable[tuple[str, str]]) -> str:
    """Union the base classes of all ``(tool, raw)`` predictions, then reduce.

    Returns ``""`` when no prediction carries any class signal (so callers can
    distinguish "unclassified" from an explicit ``Other``).
    """
    cats: set[str] = set()
    for tool, raw in predictions:
        cats |= categories_for(tool, raw)
    if not cats:
        return ""
    return reduce_to_label(cats)
