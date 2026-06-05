"""BGC product-class normalisation.

Turns per-tool ``classification_path`` strings (antiSMASH, GECCO, SanntiS,
mibig) into one of nine canonical labels, with raw paths preserved upstream
for per-prediction display.
"""

from .bgc_types import (
    ANTISMASH_CATEGORY_TO_BASE,
    ANTISMASH_TYPE_TO_CATEGORY,
    BASE_CLASSES,
    BASE_NAME_TO_CLASS,
    CANONICAL_LABELS,
)
from .normalize import (
    categories_for,
    classify_ibgc,
    normalize_single,
    reduce_to_label,
)

__all__ = [
    "ANTISMASH_CATEGORY_TO_BASE",
    "ANTISMASH_TYPE_TO_CATEGORY",
    "BASE_CLASSES",
    "BASE_NAME_TO_CLASS",
    "CANONICAL_LABELS",
    "categories_for",
    "classify_ibgc",
    "normalize_single",
    "reduce_to_label",
]
