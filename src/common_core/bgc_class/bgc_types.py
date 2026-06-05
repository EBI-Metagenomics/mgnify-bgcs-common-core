"""antiSMASH BGC-type vocabulary and base-class mappings.

The antiSMASH product-type → category table is vendored from
``antismash/db-schema`` (``bgc_types.sql``). Type names are normalised here
to lowercase with ``-`` replaced by ``_`` so they match the staged-file
convention (where antiSMASH product names already have ``-`` rewritten to
``_`` — which is *also* the delimiter joining several products in one
``classification_path``). Because some type names are substrings of others
(e.g. ``cdps`` vs ``fungal_cdps``), the tokeniser in :mod:`.normalize` always
matches the longest atom-run first.

antiSMASH itself has no ``alkaloid`` category — ``Alkaloid`` only ever comes
from GECCO/SanntiS, which already emit base-class names directly.
"""

from __future__ import annotations

# Canonical base classes (pre-hybrid reduction).
BASE_CLASSES: tuple[str, ...] = (
    "Polyketide",
    "NRP",
    "RiPP",
    "Terpene",
    "Saccharide",
    "Alkaloid",
    "Other",
)

# Final normalised labels, in display order. The two hybrid buckets are
# derived in :func:`.normalize.reduce_to_label`.
CANONICAL_LABELS: tuple[str, ...] = (
    "Polyketide",
    "NRP",
    "RiPP",
    "Terpene",
    "Saccharide",
    "Alkaloid",
    "Other",
    "Hybrid(P+N)",
    "Hybrid",
)

# antiSMASH category -> canonical base class.
ANTISMASH_CATEGORY_TO_BASE: dict[str, str] = {
    "pks": "Polyketide",
    "nrps": "NRP",
    "ripp": "RiPP",
    "terpene": "Terpene",
    "saccharide": "Saccharide",
    "other": "Other",
}

# Base-class atom (as emitted by GECCO / SanntiS / mibig, lowercased) ->
# canonical base class. GECCO's "Unknown" collapses to "Other".
BASE_NAME_TO_CLASS: dict[str, str] = {
    "polyketide": "Polyketide",
    "nrp": "NRP",
    "ripp": "RiPP",
    "terpene": "Terpene",
    "saccharide": "Saccharide",
    "alkaloid": "Alkaloid",
    "other": "Other",
    "unknown": "Other",
}

# antiSMASH product type (lowercased, '-' -> '_') -> category.
# Vendored from antismash/db-schema bgc_types.sql.
_ANTISMASH_TYPES_RAW: dict[str, str] = {
    "aureonuclemycin": "other",
    "benzoxazole": "pks",
    "t1pks": "pks",
    "t2pks": "pks",
    "t3pks": "pks",
    "transat-pks": "pks",
    "hgle-ks": "pks",
    "polyhalogenated-pyrrole": "other",
    "prodigiosin": "pks",
    "ppys-ks": "pks",
    "arylpolyene": "pks",
    "ladderane": "pks",
    "leupeptin": "other",
    "hr-t2pks": "pks",
    "pufa": "pks",
    "resorcinol": "other",
    "nrps": "nrps",
    "cdps": "nrps",
    "fungal_cdps": "nrps",
    "rcdps": "nrps",
    "thioamide-nrp": "nrps",
    "napaa": "nrps",
    "mycosporine": "nrps",
    "t3nrps-iterative": "nrps",
    "terpene": "terpene",
    "atropopeptide": "ripp",
    "lanthipeptide-class-i": "ripp",
    "lanthipeptide-class-ii": "ripp",
    "lanthipeptide-class-iii": "ripp",
    "lanthipeptide-class-iv": "ripp",
    "lanthipeptide-class-v": "ripp",
    "lipolanthine": "ripp",
    "azole-containing-ripp": "ripp",
    "thioamitides": "ripp",
    "linaridin": "ripp",
    "cyanobactin": "ripp",
    "glycocin": "ripp",
    "lassopeptide": "ripp",
    "sactipeptide": "ripp",
    "bottromycin": "ripp",
    "microviridin": "ripp",
    "proteusin": "ripp",
    "ranthipeptide": "ripp",
    "redox-cofactor": "ripp",
    "darobactin": "ripp",
    "triceptide": "ripp",
    "archaeal-ripp": "ripp",
    "epipeptide": "ripp",
    "cyclic-lactone-autoinducer": "ripp",
    "spliceotide": "ripp",
    "ras-ripp": "ripp",
    "fungal-ripp": "ripp",
    "blactam": "other",
    "2dos": "other",
    "amglyccycl": "other",
    "aminocoumarin": "other",
    "azoxy-crosslink": "other",
    "azoxy-dimer": "other",
    "cytokinin": "other",
    "ni-siderophore": "other",
    "ectoine": "other",
    "naggn": "other",
    "butyrolactone": "other",
    "indole": "other",
    "lincosamides": "other",
    "nucleoside": "other",
    "phosphoglycolipid": "other",
    "melanin": "other",
    "oligosaccharide": "saccharide",
    "furan": "other",
    "hserlactone": "other",
    "phenazine": "other",
    "phosphonate": "other",
    "guanidinotides": "ripp",
    "other": "other",
    "acyl_amino_acids": "other",
    "pbde": "other",
    "polyyne": "other",
    "betalactone": "other",
    "tropodithietic-acid": "other",
    "pyrrolidine": "other",
    "crocagin": "ripp",
    "nrp-metallophore": "nrps",
    "methanobactin": "ripp",
    "nitropropanoic_acid": "other",
    "opine-like-metallophore": "other",
    "aminopolycarboxylic-acid": "other",
    "isocyanide": "other",
    "isocyanide-nrp": "nrps",
    "hydrogen-cyanide": "other",
    "hydroxytropolone": "other",
    "deazapurine": "other",
    "nrps-like": "nrps",
    "pks-like": "pks",
    "transat-pks-like": "pks",
    "ripp-like": "ripp",
    "rre-containing": "ripp",
    "phosphonate-like": "other",
    "terpene-precursor": "terpene",
    # antiSMASH 7.x RiPP types not present in the current master bgc_types.sql
    # (kept here so older staged runs still classify correctly).
    "thiopeptide": "ripp",
    "lap": "ripp",
    "lanthipeptide": "ripp",
    "lassopeptide-like": "ripp",
}

# Public, staged-file-normalised form: '-' -> '_'.
ANTISMASH_TYPE_TO_CATEGORY: dict[str, str] = {
    name.replace("-", "_"): category for name, category in _ANTISMASH_TYPES_RAW.items()
}
