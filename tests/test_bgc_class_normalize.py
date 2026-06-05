"""Tests for common_core.bgc_class normalisation."""

import pytest

from common_core.bgc_class import (
    categories_for,
    classify_ibgc,
    normalize_single,
    reduce_to_label,
)


# ── antiSMASH tokenisation (greedy longest atom-run) ─────────────────────────
@pytest.mark.parametrize(
    "raw,expected",
    [
        ("T1PKS", {"Polyketide"}),
        ("NRPS", {"NRP"}),
        ("RiPP_like", {"RiPP"}),
        ("RRE_containing", {"RiPP"}),
        ("terpene_precursor", {"Terpene"}),
        ("acyl_amino_acids", {"Other"}),
        ("hydrogen_cyanide", {"Other"}),
        ("NI_siderophore", {"Other"}),
        ("oligosaccharide", {"Saccharide"}),
        # multi-product hybrids
        ("NRPS_T1PKS", {"NRP", "Polyketide"}),
        ("transAT_PKS_NRPS", {"Polyketide", "NRP"}),
        ("T1PKS_NRPS_like", {"Polyketide", "NRP"}),
        ("NRPS_like_T1PKS_NRPS", {"NRP", "Polyketide"}),
        # nrp-metallophore (nrps) + nrps -> still just NRP
        ("NRP_metallophore_NRPS", {"NRP"}),
        ("NRP_metallophore", {"NRP"}),
        # substring collision: fungal_cdps must beat cdps (both nrps anyway)
        ("fungal_cdps", {"NRP"}),
        ("CDPS", {"NRP"}),
        # antiSMASH 8 also stages bare category names alongside granular types.
        # "NRP"'s antiSMASH *type* is "nrps", so the base-name fallback is what
        # keeps it out of "Other".
        ("NRP", {"NRP"}),
        ("Terpene", {"Terpene"}),
        ("Polyketide", {"Polyketide"}),
        ("RiPP", {"RiPP"}),
        ("Saccharide", {"Saccharide"}),
        ("Other", {"Other"}),
        # mixed granular + category in one hybrid path
        ("NRP_T1PKS", {"NRP", "Polyketide"}),
        # antiSMASH 7.x RiPP types absent from the master schema
        ("thiopeptide", {"RiPP"}),
        ("LAP", {"RiPP"}),
        ("LAP_thiopeptide", {"RiPP"}),
    ],
)
def test_antismash_categories(raw, expected):
    assert categories_for("antiSMASH:8.0.1", raw) == expected


def test_antismash_unknown_atom_falls_back_to_other():
    assert categories_for("antiSMASH", "totally_made_up") == {"Other"}


# ── GECCO / SanntiS / mibig (base-class names) ───────────────────────────────
@pytest.mark.parametrize(
    "tool,raw,expected",
    [
        ("GECCO v0.9.8", "Unknown", {"Other"}),
        ("GECCO v0.9.8", "Terpene", {"Terpene"}),
        ("GECCO v0.9.8", "NRP_Polyketide", {"NRP", "Polyketide"}),
        ("SanntiSv0.9.3.3", "Other", {"Other"}),
        ("SanntiSv0.9.3.3", "NRP_Other_Polyketide", {"NRP", "Other", "Polyketide"}),
        ("SanntiSv0.9.3.3", "Alkaloid_Terpene", {"Alkaloid", "Terpene"}),
        ("mibig:4.0", "NRP", {"NRP"}),
    ],
)
def test_base_categories(tool, raw, expected):
    assert categories_for(tool, raw) == expected


def test_blank_path_yields_no_categories():
    assert categories_for("antiSMASH", "") == set()
    assert categories_for("GECCO", "   ") == set()


# ── reduce_to_label: P+N wins if both present ────────────────────────────────
@pytest.mark.parametrize(
    "cats,label",
    [
        (set(), "Other"),
        ({"Other"}, "Other"),
        ({"Polyketide"}, "Polyketide"),
        ({"Polyketide", "Other"}, "Polyketide"),
        ({"Polyketide", "NRP"}, "Hybrid(P+N)"),
        ({"Polyketide", "NRP", "Other"}, "Hybrid(P+N)"),
        ({"Polyketide", "NRP", "RiPP"}, "Hybrid(P+N)"),
        ({"NRP", "Saccharide"}, "Hybrid"),
        ({"Alkaloid", "Terpene"}, "Hybrid"),
    ],
)
def test_reduce_to_label(cats, label):
    assert reduce_to_label(cats) == label


def test_normalize_single():
    assert normalize_single("antiSMASH", "NRPS_T1PKS") == "Hybrid(P+N)"
    assert normalize_single("GECCO", "Unknown") == "Other"
    assert normalize_single("antiSMASH", "") == ""


# ── classify_ibgc: union across tools, then reduce ───────────────────────────
def test_classify_ibgc_union_creates_hybrid():
    # antiSMASH says Polyketide, GECCO says NRP -> union -> Hybrid(P+N)
    preds = [("antiSMASH:8.0.1", "T1PKS"), ("GECCO v0.9.8", "NRP")]
    assert classify_ibgc(preds) == "Hybrid(P+N)"


def test_classify_ibgc_agreeing_tools():
    preds = [("antiSMASH:8.0.1", "terpene"), ("SanntiSv0.9.3.3", "Terpene")]
    assert classify_ibgc(preds) == "Terpene"


def test_classify_ibgc_other_only():
    preds = [("GECCO v0.9.8", "Unknown"), ("SanntiSv0.9.3.3", "Other")]
    assert classify_ibgc(preds) == "Other"


def test_classify_ibgc_no_signal_is_empty():
    assert classify_ibgc([("antiSMASH", ""), ("GECCO", "")]) == ""
    assert classify_ibgc([]) == ""
