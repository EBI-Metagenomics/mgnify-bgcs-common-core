"""Tests for Lin-based ChemOnt semantic similarity.

``semantic_similarity`` (BMA over Lin) is what the chemical search uses: it
must map onto a 0–1 threshold slider, scoring 1.0 for identical term sets and
0.0 for terms sharing only the uninformative root.
"""

from __future__ import annotations

import pytest

from common_core.chemont.similarity import (
    coverage_similarity,
    lin_similarity,
    semantic_similarity,
)

# Hierarchy:  A (root) ─ B ─┬─ C
#                            └─ D
_ANCESTORS = {  # inclusive of self
    "A": {"A"},
    "B": {"B", "A"},
    "C": {"C", "B", "A"},
    "D": {"D", "B", "A"},
}
_IC = {"A": 0.0, "B": 1.0, "C": 3.0, "D": 3.0}


class _FakeOntology:
    def get_ancestor_ids(self, tid: str) -> set[str]:
        return _ANCESTORS.get(tid, {tid})


ONT = _FakeOntology()


def test_lin_identical_is_one():
    assert lin_similarity("C", "C", _IC, ONT) == pytest.approx(1.0)


def test_lin_siblings_share_parent():
    # MICA(C, D) = B (IC 1.0); Lin = 2·1 / (3+3) = 1/3.
    assert lin_similarity("C", "D", _IC, ONT) == pytest.approx(1.0 / 3.0)


def test_lin_only_root_in_common_is_zero():
    # MICA(C, A) = A (IC 0.0) → Lin 0.
    assert lin_similarity("C", "A", _IC, ONT) == pytest.approx(0.0)


def test_semantic_identical_sets_is_one():
    assert semantic_similarity(["B", "C"], ["B", "C"], _IC, ONT) == pytest.approx(1.0)


def test_semantic_partial_overlap_between_zero_and_one():
    score = semantic_similarity(["C"], ["D"], _IC, ONT)
    assert 0.0 < score < 1.0
    assert score == pytest.approx(1.0 / 3.0)


def test_semantic_empty_set_is_zero():
    assert semantic_similarity([], ["C"], _IC, ONT) == 0.0
    assert semantic_similarity(["C"], [], _IC, ONT) == 0.0


def test_coverage_full_when_query_contains_target():
    # Target [C] fully covered by query [C, D] (exact match present) → 1.0,
    # regardless of the extra query term D.
    assert coverage_similarity(["C", "D"], ["C"], _IC, ONT) == pytest.approx(1.0)


def test_coverage_is_asymmetric():
    # query [C] covering target [C, D]: lin(C,C)=1, lin(D,C)=1/3 → mean 2/3.
    fwd = coverage_similarity(["C"], ["C", "D"], _IC, ONT)
    rev = coverage_similarity(["C", "D"], ["C"], _IC, ONT)
    assert fwd == pytest.approx((1.0 + 1.0 / 3.0) / 2.0)
    assert rev == pytest.approx(1.0)
    assert fwd != rev


def test_coverage_empty_set_is_zero():
    assert coverage_similarity([], ["C"], _IC, ONT) == 0.0
    assert coverage_similarity(["C"], [], _IC, ONT) == 0.0
