#!/usr/bin/env python3

# Copyright 2026 EMBL - European Bioinformatics Institute
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for common_core.npc (NPClassifier ontology and label parsing)."""

import pytest

from common_core.npc import (
    LEVELS,
    NPC_TSV_COLUMNS,
    SOURCE_CURATED,
    SOURCE_PREDICTED,
    labels_from_curated_attrs,
    labels_from_sanntis_attrs,
    labels_to_rows,
    load_ontology,
)


@pytest.fixture(scope="module")
def ont():
    return load_ontology()


def test_ontology_shape(ont):
    assert LEVELS == ("pathway", "superclass", "class")
    assert len(ont.at_level("pathway")) == 7
    # 77 NPClassifier superclasses + the 6 "Other <pathway>" buckets SanntiS 1.0 emits
    assert len(ont.at_level("superclass")) == 83
    assert len(ont.at_level("class")) == 687
    assert len(ont) == 777
    # every non-root term but one orphan superclass has parents exactly one level up
    assert [t.name for t in ont.orphans()] == ["Pseudoalkaloids (transamidation)"]
    for t in ont.terms.values():
        if t.level != "pathway" and t not in ont.orphans():
            parents = ont.parents[t.term_id]
            assert parents
            assert all(ont.terms[p].depth == t.depth - 1 for p in parents)


def test_find_is_case_insensitive(ont):
    t = ont.find("pathway", "amino acids AND peptides")
    assert t is not None and t.term_id == "NPC:pathway:amino_acids_and_peptides"
    assert ont.find("superclass", "Oligopeptides") is not None
    # SanntiS 1.0 writes the bucket superclasses in lower case ("other Terpenoids")
    other = ont.find("superclass", "other Terpenoids")
    assert other is not None and ont.parents[other.term_id] == (
        "NPC:pathway:terpenoids",
    )
    assert ont.find("pathway", "Oligopeptides") is None
    assert ont.find("class", "no such class") is None


def test_dag_multi_parent_and_paths(ont):
    macrolides = ont.find("superclass", "Macrolides").term_id
    assert len(ont.parents[macrolides]) == 2
    paths = ont.paths(macrolides)
    assert len(paths) == 2 and all(p[-1] == macrolides and len(p) == 2 for p in paths)
    anc = ont.ancestors([macrolides], include_self=False)
    assert anc == set(ont.parents[macrolides])


def test_descendants(ont):
    pk = ont.find("pathway", "Polyketides").term_id
    desc = ont.descendants([pk])
    assert pk in desc
    assert ont.find("superclass", "Macrolides").term_id in desc
    assert all(ont.terms[d].level in LEVELS for d in desc)
    assert ont.descendants([pk], include_self=False) == desc - {pk}
    assert ont.descendants(["NPC:pathway:does_not_exist"]) == set()


def test_tree_lists_roots_first(ont):
    tree = ont.tree()
    assert [n["name"] for n in tree] == [t.name for t in ont.roots()] + [
        "Pseudoalkaloids (transamidation)"
    ]
    assert [n["level"] for n in tree] == ["pathway"] * 7 + ["superclass"]
    # a multi-parent superclass appears under each of its parents
    hits = sum(1 for n in tree for c in n["children"] if c["name"] == "Macrolides")
    assert hits == 2


SANNTIS_ATTRS = {
    "ID": "ctg_sanntis_1",
    "nearest_MiBIG_class": "NRPS",
    "bgc_class": "NRPS",
    "p_non_bgc": "0.0240",
    "n_orfs": "43",
    "npc_pathway": "Amino acids and Peptides|Polyketides",
    "npc_pathway_score": "0.9495",
    "npc_pathway_confident": "true",
    "npc_superclass": "Oligopeptides",
    "npc_superclass_top": "Oligopeptides",
    "npc_superclass_score": "0.9725",
}


def test_sanntis_labels_confident(ont):
    labels, unknown = labels_from_sanntis_attrs(SANNTIS_ATTRS, ont)
    assert unknown == []
    assert [(l.level, l.name, l.rank, l.score, l.confident) for l in labels] == [
        ("pathway", "Amino acids and Peptides", 1, 0.9495, True),
        ("pathway", "Polyketides", 2, None, True),
        ("superclass", "Oligopeptides", 1, 0.9725, True),
    ]
    assert {l.source for l in labels} == {SOURCE_PREDICTED}


def test_sanntis_labels_unassigned_superclass(ont):
    attrs = dict(
        SANNTIS_ATTRS,
        npc_pathway="Terpenoids",
        npc_pathway_confident="false",
        npc_superclass="unassigned",
        npc_superclass_top="Diterpenoids",
        npc_superclass_score="0.41",
    )
    labels, unknown = labels_from_sanntis_attrs(attrs, ont)
    assert unknown == []
    assert [(l.level, l.name, l.confident, l.score) for l in labels] == [
        ("pathway", "Terpenoids", False, 0.9495),
        ("superclass", "Diterpenoids", False, 0.41),
    ]


def test_sanntis_labels_unknown_names_reported(ont):
    attrs = dict(
        SANNTIS_ATTRS,
        npc_pathway="Polyketides|Not a pathway",
        npc_superclass_top="Not a superclass",
        npc_superclass="unassigned",
    )
    labels, unknown = labels_from_sanntis_attrs(attrs, ont)
    assert [l.name for l in labels] == ["Polyketides"]
    assert unknown == [("pathway", "Not a pathway"), ("superclass", "Not a superclass")]


def test_sanntis_labels_absent():
    labels, unknown = labels_from_sanntis_attrs({"ID": "x", "bgc_class": "PKS"})
    assert labels == [] and unknown == []


def test_curated_labels(ont):
    attrs = {
        "npc_pathway": "Polyketides",
        "npc_superclass": "Macrolides",
        "npc_class": "Spirotetronate macrolides|Erythromycins",
    }
    labels, unknown = labels_from_curated_attrs(attrs, ont)
    assert unknown == []
    assert [(l.level, l.name, l.rank) for l in labels] == [
        ("pathway", "Polyketides", 1),
        ("superclass", "Macrolides", 1),
        ("class", "Spirotetronate macrolides", 1),
        ("class", "Erythromycins", 2),
    ]
    assert all(
        l.confident and l.score is None and l.source == SOURCE_CURATED for l in labels
    )


def test_rows_follow_tsv_columns(ont):
    labels, _ = labels_from_sanntis_attrs(SANNTIS_ATTRS, ont)
    rows = labels_to_rows(
        labels,
        contig_sha256="abc",
        bgc_start=10,
        bgc_end=200,
        detector_name="SanntiSv1.0.0",
    )
    assert [tuple(r) == NPC_TSV_COLUMNS for r in rows] == [True] * 3
    assert (
        rows[0]["score"] == "0.9495"
        and rows[1]["score"] == ""
        and rows[0]["confident"] == "true"
    )
    assert rows[0]["np_term_id"] == "NPC:pathway:amino_acids_and_peptides"
