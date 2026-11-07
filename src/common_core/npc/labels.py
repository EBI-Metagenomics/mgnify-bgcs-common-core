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

"""NPClassifier labels of one BGC, from detector GFF attributes to ``bgc_np_classes.tsv`` rows.

Two producers:

* **SanntiS 1.0** (``source = predicted``): ``npc_pathway`` holds the ``|``-joined
  pathways at or above their thresholds (or only the top one), ``npc_pathway_score``
  and ``npc_pathway_confident`` describe the top pathway, ``npc_superclass_top`` /
  ``npc_superclass_score`` the top superclass, and ``npc_superclass`` is that
  superclass or ``unassigned`` when it misses the abstention tier.
* **Curated** (``source = curated``, MIBiG): ``npc_pathway``, ``npc_superclass`` and
  ``npc_class`` are ``|``-joined term names; every label is confident and unscored.

Names are resolved against the vendored ontology; names it does not know are
reported back to the caller instead of being written.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from .ontology import LEVELS, Ontology, load_ontology

log = logging.getLogger(__name__)

SOURCE_PREDICTED = "predicted"
SOURCE_CURATED = "curated"

#: Columns of ``bgc_np_classes.tsv`` in the staged bundle (ETL writer, portal loader).
NPC_TSV_COLUMNS: tuple[str, ...] = (
    "contig_sha256",
    "bgc_start",
    "bgc_end",
    "detector_name",
    "np_term_id",
    "level",
    "name",
    "rank",
    "score",
    "confident",
    "source",
)

UNASSIGNED = "unassigned"
LIST_SEP = "|"


@dataclass(frozen=True)
class NpcLabel:
    term_id: str
    level: str
    name: str
    rank: int
    score: float | None
    confident: bool
    source: str

    def as_row(self) -> dict[str, str]:
        return {
            "np_term_id": self.term_id,
            "level": self.level,
            "name": self.name,
            "rank": str(self.rank),
            "score": "" if self.score is None else f"{self.score:.4f}",
            "confident": "true" if self.confident else "false",
            "source": self.source,
        }


def _split(value: str | None) -> list[str]:
    return [v.strip() for v in (value or "").split(LIST_SEP) if v.strip()]


def _float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _bool(value: str | None) -> bool:
    return (value or "").strip().lower() in ("true", "1", "yes")


def _resolve(
    ont: Ontology, level: str, names: Iterable[str], unknown: list[tuple[str, str]]
):
    for name in names:
        term = ont.find(level, name)
        if term is None:
            unknown.append((level, name))
            continue
        yield term


def labels_from_sanntis_attrs(
    attrs: Mapping[str, str], ont: Ontology | None = None
) -> tuple[list[NpcLabel], list[tuple[str, str]]]:
    """SanntiS 1.0 ``CLUSTER`` attributes → labels, plus the ``(level, name)`` pairs not in the ontology."""
    ont = ont or load_ontology()
    unknown: list[tuple[str, str]] = []
    out: list[NpcLabel] = []

    pathways = _split(attrs.get("npc_pathway"))
    if pathways:
        confident = _bool(attrs.get("npc_pathway_confident"))
        top_score = _float(attrs.get("npc_pathway_score"))
        for rank, term in enumerate(
            _resolve(ont, "pathway", pathways, unknown), start=1
        ):
            out.append(
                NpcLabel(
                    term.term_id,
                    "pathway",
                    term.name,
                    rank,
                    top_score if rank == 1 else None,
                    confident,
                    SOURCE_PREDICTED,
                )
            )

    top = (attrs.get("npc_superclass_top") or "").strip()
    assigned = (attrs.get("npc_superclass") or "").strip()
    if top and top.casefold() != UNASSIGNED:
        confident = bool(assigned) and assigned.casefold() != UNASSIGNED
        for term in _resolve(ont, "superclass", [top], unknown):
            out.append(
                NpcLabel(
                    term.term_id,
                    "superclass",
                    term.name,
                    1,
                    _float(attrs.get("npc_superclass_score")),
                    confident,
                    SOURCE_PREDICTED,
                )
            )
    return out, unknown


CURATED_KEYS = {
    "pathway": "npc_pathway",
    "superclass": "npc_superclass",
    "class": "npc_class",
}


def labels_from_curated_attrs(
    attrs: Mapping[str, str], ont: Ontology | None = None
) -> tuple[list[NpcLabel], list[tuple[str, str]]]:
    """Curated (MIBiG) region attributes → confident, unscored labels at every level present."""
    ont = ont or load_ontology()
    unknown: list[tuple[str, str]] = []
    out: list[NpcLabel] = []
    for level in LEVELS:
        names = _split(attrs.get(CURATED_KEYS[level]))
        for rank, term in enumerate(_resolve(ont, level, names, unknown), start=1):
            out.append(
                NpcLabel(
                    term.term_id, level, term.name, rank, None, True, SOURCE_CURATED
                )
            )
    return out, unknown


def labels_to_rows(
    labels: Iterable[NpcLabel],
    *,
    contig_sha256: str,
    bgc_start: int,
    bgc_end: int,
    detector_name: str,
) -> list[dict[str, str]]:
    """Rows of ``bgc_np_classes.tsv`` (column order :data:`NPC_TSV_COLUMNS`)."""
    head = {
        "contig_sha256": contig_sha256,
        "bgc_start": str(bgc_start),
        "bgc_end": str(bgc_end),
        "detector_name": detector_name,
    }
    return [{**head, **lab.as_row()} for lab in labels]
