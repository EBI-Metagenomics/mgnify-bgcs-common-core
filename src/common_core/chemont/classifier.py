"""SMILES classification against ChemOnt SMARTS and ChemOnt ID lookup."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

from common_core.chemont.ontology import ChemOntOntology, get_ontology

log = logging.getLogger(__name__)


@dataclass
class ClassificationResult:
    """A single ChemOnt match for a SMILES string."""

    chemont_id: str
    name: str
    smarts: str
    depth: int


def classify_smiles(
    smiles: str,
    ontology: ChemOntOntology | None = None,
) -> list[ClassificationResult]:
    """Classify a SMILES string against all ChemOnt SMARTS patterns.

    Returns matches ranked by specificity (deepest / most specific first).
    Returns an empty list for invalid SMILES.
    """
    from rdkit import Chem

    ont = ontology or get_ontology()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        log.warning("invalid SMILES: %s", smiles)
        return []

    matches: list[ClassificationResult] = []
    for term in ont.terms_with_smarts():
        smarts_mol = ont.compiled_smarts[term.id]
        if mol.HasSubstructMatch(smarts_mol):
            matches.append(
                ClassificationResult(
                    chemont_id=term.id,
                    name=term.name,
                    smarts=term.smarts,  # type: ignore[arg-type]
                    depth=term.depth,
                )
            )

    return matches


def classify_smiles_batch(
    smiles_list: Sequence[str],
    ontology: ChemOntOntology | None = None,
) -> list[list[ClassificationResult]]:
    """Classify multiple SMILES strings. Returns one result list per input."""
    ont = ontology or get_ontology()
    return [classify_smiles(s, ontology=ont) for s in smiles_list]


def lookup_chemont_ids(
    chemont_ids: Sequence[str],
    ontology: ChemOntOntology | None = None,
) -> dict[str, list[tuple[str, str, str]]]:
    """Given ChemOnt IDs, return SMARTS for each ID and all ancestor nodes.

    Returns a dict mapping each input ID to a list of
    ``(chemont_id, name, smarts)`` tuples, ordered from the term itself up to
    root. Only includes terms that have a SMARTS pattern.
    """
    ont = ontology or get_ontology()
    result: dict[str, list[tuple[str, str, str]]] = {}
    for cid in chemont_ids:
        result[cid] = ont.get_lineage_smarts(cid)
    return result
