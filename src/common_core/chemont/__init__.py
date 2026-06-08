"""ChemOnt chemical classification library."""

from common_core.chemont.classifier import (
    ClassificationResult,
    classify_smiles,
    classify_smiles_batch,
    lookup_chemont_ids,
)
from common_core.chemont.classyfire_client import (
    ClassyFireError,
    ClassyFireResult,
    ClassyFireUnavailable,
    classify,
    smiles_to_inchikey,
)
from common_core.chemont.ontology import (
    ChemOntOntology,
    ChemOntTerm,
    get_ontology,
    reset_ontology,
)
from common_core.chemont.similarity import (
    best_match_average,
    compute_ic_values,
    lin_similarity,
    normalize_similarity,
    resnik_similarity,
    semantic_similarity,
)

__all__ = [
    "ChemOntOntology",
    "ChemOntTerm",
    "ClassificationResult",
    "ClassyFireError",
    "ClassyFireResult",
    "ClassyFireUnavailable",
    "best_match_average",
    "classify",
    "classify_smiles",
    "classify_smiles_batch",
    "compute_ic_values",
    "get_ontology",
    "lin_similarity",
    "lookup_chemont_ids",
    "normalize_similarity",
    "reset_ontology",
    "resnik_similarity",
    "semantic_similarity",
    "smiles_to_inchikey",
]
