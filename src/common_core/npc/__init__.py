"""NPClassifier ontology (pathway → superclass → class) shared by the ETL and the portal.

The ontology is the frozen vocabulary that SanntiS 1.0 ships (``np_terms.tsv`` and
``np_term_edges.tsv``, NPClassifier index ``npclassifier_index_v1_250fd53c``). It is a
DAG: a few superclasses hang under two pathways, so a term can have several parents.

The ETL turns detector GFF attributes into :class:`NpcLabel` rows
(``bgc_np_classes.tsv``); the portal loads the same ontology into its tables and
expands a selected term to its descendants for search.
"""

from .labels import (
    NPC_TSV_COLUMNS,
    SOURCE_CURATED,
    SOURCE_PREDICTED,
    NpcLabel,
    labels_from_curated_attrs,
    labels_from_sanntis_attrs,
    labels_to_rows,
)
from .ontology import (
    LEVELS,
    ONTOLOGY_VERSION,
    Ontology,
    Term,
    load_ontology,
    ontology_files,
)

__all__ = [
    "LEVELS",
    "NPC_TSV_COLUMNS",
    "ONTOLOGY_VERSION",
    "SOURCE_CURATED",
    "SOURCE_PREDICTED",
    "NpcLabel",
    "Ontology",
    "Term",
    "labels_from_curated_attrs",
    "labels_from_sanntis_attrs",
    "labels_to_rows",
    "load_ontology",
    "ontology_files",
]
