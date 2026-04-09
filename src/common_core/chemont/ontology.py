"""ChemOnt ontology parser with hierarchy, SMARTS extraction, and lazy caching."""

from __future__ import annotations

import logging
import os
import re
import threading
import zipfile
from collections import deque
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

_SMARTS_XREF_RE = re.compile(r'^xref:\s+SMARTS\s+"(.+)"', re.MULTILINE)

_DEFAULT_OBO_PATH = "/data/chemont/ChemOnt_2_1.obo"


@dataclass(frozen=True)
class ChemOntTerm:
    """A single ChemOnt ontology term."""

    id: str
    name: str
    smarts: str | None
    parent_ids: tuple[str, ...]
    depth: int = 0


class ChemOntOntology:
    """Parsed ChemOnt ontology with hierarchy and SMARTS lookup.

    Attributes:
        terms: mapping of ChemOnt ID → ChemOntTerm.
        compiled_smarts: mapping of ChemOnt ID → pre-compiled RDKit Mol
            (only for terms whose SMARTS compiled successfully).
    """

    def __init__(self, obo_path: str | Path) -> None:
        obo_path = Path(obo_path)
        if not obo_path.exists():
            raise FileNotFoundError(f"OBO file not found: {obo_path}")

        raw_terms = _parse_obo(obo_path)
        depths = _compute_depths(raw_terms)
        self.terms: dict[str, ChemOntTerm] = {
            tid: ChemOntTerm(
                id=t["id"],
                name=t["name"],
                smarts=t["smarts"],
                parent_ids=tuple(t["parent_ids"]),
                depth=depths.get(tid, 0),
            )
            for tid, t in raw_terms.items()
        }
        self.compiled_smarts: dict[str, object] = _compile_smarts(self.terms)

        n_smarts = len(self.compiled_smarts)
        log.info(
            "loaded %d ChemOnt terms (%d with compiled SMARTS) from %s",
            len(self.terms),
            n_smarts,
            obo_path,
        )

    @classmethod
    def from_zip(cls, zip_path: str | Path) -> ChemOntOntology:
        """Extract the OBO file from a zip archive and parse it."""
        zip_path = Path(zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            obo_names = [n for n in zf.namelist() if n.endswith(".obo")]
            if not obo_names:
                raise ValueError(f"No .obo file found in {zip_path}")
            # Extract to same directory as zip
            extract_dir = zip_path.parent
            zf.extract(obo_names[0], extract_dir)
            return cls(extract_dir / obo_names[0])

    def get_term(self, chemont_id: str) -> ChemOntTerm | None:
        return self.terms.get(chemont_id)

    def get_ancestors(self, chemont_id: str) -> list[ChemOntTerm]:
        """Return all ancestors from term to root, ordered child-to-root."""
        result: list[ChemOntTerm] = []
        visited: set[str] = set()
        queue = deque[str]()

        term = self.terms.get(chemont_id)
        if term is None:
            return result

        for pid in term.parent_ids:
            if pid not in visited:
                queue.append(pid)
                visited.add(pid)

        while queue:
            tid = queue.popleft()
            t = self.terms.get(tid)
            if t is None:
                continue
            result.append(t)
            for pid in t.parent_ids:
                if pid not in visited:
                    queue.append(pid)
                    visited.add(pid)

        return result

    def get_lineage_smarts(
        self, chemont_id: str
    ) -> list[tuple[str, str, str]]:
        """Return (chemont_id, name, smarts) for the term and all ancestors that have SMARTS.

        Ordered from the term itself up to root.
        """
        result: list[tuple[str, str, str]] = []
        term = self.terms.get(chemont_id)
        if term is None:
            return result
        if term.smarts is not None:
            result.append((term.id, term.name, term.smarts))
        for ancestor in self.get_ancestors(chemont_id):
            if ancestor.smarts is not None:
                result.append((ancestor.id, ancestor.name, ancestor.smarts))
        return result

    def terms_with_smarts(self) -> list[ChemOntTerm]:
        """Return all terms that have compiled SMARTS, sorted by depth (deepest first)."""
        return sorted(
            (t for t in self.terms.values() if t.id in self.compiled_smarts),
            key=lambda t: t.depth,
            reverse=True,
        )


# ---------------------------------------------------------------------------
# OBO parsing helpers
# ---------------------------------------------------------------------------


def _parse_obo(obo_path: Path) -> dict[str, dict]:
    """Parse OBO file into raw term dicts keyed by ID."""
    terms: dict[str, dict] = {}
    current: dict | None = None

    with open(obo_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line == "[Term]":
                current = {
                    "id": "",
                    "name": "",
                    "smarts": None,
                    "parent_ids": [],
                }
                continue
            if line.startswith("[") and line.endswith("]"):
                # End of a Term stanza (new stanza type)
                if current and current["id"]:
                    terms[current["id"]] = current
                current = None
                continue
            if current is None:
                continue
            if line.startswith("id: "):
                current["id"] = line[4:].strip()
            elif line.startswith("name: "):
                current["name"] = line[6:].strip()
            elif line.startswith("is_a: "):
                parent_id = line[6:].split("!")[0].strip()
                current["parent_ids"].append(parent_id)
            elif line.startswith("xref: SMARTS"):
                m = _SMARTS_XREF_RE.match(line)
                if m:
                    current["smarts"] = m.group(1)

    # Capture last term if file doesn't end with a new stanza
    if current and current.get("id"):
        terms[current["id"]] = current

    return terms


def _compute_depths(raw_terms: dict[str, dict]) -> dict[str, int]:
    """BFS from root(s) to compute depth for every term."""
    children: dict[str, list[str]] = {}
    roots: list[str] = []

    for tid, t in raw_terms.items():
        if not t["parent_ids"]:
            roots.append(tid)
        for pid in t["parent_ids"]:
            children.setdefault(pid, []).append(tid)

    depths: dict[str, int] = {}
    queue: deque[tuple[str, int]] = deque()
    for r in roots:
        depths[r] = 0
        queue.append((r, 0))

    while queue:
        tid, d = queue.popleft()
        for child in children.get(tid, []):
            if child not in depths:
                depths[child] = d + 1
                queue.append((child, d + 1))

    return depths


def _compile_smarts(terms: dict[str, ChemOntTerm]) -> dict[str, object]:
    """Pre-compile SMARTS patterns with RDKit. Returns {chemont_id: Mol}."""
    from rdkit import Chem

    compiled: dict[str, object] = {}
    failed = 0
    for tid, term in terms.items():
        if term.smarts is None:
            continue
        mol = Chem.MolFromSmarts(term.smarts)
        if mol is None:
            failed += 1
            log.debug("failed to compile SMARTS for %s (%s)", tid, term.name)
            continue
        compiled[tid] = mol
    if failed:
        log.warning("%d ChemOnt SMARTS patterns failed to compile", failed)
    return compiled


# ---------------------------------------------------------------------------
# Lazy singleton
# ---------------------------------------------------------------------------

_instance: ChemOntOntology | None = None
_lock = threading.Lock()


def get_ontology(obo_path: str | Path | None = None) -> ChemOntOntology:
    """Get or create the cached ontology singleton.

    On first call, parses the OBO file and compiles SMARTS. Subsequent calls
    return the cached instance. Thread-safe.

    Args:
        obo_path: Path to the OBO file. Only used on first call. Falls back to
            the ``CHEMONT_OBO_PATH`` env var, then ``/data/chemont/ChemOnt_2_1.obo``.
    """
    global _instance
    if _instance is not None:
        return _instance
    with _lock:
        if _instance is not None:
            return _instance
        if obo_path is None:
            obo_path = os.environ.get("CHEMONT_OBO_PATH", _DEFAULT_OBO_PATH)
        _instance = ChemOntOntology(obo_path)
        return _instance


def reset_ontology() -> None:
    """Clear the cached singleton (useful for testing)."""
    global _instance
    with _lock:
        _instance = None
