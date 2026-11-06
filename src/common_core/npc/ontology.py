"""Load the vendored NPClassifier ontology and walk it."""

from __future__ import annotations

import csv
from collections.abc import Iterable
from dataclasses import dataclass, field
from functools import lru_cache
from importlib.resources import files
from pathlib import Path

#: Identifier of the frozen NPClassifier index the term tables were built from
#: (the same string SanntiS 1.0 records in its training-run provenance).
ONTOLOGY_VERSION = "npclassifier_index_v1_250fd53c+sanntis_other_v1"
#: The "+sanntis_other_v1" suffix: six superclass terms ``Other <pathway>`` were added to
#: the SanntiS index because the SanntiS 1.0 superclass head emits them (rare superclasses
#: bucketed per pathway at training time); they are not NPClassifier terms.

#: Levels from the root down. Every edge goes from level ``i+1`` to level ``i``.
LEVELS: tuple[str, ...] = ("pathway", "superclass", "class")

TERMS_FILE = "np_terms.tsv"
EDGES_FILE = "np_term_edges.tsv"


def ontology_files() -> dict[str, Path]:
    """Paths of the vendored term and edge tables."""
    base = files("common_core") / "npc" / "data"
    return {
        TERMS_FILE: Path(str(base / TERMS_FILE)),
        EDGES_FILE: Path(str(base / EDGES_FILE)),
    }


@dataclass(frozen=True)
class Term:
    term_id: str
    level: str
    name: str
    is_target_level: bool = False

    @property
    def depth(self) -> int:
        return LEVELS.index(self.level)


def _norm(name: str) -> str:
    """Case- and whitespace-insensitive key for term names."""
    return " ".join((name or "").split()).casefold()


@dataclass
class Ontology:
    terms: dict[str, Term]
    parents: dict[str, tuple[str, ...]] = field(default_factory=dict)
    children: dict[str, tuple[str, ...]] = field(default_factory=dict)
    _by_name: dict[tuple[str, str], str] = field(default_factory=dict, repr=False)

    # ── lookup ────────────────────────────────────────────────────────────
    def __contains__(self, term_id: str) -> bool:
        return term_id in self.terms

    def __len__(self) -> int:
        return len(self.terms)

    def get(self, term_id: str) -> Term | None:
        return self.terms.get(term_id)

    def find(self, level: str, name: str) -> Term | None:
        """Term of ``level`` whose name matches ``name`` (case-insensitive)."""
        tid = self._by_name.get((level, _norm(name)))
        return self.terms[tid] if tid else None

    def at_level(self, level: str) -> list[Term]:
        return sorted(
            (t for t in self.terms.values() if t.level == level),
            key=lambda t: t.name.casefold(),
        )

    def roots(self) -> list[Term]:
        return self.at_level(LEVELS[0])

    # ── traversal ─────────────────────────────────────────────────────────
    def descendants(
        self, term_ids: Iterable[str], include_self: bool = True
    ) -> set[str]:
        """All terms below any of ``term_ids`` (DAG-safe), optionally with the seeds."""
        seeds = [t for t in term_ids if t in self.terms]
        out: set[str] = set(seeds) if include_self else set()
        stack = list(seeds)
        seen: set[str] = set()
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            for child in self.children.get(cur, ()):
                out.add(child)
                stack.append(child)
        return out

    def ancestors(self, term_ids: Iterable[str], include_self: bool = True) -> set[str]:
        seeds = [t for t in term_ids if t in self.terms]
        out: set[str] = set(seeds) if include_self else set()
        stack = list(seeds)
        seen: set[str] = set()
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            for parent in self.parents.get(cur, ()):
                out.add(parent)
                stack.append(parent)
        return out

    def paths(self, term_id: str) -> list[tuple[str, ...]]:
        """Every root-to-term chain of ids (several when an ancestor has two parents)."""
        parents = self.parents.get(term_id, ())
        if not parents:
            return [(term_id,)]
        out: list[tuple[str, ...]] = []
        for p in parents:
            out.extend(path + (term_id,) for path in self.paths(p))
        return out

    def orphans(self) -> list[Term]:
        """Non-root terms the edge table gives no parent (one superclass in index v1)."""
        return sorted(
            (
                t
                for t in self.terms.values()
                if t.level != LEVELS[0] and t.term_id not in self.parents
            ),
            key=lambda t: (t.depth, t.name.casefold()),
        )

    def tree(self) -> list[dict]:
        """Nested ``{term_id, level, name, children}`` dicts: the pathways by name, then the orphans.

        A multi-parent term appears once under each parent, so every term is reachable.
        """

        def node(tid: str) -> dict:
            t = self.terms[tid]
            kids = sorted(
                self.children.get(tid, ()), key=lambda c: self.terms[c].name.casefold()
            )
            return {
                "term_id": tid,
                "level": t.level,
                "name": t.name,
                "children": [node(c) for c in kids],
            }

        return [node(r.term_id) for r in self.roots()] + [
            node(o.term_id) for o in self.orphans()
        ]


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def load_ontology_from(terms_path: Path, edges_path: Path) -> Ontology:
    terms: dict[str, Term] = {}
    for row in _read_tsv(terms_path):
        level = row["level"]
        if level not in LEVELS:
            raise ValueError(
                f"{terms_path}: unknown level {level!r} for {row['np_term_id']}"
            )
        terms[row["np_term_id"]] = Term(
            term_id=row["np_term_id"],
            level=level,
            name=row["name"],
            is_target_level=(row.get("is_target_level", "").strip().lower() == "true"),
        )
    parents: dict[str, list[str]] = {}
    children: dict[str, list[str]] = {}
    for row in _read_tsv(edges_path):
        c, p = row["child_np_term_id"], row["parent_np_term_id"]
        if c not in terms or p not in terms:
            raise ValueError(
                f"{edges_path}: edge {c} -> {p} references an unknown term"
            )
        if terms[c].depth != terms[p].depth + 1:
            raise ValueError(
                f"{edges_path}: edge {c} -> {p} does not go up exactly one level"
            )
        parents.setdefault(c, []).append(p)
        children.setdefault(p, []).append(c)
    by_name = {(t.level, _norm(t.name)): t.term_id for t in terms.values()}
    if len(by_name) != len(terms):
        raise ValueError(f"{terms_path}: duplicate (level, name) pairs")
    return Ontology(
        terms=terms,
        parents={k: tuple(sorted(v)) for k, v in parents.items()},
        children={k: tuple(sorted(v)) for k, v in children.items()},
        _by_name=by_name,
    )


@lru_cache(maxsize=1)
def load_ontology() -> Ontology:
    """The vendored ontology (cached)."""
    f = ontology_files()
    return load_ontology_from(f[TERMS_FILE], f[EDGES_FILE])
