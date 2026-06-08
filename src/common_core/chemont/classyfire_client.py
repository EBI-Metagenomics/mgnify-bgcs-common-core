"""ClassyFire client: SMILES → ChemOnt term set via the ClassyFire web API.

The deployed ChemOnt OBO ships no SMARTS, and ClassyFire's own SMARTS rule set
is not public, so a local structure→ChemOnt classifier is not feasible. Instead
we classify a query SMILES with the canonical ClassyFire service and reuse the
ChemOnt terms it returns (the same vocabulary CHAMOIS produces for stored BGCs).

Resolution order for a query, fastest first:
  1. ``GET /entities/{InChIKey}.json`` — ClassyFire's own cache; instant for any
     already-classified compound (most known natural products).
  2. ``POST /queries.json`` + poll ``GET /queries/{id}.json`` — runs the
     classification for a novel structure; slower and rate-limited.

This module is intentionally free of Django/Redis: callers (e.g. the portal
Celery task) layer their own InChIKey cache on top of :func:`classify`.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://classyfire.wishartlab.com"

# Single-node ChemOnt assignments present on every ClassyFire entity record.
_SINGLE_NODE_KEYS = ("kingdom", "superclass", "class", "subclass", "direct_parent")
# List-valued node collections.
_LIST_NODE_KEYS = ("intermediate_nodes", "alternative_parents")


class ClassyFireError(Exception):
    """Base error for ClassyFire interactions."""


class ClassyFireUnavailable(ClassyFireError):
    """The ClassyFire service could not be reached or did not respond in time.

    Distinct from "classified but no terms" so callers can surface a
    retryable "try again later" state rather than a silent empty result.
    """


@dataclass
class ClassyFireResult:
    """Parsed ChemOnt classification for one structure."""

    inchikey: str
    chemont_ids: list[str] = field(default_factory=list)
    direct_parent: str = ""
    classification: dict = field(default_factory=dict)


def smiles_to_inchikey(smiles: str) -> str | None:
    """Return the InChIKey for a SMILES, or ``None`` if RDKit can't parse it."""
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToInchiKey(mol)


def _extract_chemont_ids(entity: dict) -> tuple[list[str], str]:
    """Pull the ChemOnt id set (and direct-parent id) out of an entity record.

    Order is preserved kingdom→direct_parent then the list collections, with
    duplicates removed. The hierarchy walk for similarity happens downstream
    (Resnik/BMA over the ontology), so direct nodes are sufficient here.
    """
    ids: list[str] = []
    seen: set[str] = set()

    def add(node: dict | None) -> None:
        if not node:
            return
        cid = node.get("chemont_id")
        if cid and cid not in seen:
            seen.add(cid)
            ids.append(cid)

    for key in _SINGLE_NODE_KEYS:
        add(entity.get(key))
    for key in _LIST_NODE_KEYS:
        for node in entity.get(key) or []:
            add(node)

    direct = (entity.get("direct_parent") or {}).get("chemont_id", "")
    return ids, direct


def _http_get_json(url: str, timeout: float) -> dict | None:
    """GET a JSON document. Returns ``None`` on HTTP 404, raises on other errors."""
    req = urllib.request.Request(
        url, headers={"Accept": "application/json", "User-Agent": "bgc-data-portal"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise ClassyFireUnavailable(f"ClassyFire GET {url} → HTTP {exc.code}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise ClassyFireUnavailable(f"ClassyFire GET {url} failed: {exc}") from exc


def _http_post_json(url: str, payload: dict, timeout: float) -> dict:
    """POST a JSON body and return the parsed JSON response."""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "bgc-data-portal",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise ClassyFireUnavailable(
            f"ClassyFire POST {url} → HTTP {exc.code}"
        ) from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise ClassyFireUnavailable(f"ClassyFire POST {url} failed: {exc}") from exc


def _lookup_entity(base_url: str, inchikey: str, timeout: float) -> dict | None:
    """Try ClassyFire's entity cache for an already-classified compound."""
    return _http_get_json(f"{base_url}/entities/{inchikey}.json", timeout)


def _submit_and_poll(
    base_url: str,
    smiles: str,
    *,
    timeout: float,
    poll_timeout: float,
    poll_interval: float,
    label: str,
) -> dict | None:
    """Submit a structure query and poll until classification finishes.

    Returns the first classified entity dict, or ``None`` if the query
    completed with no classification.
    """
    submission = _http_post_json(
        f"{base_url}/queries.json",
        {"label": label, "query_input": smiles, "query_type": "STRUCTURE"},
        timeout,
    )
    query_id = submission.get("id")
    if not query_id:
        raise ClassyFireUnavailable(
            f"ClassyFire submission returned no query id: {submission!r}"
        )

    deadline = time.monotonic() + poll_timeout
    poll_url = f"{base_url}/queries/{query_id}.json"
    while time.monotonic() < deadline:
        doc = _http_get_json(poll_url, timeout)
        status = (doc or {}).get("classification_status", "")
        if status == "Done":
            entities = doc.get("entities") or []
            return entities[0] if entities else None
        # "In Queue" / "In progress" → keep waiting.
        time.sleep(poll_interval)

    raise ClassyFireUnavailable(
        f"ClassyFire query {query_id} did not finish within {poll_timeout:.0f}s"
    )


def classify(
    smiles: str,
    *,
    base_url: str = DEFAULT_BASE_URL,
    timeout: float = 30.0,
    poll_timeout: float = 90.0,
    poll_interval: float = 3.0,
    label: str = "bgc-data-portal",
) -> ClassyFireResult | None:
    """Classify a SMILES into ChemOnt terms via ClassyFire.

    Returns a :class:`ClassyFireResult` (possibly with an empty
    ``chemont_ids`` if ClassyFire classified the structure but assigned no
    terms), or ``None`` if RDKit cannot parse the SMILES.

    Raises :class:`ClassyFireUnavailable` if the service can't be reached or
    a novel-structure query times out — callers should treat this as a
    retryable failure, not "no results".
    """
    inchikey = smiles_to_inchikey(smiles)
    if inchikey is None:
        log.warning("invalid SMILES, cannot classify: %s", smiles[:60])
        return None

    base_url = base_url.rstrip("/")

    entity = _lookup_entity(base_url, inchikey, timeout)
    source = "entity-cache"
    if entity is None:
        entity = _submit_and_poll(
            base_url,
            smiles,
            timeout=timeout,
            poll_timeout=poll_timeout,
            poll_interval=poll_interval,
            label=label,
        )
        source = "query"

    if not entity:
        log.info("ClassyFire returned no classification for %s (%s)", inchikey, source)
        return ClassyFireResult(inchikey=inchikey)

    chemont_ids, direct = _extract_chemont_ids(entity)
    log.info(
        "ClassyFire classified %s (%s): %d ChemOnt terms",
        inchikey, source, len(chemont_ids),
    )
    return ClassyFireResult(
        inchikey=inchikey,
        chemont_ids=chemont_ids,
        direct_parent=direct,
        classification=entity,
    )
