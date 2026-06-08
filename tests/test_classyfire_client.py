"""Tests for the ClassyFire client (SMILES → ChemOnt terms).

HTTP is stubbed at the module's ``_http_get_json`` / ``_http_post_json``
helpers so no network is touched; RDKit (a real dependency) computes the
InChIKey.
"""

from __future__ import annotations

import pytest

from common_core.chemont import classyfire_client as cf

# Ethanol — trivially valid for RDKit.
SMILES = "CCO"

# A minimal ClassyFire entity record covering all node shapes the parser reads.
ENTITY = {
    "kingdom": {"chemont_id": "CHEMONTID:0000000", "name": "Organic compounds"},
    "superclass": {"chemont_id": "CHEMONTID:0000012", "name": "Lipids"},
    "class": {"chemont_id": "CHEMONTID:0000259", "name": "Prenol lipids"},
    "subclass": None,
    "direct_parent": {"chemont_id": "CHEMONTID:0001755", "name": "Diterpene glycosides"},
    "intermediate_nodes": [{"chemont_id": "CHEMONTID:0002012", "name": "Oxanes"}],
    "alternative_parents": [
        {"chemont_id": "CHEMONTID:0000147", "name": "Macrolides"},
        {"chemont_id": "CHEMONTID:0000147", "name": "Macrolides (dup)"},  # dedup
    ],
}


def test_invalid_smiles_returns_none():
    assert cf.classify("not a smiles!!!") is None


def test_entity_cache_hit_skips_submission(monkeypatch):
    """A compound already in ClassyFire's entity cache must not be submitted."""
    monkeypatch.setattr(cf, "_http_get_json", lambda url, timeout: dict(ENTITY))

    def _no_post(*a, **k):  # pragma: no cover - must never run
        raise AssertionError("submission attempted despite cache hit")

    monkeypatch.setattr(cf, "_http_post_json", _no_post)

    result = cf.classify(SMILES)

    assert result is not None
    assert result.direct_parent == "CHEMONTID:0001755"
    # Order preserved (single nodes first), duplicates removed.
    assert result.chemont_ids == [
        "CHEMONTID:0000000",
        "CHEMONTID:0000012",
        "CHEMONTID:0000259",
        "CHEMONTID:0001755",
        "CHEMONTID:0002012",
        "CHEMONTID:0000147",
    ]


def test_submit_and_poll_on_cache_miss(monkeypatch):
    """Entity cache 404 → submit a query and poll until Done."""
    monkeypatch.setattr(cf, "_http_get_json", _fake_get_seq([
        None,                                              # entity cache miss (404)
        {"classification_status": "In progress"},          # first poll
        {"classification_status": "Done", "entities": [dict(ENTITY)]},  # second poll
    ]))
    monkeypatch.setattr(cf, "_http_post_json", lambda url, payload, timeout: {"id": 42})
    monkeypatch.setattr(cf.time, "sleep", lambda s: None)

    result = cf.classify(SMILES, poll_interval=0)

    assert result is not None
    assert "CHEMONTID:0000259" in result.chemont_ids


def test_unreachable_service_raises(monkeypatch):
    def _boom(url, timeout):
        raise cf.ClassyFireUnavailable("connection refused")

    monkeypatch.setattr(cf, "_http_get_json", _boom)

    with pytest.raises(cf.ClassyFireUnavailable):
        cf.classify(SMILES)


def test_classified_but_no_terms_returns_empty(monkeypatch):
    """A Done query with no entity → result with empty term list, not an error."""
    monkeypatch.setattr(cf, "_http_get_json", _fake_get_seq([
        None,
        {"classification_status": "Done", "entities": []},
    ]))
    monkeypatch.setattr(cf, "_http_post_json", lambda url, payload, timeout: {"id": 7})
    monkeypatch.setattr(cf.time, "sleep", lambda s: None)

    result = cf.classify(SMILES, poll_interval=0)

    assert result is not None
    assert result.chemont_ids == []


def _fake_get_seq(responses):
    """Return a ``_http_get_json`` stub that yields ``responses`` in order."""
    it = iter(responses)

    def _get(url, timeout):
        return next(it)

    return _get
