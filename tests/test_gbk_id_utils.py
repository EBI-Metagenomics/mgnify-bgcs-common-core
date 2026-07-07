"""Unit tests for common_core.gbk_id_utils.unwrap_id_qualifiers_inplace."""
from __future__ import annotations

from pathlib import Path

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqFeature import FeatureLocation, SeqFeature
from Bio.SeqRecord import SeqRecord

from common_core.gbk_id_utils import unwrap_id_qualifiers_inplace


LONG_ID = "ERZ12818450.521-NODE-521-length-3301-cov-2.911275_5"


def _write_raw(records, path: Path) -> None:
    with open(path, "w") as fh:
        SeqIO.write(records, fh, "genbank")


def _cds_qualifiers(path: Path) -> dict:
    rec = next(SeqIO.parse(str(path), "genbank"))
    cds = next(f for f in rec.features if f.type == "CDS")
    return {k: str(v[0]) for k, v in cds.qualifiers.items()}


def _make_record() -> SeqRecord:
    rec = SeqRecord(Seq("A" * 3000), id="c1", name="c1")
    rec.annotations["molecule_type"] = "DNA"
    rec.annotations["topology"] = "linear"
    rec.features = [
        SeqFeature(
            location=FeatureLocation(0, len(rec.seq), strand=1),
            type="source",
            qualifiers={"note": ["test"]},
        ),
        SeqFeature(
            location=FeatureLocation(0, 300, strand=1),
            type="CDS",
            qualifiers={
                "protein_id": [LONG_ID],
                "locus_tag": [LONG_ID],
                "product": ["hypothetical protein with spaces"],
                "translation": ["M" * 100],
            },
        ),
    ]
    return rec


def test_unwrap_recovers_long_locus_tag_and_protein_id(tmp_path: Path) -> None:
    gbk = tmp_path / "wrapped.gbk"
    _write_raw([_make_record()], gbk)

    before = _cds_qualifiers(gbk)
    assert " " in before["locus_tag"]
    assert " " in before["protein_id"]

    unwrap_id_qualifiers_inplace(gbk)

    after = _cds_qualifiers(gbk)
    assert after["locus_tag"] == LONG_ID
    assert after["protein_id"] == LONG_ID


def test_unwrap_preserves_product_whitespace(tmp_path: Path) -> None:
    gbk = tmp_path / "wrapped.gbk"
    _write_raw([_make_record()], gbk)
    unwrap_id_qualifiers_inplace(gbk)
    after = _cds_qualifiers(gbk)
    assert after["product"] == "hypothetical protein with spaces"


def test_unwrap_is_idempotent(tmp_path: Path) -> None:
    gbk = tmp_path / "wrapped.gbk"
    _write_raw([_make_record()], gbk)
    unwrap_id_qualifiers_inplace(gbk)
    first = gbk.read_text()
    unwrap_id_qualifiers_inplace(gbk)
    assert gbk.read_text() == first
