"""Unit tests for common_core.contig_length_filter."""

from __future__ import annotations

import logging
from pathlib import Path

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqFeature import FeatureLocation, SeqFeature
from Bio.SeqRecord import SeqRecord

from common_core.contig_length_filter import (
    FilterSettings,
    IOConfig,
    _protein_contig_map_from_gbk,
    _protein_contig_map_from_gff,
    filter_faa,
    run,
)

# Two contigs: one passes a 3000 bp threshold, one does not.
LONG = "MGYG000296006_1"
SHORT = "MGYG000296006_2"
LENGTHS = {LONG: 5000, SHORT: 1000}


def _write_fasta(path: Path, entries: list[tuple[str, str, str]]) -> None:
    """entries: (id, description, sequence)."""
    with open(path, "w") as fh:
        for rid, desc, seq in entries:
            fh.write(f">{rid} {desc}".rstrip() + f"\n{seq}\n")


def _write_fna(path: Path) -> None:
    _write_fasta(path, [(cid, "", "A" * n) for cid, n in LENGTHS.items()])


def _write_prokka_gff(path: Path) -> None:
    # Prokka-style: protein IDs are locus tags that do not embed the contig ID.
    rows = [
        (LONG, 837, 1904, "MGYG000296006_00001"),
        (LONG, 2148, 2705, "MGYG000296006_00002"),
        (SHORT, 10, 600, "MGYG000296006_00003"),
    ]
    with open(path, "w") as fh:
        fh.write("##gff-version 3\n")
        for seqid, start, end, pid in rows:
            fh.write(
                f"{seqid}\tProdigal:002006\tCDS\t{start}\t{end}\t.\t+\t0\t"
                f"ID={pid};Name=rssB_1;locus_tag={pid};product=hypothetical protein\n"
            )
        fh.write("##FASTA\n>ignored\nACGT\n")


def _ids(path: Path) -> list[str]:
    return [r.id for r in SeqIO.parse(str(path), "fasta")]


def test_gff_map_handles_prokka_locus_tags(tmp_path: Path) -> None:
    gff = tmp_path / "g.gff"
    _write_prokka_gff(gff)
    mapping = _protein_contig_map_from_gff(gff)
    assert mapping["MGYG000296006_00001"] == LONG
    assert mapping["MGYG000296006_00003"] == SHORT
    # Gene symbols in Name= are not used as protein identifiers.
    assert "rssB_1" not in mapping


def test_prokka_faa_kept_with_gff_mapping(tmp_path: Path) -> None:
    """Regression: catalogue FAAs used to be emptied entirely."""
    faa = tmp_path / "p.faa"
    gff = tmp_path / "g.gff"
    out = tmp_path / "out.faa"
    _write_prokka_gff(gff)
    _write_fasta(
        faa,
        [
            ("MGYG000296006_00001", "hypothetical protein", "MKV"),
            ("MGYG000296006_00002", "Regulator of RpoS", "MKL"),
            ("MGYG000296006_00003", "hypothetical protein", "MKA"),
        ],
    )
    n = filter_faa(
        faa,
        out,
        passing={LONG},
        known_contigs=set(LENGTHS),
        protein_to_contig=_protein_contig_map_from_gff(gff),
    )
    assert n == 2
    assert _ids(out) == ["MGYG000296006_00001", "MGYG000296006_00002"]


def test_pyrodigal_ids_without_mapping(tmp_path: Path) -> None:
    faa = tmp_path / "p.faa"
    out = tmp_path / "out.faa"
    _write_fasta(
        faa,
        [(f"{LONG}_1", "# 1 # 300 # 1", "MKV"), (f"{SHORT}_1", "# 1 # 300 # 1", "MKL")],
    )
    n = filter_faa(faa, out, passing={LONG}, known_contigs=set(LENGTHS))
    assert n == 1
    assert _ids(out) == [f"{LONG}_1"]


def test_bracket_description_fallback(tmp_path: Path) -> None:
    faa = tmp_path / "p.faa"
    out = tmp_path / "out.faa"
    _write_fasta(
        faa,
        [
            ("MGYG000296006_00351", f"[{LONG}] product=Elongation factor G", "MKV"),
            ("MGYG000296006_00352", f"[{SHORT}] product=hypothetical", "MKL"),
        ],
    )
    n = filter_faa(faa, out, passing={LONG}, known_contigs=set(LENGTHS))
    assert n == 1
    assert _ids(out) == ["MGYG000296006_00351"]


def test_gbk_map_uses_locus_tag_and_protein_id(tmp_path: Path) -> None:
    rec = SeqRecord(Seq("A" * 5000), id=LONG, name="c1")
    rec.annotations["molecule_type"] = "DNA"
    rec.features = [
        SeqFeature(
            FeatureLocation(0, 300, strand=1),
            type="CDS",
            qualifiers={"locus_tag": ["LT_00001"], "protein_id": ["WP_000001.1"]},
        )
    ]
    gbk = tmp_path / "g.gbk"
    SeqIO.write([rec], str(gbk), "genbank")
    mapping = _protein_contig_map_from_gbk(gbk)
    assert mapping == {"LT_00001": LONG, "WP_000001.1": LONG}


def test_all_unmapped_logs_error(tmp_path: Path, caplog) -> None:
    faa = tmp_path / "p.faa"
    out = tmp_path / "out.faa"
    _write_fasta(faa, [("MGYG000296006_00001", "hypothetical protein", "MKV")])
    with caplog.at_level(logging.WARNING):
        n = filter_faa(faa, out, passing={LONG}, known_contigs=set(LENGTHS))
    assert n == 0
    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert errors and "could not be mapped" in errors[0].getMessage()


def test_length_filtered_proteins_are_not_reported_unmapped(
    tmp_path: Path, caplog
) -> None:
    faa = tmp_path / "p.faa"
    out = tmp_path / "out.faa"
    _write_fasta(faa, [(f"{SHORT}_1", "", "MKV"), (f"{LONG}_1", "", "MKL")])
    with caplog.at_level(logging.WARNING):
        filter_faa(faa, out, passing={LONG}, known_contigs=set(LENGTHS))
    assert not [r for r in caplog.records if "could not be mapped" in r.getMessage()]


def test_run_end_to_end_with_prokka_inputs(tmp_path: Path) -> None:
    fna = tmp_path / "g.fna"
    faa = tmp_path / "g.faa"
    gff = tmp_path / "g.gff"
    _write_fna(fna)
    _write_prokka_gff(gff)
    _write_fasta(
        faa,
        [
            ("MGYG000296006_00001", "hypothetical protein", "MKV"),
            ("MGYG000296006_00003", "hypothetical protein", "MKA"),
        ],
    )
    out_dir = tmp_path / "filtered"
    cfg = FilterSettings(
        min_length=3000,
        io=IOConfig(fna=str(fna), faa=str(faa), gff=str(gff), out_dir=str(out_dir)),
    )
    run(cfg)
    assert _ids(out_dir / "filtered.fna") == [LONG]
    assert _ids(out_dir / "filtered.faa") == ["MGYG000296006_00001"]
    gff_lines = [
        line
        for line in (out_dir / "filtered.gff").read_text().splitlines()
        if not line.startswith("#")
    ]
    assert [line.split("\t")[0] for line in gff_lines] == [LONG, LONG]
