"""Filter contigs (and associated FAA / GBK / GFF) by minimum sequence length.

Supports any combination of input files: at least one of --fna or --gbk is
required so we can determine contig lengths. When additional files (FAA, GFF,
GBK, FNA) are provided they are filtered consistently by the set of contig IDs
that pass the length threshold.
"""

from __future__ import annotations

import argparse
import gzip
import logging
import re
from pathlib import Path
from typing import Dict, Optional, Set

from Bio import SeqIO
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

from common_core.config_loader import LoaderOptions, load_settings
from common_core.gbk_id_utils import unwrap_id_qualifiers_inplace
from common_core.io import open_text
from common_core.logging_setup import LoggingConfig, setup_logging
from common_core.versioning import dist_version

log = logging.getLogger(__name__)


def _open_seq(path: Path):
    """Open a file for BioPython SeqIO, transparently decompressing .gz."""
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path, "r")


# ---------------------------------------------------------------------------
# Settings (worker-template pattern)
# ---------------------------------------------------------------------------


class IOConfig(BaseModel):
    fna: Optional[str] = None
    faa: Optional[str] = None
    gbk: Optional[str] = None
    gff: Optional[str] = None
    out_dir: str = "."


class FilterSettings(BaseSettings):
    job_name: str = "contig-length-filter"
    min_length: int
    io: IOConfig = Field(default_factory=IOConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


# ---------------------------------------------------------------------------
# Filtering helpers
# ---------------------------------------------------------------------------


def _contig_lengths_from_fna(fna_path: Path) -> Dict[str, int]:
    """Return ``{contig_id: length}`` for every FASTA record."""
    with _open_seq(fna_path) as fh:
        return {record.id: len(record.seq) for record in SeqIO.parse(fh, "fasta")}


def _contig_lengths_from_gbk(gbk_path: Path) -> Dict[str, int]:
    """Return ``{contig_id: length}`` for every GenBank record."""
    with _open_seq(gbk_path) as fh:
        return {record.id: len(record.seq) for record in SeqIO.parse(fh, "genbank")}


def filter_fna(fna_path: Path, out_path: Path, passing: Set[str]) -> int:
    """Write only passing contigs. Return count of written records."""
    with _open_seq(fna_path) as fh:
        kept = [r for r in SeqIO.parse(fh, "fasta") if r.id in passing]
    SeqIO.write(kept, str(out_path), "fasta")
    return len(kept)


def filter_gbk(gbk_path: Path, out_path: Path, passing: Set[str]) -> int:
    """Write only GenBank records whose id is in *passing*."""
    with _open_seq(gbk_path) as fh:
        kept = [r for r in SeqIO.parse(fh, "genbank") if r.id in passing]
    SeqIO.write(kept, str(out_path), "genbank")
    unwrap_id_qualifiers_inplace(out_path)
    return len(kept)


_BRACKET_CONTIG_RE = re.compile(r"\[([^\]]+)\]")


def _contig_id_from_protein_id(protein_id: str) -> str:
    """Derive contig ID from Pyrodigal-style protein ID.

    Pyrodigal names proteins ``{contig_id}_{gene_number}``.  We strip the
    trailing ``_N`` token to recover the contig ID.
    """
    parts = protein_id.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return protein_id


def _contig_id_from_description(description: str) -> str | None:
    """Extract contig ID from bracket notation in GBK-reversed FAA headers.

    GBK-reversed FAA headers embed the source contig as the first bracketed
    token, e.g.::

        MGYG000296008_00351 [MGYG000296008_6] product=Elongation factor G

    Returns the bracketed value, or ``None`` if no bracket is found.
    """
    m = _BRACKET_CONTIG_RE.search(description)
    return m.group(1) if m else None


# CDS attributes that carry the identifier used as the FAA record ID.
# ``Name`` is deliberately excluded: Prokka/Bakta use it for gene symbols
# (e.g. ``rssB_1``), which are not unique protein identifiers.
_GFF_PROTEIN_ID_KEYS = ("ID", "locus_tag", "protein_id")
_GBK_PROTEIN_ID_KEYS = ("locus_tag", "protein_id")


def _protein_contig_map_from_gff(gff_path: Path) -> Dict[str, str]:
    """Map protein identifiers to their contig using the CDS features of a GFF.

    Covers annotators whose protein IDs do not embed the contig ID, e.g.
    Prokka locus tags (``MGYG000296006_00001`` on contig ``MGYG000296006_1``).
    """
    mapping: Dict[str, str] = {}
    with open_text(gff_path) as fh:
        for line in fh:
            if line.startswith("##FASTA"):
                break
            if line.startswith("#"):
                continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 9 or cols[2] != "CDS":
                continue
            for attr in cols[8].split(";"):
                key, sep, value = attr.partition("=")
                if sep and key.strip() in _GFF_PROTEIN_ID_KEYS and value.strip():
                    mapping.setdefault(value.strip(), cols[0])
    return mapping


def _protein_contig_map_from_gbk(gbk_path: Path) -> Dict[str, str]:
    """Map protein identifiers to their contig using the CDS features of a GBK."""
    mapping: Dict[str, str] = {}
    with _open_seq(gbk_path) as fh:
        for record in SeqIO.parse(fh, "genbank"):
            for feature in record.features:
                if feature.type != "CDS":
                    continue
                for key in _GBK_PROTEIN_ID_KEYS:
                    for value in feature.qualifiers.get(key, []):
                        mapping.setdefault(value, record.id)
    return mapping


def _resolve_contig_id(
    record_id: str,
    description: str,
    known_contigs: Set[str],
    protein_to_contig: Optional[Dict[str, str]],
) -> Optional[str]:
    """Return the contig a protein belongs to, or ``None`` if unresolvable.

    Resolution order: explicit annotation mapping (GFF/GBK CDS features),
    Pyrodigal-style ``{contig_id}_{n}`` ID, then the first bracketed token
    of a GBK-reversed FAA description.
    """
    if protein_to_contig:
        contig_id = protein_to_contig.get(record_id)
        if contig_id is not None:
            return contig_id
    contig_id = _contig_id_from_protein_id(record_id)
    if contig_id in known_contigs:
        return contig_id
    contig_id = _contig_id_from_description(description)
    if contig_id in known_contigs:
        return contig_id
    return None


def filter_faa(
    faa_path: Path,
    out_path: Path,
    passing: Set[str],
    known_contigs: Optional[Set[str]] = None,
    protein_to_contig: Optional[Dict[str, str]] = None,
) -> int:
    """Keep proteins whose parent contig is in *passing*.

    The parent contig is resolved with :func:`_resolve_contig_id`, which
    supports three protein-ID conventions:

    * **Annotation-mapped** — *protein_to_contig* (built from GFF/GBK CDS
      features) maps IDs that do not embed the contig, e.g. Prokka locus tags.
    * **Pyrodigal-style** ``{contig_id}_{gene_number}`` — contig ID is derived
      by stripping the trailing ``_N`` numeric token from the sequence ID.
    * **GBK-reversed** — the contig ID is embedded in the FASTA description as
      the first bracketed token, e.g. ``[MGYG000296008_6]``.

    *known_contigs* is the full set of contig IDs before length filtering
    (defaults to *passing*). Proteins that cannot be placed on any known
    contig are dropped and counted; the count is logged at ERROR level when
    *every* protein is unmapped (incompatible protein / contig IDs) so it is
    never silent.
    """
    known = known_contigs if known_contigs is not None else passing
    kept = []
    total = 0
    unresolved = []
    with _open_seq(faa_path) as fh:
        for r in SeqIO.parse(fh, "fasta"):
            total += 1
            contig_id = _resolve_contig_id(
                r.id, r.description, known, protein_to_contig
            )
            if contig_id is None:
                unresolved.append(r.id)
            elif contig_id in passing:
                kept.append(r)

    if unresolved:
        # Logged rather than raised: the workflow's errorStrategy ``ignore``
        # would otherwise drop the whole genome, including from runs that
        # never consume the filtered FAA (e.g. precomputed InterProScan).
        level = (
            logging.ERROR if len(unresolved) == total and passing else logging.WARNING
        )
        log.log(
            level,
            "FAA: %d of %d proteins could not be mapped to a contig and were "
            "dropped (e.g. %s). Provide a GFF/GBK whose CDS features carry these "
            "IDs (ID / locus_tag / protein_id).",
            len(unresolved),
            total,
            unresolved[:3],
        )

    SeqIO.write(kept, str(out_path), "fasta")
    return len(kept)


def filter_gff(gff_path: Path, out_path: Path, passing: Set[str]) -> int:
    """Keep GFF feature lines whose seqid (col 1) is in *passing*."""
    kept = 0
    with open_text(gff_path) as fin, open(out_path, "w") as fout:
        for line in fin:
            if line.startswith("#"):
                fout.write(line)
                continue
            cols = line.split("\t", 2)
            if cols[0] in passing:
                fout.write(line)
                kept += 1
    return kept


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run(cfg: FilterSettings) -> None:
    io = cfg.io
    out_dir = Path(io.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    has_fna = io.fna is not None
    has_gbk = io.gbk is not None

    if not has_fna and not has_gbk:
        raise SystemExit("At least one of --fna or --gbk must be provided.")

    # Determine contig lengths from whichever source is available.
    # Prefer FNA if both are given (FASTA parsing is faster).
    if has_fna:
        lengths = _contig_lengths_from_fna(Path(io.fna))
        source = "FNA"
    else:
        lengths = _contig_lengths_from_gbk(Path(io.gbk))
        source = "GBK"
    known = set(lengths)
    passing = {cid for cid, n in lengths.items() if n >= cfg.min_length}
    log.info(
        "%s: %d of %d contigs pass >= %d bp filter",
        source,
        len(passing),
        len(known),
        cfg.min_length,
    )

    if not passing:
        log.warning("No contigs passed the length filter — all outputs will be empty.")

    # Filter each provided file type.
    if has_fna:
        n = filter_fna(Path(io.fna), out_dir / "filtered.fna", passing)
        log.info("FNA: wrote %d records", n)

    if has_gbk:
        n = filter_gbk(Path(io.gbk), out_dir / "filtered.gbk", passing)
        log.info("GBK: wrote %d records", n)

    if io.faa is not None:
        # Map protein IDs to contigs from the annotation the FAA came with;
        # needed when protein IDs do not embed the contig (Prokka locus tags).
        if io.gff is not None:
            protein_to_contig = _protein_contig_map_from_gff(Path(io.gff))
        elif has_gbk:
            protein_to_contig = _protein_contig_map_from_gbk(Path(io.gbk))
        else:
            protein_to_contig = None
        n = filter_faa(
            Path(io.faa),
            out_dir / "filtered.faa",
            passing,
            known_contigs=known,
            protein_to_contig=protein_to_contig,
        )
        log.info("FAA: wrote %d records", n)

    if io.gff is not None:
        n = filter_gff(Path(io.gff), out_dir / "filtered.gff", passing)
        log.info("GFF: wrote %d feature lines", n)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Filter contigs and associated files by minimum length."
    )
    p.add_argument("--config", help="Path to YAML config")
    p.add_argument("--min-length", type=int, help="Minimum contig length (bp)")
    p.add_argument("--fna", help="Input FASTA nucleotide file")
    p.add_argument("--faa", help="Input FASTA amino-acid file")
    p.add_argument("--gbk", help="Input GenBank file")
    p.add_argument("--gff", help="Input GFF file")
    p.add_argument("--out-dir", default=".", help="Output directory (default: .)")
    p.add_argument("--log-level", help="Log level")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    cli_overrides: dict = {}
    if args.min_length is not None:
        cli_overrides["min_length"] = args.min_length

    io_overrides: dict = {}
    if args.fna:
        io_overrides["fna"] = args.fna
    if args.faa:
        io_overrides["faa"] = args.faa
    if args.gbk:
        io_overrides["gbk"] = args.gbk
    if args.gff:
        io_overrides["gff"] = args.gff
    if args.out_dir:
        io_overrides["out_dir"] = args.out_dir
    if io_overrides:
        cli_overrides["io"] = io_overrides

    if args.log_level:
        cli_overrides.setdefault("logging", {})["level"] = args.log_level

    cfg = load_settings(
        FilterSettings,
        yaml_path=args.config,
        cli_overrides=cli_overrides,
        options=LoaderOptions(env_prefix="CONTIGFILTER_"),
    )

    setup_logging(cfg.logging)

    pkg_version = dist_version(__name__)
    log.info("running %s version %s", cfg.job_name, pkg_version)

    run(cfg)
    log.info("done")


if __name__ == "__main__":
    main()
