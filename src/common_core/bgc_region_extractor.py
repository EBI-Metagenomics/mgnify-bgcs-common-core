"""Extract per-BGC GenBank sub-records from a whole-genome GBK and BGC GFF.

Each BGC region found in the GFF is sliced from the matching genome record and
written as a standalone GenBank file suitable for tools like CHAMOIS.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

from common_core.config_loader import LoaderOptions, load_settings
from common_core.logging_setup import LoggingConfig, setup_logging
from common_core.versioning import dist_version

log = logging.getLogger(__name__)

# Default GFF feature types per caller.
CALLER_FEATURE_TYPES = {
    "gecco": "biosynthetic_gene_cluster",
    "sanntis": "biosynthetic_gene_cluster",
    "antismash": "region",
}


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


class IOConfig(BaseModel):
    gbk: str
    gff: str
    out_dir: str = "."


class ExtractorSettings(BaseSettings):
    job_name: str = "bgc-region-extractor"
    io: IOConfig
    caller_name: str = "unknown"
    feature_type: Optional[str] = None
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


# ---------------------------------------------------------------------------
# GFF parsing
# ---------------------------------------------------------------------------


def parse_bgc_regions(
    gff_path: Path, feature_type: str
) -> List[Tuple[str, int, int, str]]:
    """Parse a GFF and return BGC region coordinates.

    Returns a list of ``(seqid, start, end, attributes)`` tuples.
    Coordinates are converted to 0-based half-open to match BioPython slicing.
    """
    regions: List[Tuple[str, int, int, str]] = []
    ft_lower = feature_type.lower()

    with open(gff_path, "r") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 9:
                continue
            if cols[2].lower() == ft_lower:
                seqid = cols[0]
                start = int(cols[3]) - 1  # GFF is 1-based inclusive
                end = int(cols[4])  # GFF end is inclusive → half-open
                attrs = cols[8]
                regions.append((seqid, start, end, attrs))

    return regions


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def extract_bgc_regions(
    genome_records: Dict[str, SeqRecord],
    regions: List[Tuple[str, int, int, str]],
    caller_name: str,
    prefix: str,
    out_dir: Path,
) -> int:
    """Extract and write per-BGC GenBank files. Return count written."""
    written = 0
    for idx, (seqid, start, end, _attrs) in enumerate(regions, 1):
        record = genome_records.get(seqid)
        if record is None:
            log.warning(
                "Contig %s from GFF not found in GBK — skipping region %d", seqid, idx
            )
            continue

        # Clamp coordinates to record length.
        end = min(end, len(record.seq))
        start = max(start, 0)

        sub = record[start:end]
        sub.id = f"{prefix}_{caller_name}_bgc_{idx}"
        sub.name = sub.id
        sub.description = (
            f"BGC region {idx} from {caller_name} | {seqid}:{start + 1}-{end}"
        )

        out_path = out_dir / f"{sub.id}.gbk"
        SeqIO.write([sub], str(out_path), "genbank")
        written += 1

    return written


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run(cfg: ExtractorSettings) -> None:
    io = cfg.io
    out_dir = Path(io.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve feature type: explicit > caller default > fallback
    feature_type = cfg.feature_type
    if feature_type is None:
        feature_type = CALLER_FEATURE_TYPES.get(
            cfg.caller_name.lower(), "biosynthetic_gene_cluster"
        )
    log.info(
        "Extracting BGC regions (caller=%s, feature_type=%s)",
        cfg.caller_name,
        feature_type,
    )

    # Parse genome GBK into dict keyed by record.id
    gbk_path = Path(io.gbk)
    genome_records: Dict[str, SeqRecord] = {
        r.id: r for r in SeqIO.parse(str(gbk_path), "genbank")
    }
    log.info("Loaded %d genome records from %s", len(genome_records), gbk_path.name)

    # Parse BGC regions from GFF
    gff_path = Path(io.gff)
    regions = parse_bgc_regions(gff_path, feature_type)
    log.info("Found %d BGC regions in %s", len(regions), gff_path.name)

    if not regions:
        log.info("No BGC regions found — nothing to extract.")
        return

    # Derive prefix from GBK filename (strip extension)
    prefix = gbk_path.stem
    written = extract_bgc_regions(genome_records, regions, cfg.caller_name, prefix, out_dir)
    log.info("Wrote %d per-BGC GenBank files to %s", written, out_dir)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract per-BGC GenBank sub-records from genome GBK + BGC GFF."
    )
    p.add_argument("--config", help="Path to YAML config")
    p.add_argument("--gbk", required=True, help="Whole-genome GenBank file")
    p.add_argument("--gff", required=True, help="BGC annotation GFF file")
    p.add_argument("--caller-name", default="unknown", help="Name of BGC caller")
    p.add_argument(
        "--feature-type",
        help="GFF feature type to extract (default: auto from caller name)",
    )
    p.add_argument("--out-dir", default=".", help="Output directory")
    p.add_argument("--log-level", help="Log level")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    cli_overrides: dict = {}
    io_overrides: dict = {"gbk": args.gbk, "gff": args.gff}
    if args.out_dir:
        io_overrides["out_dir"] = args.out_dir
    cli_overrides["io"] = io_overrides

    if args.caller_name:
        cli_overrides["caller_name"] = args.caller_name
    if args.feature_type:
        cli_overrides["feature_type"] = args.feature_type
    if args.log_level:
        cli_overrides.setdefault("logging", {})["level"] = args.log_level

    cfg = load_settings(
        ExtractorSettings,
        yaml_path=args.config,
        cli_overrides=cli_overrides,
        options=LoaderOptions(env_prefix="BGCEXTRACT_"),
    )

    setup_logging(cfg.logging)

    pkg_version = dist_version(__name__)
    log.info("running %s version %s", cfg.job_name, pkg_version)

    run(cfg)
    log.info("done")


if __name__ == "__main__":
    main()
