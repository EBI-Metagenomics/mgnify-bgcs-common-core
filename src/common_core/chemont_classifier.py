"""ChemOnt chemical classification worker.

CLI usage::

    chemont-classify --smiles "CCO"
    chemont-classify --input compounds.tsv --output results.tsv
    chemont-classify --config chemont.yaml --input compounds.tsv --output results.tsv

Library usage::

    from common_core.chemont import classify_smiles, lookup_chemont_ids
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

from common_core.chemont.classifier import classify_smiles
from common_core.chemont.ontology import get_ontology
from common_core.config_loader import LoaderOptions, load_settings
from common_core.logging_setup import LoggingConfig, setup_logging
from common_core.versioning import dist_version

log = logging.getLogger(__name__)


class ChemOntConfig(BaseModel):
    obo_path: str = "/data/chemont/ChemOnt_2_1.obo"


class IOConfig(BaseModel):
    smiles: str | None = None
    input: str | None = None
    output: str | None = None
    smiles_column: str = "smiles"
    id_column: str | None = "id"


class ClassifierSettings(BaseSettings):
    job_name: str = "chemont-classify"
    chemont: ChemOntConfig = Field(default_factory=ChemOntConfig)
    io: IOConfig = Field(default_factory=IOConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Classify SMILES against ChemOnt ontology via SMARTS matching"
    )
    p.add_argument("--config", help="Path to YAML config for this job")
    p.add_argument("--job-name")
    p.add_argument("--obo-path", help="Path to ChemOnt OBO file")
    p.add_argument("--smiles", help="Single SMILES string to classify")
    p.add_argument("--input", help="Input TSV file with SMILES column")
    p.add_argument("--output", help="Output TSV file (default: stdout)")
    p.add_argument(
        "--smiles-column", help="Column name containing SMILES (default: smiles)"
    )
    p.add_argument(
        "--id-column", help="Column name for compound ID (default: id)"
    )
    p.add_argument("--log-level")
    return p.parse_args()


def _run_single(cfg: ClassifierSettings) -> None:
    """Classify a single SMILES and print results."""
    smiles = cfg.io.smiles
    assert smiles is not None
    ont = get_ontology(cfg.chemont.obo_path)
    results = classify_smiles(smiles, ontology=ont)

    if not results:
        print(f"No ChemOnt matches for: {smiles}")
        return

    print(f"ChemOnt classification for: {smiles}")
    print(f"{'Rank':<6}{'ChemOnt ID':<22}{'Name':<40}{'Depth':<6}")
    print("-" * 74)
    for rank, r in enumerate(results, 1):
        print(f"{rank:<6}{r.chemont_id:<22}{r.name:<40}{r.depth:<6}")


def _run_batch(cfg: ClassifierSettings) -> None:
    """Classify SMILES from a TSV file."""
    assert cfg.io.input is not None
    ont = get_ontology(cfg.chemont.obo_path)

    out_fh = (
        open(cfg.io.output, "w", newline="")
        if cfg.io.output
        else sys.stdout
    )
    try:
        writer = csv.writer(out_fh, delimiter="\t")
        writer.writerow(
            ["id", "smiles", "chemont_id", "chemont_name", "smarts", "depth", "rank"]
        )

        with open(cfg.io.input, newline="") as in_fh:
            reader = csv.DictReader(in_fh, delimiter="\t")
            smiles_col = cfg.io.smiles_column
            id_col = cfg.io.id_column

            if reader.fieldnames and smiles_col not in reader.fieldnames:
                raise SystemExit(
                    f"Column '{smiles_col}' not found in input. "
                    f"Available: {reader.fieldnames}"
                )

            processed = 0
            for row in reader:
                smi = row[smiles_col]
                compound_id = row.get(id_col or "", "") if id_col else ""
                results = classify_smiles(smi, ontology=ont)
                for rank, r in enumerate(results, 1):
                    writer.writerow(
                        [compound_id, smi, r.chemont_id, r.name, r.smarts, r.depth, rank]
                    )
                processed += 1
                if processed % 1000 == 0:
                    log.info("processed %d compounds", processed)

            log.info("finished: %d compounds processed", processed)
    finally:
        if out_fh is not sys.stdout:
            out_fh.close()


def run(cfg: ClassifierSettings) -> None:
    """Run the classifier in single or batch mode."""
    if cfg.io.smiles:
        _run_single(cfg)
    elif cfg.io.input:
        _run_batch(cfg)
    else:
        raise SystemExit(
            "Provide either --smiles for single mode or --input for batch mode."
        )


def main() -> None:
    args = parse_args()

    cli_overrides: dict = {}
    if args.job_name:
        cli_overrides["job_name"] = args.job_name
    if args.obo_path:
        cli_overrides.setdefault("chemont", {})["obo_path"] = args.obo_path
    if args.smiles:
        cli_overrides.setdefault("io", {})["smiles"] = args.smiles
    if args.input:
        cli_overrides.setdefault("io", {})["input"] = args.input
    if args.output:
        cli_overrides.setdefault("io", {})["output"] = args.output
    if args.smiles_column:
        cli_overrides.setdefault("io", {})["smiles_column"] = args.smiles_column
    if args.id_column:
        cli_overrides.setdefault("io", {})["id_column"] = args.id_column
    if args.log_level:
        cli_overrides.setdefault("logging", {})["level"] = args.log_level

    cfg = load_settings(
        ClassifierSettings,
        yaml_path=args.config,
        cli_overrides=cli_overrides,
        options=LoaderOptions(env_prefix="CHEMONT_"),
    )

    setup_logging(cfg.logging)

    pkg_version = dist_version(__name__)
    log.info("running %s version %s", cfg.job_name, pkg_version)
    log.info("config: %s", cfg.model_dump())

    run(cfg)


if __name__ == "__main__":
    main()
