# mgnify-bgcs-common-core

Shared Python utilities for the MGnify BGCs services. It is used by the
[ETL pipelines](https://github.com/EBI-Metagenomics/mgnify-bgcs-etl) and the
[BGC data portal](https://github.com/EBI-Metagenomics/bgc_data_portal).

## Install

Consumers pin a commit SHA:

```bash
pip install "mgnify-bgcs-common-core @ git+https://github.com/EBI-Metagenomics/mgnify-bgcs-common-core.git@<sha>"
```

Pin a full commit SHA until v2.0.0 is released, then pin that tag. Never pin
`@v1.9.1`: it predates the `npc` package, which both consumers import.

For development:

```bash
pip install -e ".[dev,hpc-cpu]"
pytest
ruff check .
```

Optional extras:

- `dev`: pytest and ruff.
- `hpc-cpu`: the clustering stack used by `bgc-cluster` (numpy, scipy, pyarrow,
  joblib, `igraph`, `leidenalg`). `igraph` and `leidenalg` are GPL-licensed.
  Without this extra the clustering tests skip.

## Command-line tools

| Command | Purpose | Used by |
|---|---|---|
| `contig-length-filter` | Drops contigs below a minimum length from matching FNA/FAA/GBK/GFF files | ETL (`CONTIG_LENGTH_FILTER` step) |
| `bgc-cluster` | iBGC similarity and hierarchical Leiden clustering on HPC (`run`, `project-partials`) | Portal clustering cycle, see `docs/runbooks/services/bgc_data_portal/clustering-compute.md` in the monorepo |

## Contents

| Module | Purpose | Imported by |
|---|---|---|
| `bgc_class` | BGC class normalisation | portal |
| `clustering` | iBGC similarity, weights, scoring and Leiden clustering | portal, `bgc-cluster` |
| `npc` | Vendored NPClassifier ontology | portal, ETL |
| `contig_length_filter` | Contig-length filter | `contig-length-filter` |
| `gbk_id_utils`, `io`, `helpers` | GenBank id handling and I/O helpers | ETL |
| `config_loader`, `logging_setup`, `versioning` | Configuration, logging and tool-version helpers | ETL |

## Clustering tarball contract

The portal exports an input tarball (`manage.py export_clustering_inputs`),
`bgc-cluster run` turns it into an output tarball, and the portal imports that.
Both sides read and write it only through `common_core.clustering.io`; the
members and the schema version (`SCHEMA_VERSION` in `clustering/schema.py`) are
documented in the `clustering/io.py` module docstring. Change the schema there,
bump `SCHEMA_VERSION`, and keep older tarballs readable.

## Licence

Apache-2.0. See [LICENSE](LICENSE).
