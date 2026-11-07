# mgnify-bgcs-common-core

Shared Python utilities for the MGnify BGCs services. It is used by the
[ETL pipelines](https://github.com/EBI-Metagenomics/mgnify-bgcs-etl) and the
[BGC data portal](https://github.com/EBI-Metagenomics/bgc_data_portal).

## Install

Consumers pin a commit SHA:

```bash
pip install "mgnify-bgcs-common-core @ git+https://github.com/EBI-Metagenomics/mgnify-bgcs-common-core.git@<sha>"
```

For development:

```bash
pip install -e ".[dev]"
pytest
ruff check .
```

Optional extras:

- `hpc-cpu`: the clustering stack used by `bgc-cluster`. It pulls in the GPL-licensed `igraph` and `leidenalg`.
- `ml`: model-side helpers.

## Contents

| Module | Purpose |
|---|---|
| `bgc_class` | BGC class normalisation |
| `clustering` | iBGC similarity and Leiden clustering (`bgc-cluster` CLI) |
| `npc` | Vendored NPClassifier ontology |
| `contig_length_filter` | Contig-length filter (`contig-length-filter` CLI) |
| `gbk_id_utils`, `io`, `helpers` | GenBank id handling and I/O helpers |
| `config_loader`, `logging_setup`, `versioning` | Configuration, logging and tool-version helpers |

## Licence

Apache-2.0. See [LICENSE](LICENSE).
