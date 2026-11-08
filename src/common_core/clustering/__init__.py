#!/usr/bin/env python3

# Copyright 2026 EMBL - European Bioinformatics Institute
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""HPC-side BGC clustering: similarity (composite Dice or weighted cosine), KNN,
hierarchical Leiden, and novelty.

The portal exports per-iBGC signature matrices (``export_clustering_inputs``)
and an HPC job (``bgc-cluster run``) returns the hierarchy, partial-iBGC
projections and novelty scores; the tarball contract is in ``io.py``. Everything
in this package is pure compute, no Django. Parallelism is opt-in and bounded:
most stages are single-threaded because the libraries they call don't benefit
from worker pools.
"""

from common_core.clustering.schema import (
    ClusteringInputs,
    ClusteringOutputs,
    RunParams,
)

__all__ = ["ClusteringInputs", "ClusteringOutputs", "RunParams"]
