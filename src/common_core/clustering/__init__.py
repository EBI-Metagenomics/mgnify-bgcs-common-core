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

"""HPC-side BGC clustering: composite similarity, KNN, hierarchical Leiden, UMAP.

The portal exports per-iBGC signature matrices and asks an HPC job to produce
the hierarchy, coordinates, and novelty scores. The ``bgc-cluster`` CLI is
the entry point; everything in this package is pure compute, no Django.

Parallelism is opt-in and bounded — most stages are single-threaded because
the libraries they call don't benefit from worker pools. See the plan in
``/Users/fragoso/.claude/plans/`` (or the README) for the rationale.
"""

from common_core.clustering.schema import (
    ClusteringInputs,
    ClusteringOutputs,
    RunParams,
)

__all__ = ["ClusteringInputs", "ClusteringOutputs", "RunParams"]
