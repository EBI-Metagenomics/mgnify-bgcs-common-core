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

"""Shared BGC-detection metric primitives.

This package is the single source of truth for the metric definitions used both
during training (per-epoch validation, threshold sweeps) and by the post-hoc
``bgc-eval`` benchmark. The split keeps the two paths in numeric lock-step and
lets the portal / ETL side of the monorepo depend on ``common_core`` without
pulling in ``torch``.

Modules:

- ``intervals``: pure-Python interval data types (``BGCInterval``, ``CDSFeature``,
  ``GroundTruth``) and helpers. No heavy deps.
- ``range``: event-based (cluster) and range-based (ORF / average) scoring —
  Precision, Recall, F1/F2/F0.5, plus micro/macro aggregation. Pure Python.
- ``curves``: threshold sweeps and AUCPR. Requires ``scikit-learn``.
- ``decoding``: per-ORF max normalisation + contiguous-above-threshold region
  calling. NumPy-only (no torch); accepts either NumPy arrays or torch tensors.
- ``losses``: focal loss for the detection ANN. Requires ``torch``.

The heavy-dep modules (``curves``, ``decoding``, ``losses``) are imported
lazily; importing ``common_core.metrics`` on its own does not import torch.
"""
from __future__ import annotations

from .intervals import BGCInterval, CDSFeature, GroundTruth, cds_in_region
from .range import (
    ClusterStats,
    ORFStats,
    aggregate_macro,
    aggregate_micro,
    cluster_eval,
    fbeta,
    orf_eval,
)

__all__ = [
    "BGCInterval",
    "CDSFeature",
    "ClusterStats",
    "GroundTruth",
    "ORFStats",
    "aggregate_macro",
    "aggregate_micro",
    "cds_in_region",
    "cluster_eval",
    "fbeta",
    "orf_eval",
]
