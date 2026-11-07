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

"""BGC product-class normalisation.

Turns per-tool ``classification_path`` strings (antiSMASH, GECCO, SanntiS,
mibig) into one of nine canonical labels, with raw paths preserved upstream
for per-prediction display.
"""

from .bgc_types import (
    ANTISMASH_CATEGORY_TO_BASE,
    ANTISMASH_TYPE_TO_CATEGORY,
    BASE_CLASSES,
    BASE_NAME_TO_CLASS,
    CANONICAL_LABELS,
)
from .normalize import (
    categories_for,
    classify_ibgc,
    normalize_single,
    reduce_to_label,
)

__all__ = [
    "ANTISMASH_CATEGORY_TO_BASE",
    "ANTISMASH_TYPE_TO_CATEGORY",
    "BASE_CLASSES",
    "BASE_NAME_TO_CLASS",
    "CANONICAL_LABELS",
    "categories_for",
    "classify_ibgc",
    "normalize_single",
    "reduce_to_label",
]
