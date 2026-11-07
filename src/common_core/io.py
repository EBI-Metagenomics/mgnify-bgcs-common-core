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

from __future__ import annotations

import gzip
from pathlib import Path


def open_text(path: str | Path, mode: str = "rt", encoding: str = "utf-8"):
    """Open a text file, transparently decompressing .gz if needed."""
    opener = gzip.open if str(path).endswith(".gz") else open
    if opener is gzip.open and mode in ("r", "w", "a", "x"):
        mode = mode + "t"
    return opener(path, mode, encoding=encoding)
