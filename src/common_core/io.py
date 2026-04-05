from __future__ import annotations

import gzip
from pathlib import Path


def open_text(path: str | Path, mode: str = "rt", encoding: str = "utf-8"):
    """Open a text file, transparently decompressing .gz if needed."""
    opener = gzip.open if str(path).endswith(".gz") else open
    return opener(path, mode, encoding=encoding)
