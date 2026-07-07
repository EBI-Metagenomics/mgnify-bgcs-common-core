"""Repair identifier qualifiers in Biopython-written GenBank files.

Biopython's ``SeqIO.write(..., "genbank")`` wraps quoted qualifier values at
~45 chars of value (79-col line width minus the 21-col indent and framing).
On re-read Biopython joins continuation lines for non-``translation``
qualifiers by inserting a space, corrupting single-token identifiers like
``/locus_tag`` and ``/protein_id``. Downstream consumers (InterProScan
matching, CHAMOIS, our own metadata collector) then miss the join and drop
or corrupt rows.

The identifier qualifiers written by our pipeline never contain legitimate
whitespace, so we post-process the freshly-written file and join wrapped
continuation lines back into a single line. We do NOT touch ``/product``
(may legitimately contain spaces) or ``/translation`` (Biopython joins it
without spaces already).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

_INDENT = " " * 21
_ID_QUAL_RE = re.compile(r'^ {21}/(locus_tag|protein_id|gene)="')


def unwrap_id_qualifiers_inplace(
    gbk_path: str | Path,
    qualifiers: Iterable[str] = ("locus_tag", "protein_id", "gene"),
) -> None:
    """Rewrite *gbk_path* with wrapped id qualifier values joined onto one line.

    Safe to call multiple times; a value that already fits on one line is
    passed through unchanged.
    """
    qual_re = re.compile(
        r'^ {21}/(' + "|".join(re.escape(q) for q in qualifiers) + r')="'
    )
    p = Path(gbk_path)
    text = p.read_text()
    lines = text.split("\n")
    out: list[str] = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        if qual_re.match(line) and line.count('"') < 2:
            joined = line
            j = i + 1
            while j < n:
                cont = lines[j]
                if not cont.startswith(_INDENT):
                    break
                tail = cont[len(_INDENT):]
                if tail.lstrip().startswith("/"):
                    break
                joined = joined.rstrip() + tail.strip()
                j += 1
                if '"' in tail:
                    break
            out.append(joined)
            i = j
        else:
            out.append(line)
            i += 1
    p.write_text("\n".join(out))
