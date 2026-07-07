"""Interval data types used by both range-based scoring and decoding.

Coordinates are 0-based, half-open ``[start, end)`` throughout the metrics
package. GFF parsers upstream are responsible for converting 1-based inclusive
GFF3 coordinates.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BGCInterval:
    """A genomic interval representing a BGC region."""

    seqid: str
    start: int
    end: int


@dataclass
class CDSFeature:
    """A CDS feature extracted from a GenBank record."""

    locus_tag: str
    start: int
    end: int


@dataclass
class GroundTruth:
    """Ground truth for a single synthetic contig.

    ``bgc_cds`` is the subset of ``all_cds`` that any-overlaps ``bgc_region``.
    """

    prefix: str
    bgc_region: BGCInterval
    bgc_cds: list[CDSFeature]
    all_cds: list[CDSFeature]


def cds_in_region(cds_list: list[CDSFeature], region: BGCInterval) -> list[CDSFeature]:
    """Return CDS features that any-overlap the region."""
    return [c for c in cds_list if c.start < region.end and c.end > region.start]


def overlap(a: BGCInterval, b: BGCInterval) -> bool:
    """Return True if two intervals any-overlap.

    ``seqid`` is intentionally ignored: predictions from different tools use
    different sequence identifiers even for the same contig, and BGC-scale
    scoring only cares about coordinate overlap.
    """
    return a.start < b.end and a.end > b.start
