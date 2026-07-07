"""Decode per-position BGC probabilities into called regions.

Implements the SanntiS §3.4 decoding step:

1. **Per-ORF max normalisation** — several annotations map to the same ORF;
   the score for the ORF is the maximum score over its annotations.
2. **Contiguous-above-threshold region calls** — any run of consecutive ORFs
   whose scores stay ``>= threshold`` becomes one predicted BGC region.

Inputs are NumPy arrays or Python sequences; torch tensors are accepted and
detached to CPU numpy. Nothing here requires torch itself.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from .intervals import BGCInterval


def _to_numpy(x: Any) -> np.ndarray:
    """Coerce torch tensors, lists, or arrays to a 1-D float numpy array."""
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    return arr


def per_orf_max(
    scores: Any,
    orf_index: Sequence[int] | np.ndarray,
    n_orfs: int | None = None,
) -> np.ndarray:
    """Collapse per-annotation scores to per-ORF scores by taking the max.

    Args:
        scores: per-annotation BGC probability, shape ``(n_annotations,)``.
        orf_index: for each annotation position, the index of the ORF it
            belongs to. Length must match ``scores``. ORF indices are the
            positional index of an ORF on the contig, so ties between
            annotations of the same ORF collapse to a single value.
        n_orfs: total number of ORFs on the contig. Inferred from
            ``max(orf_index) + 1`` when omitted; pass explicitly when trailing
            ORFs have no annotations.

    Returns:
        Length-``n_orfs`` float array. ORFs with no annotations get 0.0.
    """
    s = _to_numpy(scores)
    idx = np.asarray(orf_index, dtype=np.int64).reshape(-1)
    if s.shape != idx.shape:
        raise ValueError(
            f"scores length {s.shape[0]} != orf_index length {idx.shape[0]}"
        )
    if n_orfs is None:
        n_orfs = int(idx.max()) + 1 if idx.size else 0

    out = np.zeros(n_orfs, dtype=np.float64)
    if s.size == 0:
        return out
    # np.maximum.at is an unbuffered scatter-max; correct under duplicate
    # indices, unlike ``out[idx] = np.maximum(out[idx], s)``.
    np.maximum.at(out, idx, s)
    return out


def contiguous_above_threshold(
    per_orf_scores: np.ndarray,
    threshold: float,
    *,
    orf_starts: Sequence[int] | np.ndarray | None = None,
    orf_ends: Sequence[int] | np.ndarray | None = None,
    seqid: str = "",
) -> list[BGCInterval]:
    """Emit one interval per maximal run of ORFs with score ``>= threshold``.

    Args:
        per_orf_scores: length-``n_orfs`` per-ORF score array.
        threshold: score cutoff; SanntiS default is 0.85.
        orf_starts / orf_ends: optional nucleotide coordinates per ORF. When
            provided, emitted intervals use nucleotide coordinates
            (``[first_orf_start, last_orf_end)``); when omitted, coordinates
            are ORF-index-based (``[first_orf_idx, last_orf_idx + 1)``),
            useful for training-time metrics that never leave ORF space.
        seqid: label for the emitted intervals.
    """
    scores = np.asarray(per_orf_scores, dtype=np.float64).reshape(-1)
    if scores.size == 0:
        return []
    above = scores >= threshold
    if not above.any():
        return []

    use_nt = orf_starts is not None and orf_ends is not None
    if use_nt:
        starts = np.asarray(orf_starts, dtype=np.int64).reshape(-1)
        ends = np.asarray(orf_ends, dtype=np.int64).reshape(-1)
        if starts.shape[0] != scores.shape[0] or ends.shape[0] != scores.shape[0]:
            raise ValueError("orf_starts / orf_ends must match per_orf_scores length")

    intervals: list[BGCInterval] = []
    n = scores.shape[0]
    i = 0
    while i < n:
        if not above[i]:
            i += 1
            continue
        j = i
        while j < n and above[j]:
            j += 1
        if use_nt:
            intervals.append(
                BGCInterval(seqid=seqid, start=int(starts[i]), end=int(ends[j - 1]))
            )
        else:
            intervals.append(BGCInterval(seqid=seqid, start=i, end=j))
        i = j
    return intervals


def decode_regions(
    scores: Any,
    orf_index: Sequence[int] | np.ndarray,
    threshold: float,
    *,
    n_orfs: int | None = None,
    orf_starts: Sequence[int] | np.ndarray | None = None,
    orf_ends: Sequence[int] | np.ndarray | None = None,
    seqid: str = "",
) -> list[BGCInterval]:
    """End-to-end decoder: per-annotation scores → called BGC regions."""
    per_orf = per_orf_max(scores, orf_index, n_orfs=n_orfs)
    return contiguous_above_threshold(
        per_orf,
        threshold,
        orf_starts=orf_starts,
        orf_ends=orf_ends,
        seqid=seqid,
    )
