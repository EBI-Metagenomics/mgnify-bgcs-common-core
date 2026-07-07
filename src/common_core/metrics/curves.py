"""Threshold sweeps and AUCPR — the §5 evaluation gaps.

- :func:`pr_curve` / :func:`aucpr` — position-level (per-annotation)
  precision-recall analysis. Cheap; safe to call every validation epoch.
- :func:`threshold_sweep` — for each threshold in ``[0, 1]`` steps ``0.005``
  by default, decode the score sequence into regions and score them with
  ``cluster_eval`` + ``orf_eval``. This is the step the SanntiS paper takes
  before picking the ``0.85`` operating threshold and drawing the AUCPR
  headline in Supplementary Figure 1c.

All heavy deps (numpy, sklearn) are imported at module load; guard the import
if you need to keep them out of a downstream install.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve

from .decoding import decode_regions
from .intervals import BGCInterval, CDSFeature, GroundTruth
from .range import cluster_eval, orf_eval


def _to_numpy(x: Any) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64).reshape(-1)


@dataclass
class PRPoint:
    """A single (precision, recall) point on a PR curve."""

    threshold: float
    precision: float
    recall: float


@dataclass
class PRCurve:
    """Position-level precision-recall curve + AUCPR."""

    thresholds: np.ndarray
    precision: np.ndarray
    recall: np.ndarray
    aucpr: float

    def as_points(self) -> list[PRPoint]:
        return [
            PRPoint(threshold=float(t), precision=float(p), recall=float(r))
            for t, p, r in zip(self.thresholds, self.precision, self.recall)
        ]


def pr_curve(scores: Any, targets: Any) -> PRCurve:
    """Position-level precision-recall curve, computed with sklearn.

    ``scores`` are sigmoid probabilities in ``[0, 1]`` (per-annotation or
    per-ORF; the aggregation choice is the caller's). ``targets`` are 0/1.
    """
    s = _to_numpy(scores)
    t = _to_numpy(targets).astype(np.int64)
    if s.shape != t.shape:
        raise ValueError(
            f"scores shape {s.shape} != targets shape {t.shape}"
        )
    precision, recall, thresholds = precision_recall_curve(t, s)
    # sklearn appends a synthetic (0, 1) endpoint so precision/recall have
    # length ``len(thresholds) + 1``. Pad ``thresholds`` with 1.0 so all three
    # arrays align element-wise for downstream plotting/logging.
    thresholds = np.concatenate([thresholds, np.array([1.0])])
    aucpr = float(average_precision_score(t, s))
    return PRCurve(
        thresholds=thresholds,
        precision=precision,
        recall=recall,
        aucpr=aucpr,
    )


def aucpr(scores: Any, targets: Any) -> float:
    """Position-level AUCPR (a.k.a. average precision).

    Cheap enough to log every validation epoch. Serves as the primary
    training-time monitoring metric per the design brief.
    """
    s = _to_numpy(scores)
    t = _to_numpy(targets).astype(np.int64)
    if s.shape != t.shape:
        raise ValueError(
            f"scores shape {s.shape} != targets shape {t.shape}"
        )
    return float(average_precision_score(t, s))


@dataclass
class ThresholdRow:
    """Aggregated cluster/range metrics at one score threshold."""

    threshold: float
    cluster_tp: int = 0
    cluster_fp: int = 0
    cluster_fn: int = 0
    cluster_precision: float = 0.0
    cluster_recall: float = 0.0
    cluster_f1: float = 0.0
    orf_precision: float = 0.0
    orf_recall: float = 0.0
    orf_f1: float = 0.0
    n_samples: int = 0
    _per_sample: list[dict] = field(default_factory=list, repr=False)


@dataclass
class SweepSample:
    """One contig's worth of data for a threshold sweep.

    Kept separate from ``GroundTruth`` so callers can supply the per-annotation
    score sequence + ORF index mapping alongside the ground-truth regions.
    """

    prefix: str
    seqid: str
    scores: np.ndarray
    orf_index: np.ndarray
    gt: GroundTruth
    all_cds: list[CDSFeature]
    orf_starts: np.ndarray | None = None
    orf_ends: np.ndarray | None = None


def threshold_sweep(
    samples: Sequence[SweepSample],
    *,
    thresholds: Sequence[float] | np.ndarray | None = None,
    step: float = 0.005,
) -> list[ThresholdRow]:
    """Cluster + range-based metrics across a grid of decoding thresholds.

    Args:
        samples: one per validation contig.
        thresholds: explicit threshold grid; if omitted, uses
            ``np.arange(0.0, 1.0 + step, step)``.
        step: step size when ``thresholds`` is omitted. ``0.005`` matches the
            paper's benchmark protocol (§4, "0→1 in steps of 0.005").

    Returns:
        One :class:`ThresholdRow` per threshold, with micro-averaged
        cluster/range metrics across all supplied samples.
    """
    if thresholds is None:
        thresholds = np.arange(0.0, 1.0 + step / 2, step)
    grid = np.asarray(thresholds, dtype=np.float64).reshape(-1)

    rows: list[ThresholdRow] = []
    for thr in grid:
        row = ThresholdRow(threshold=float(thr))
        for sample in samples:
            pred_regions: list[BGCInterval] = decode_regions(
                sample.scores,
                sample.orf_index,
                threshold=float(thr),
                orf_starts=sample.orf_starts,
                orf_ends=sample.orf_ends,
                seqid=sample.seqid,
            )
            c = cluster_eval([sample.gt.bgc_region], pred_regions)
            o = orf_eval(sample.gt, pred_regions, sample.all_cds)
            row.cluster_tp += c.tp
            row.cluster_fp += c.fp
            row.cluster_fn += c.fn
            row._per_sample.append(
                {
                    "orf_n_gt_cds": o.n_gt_cds,
                    "orf_covered_cds": o.covered_cds,
                    "orf_n_pred_cds": o.n_pred_cds,
                    "orf_correct_pred_cds": o.correct_pred_cds,
                }
            )
            row.n_samples += 1

        tp, fp, fn = row.cluster_tp, row.cluster_fp, row.cluster_fn
        row.cluster_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        row.cluster_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        row.cluster_f1 = (
            2 * row.cluster_precision * row.cluster_recall
            / (row.cluster_precision + row.cluster_recall)
            if (row.cluster_precision + row.cluster_recall) > 0
            else 0.0
        )

        n_gt = sum(r["orf_n_gt_cds"] for r in row._per_sample)
        covered = sum(r["orf_covered_cds"] for r in row._per_sample)
        n_pred = sum(r["orf_n_pred_cds"] for r in row._per_sample)
        correct = sum(r["orf_correct_pred_cds"] for r in row._per_sample)
        row.orf_recall = covered / n_gt if n_gt > 0 else 0.0
        row.orf_precision = correct / n_pred if n_pred > 0 else 0.0
        row.orf_f1 = (
            2 * row.orf_precision * row.orf_recall
            / (row.orf_precision + row.orf_recall)
            if (row.orf_precision + row.orf_recall) > 0
            else 0.0
        )
        rows.append(row)
    return rows


def sweep_aucpr(rows: Sequence[ThresholdRow], metric: str = "cluster") -> float:
    """AUC of the P–R curve traced out by a threshold sweep.

    ``metric`` is ``"cluster"`` or ``"orf"``. Uses the trapezoidal rule on
    recall-sorted points, mirroring the "AUCPR across thresholds" the paper
    reports for the 9-genomes benchmark.
    """
    if metric == "cluster":
        p = np.array([r.cluster_precision for r in rows])
        r = np.array([r.cluster_recall for r in rows])
    elif metric == "orf":
        p = np.array([r.orf_precision for r in rows])
        r = np.array([r.orf_recall for r in rows])
    else:
        raise ValueError(f"unknown metric {metric!r}; use 'cluster' or 'orf'")
    order = np.argsort(r)
    return float(np.trapezoid(p[order], r[order]))
