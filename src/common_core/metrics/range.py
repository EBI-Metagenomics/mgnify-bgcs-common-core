"""Event-based (cluster) and range-based (ORF) BGC-detection scoring.

Implements the two-metric protocol from the SanntiS paper (§5 of the baseline
strategy) and the range-based F1_T family from Tatbul 2018 that
`03-ml_problem-framing.md` designates as the primary optimisation target.

- ``cluster_eval`` — cluster/event-based TP/FP/FN via any-overlap.
- ``orf_eval`` — per-region Recall_T and per-prediction Precision_T with raw
  counts retained for downstream micro-averaging.
- ``aggregate_micro`` / ``aggregate_macro`` — cross-sample aggregation over
  the row dicts emitted by ``bgc-eval``.
"""
from __future__ import annotations

from dataclasses import dataclass

from .intervals import BGCInterval, CDSFeature, GroundTruth, cds_in_region, overlap


def fbeta(precision: float, recall: float, beta: float) -> float:
    """Fβ score. Returns 0.0 when the denominator is zero."""
    b2 = beta * beta
    denom = b2 * precision + recall
    return (1 + b2) * precision * recall / denom if denom > 0 else 0.0


@dataclass
class ClusterStats:
    """Cluster-based evaluation results for a single sample."""

    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
    f2: float
    f0_5: float


@dataclass
class ORFStats:
    """ORF-based (range-based) results for a single sample.

    Raw counts (``n_gt_cds``, ``covered_cds``, ``n_pred_cds``,
    ``correct_pred_cds``) are kept so a true micro-average can be recomputed
    by summing across samples.
    """

    avg_recall_t: float
    avg_precision_t: float
    f1: float
    f2: float
    f0_5: float
    n_gt_cds: int
    covered_cds: int
    n_pred_cds: int
    correct_pred_cds: int


def cluster_eval(
    gt_regions: list[BGCInterval],
    pred_regions: list[BGCInterval],
) -> ClusterStats:
    """Cluster-based evaluation.

    TP: predicted region overlapping ≥1 ground-truth region.
    FP: predicted region not overlapping any ground-truth region.
    FN: ground-truth region not overlapping any predicted region.
    """
    tp = sum(1 for p in pred_regions if any(overlap(p, g) for g in gt_regions))
    fp = len(pred_regions) - tp
    fn = sum(1 for g in gt_regions if not any(overlap(g, p) for p in pred_regions))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return ClusterStats(
        tp=tp,
        fp=fp,
        fn=fn,
        precision=precision,
        recall=recall,
        f1=fbeta(precision, recall, 1.0),
        f2=fbeta(precision, recall, 2.0),
        f0_5=fbeta(precision, recall, 0.5),
    )


def orf_eval(
    gt: GroundTruth,
    pred_regions: list[BGCInterval],
    all_contig_cds: list[CDSFeature],
) -> ORFStats:
    """Range-based (ORF-level) evaluation.

    Recall_T  = |CDS in gt_region ∩ any pred_region| / |CDS in gt_region|
    Precision_T per predicted region
              = |CDS in pred_region ∩ gt_region| / |CDS in pred_region|

    ``avg_recall_t`` collapses to the single Recall_T for this sample (one BGC
    per synthetic contig); ``avg_precision_t`` is the mean of per-predicted-
    region Precision_T values.
    """
    n_gt_cds = len(gt.bgc_cds)
    gt_cds_set = {cds.locus_tag for cds in gt.bgc_cds}

    if not pred_regions:
        return ORFStats(
            avg_recall_t=0.0,
            avg_precision_t=0.0,
            f1=0.0,
            f2=0.0,
            f0_5=0.0,
            n_gt_cds=n_gt_cds,
            covered_cds=0,
            n_pred_cds=0,
            correct_pred_cds=0,
        )

    covered_cds = sum(
        1
        for cds in gt.bgc_cds
        if any(cds.start < p.end and cds.end > p.start for p in pred_regions)
    )
    recall_t = covered_cds / n_gt_cds if n_gt_cds > 0 else 0.0

    precision_t_vals: list[float] = []
    total_pred_cds = 0
    total_correct = 0
    for pred in pred_regions:
        pred_cds = cds_in_region(all_contig_cds, pred)
        n_pred = len(pred_cds)
        n_correct = sum(1 for cds in pred_cds if cds.locus_tag in gt_cds_set)
        total_pred_cds += n_pred
        total_correct += n_correct
        precision_t_vals.append(n_correct / n_pred if n_pred > 0 else 0.0)

    avg_precision_t = sum(precision_t_vals) / len(precision_t_vals)

    return ORFStats(
        avg_recall_t=recall_t,
        avg_precision_t=avg_precision_t,
        f1=fbeta(avg_precision_t, recall_t, 1.0),
        f2=fbeta(avg_precision_t, recall_t, 2.0),
        f0_5=fbeta(avg_precision_t, recall_t, 0.5),
        n_gt_cds=n_gt_cds,
        covered_cds=covered_cds,
        n_pred_cds=total_pred_cds,
        correct_pred_cds=total_correct,
    )


def aggregate_micro(rows: list[dict]) -> dict:
    """Micro-average: sum raw TP/FP/FN and ORF counts, then compute metrics."""
    if not rows:
        return {}

    tp = sum(r["cluster_tp"] for r in rows)
    fp = sum(r["cluster_fp"] for r in rows)
    fn = sum(r["cluster_fn"] for r in rows)
    c_p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    c_r = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    n_gt = sum(r["orf_n_gt_cds"] for r in rows)
    covered = sum(r["orf_covered_cds"] for r in rows)
    n_pred = sum(r["orf_n_pred_cds"] for r in rows)
    correct = sum(r["orf_correct_pred_cds"] for r in rows)
    o_r = covered / n_gt if n_gt > 0 else 0.0
    o_p = correct / n_pred if n_pred > 0 else 0.0

    return {
        "avg_mode": "micro",
        "n_samples": len(rows),
        "cluster_tp": tp,
        "cluster_fp": fp,
        "cluster_fn": fn,
        "cluster_precision": round(c_p, 4),
        "cluster_recall": round(c_r, 4),
        "cluster_f1": round(fbeta(c_p, c_r, 1.0), 4),
        "cluster_f2": round(fbeta(c_p, c_r, 2.0), 4),
        "cluster_f0_5": round(fbeta(c_p, c_r, 0.5), 4),
        "orf_precision": round(o_p, 4),
        "orf_recall": round(o_r, 4),
        "orf_f1": round(fbeta(o_p, o_r, 1.0), 4),
        "orf_f2": round(fbeta(o_p, o_r, 2.0), 4),
        "orf_f0_5": round(fbeta(o_p, o_r, 0.5), 4),
    }


def aggregate_macro(rows: list[dict]) -> dict:
    """Macro-average: average per-sample precision/recall, then compute F-scores.

    This is the macro-average Range-based F1_T family that
    ``03-ml_problem-framing.md`` designates as the primary optimisation target.
    """
    if not rows:
        return {}

    def _mean(key: str) -> float:
        vals = [r[key] for r in rows if r.get(key) is not None]
        return sum(vals) / len(vals) if vals else 0.0

    c_p, c_r = _mean("cluster_precision"), _mean("cluster_recall")
    o_p, o_r = _mean("orf_avg_precision_t"), _mean("orf_avg_recall_t")

    return {
        "avg_mode": "macro",
        "n_samples": len(rows),
        "cluster_precision": round(c_p, 4),
        "cluster_recall": round(c_r, 4),
        "cluster_f1": round(fbeta(c_p, c_r, 1.0), 4),
        "cluster_f2": round(fbeta(c_p, c_r, 2.0), 4),
        "cluster_f0_5": round(fbeta(c_p, c_r, 0.5), 4),
        "orf_precision": round(o_p, 4),
        "orf_recall": round(o_r, 4),
        "orf_f1": round(fbeta(o_p, o_r, 1.0), 4),
        "orf_f2": round(fbeta(o_p, o_r, 2.0), 4),
        "orf_f0_5": round(fbeta(o_p, o_r, 0.5), 4),
    }
