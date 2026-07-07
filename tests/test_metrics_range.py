"""Tests for the pure-Python range/cluster BGC-detection metrics."""
from __future__ import annotations

from common_core.metrics import (
    BGCInterval,
    CDSFeature,
    GroundTruth,
    aggregate_macro,
    aggregate_micro,
    cluster_eval,
    fbeta,
    orf_eval,
)


def _cds(tag: str, start: int, end: int) -> CDSFeature:
    return CDSFeature(locus_tag=tag, start=start, end=end)


def test_fbeta_zero_denominator():
    assert fbeta(0.0, 0.0, 1.0) == 0.0


def test_fbeta_beta1_is_harmonic_mean():
    assert fbeta(0.5, 0.5, 1.0) == 0.5


def test_cluster_eval_all_correct():
    gt = [BGCInterval("c", 100, 200)]
    pred = [BGCInterval("c", 120, 180)]
    s = cluster_eval(gt, pred)
    assert (s.tp, s.fp, s.fn) == (1, 0, 0)
    assert s.f1 == 1.0


def test_cluster_eval_all_wrong():
    gt = [BGCInterval("c", 100, 200)]
    pred = [BGCInterval("c", 300, 400)]
    s = cluster_eval(gt, pred)
    assert (s.tp, s.fp, s.fn) == (0, 1, 1)
    assert s.precision == 0.0
    assert s.recall == 0.0


def test_orf_eval_perfect():
    all_cds = [_cds("a", 0, 10), _cds("b", 10, 20), _cds("c", 20, 30)]
    gt_region = BGCInterval("c", 0, 20)
    gt_cds = [c for c in all_cds if c.start < 20]
    gt = GroundTruth(prefix="x", bgc_region=gt_region, bgc_cds=gt_cds, all_cds=all_cds)
    pred = [BGCInterval("c", 0, 20)]
    s = orf_eval(gt, pred, all_cds)
    assert s.avg_recall_t == 1.0
    assert s.avg_precision_t == 1.0
    assert s.f1 == 1.0


def test_orf_eval_no_predictions():
    all_cds = [_cds("a", 0, 10)]
    gt_region = BGCInterval("c", 0, 10)
    gt = GroundTruth(prefix="x", bgc_region=gt_region, bgc_cds=all_cds, all_cds=all_cds)
    s = orf_eval(gt, [], all_cds)
    assert s.n_gt_cds == 1
    assert s.n_pred_cds == 0
    assert s.avg_recall_t == 0.0
    assert s.f1 == 0.0


def test_aggregate_micro_and_macro_agree_on_uniform_rows():
    rows = [
        {
            "cluster_tp": 1, "cluster_fp": 0, "cluster_fn": 0,
            "cluster_precision": 1.0, "cluster_recall": 1.0,
            "orf_avg_precision_t": 1.0, "orf_avg_recall_t": 1.0,
            "orf_n_gt_cds": 4, "orf_covered_cds": 4,
            "orf_n_pred_cds": 4, "orf_correct_pred_cds": 4,
        }
        for _ in range(3)
    ]
    micro = aggregate_micro(rows)
    macro = aggregate_macro(rows)
    assert micro["cluster_f1"] == 1.0
    assert macro["cluster_f1"] == 1.0
    assert micro["orf_f1"] == 1.0
    assert macro["orf_f1"] == 1.0
