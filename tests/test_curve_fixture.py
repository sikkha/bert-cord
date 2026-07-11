"""Smoke test on the tiny tracked metrics fixture (end-to-end analyze + summary + plots)."""

from __future__ import annotations

import json
import os

from coordinator_bert.curve_analysis import AnalysisConfig, analyze, load_metrics
from coordinator_bert.curve_plots import generate_plots, write_summary_json

_FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "curve_metrics.jsonl")


def test_fixture_loads_and_analyzes():
    records = load_metrics(_FIXTURE)
    assert len(records) == 10
    r = analyze(records, AnalysisConfig(run_id="fixture", future_steps=(2000, 4000)))
    # A clean decreasing curve with no instability and enough points -> CONTINUE or PLATEAU.
    assert r.status in ("CONTINUE", "PLATEAU")
    assert r.chosen_model in ("power", "exp", "sqrt")
    assert r.instability["has_nan_inf"] is False
    # best_val_loss is measured on the EMA-smoothed series (lags the raw minimum of 2.01).
    assert r.best_val_loss < 2.5


def test_fixture_end_to_end_plots_and_summary(tmp_path):
    records = load_metrics(_FIXTURE)
    r = analyze(records, AnalysisConfig(run_id="fixture"))
    paths = generate_plots(r, str(tmp_path), formats="png")
    # gradient_norm + masked_accuracy + learning_rate present -> all six figures.
    for name in ("validation_loss_curve", "perplexity_curve", "task_metric_curve",
                 "learning_rate_curve", "gradient_norm_curve", "improvement_rate_curve"):
        assert name in paths
    sp = str(tmp_path / "analysis_summary.json")
    write_summary_json(r, sp)
    summary = json.load(open(sp))
    assert summary["status"] == r.status
