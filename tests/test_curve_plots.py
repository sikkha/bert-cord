"""Tests for the training-curve plotting module (headless, matplotlib Agg).

Annotations are asserted by reading the generated SVG text (matplotlib embeds legend/label
text in SVG), which is a robust, backend-safe way to check what was drawn.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from coordinator_bert.curve_analysis import AnalysisConfig, analyze
from coordinator_bert.curve_plots import (
    generate_plots,
    write_markdown_report,
    write_summary_json,
)


def _records(steps, val, **extra):
    out = []
    for i, s in enumerate(steps):
        r = {"step": int(s), "tokens_seen": int(s) * 1000, "val_loss": float(val[i])}
        for k, v in extra.items():
            if v is not None:
                r[k] = float(v[i])
        out.append(r)
    return out


def _cfg(**kw):
    base = dict(min_evals=6, min_fit_points=6, patience=4, future_steps=(3000, 5000), n_boot=60)
    base.update(kw)
    return AnalysisConfig(**base)


def _svg_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


def test_plots_created_for_valid_metrics(tmp_path):
    steps = np.arange(1, 25) * 100
    val = 2.0 + 8.0 * steps.astype(float) ** (-0.5)
    r = analyze(_records(steps, val,
                         masked_accuracy=np.linspace(0.0, 0.5, steps.size),
                         learning_rate=np.full(steps.size, 5e-4),
                         gradient_norm=np.full(steps.size, 1.0)),
                _cfg(run_id="valid"))
    paths = generate_plots(r, str(tmp_path), formats="png", show_confidence=True)
    # All six figures should be produced when every field is present.
    for name in ("validation_loss_curve", "perplexity_curve", "task_metric_curve",
                 "learning_rate_curve", "gradient_norm_curve", "improvement_rate_curve"):
        assert name in paths, f"missing {name}"
        for p in paths[name]:
            assert os.path.exists(p) and os.path.getsize(p) > 0


def test_both_formats_produced(tmp_path):
    steps = np.arange(1, 20) * 100
    val = 2.0 + 6.0 * steps.astype(float) ** (-0.5)
    r = analyze(_records(steps, val), _cfg(run_id="fmt"))
    paths = generate_plots(r, str(tmp_path), formats="both")
    vpaths = paths["validation_loss_curve"]
    assert any(p.endswith(".png") for p in vpaths)
    assert any(p.endswith(".svg") for p in vpaths)


def test_flat_curve_includes_plateau_annotation(tmp_path):
    steps = np.arange(1, 22) * 100
    val = 2.0 + 5e-4 * np.random.default_rng(0).standard_normal(steps.size)
    r = analyze(_records(steps, val), _cfg(run_id="flat"))
    assert r.status == "PLATEAU"
    paths = generate_plots(r, str(tmp_path), formats="svg")
    svg = _svg_text([p for p in paths["validation_loss_curve"] if p.endswith(".svg")][0])
    assert "plateau region" in svg


def test_unstable_curve_marks_spikes(tmp_path):
    steps = np.arange(1, 22) * 100
    val = (2.0 + 8.0 * steps.astype(float) ** (-0.5))
    val[16] = val[16] * 3.0
    r = analyze(_records(steps, val), _cfg(run_id="spike"))
    assert r.status == "UNSTABLE"
    assert r.instability["loss_spike_steps"]
    paths = generate_plots(r, str(tmp_path), formats="svg")
    svg = _svg_text([p for p in paths["validation_loss_curve"] if p.endswith(".svg")][0])
    assert "instability" in svg  # spike marker legend label present


def test_insufficient_data_makes_observed_only_no_forecast(tmp_path):
    steps = np.array([100, 200, 300])
    val = np.array([3.0, 2.9, 2.85])
    r = analyze(_records(steps, val), _cfg(run_id="few"))
    assert r.status == "INSUFFICIENT_DATA"
    assert r.forecast == {}  # no fit -> no forecast
    paths = generate_plots(r, str(tmp_path), formats="svg")
    # Observed validation curve still drawn...
    assert "validation_loss_curve" in paths
    svg = _svg_text([p for p in paths["validation_loss_curve"] if p.endswith(".svg")][0])
    # ...but with no forecast series.
    assert "forecast" not in svg
    assert "observed val loss" in svg


def test_missing_gradient_norm_skips_figure(tmp_path):
    steps = np.arange(1, 20) * 100
    val = 2.0 + 6.0 * steps.astype(float) ** (-0.5)
    r = analyze(_records(steps, val), _cfg(run_id="nograd"))  # no gradient_norm field
    paths = generate_plots(r, str(tmp_path), formats="png")
    assert "gradient_norm_curve" not in paths      # skipped, not failed
    assert "validation_loss_curve" in paths        # others still produced


def test_missing_task_and_lr_fields_skip_those_figures(tmp_path):
    steps = np.arange(1, 20) * 100
    val = 2.0 + 6.0 * steps.astype(float) ** (-0.5)
    r = analyze(_records(steps, val), _cfg(run_id="minimal"))
    paths = generate_plots(r, str(tmp_path), formats="png")
    assert "task_metric_curve" not in paths
    assert "learning_rate_curve" not in paths
    assert "gradient_norm_curve" not in paths
    assert "validation_loss_curve" in paths and "perplexity_curve" in paths


def test_summary_json_matches_recommendation(tmp_path):
    steps = np.arange(1, 22) * 100
    val = 2.0 + 5e-4 * np.random.default_rng(3).standard_normal(steps.size)  # flat -> plateau
    r = analyze(_records(steps, val), _cfg(run_id="match"))
    sp = str(tmp_path / "analysis_summary.json")
    write_summary_json(r, sp)
    with open(sp) as fh:
        summary = json.load(fh)
    assert summary["status"] == r.status
    assert summary["recommended_stop_step"] == r.recommended_stop_step
    assert summary["run_id"] == "match"
    assert summary["best_val_loss"] == r.best_val_loss


def test_markdown_report_written_and_links_plots(tmp_path):
    steps = np.arange(1, 24) * 100
    val = 2.0 + 8.0 * steps.astype(float) ** (-0.5)
    r = analyze(_records(steps, val, gradient_norm=np.full(steps.size, 1.0)),
                _cfg(run_id="report"))
    paths = generate_plots(r, str(tmp_path), formats="png")
    rp = str(tmp_path / "training_curve_report.md")
    write_markdown_report(r, paths, rp)
    text = open(rp).read()
    assert "Training-curve analysis" in text
    assert "Heuristic" in text  # limitation disclaimer present
    assert "validation_loss_curve.png" in text  # figure linked
    assert f"`{r.status}`" in text


def test_generate_plots_headless_does_not_require_display(tmp_path, monkeypatch):
    # Simulate a server with no DISPLAY; Agg must still work.
    monkeypatch.delenv("DISPLAY", raising=False)
    steps = np.arange(1, 18) * 100
    val = 2.0 + 6.0 * steps.astype(float) ** (-0.5)
    r = analyze(_records(steps, val), _cfg(run_id="headless"))
    paths = generate_plots(r, str(tmp_path), formats="png")
    assert paths and os.path.exists(paths["validation_loss_curve"][0])
