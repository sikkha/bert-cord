"""Tests for the conservative training-curve analysis (status logic + robust fitting)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from coordinator_bert.curve_analysis import (
    AnalysisConfig,
    analyze,
    fit_curves,
    load_metrics,
    select_best_fit,
    to_series,
)


def _records(steps, val_loss, **extra):
    out = []
    for i, s in enumerate(steps):
        r = {"step": int(s), "tokens_seen": int(s) * 1000, "val_loss": float(val_loss[i])}
        for k, v in extra.items():
            if v is not None:
                r[k] = float(v[i])
        out.append(r)
    return out


def _cfg(**kw):
    base = dict(min_evals=6, min_fit_points=6, patience=4, future_steps=(1000, 2000),
               n_boot=80)
    base.update(kw)
    return AnalysisConfig(**base)


# --------------------------------------------------------------------------------------- #
# Status classification
# --------------------------------------------------------------------------------------- #
def test_improving_curve_is_continue():
    steps = np.arange(1, 21) * 100
    val = 2.0 + 8.0 * steps.astype(float) ** (-0.5)
    r = analyze(_records(steps, val), _cfg(run_id="improving"))
    assert r.status == "CONTINUE"
    assert r.chosen_model in ("power", "exp", "sqrt")
    # Asymptote should be well below the current loss for an improving curve.
    assert r.predicted_asymptote["point"] < r.best_val_loss


def test_flat_curve_is_plateau():
    steps = np.arange(1, 21) * 100
    val = 2.0 + 5e-4 * np.random.default_rng(0).standard_normal(steps.size)
    r = analyze(_records(steps, val), _cfg(run_id="flat"))
    assert r.status == "PLATEAU"
    assert r.recommended_stop_step is not None


def test_noisy_but_improving_is_continue():
    steps = np.arange(1, 26) * 100
    val = 2.0 + 8.0 * steps.astype(float) ** (-0.5) + 0.05 * np.random.default_rng(1).standard_normal(steps.size)
    r = analyze(_records(steps, val), _cfg(run_id="noisy"))
    assert r.status == "CONTINUE"
    assert not r.instability["loss_spike_steps"]  # noise must not be flagged as a spike


def test_spike_curve_is_unstable():
    steps = np.arange(1, 21) * 100
    val = 2.0 + 8.0 * steps.astype(float) ** (-0.5)
    val[15] = val[15] * 3.0  # inject a large spike
    r = analyze(_records(steps, val), _cfg(run_id="spike"))
    assert r.status == "UNSTABLE"
    assert 1600.0 in r.instability["loss_spike_steps"]


def test_nan_curve_is_unstable():
    steps = np.arange(1, 21) * 100
    val = 2.0 + 8.0 * steps.astype(float) ** (-0.5)
    val = val.tolist()
    val[10] = float("nan")  # reported NaN
    r = analyze(_records(steps, val), _cfg(run_id="nan"))
    assert r.status == "UNSTABLE"
    assert r.instability["has_nan_inf"] is True
    assert 1100.0 in r.instability["nan_inf_steps"]


def test_inf_is_unstable():
    steps = np.arange(1, 21) * 100
    val = (2.0 + 8.0 * steps.astype(float) ** (-0.5)).tolist()
    val[12] = float("inf")
    r = analyze(_records(steps, val), _cfg(run_id="inf"))
    assert r.status == "UNSTABLE"
    assert r.instability["has_nan_inf"] is True


def test_too_few_points_is_insufficient_data():
    steps = np.array([100, 200, 300])
    val = np.array([3.0, 2.8, 2.7])
    r = analyze(_records(steps, val), _cfg(run_id="few"))
    assert r.status == "INSUFFICIENT_DATA"
    assert r.chosen_model is None
    assert any("evaluation" in w for w in r.warnings)


# --------------------------------------------------------------------------------------- #
# Robustness: never crash on degenerate input
# --------------------------------------------------------------------------------------- #
@pytest.mark.parametrize("records", [
    [],
    [{"step": 1, "val_loss": 2.0}],
    [{"step": s, "val_loss": 2.0} for s in range(10)],            # constant loss
    [{"step": s, "val_loss": float("nan")} for s in range(10)],   # all NaN
    [{"step": 0, "val_loss": 1.0}, {"step": 0, "val_loss": 1.0}], # zero step-spread
    [{"step": s, "val_loss": 1e300} for s in range(10)],          # huge values
    [{"foo": "bar"}, {"step": "x", "val_loss": "y"}],             # junk / non-numeric
])
def test_analyze_never_crashes_on_degenerate_input(records):
    r = analyze(records, _cfg(run_id="degenerate"))
    assert r.status in ("CONTINUE", "PLATEAU", "UNSTABLE", "INSUFFICIENT_DATA")
    # Result must be JSON-serializable.
    json.dumps(r.to_dict())


def test_fit_curves_returns_empty_on_too_few_points():
    steps = np.array([1.0, 2.0, 3.0])
    vals = np.array([3.0, 2.0, 1.5])
    assert fit_curves(steps, vals, min_points=6) == []
    assert select_best_fit([]) is None


def test_fit_curves_selects_reasonable_model_on_power_data():
    steps = np.arange(1, 40) * 50.0
    vals = 1.5 + 6.0 * steps ** (-0.5)
    fits = fit_curves(steps, vals, min_points=6)
    best = select_best_fit(fits)
    assert best is not None
    # Recovered asymptote should be close to the true 1.5.
    assert abs(best.asymptote - 1.5) < 0.3
    assert best.r2 > 0.95


# --------------------------------------------------------------------------------------- #
# Loading CSV / JSONL
# --------------------------------------------------------------------------------------- #
def test_load_jsonl_and_csv_roundtrip(tmp_path):
    steps = np.arange(1, 13) * 100
    val = 2.0 + 5.0 * steps.astype(float) ** (-0.5)
    recs = _records(steps, val, learning_rate=[1e-4] * len(steps))

    jl = tmp_path / "m.jsonl"
    with open(jl, "w") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")
    loaded = load_metrics(str(jl))
    assert len(loaded) == len(recs)

    csvp = tmp_path / "m.csv"
    import csv as _csv
    with open(csvp, "w", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=["step", "tokens_seen", "val_loss", "learning_rate"])
        w.writeheader()
        for r in recs:
            w.writerow({k: r.get(k, "") for k in
                        ["step", "tokens_seen", "val_loss", "learning_rate"]})
    loaded_csv = load_metrics(str(csvp))
    s = to_series(loaded_csv)
    assert s.finite_val_count() == len(recs)
    assert s.present["learning_rate"] is True


def test_missing_val_loss_rows_are_ignored_not_flagged():
    # Non-eval rows carry train_loss only; their missing val_loss must NOT be flagged NaN.
    recs = []
    for s in range(1, 40):
        row = {"step": s * 50, "train_loss": 3.0 - s * 0.01}
        if s % 5 == 0:  # sparse eval rows
            row["val_loss"] = 3.0 - s * 0.02
        recs.append(row)
    r = analyze(recs, _cfg(min_evals=5, min_fit_points=5, run_id="sparse"))
    assert r.instability["has_nan_inf"] is False
    assert r.status in ("CONTINUE", "PLATEAU", "INSUFFICIENT_DATA")


def test_gradient_norm_excursion_flags_unstable():
    steps = np.arange(1, 21) * 100
    val = 2.0 + 8.0 * steps.astype(float) ** (-0.5)
    gn = np.full(steps.size, 1.0)
    gn[14] = 50.0  # gradient explosion
    r = analyze(_records(steps, val, gradient_norm=gn), _cfg(run_id="grad"))
    assert r.status == "UNSTABLE"
    assert 1500.0 in r.instability["grad_spike_steps"]


def test_prob_beats_target_is_between_0_and_1():
    steps = np.arange(1, 26) * 100
    val = 2.0 + 8.0 * steps.astype(float) ** (-0.5) + 0.03 * np.random.default_rng(2).standard_normal(steps.size)
    r = analyze(_records(steps, val), _cfg(run_id="tgt", target_loss=2.05))
    for v in r.prob_beats_target.values():
        assert 0.0 <= v <= 1.0
