"""Static research figures for training-curve analysis (matplotlib only, headless).

Separation of concerns: this module **only plots** structured data produced by
``curve_analysis.analyze`` (an ``AnalysisResult`` or its ``to_dict()``). It performs no model
fitting or status decisions — it visualizes what analysis already computed. matplotlib runs on
the non-interactive ``Agg`` backend so it works on servers / DGX with no display. seaborn is
not used. Each figure is a separate file (no subplots).

Honesty in visuals: observed and forecast regions are drawn distinctly; lines break across
NaN gaps (never interpolated as if continuous); forecasts appear only when analysis produced a
fit; axes are labelled with units and titled with the run identifier. Forecasts here are
heuristic learning-curve extrapolation — not a claim of optimality.
"""

from __future__ import annotations

import json
import math
import os
from typing import Optional

import numpy as np

import matplotlib
matplotlib.use("Agg")  # headless: must be set before pyplot import
import matplotlib.pyplot as plt  # noqa: E402


# --------------------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------------------- #
def _arr(values) -> np.ndarray:
    """List with None -> float array with NaN (so matplotlib breaks lines at gaps)."""
    return np.asarray([np.nan if v is None else float(v) for v in (values or [])], dtype=float)


def _as_dict(result) -> dict:
    return result.to_dict() if hasattr(result, "to_dict") else dict(result)


def _finite(x: np.ndarray, y: np.ndarray):
    m = np.isfinite(x) & np.isfinite(y)
    return x[m], y[m]


def _apply_axes(ax, log_x: bool, log_y: bool) -> None:
    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")


def _save(fig, outdir: str, name: str, formats: str) -> list[str]:
    os.makedirs(outdir, exist_ok=True)
    fmts = {"png": ["png"], "svg": ["svg"], "both": ["png", "svg"]}[formats]
    paths = []
    for f in fmts:
        p = os.path.join(outdir, f"{name}.{f}")
        fig.savefig(p, dpi=120, bbox_inches="tight")
        paths.append(p)
    plt.close(fig)
    return paths


def _markers(ax, result: dict) -> None:
    """Vertical markers for current step and recommended stop step."""
    cur = result.get("current_step")
    if cur is not None and np.isfinite(cur):
        ax.axvline(cur, color="#444444", linestyle=":", linewidth=1.2,
                   label=f"current step ({int(cur)})")
    stop = result.get("recommended_stop_step")
    if stop is not None and np.isfinite(stop):
        ax.axvline(stop, color="#b30000", linestyle="--", linewidth=1.4,
                   label=f"recommended stop ({int(stop)})")


def _instability_marks(ax, result: dict, label: str = "instability") -> None:
    inst = result.get("instability", {})
    steps = list(inst.get("loss_spike_steps", [])) + list(inst.get("nan_inf_steps", []))
    first = True
    for s in steps:
        if s is None or not np.isfinite(s):
            continue
        ax.axvline(s, color="#ff7f0e", linestyle="-", linewidth=0.9, alpha=0.7,
                   label=label if first else None)
        first = False


def _plateau_span(ax, result: dict) -> None:
    pl = result.get("plateau", {})
    if pl.get("is_plateau") and pl.get("plateau_start_step") is not None:
        start = pl["plateau_start_step"]
        cur = result.get("current_step")
        if start is not None and cur is not None and np.isfinite(start) and np.isfinite(cur) \
                and cur > start:
            ax.axvspan(start, cur, color="#cccccc", alpha=0.35, label="plateau region")


# --------------------------------------------------------------------------------------- #
# A. validation loss curve
# --------------------------------------------------------------------------------------- #
def plot_validation_loss(result, formats: str, outdir: str, show_confidence: bool = True,
                         log_x: bool = False, log_y: bool = False) -> Optional[list[str]]:
    r = _as_dict(result)
    s = r["series"]
    vstep = _arr(s.get("val_step"))
    vloss = _arr(s.get("val_loss_finite"))
    vsmooth = _arr(s.get("val_loss_smoothed"))
    if vstep.size == 0 or not np.any(np.isfinite(vloss)):
        return None

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(*_finite(vstep, vloss), s=22, color="#1f77b4", alpha=0.8,
               label="observed val loss", zorder=3)
    fx, fy = _finite(vstep, vsmooth)
    if fx.size:
        ax.plot(fx, fy, color="#1f77b4", linewidth=1.6, label="EMA-smoothed")

    # Best point.
    if r.get("best_step") is not None and r.get("best_val_loss") is not None:
        ax.scatter([r["best_step"]], [r["best_val_loss"]], marker="*", s=180,
                   color="#2ca02c", edgecolor="black", zorder=5,
                   label=f"best ({r['best_val_loss']:.3f})")

    # Forecast region (distinct dashed) + CI band.
    fc = r.get("forecast", {})
    if fc.get("grid"):
        g = _arr(fc["grid"])
        pt = _arr(fc.get("point"))
        cur = r.get("current_step")
        fut = np.isfinite(g) & (g >= (cur if cur is not None else -np.inf))
        ax.plot(g[fut], pt[fut], color="#d62728", linestyle="--", linewidth=1.6,
                label=f"forecast ({r.get('chosen_model','?')} fit)")
        # thin fit over observed region for context
        obs = np.isfinite(g) & (g < (cur if cur is not None else np.inf))
        ax.plot(g[obs], pt[obs], color="#d62728", linestyle=":", linewidth=1.0, alpha=0.6)
        if show_confidence and fc.get("low") and fc.get("high"):
            lo = _arr(fc["low"])
            hi = _arr(fc["high"])
            band = np.isfinite(g) & np.isfinite(lo) & np.isfinite(hi)
            if np.any(band):
                ci = int(round(100 * r.get("confidence", {}).get("ci", 0.9)))
                ax.fill_between(g[band], lo[band], hi[band], color="#d62728", alpha=0.15,
                                label=f"{ci}% bootstrap CI")
        # asymptote line
        asy = r.get("predicted_asymptote", {}).get("point")
        if asy is not None and np.isfinite(asy):
            ax.axhline(asy, color="#9467bd", linestyle="-.", linewidth=1.0,
                       label=f"est. asymptote ({asy:.3f})")

    _plateau_span(ax, r)
    _instability_marks(ax, r)
    _markers(ax, r)
    _apply_axes(ax, log_x, log_y)
    ax.set_xlabel("training step")
    ax.set_ylabel("validation MLM loss (nats)")
    ax.set_title(f"Validation loss — {r.get('run_id','run')}  [status: {r.get('status')}]")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.25)
    return _save(fig, outdir, "validation_loss_curve", formats)


# --------------------------------------------------------------------------------------- #
# B. perplexity curve
# --------------------------------------------------------------------------------------- #
def plot_perplexity(result, formats: str, outdir: str, show_confidence: bool = True,
                    log_x: bool = False, log_y: Optional[bool] = None) -> Optional[list[str]]:
    r = _as_dict(result)
    s = r["series"]
    vstep = _arr(s.get("val_step"))
    vloss = _arr(s.get("val_loss_finite"))
    if vstep.size == 0 or not np.any(np.isfinite(vloss)):
        return None
    ppl = np.exp(np.clip(vloss, -20, 20))

    fig, ax = plt.subplots(figsize=(8, 5))
    fx, fy = _finite(vstep, ppl)
    ax.scatter(fx, fy, s=22, color="#1f77b4", alpha=0.8, label="observed perplexity", zorder=3)

    fc = r.get("forecast", {})
    if fc.get("grid"):
        g = _arr(fc["grid"])
        pt = np.exp(np.clip(_arr(fc.get("point")), -20, 20))
        cur = r.get("current_step")
        fut = np.isfinite(g) & (g >= (cur if cur is not None else -np.inf))
        ax.plot(g[fut], pt[fut], color="#d62728", linestyle="--", linewidth=1.6,
                label="forecast perplexity")
        if show_confidence and fc.get("low") and fc.get("high"):
            lo = np.exp(np.clip(_arr(fc["low"]), -20, 20))
            hi = np.exp(np.clip(_arr(fc["high"]), -20, 20))
            band = np.isfinite(g) & np.isfinite(lo) & np.isfinite(hi)
            if np.any(band):
                ax.fill_between(g[band], lo[band], hi[band], color="#d62728", alpha=0.15,
                                label="bootstrap CI")

    # Auto log-y when values span a large range.
    if log_y is None:
        vals = fy[np.isfinite(fy)]
        log_y = bool(vals.size and (vals.max() / max(1e-9, vals.min()) > 20))
    _apply_axes(ax, log_x, log_y)
    _markers(ax, r)
    ax.set_xlabel("training step")
    ax.set_ylabel("validation perplexity  (= exp(loss))")
    ax.set_title(f"Perplexity — {r.get('run_id','run')}   (perplexity = exp(loss))")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.25)
    return _save(fig, outdir, "perplexity_curve", formats)


# --------------------------------------------------------------------------------------- #
# C. task-metric curve (masked-token accuracy)
# --------------------------------------------------------------------------------------- #
def plot_task_metric(result, formats: str, outdir: str, chance_baseline: Optional[float] = None,
                     log_x: bool = False) -> Optional[list[str]]:
    r = _as_dict(result)
    s = r["series"]
    if not s.get("present", {}).get("masked_accuracy"):
        return None
    step = _arr(s.get("step"))
    acc = _arr(s.get("masked_accuracy"))
    fx, fy = _finite(step, acc)
    if fx.size == 0:
        return None

    # EMA smoothing of the finite accuracy values.
    sm = _ema_finite(fy, 0.3)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(fx, fy, s=22, color="#8c564b", alpha=0.8, label="masked-token accuracy", zorder=3)
    ax.plot(fx, sm, color="#8c564b", linewidth=1.5, label="EMA-smoothed")

    best_i = int(np.argmax(fy))
    ax.scatter([fx[best_i]], [fy[best_i]], marker="*", s=160, color="#2ca02c",
               edgecolor="black", zorder=5, label=f"best ({fy[best_i]:.3f})")

    if chance_baseline is not None and np.isfinite(chance_baseline):
        ax.axhline(chance_baseline, color="#777777", linestyle="--", linewidth=1.0,
                   label=f"chance baseline ({chance_baseline:.3g})")

    # Honesty note: warn when loss is improving but task accuracy is flat.
    acc_slope = _slope(fx, sm)
    loss_improving = np.isfinite(r.get("recent_improvement_per_100", np.nan)) and \
        r.get("recent_improvement_per_100", 0) > r.get("config", {}).get(
            "negligible_gain_per_100", 1e-3)
    acc_flat = np.isfinite(acc_slope) and abs(acc_slope) < 1e-6
    if loss_improving and acc_flat:
        ax.text(0.02, 0.02,
                "note: val loss improving but task accuracy flat\n"
                "(loss gains are not yet task gains)",
                transform=ax.transAxes, fontsize=8, color="#b30000",
                verticalalignment="bottom",
                bbox=dict(boxstyle="round", fc="#fff3f3", ec="#b30000", alpha=0.8))

    _apply_axes(ax, log_x, False)
    _markers(ax, r)
    ax.set_xlabel("training step")
    ax.set_ylabel("masked-token accuracy (top-1)")
    ax.set_title(f"Task metric — {r.get('run_id','run')}  (forecast omitted: not justified)")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.25)
    return _save(fig, outdir, "task_metric_curve", formats)


# --------------------------------------------------------------------------------------- #
# D. learning-rate curve
# --------------------------------------------------------------------------------------- #
def plot_learning_rate(result, formats: str, outdir: str, log_x: bool = False
                       ) -> Optional[list[str]]:
    r = _as_dict(result)
    s = r["series"]
    if not s.get("present", {}).get("learning_rate"):
        return None
    step = _arr(s.get("step"))
    lr = _arr(s.get("learning_rate"))
    fx, fy = _finite(step, lr)
    if fx.size == 0:
        return None

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(fx, fy, color="#17becf", linewidth=1.6, label="learning rate")

    # Warmup/decay boundary = step of peak LR.
    peak_i = int(np.argmax(fy))
    ax.axvline(fx[peak_i], color="#2ca02c", linestyle="--", linewidth=1.1,
               label=f"warmup→decay (peak @ {int(fx[peak_i])})")
    ax.annotate("warmup", xy=(fx[0], fy[peak_i] * 0.5), fontsize=8, color="#2ca02c")
    if peak_i < fx.size - 1:
        ax.annotate("decay", xy=(fx[-1], fy[peak_i] * 0.5), fontsize=8, color="#2ca02c",
                    horizontalalignment="right")

    _instability_marks(ax, r, label="loss instability")
    _apply_axes(ax, log_x, False)
    ax.set_xlabel("training step")
    ax.set_ylabel("learning rate")
    ax.set_title(f"Learning rate — {r.get('run_id','run')}")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.25)
    return _save(fig, outdir, "learning_rate_curve", formats)


# --------------------------------------------------------------------------------------- #
# E. gradient-norm curve (only when present)
# --------------------------------------------------------------------------------------- #
def plot_gradient_norm(result, formats: str, outdir: str, log_x: bool = False,
                       log_y: bool = False) -> Optional[list[str]]:
    r = _as_dict(result)
    s = r["series"]
    if not s.get("present", {}).get("gradient_norm"):
        return None  # missing gradient_norm -> skip this figure without failing
    step = _arr(s.get("step"))
    gn = _arr(s.get("gradient_norm"))
    fx, fy = _finite(step, gn)
    if fx.size == 0:
        return None

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(fx, fy, color="#7f7f7f", linewidth=1.0, alpha=0.7, label="gradient norm")
    ax.plot(fx, _ema_finite(fy, 0.3), color="#333333", linewidth=1.6, label="EMA-smoothed")

    thr = r.get("instability", {}).get("grad_anomaly_threshold")
    if thr is not None and np.isfinite(thr):
        ax.axhline(thr, color="#d62728", linestyle="--", linewidth=1.1,
                   label=f"anomaly threshold ({thr:.2f})")
    for sp in r.get("instability", {}).get("grad_spike_steps", []):
        if sp is not None and np.isfinite(sp):
            ax.axvline(sp, color="#ff7f0e", linewidth=0.9, alpha=0.7)

    _apply_axes(ax, log_x, log_y)
    ax.set_xlabel("training step")
    ax.set_ylabel("gradient norm (L2)")
    ax.set_title(f"Gradient norm — {r.get('run_id','run')}")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.25)
    return _save(fig, outdir, "gradient_norm_curve", formats)


# --------------------------------------------------------------------------------------- #
# F. improvement-rate curve
# --------------------------------------------------------------------------------------- #
def plot_improvement_rate(result, formats: str, outdir: str, log_x: bool = False
                          ) -> Optional[list[str]]:
    r = _as_dict(result)
    s = r["series"]
    vstep = _arr(s.get("val_step"))
    vsmooth = _arr(s.get("val_loss_smoothed"))
    fx, fy = _finite(vstep, vsmooth)
    if fx.size < 3:
        return None

    # Rolling improvement per 100 steps (local slope over a trailing window).
    window = int(r.get("config", {}).get("slope_window", 8))
    imp = np.full(fx.size, np.nan)
    for i in range(1, fx.size):
        lo = max(0, i - window + 1)
        xs, ys = fx[lo:i + 1], fy[lo:i + 1]
        if xs.size >= 2 and np.ptp(xs) > 0:
            slope = np.polyfit(xs, ys, 1)[0]
            imp[i] = -slope * 100.0

    fig, ax = plt.subplots(figsize=(8, 5))
    mi = np.isfinite(imp)
    ax.plot(fx[mi], imp[mi], color="#1f77b4", linewidth=1.6, marker="o", markersize=3,
            label="Δloss per 100 steps (improving>0)")
    ax.axhline(0.0, color="#000000", linewidth=1.0, label="zero improvement")

    neg = float(r.get("config", {}).get("negligible_gain_per_100", 1e-3))
    ax.axhspan(-neg, neg, color="#cccccc", alpha=0.4,
               label=f"economically negligible (±{neg:g})")

    # Predicted improvement over remaining budget (current -> furthest forecast).
    fc = r.get("forecast", {})
    cur = r.get("current_step")
    if fc.get("grid") and cur is not None:
        g = _arr(fc["grid"])
        pt = _arr(fc.get("point"))
        fin = np.isfinite(g) & np.isfinite(pt)
        if np.any(fin):
            g2, pt2 = g[fin], pt[fin]
            far = float(g2.max())
            cur_pred = float(np.interp(cur, g2, pt2))
            far_pred = float(pt2[np.argmax(g2)])
            span = max(1.0, far - cur)
            per100 = -(far_pred - cur_pred) / span * 100.0
            ax.annotate(
                f"forecast mean Δloss/100 over remaining\nbudget (→{int(far)}): {per100:.4f}",
                xy=(0.98, 0.95), xycoords="axes fraction", ha="right", va="top", fontsize=8,
                bbox=dict(boxstyle="round", fc="#eef5ff", ec="#1f77b4", alpha=0.8))

    _apply_axes(ax, log_x, False)
    _markers(ax, r)
    ax.set_xlabel("training step")
    ax.set_ylabel("val-loss improvement per 100 steps")
    ax.set_title(f"Improvement rate — {r.get('run_id','run')}")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.25)
    return _save(fig, outdir, "improvement_rate_curve", formats)


# --------------------------------------------------------------------------------------- #
# Orchestration + outputs
# --------------------------------------------------------------------------------------- #
def generate_plots(result, outdir: str, formats: str = "png", show_confidence: bool = True,
                   log_x: bool = False, log_y: bool = False,
                   chance_baseline: Optional[float] = None) -> dict:
    """Generate all applicable figures. Returns {figure_name: [paths]} (skips N/A figures)."""
    r = _as_dict(result)
    os.makedirs(outdir, exist_ok=True)
    out: dict = {}

    def _add(name, paths):
        if paths:
            out[name] = paths

    _add("validation_loss_curve",
         plot_validation_loss(r, formats, outdir, show_confidence, log_x, log_y))
    _add("perplexity_curve",
         plot_perplexity(r, formats, outdir, show_confidence, log_x, None))
    _add("task_metric_curve", plot_task_metric(r, formats, outdir, chance_baseline, log_x))
    _add("learning_rate_curve", plot_learning_rate(r, formats, outdir, log_x))
    _add("gradient_norm_curve", plot_gradient_norm(r, formats, outdir, log_x, log_y))
    _add("improvement_rate_curve", plot_improvement_rate(r, formats, outdir, log_x))
    return out


def write_summary_json(result, path: str) -> str:
    """Write the machine-readable analysis_summary.json (matches the plotted recommendation)."""
    r = _as_dict(result)
    summary = {
        "status": r.get("status"),
        "run_id": r.get("run_id"),
        "current_step": r.get("current_step"),
        "best_step": r.get("best_step"),
        "recommended_stop_step": r.get("recommended_stop_step"),
        "best_val_loss": r.get("best_val_loss"),
        "predicted_val_loss": r.get("predicted_val_loss"),
        "predicted_asymptote": r.get("predicted_asymptote"),
        "confidence": r.get("confidence"),
        "recent_slope_per_step": r.get("recent_slope_per_step"),
        "recent_slope_per_log_step": r.get("recent_slope_per_log_step"),
        "recent_improvement_per_100": r.get("recent_improvement_per_100"),
        "plateau_start": r.get("plateau", {}).get("plateau_start_step"),
        "instability_steps": sorted(set(
            list(r.get("instability", {}).get("loss_spike_steps", []))
            + list(r.get("instability", {}).get("nan_inf_steps", []))
            + list(r.get("instability", {}).get("grad_spike_steps", [])))),
        "chosen_curve_model": r.get("chosen_model"),
        "fit_quality": r.get("fit_quality"),
        "prob_beats_target": r.get("prob_beats_target"),
        "target_loss": r.get("target_loss"),
        "warnings": r.get("warnings"),
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=_json_default)
    return path


def write_markdown_report(result, plot_paths: dict, path: str) -> str:
    """Write a concise Markdown report embedding/linking the figures."""
    r = _as_dict(result)
    outdir = os.path.dirname(path) or "."
    status = r.get("status")

    def rel(p):
        return os.path.relpath(p, outdir)

    lines: list[str] = []
    lines.append(f"# Training-curve analysis — {r.get('run_id','run')}")
    lines.append("")
    lines.append("> **Heuristic learning-curve extrapolation.** These are simple curve fits "
                 "over past validation loss with bootstrap uncertainty — **not** a learned "
                 "forecaster and **not** a claim of optimal stopping. AdamW does not know the "
                 "optimal step. Forecasts are unreliable after phase transitions or optimizer "
                 "instability. Always weigh task-specific metrics alongside perplexity.")
    lines.append("")
    lines.append(f"**Status: `{status}`**")
    lines.append("")

    # Summary bullets
    cur = r.get("current_step")
    best = r.get("best_val_loss")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Current step: `{_fmt(cur)}` over `{r.get('n_evals')}` evaluations.")
    lines.append(f"- Best validation loss: `{_fmt(best)}` at step `{_fmt(r.get('best_step'))}`.")
    lines.append(f"- Recent improvement: `{_fmt(r.get('recent_improvement_per_100'))}` "
                 "loss per 100 steps.")
    if r.get("chosen_model"):
        fq = r.get("fit_quality", {})
        lines.append(f"- Chosen curve model: `{r['chosen_model']}` "
                     f"(R²={_fmt(fq.get('r2'))}, tail RMSE={_fmt(fq.get('tail_rmse'))}).")
        lines.append(f"- Estimated asymptotic loss: "
                     f"`{_fmt(r.get('predicted_asymptote',{}).get('point'))}`.")
        preds = r.get("predicted_val_loss", {})
        if preds:
            pred_str = ", ".join(f"step {k}: {_fmt(v)}" for k, v in preds.items())
            lines.append(f"- Predicted val loss — {pred_str}.")
        conf = r.get("confidence", {}).get("per_step", {})
        if conf:
            ci = int(round(100 * r.get("confidence", {}).get("ci", 0.9)))
            ci_str = ", ".join(
                f"step {k}: [{_fmt(v.get('low'))}, {_fmt(v.get('high'))}]"
                for k, v in conf.items())
            lines.append(f"- {ci}% bootstrap CI — {ci_str}.")
    if r.get("recommended_stop_step") is not None:
        lines.append(f"- **Recommended stop step:** `{_fmt(r.get('recommended_stop_step'))}`.")
    lines.append("")

    # Interpretation
    lines.append("## Interpretation")
    lines.append("")
    lines.append(_status_paragraph(r))
    lines.append("")

    inst = r.get("instability", {})
    inst_steps = sorted(set(list(inst.get("loss_spike_steps", []))
                            + list(inst.get("nan_inf_steps", []))
                            + list(inst.get("grad_spike_steps", []))))
    lines.append("## Instability observations")
    lines.append("")
    if inst_steps:
        lines.append(f"- Events at steps: {', '.join(_fmt(x) for x in inst_steps)}. "
                     "Forecasts are suppressed or should be distrusted around these points.")
    else:
        lines.append("- None detected (no NaN/Inf, loss spikes, or gradient excursions).")
    lines.append("")

    if r.get("warnings"):
        lines.append("## Warnings")
        lines.append("")
        for w in r["warnings"]:
            lines.append(f"- {w}")
        lines.append("")

    # Figures
    lines.append("## Figures")
    lines.append("")
    order = ["validation_loss_curve", "perplexity_curve", "task_metric_curve",
             "learning_rate_curve", "gradient_norm_curve", "improvement_rate_curve"]
    for name in order:
        paths = plot_paths.get(name)
        if not paths:
            continue
        png = next((p for p in paths if p.endswith(".png")), paths[0])
        title = name.replace("_", " ").title()
        lines.append(f"### {title}")
        lines.append("")
        lines.append(f"![{title}]({rel(png)})")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("_Generated by `scripts/analyze_training_curve.py`. Static, reproducible "
                 "figures; no interactive dashboard, no external logging._")

    os.makedirs(outdir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


# --------------------------------------------------------------------------------------- #
# tiny utilities
# --------------------------------------------------------------------------------------- #
def _ema_finite(y: np.ndarray, alpha: float) -> np.ndarray:
    out = np.full_like(y, np.nan, dtype=float)
    acc = None
    for i, v in enumerate(y):
        if not np.isfinite(v):
            out[i] = acc if acc is not None else np.nan
            continue
        acc = v if acc is None else alpha * v + (1 - alpha) * acc
        out[i] = acc
    return out


def _slope(x: np.ndarray, y: np.ndarray) -> float:
    x, y = _finite(x, y)
    if x.size < 2 or np.ptp(x) == 0:
        return float("nan")
    return float(np.polyfit(x, y, 1)[0])


def _fmt(v) -> str:
    if v is None:
        return "n/a"
    try:
        f = float(v)
        if not math.isfinite(f):
            return "n/a"
        return f"{f:.4g}"
    except (TypeError, ValueError):
        return str(v)


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _status_paragraph(r: dict) -> str:
    status = r.get("status")
    if status == "CONTINUE":
        return ("The run still shows measurable validation-loss improvement and no instability, "
                "so continued training is likely to yield further (diminishing) gains. The "
                "forecast and confidence interval below quantify the expected reduction; treat "
                "them as a rough guide, not a guarantee.")
    if status == "PLATEAU":
        return ("Validation loss has stopped improving meaningfully (improvement per 100 steps "
                "is within the negligible band and/or the best score has not improved for the "
                "patience window). Additional compute is unlikely to help much on this metric; "
                "check task-specific metrics before deciding.")
    if status == "UNSTABLE":
        return ("Instability was detected (NaN/Inf, a loss spike, or a gradient-norm excursion). "
                "Curve forecasts are unreliable across such phase transitions and are "
                "suppressed. Investigate the optimizer/learning-rate before trusting any "
                "extrapolation.")
    return ("Too few evaluations to judge the trajectory. Collect more evaluation points before "
            "drawing conclusions; no curve was fitted.")
