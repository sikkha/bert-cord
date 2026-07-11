#!/usr/bin/env python3
"""Conservative training-curve analysis + static research figures.

Estimates whether a run is *still improving enough to justify more compute*. This is
**heuristic learning-curve extrapolation**, not a learned forecaster: AdamW does not know the
optimal stopping step, forecasts are unreliable after phase transitions / optimizer
instability, and task-specific metrics must be weighed alongside perplexity. No claim of
scientific optimality is made.

The CLI only coordinates: it loads metrics, calls the analysis (structured data out), prints a
human summary, writes `analysis_summary.json` + `training_curve_report.md`, and — only if
`--plot` is given — renders static matplotlib figures via a headless backend.

Example:
  python scripts/analyze_training_curve.py \
      --metrics experiments/run_001/metrics.jsonl \
      --future-step 1000 --future-step 2000 \
      --plot --plot-dir experiments/run_001/analysis --show-confidence
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from coordinator_bert.curve_analysis import AnalysisConfig, analyze, load_metrics  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Conservative training-curve analysis + figures.")
    p.add_argument("--metrics", required=True, help="Path to metrics JSONL or CSV.")
    p.add_argument("--run-id", default=None, help="Run/config identifier for titles/summary.")
    # Analysis knobs
    p.add_argument("--ema-alpha", type=float, default=0.3)
    p.add_argument("--slope-window", type=int, default=8)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--min-delta", type=float, default=1e-3)
    p.add_argument("--min-evals", type=int, default=6)
    p.add_argument("--min-fit-points", type=int, default=6)
    p.add_argument("--future-step", type=int, action="append", default=None,
                   help="Repeatable. Future step(s) to forecast (e.g. --future-step 1000).")
    p.add_argument("--target-loss", type=float, default=None,
                   help="If set, report a heuristic 'probability' that training beats it.")
    p.add_argument("--negligible-gain-per-100", type=float, default=1e-3)
    p.add_argument("--n-boot", type=int, default=200)
    p.add_argument("--ci", type=float, default=0.90)
    p.add_argument("--bootstrap-seed", type=int, default=0)
    # Output
    p.add_argument("--out-dir", default=None,
                   help="Where to write analysis_summary.json + report (default: metrics dir).")
    # Plotting
    p.add_argument("--plot", action="store_true", help="Render static figures (matplotlib).")
    p.add_argument("--plot-dir", default=None, help="Directory for figures (default: out-dir).")
    p.add_argument("--plot-format", choices=["png", "svg", "both"], default="png")
    p.add_argument("--show-confidence", action="store_true", help="Shade bootstrap CI bands.")
    p.add_argument("--log-x", action="store_true")
    p.add_argument("--log-y", action="store_true")
    p.add_argument("--chance-baseline", type=float, default=None,
                   help="Chance-level task metric to draw on the task-metric figure.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not os.path.exists(args.metrics):
        print(f"[analyze] metrics file not found: {args.metrics}", file=sys.stderr)
        return 2

    run_id = args.run_id or os.path.splitext(os.path.basename(args.metrics))[0]
    future_steps = tuple(args.future_step) if args.future_step else (500, 1000, 2000)
    cfg = AnalysisConfig(
        ema_alpha=args.ema_alpha, slope_window=args.slope_window, patience=args.patience,
        min_delta=args.min_delta, min_evals=args.min_evals, min_fit_points=args.min_fit_points,
        future_steps=future_steps, target_loss=args.target_loss,
        negligible_gain_per_100=args.negligible_gain_per_100, n_boot=args.n_boot, ci=args.ci,
        bootstrap_seed=args.bootstrap_seed, run_id=run_id,
    )

    records = load_metrics(args.metrics)
    result = analyze(records, cfg)

    out_dir = args.out_dir or (os.path.dirname(args.metrics) or ".")
    os.makedirs(out_dir, exist_ok=True)

    # Human-readable summary to stdout.
    _print_summary(result)

    # Machine-readable summary + Markdown report (report links plots if generated).
    from coordinator_bert.curve_plots import write_markdown_report, write_summary_json

    summary_path = os.path.join(out_dir, "analysis_summary.json")
    write_summary_json(result, summary_path)

    plot_paths: dict = {}
    if args.plot:
        from coordinator_bert.curve_plots import generate_plots
        plot_dir = args.plot_dir or out_dir
        plot_paths = generate_plots(
            result, plot_dir, formats=args.plot_format,
            show_confidence=args.show_confidence, log_x=args.log_x, log_y=args.log_y,
            chance_baseline=args.chance_baseline,
        )
        print(f"[analyze] wrote {sum(len(v) for v in plot_paths.values())} figure file(s) "
              f"to {plot_dir}")
        for name, paths in plot_paths.items():
            print(f"    {name}: {', '.join(os.path.basename(p) for p in paths)}")

    report_path = os.path.join(out_dir, "training_curve_report.md")
    write_markdown_report(result, plot_paths, report_path)
    print(f"[analyze] summary -> {summary_path}")
    print(f"[analyze] report  -> {report_path}")
    return 0


def _print_summary(result) -> None:
    r = result.to_dict()
    print("=" * 70)
    print(f"training-curve analysis :: {r['run_id']}")
    print("-" * 70)
    print(f"status                 : {r['status']}")
    print(f"evaluations            : {r['n_evals']} (current step {_f(r['current_step'])})")
    print(f"best val loss          : {_f(r['best_val_loss'])} @ step {_f(r['best_step'])}")
    print(f"recent improvement/100 : {_f(r['recent_improvement_per_100'])}")
    if r.get("chosen_model"):
        print(f"chosen curve model     : {r['chosen_model']} "
              f"(R2={_f(r['fit_quality'].get('r2'))}, "
              f"tail_rmse={_f(r['fit_quality'].get('tail_rmse'))})")
        print(f"estimated asymptote    : {_f(r['predicted_asymptote'].get('point'))}")
        for k, v in r["predicted_val_loss"].items():
            ci = r["confidence"]["per_step"].get(str(k), {})
            print(f"  forecast @ {k:>6} : {_f(v)}  CI[{_f(ci.get('low'))}, {_f(ci.get('high'))}]")
        if r.get("prob_beats_target"):
            for k, v in r["prob_beats_target"].items():
                print(f"  P(beats {r['target_loss']}) @ {k:>6} : {_f(v)}  (heuristic, not calibrated)")
    if r.get("recommended_stop_step") is not None:
        print(f"recommended stop step  : {_f(r['recommended_stop_step'])}")
    inst = r["instability"]
    ev = sorted(set(list(inst["loss_spike_steps"]) + list(inst["nan_inf_steps"])
                    + list(inst["grad_spike_steps"])))
    if ev:
        print(f"instability events     : {', '.join(_f(x) for x in ev)}")
    for w in r.get("warnings", []):
        print(f"warning                : {w}")
    print("=" * 70)


def _f(v) -> str:
    if v is None:
        return "n/a"
    try:
        f = float(v)
        return "n/a" if f != f else f"{f:.4g}"
    except (TypeError, ValueError):
        return str(v)


if __name__ == "__main__":
    raise SystemExit(main())
