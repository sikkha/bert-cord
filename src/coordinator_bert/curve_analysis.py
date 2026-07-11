"""Conservative training-curve analysis (heuristic learning-curve extrapolation).

This module estimates whether a run is *still improving enough to justify more compute*. It is
**not** a learned model and makes **no claim** to know the true optimal training step. AdamW
does not tell you where to stop; these are simple curve fits over past validation loss with
honest uncertainty. Forecasts are unreliable after phase transitions or optimizer instability,
and task-specific metrics must be weighed alongside perplexity. Do not treat any output here as
scientifically optimal.

Design: analysis functions return **structured data only** (no plotting, no I/O side effects
beyond reading the metrics file). Plotting lives in ``curve_plots.py``; the CLI in
``scripts/analyze_training_curve.py`` coordinates loading, analysis, reporting, and plotting.

Dependencies are minimal: numpy is required; scipy is used only if already installed (a numpy
grid + least-squares fallback is always available and is the default).
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Callable, Optional

import numpy as np

try:  # scipy is optional; the numpy path is always available.
    from scipy.optimize import curve_fit as _scipy_curve_fit  # type: ignore
    _HAS_SCIPY = True
except Exception:  # noqa: BLE001
    _HAS_SCIPY = False


# --------------------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------------------- #
_NUMERIC_FIELDS = (
    "step", "tokens_seen", "train_loss", "val_loss", "masked_accuracy",
    "learning_rate", "gradient_norm",
)


def _coerce_float(value) -> float:
    """Return float(value), mapping empties/None/'nan'/'inf' strings to the right float."""
    if value is None or value == "":
        return math.nan
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("nan", "none", "null", ""):
            return math.nan
        if v in ("inf", "+inf", "infinity"):
            return math.inf
        if v in ("-inf", "-infinity"):
            return -math.inf
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def load_metrics(path: str) -> list[dict]:
    """Load metrics from JSONL or CSV. Format is inferred from extension then content.

    Returns a list of record dicts. Unknown columns are preserved; missing numeric fields
    become NaN. Never raises on a malformed row — bad rows are skipped defensively.
    """
    lower = path.lower()
    records: list[dict] = []
    if lower.endswith(".jsonl") or lower.endswith(".ndjson"):
        records = _load_jsonl(path)
    elif lower.endswith(".csv") or lower.endswith(".tsv"):
        records = _load_csv(path, delimiter="\t" if lower.endswith(".tsv") else ",")
    else:
        # Sniff: try JSONL first, then CSV.
        try:
            records = _load_jsonl(path)
            if not records:
                records = _load_csv(path, delimiter=",")
        except Exception:  # noqa: BLE001
            records = _load_csv(path, delimiter=",")
    return records


def _load_jsonl(path: str) -> list[dict]:
    out: list[dict] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                out.append(obj)
    return out


def _load_csv(path: str, delimiter: str = ",") -> list[dict]:
    out: list[dict] = []
    with open(path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter=delimiter)
        for row in reader:
            out.append(dict(row))
    return out


@dataclass
class MetricsSeries:
    """Column arrays extracted from records; missing values are NaN.

    ``present`` records which optional fields actually contain finite data.
    """

    step: np.ndarray
    tokens_seen: np.ndarray
    train_loss: np.ndarray
    val_loss: np.ndarray
    masked_accuracy: np.ndarray
    learning_rate: np.ndarray
    gradient_norm: np.ndarray
    present: dict[str, bool] = field(default_factory=dict)
    # Per-field mask: True where the record actually contained the key (vs missing entirely).
    reported: dict = field(default_factory=dict)

    def val_points(self) -> tuple[np.ndarray, np.ndarray]:
        """(steps, val_loss) for rows with a finite val_loss, sorted by step."""
        return _finite_pairs(self.step, self.val_loss)

    def finite_val_count(self) -> int:
        _, y = self.val_points()
        return int(y.size)


def to_series(records: list[dict]) -> MetricsSeries:
    """Turn records into aligned numeric arrays, ordered by step when present."""
    cols = {f: [] for f in _NUMERIC_FIELDS}
    rep = {f: [] for f in _NUMERIC_FIELDS}
    for rec in records:
        for f in _NUMERIC_FIELDS:
            has = f in rec and rec.get(f) not in (None, "")
            rep[f].append(bool(has))
            cols[f].append(_coerce_float(rec.get(f)))
    arrs = {f: np.asarray(cols[f], dtype=float) for f in _NUMERIC_FIELDS}
    reps = {f: np.asarray(rep[f], dtype=bool) for f in _NUMERIC_FIELDS}

    # Order by step where finite; NaN steps go last but are kept.
    step = arrs["step"]
    if np.any(np.isfinite(step)):
        order = np.argsort(np.where(np.isfinite(step), step, np.inf), kind="stable")
        arrs = {f: v[order] for f, v in arrs.items()}
        reps = {f: v[order] for f, v in reps.items()}

    present = {
        f: bool(np.any(np.isfinite(arrs[f]))) for f in _NUMERIC_FIELDS
    }
    return MetricsSeries(
        step=arrs["step"], tokens_seen=arrs["tokens_seen"], train_loss=arrs["train_loss"],
        val_loss=arrs["val_loss"], masked_accuracy=arrs["masked_accuracy"],
        learning_rate=arrs["learning_rate"], gradient_norm=arrs["gradient_norm"],
        present=present, reported=reps,
    )


def _finite_pairs(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m = np.isfinite(x) & np.isfinite(y)
    return x[m], y[m]


# --------------------------------------------------------------------------------------- #
# Smoothing, slopes
# --------------------------------------------------------------------------------------- #
def ema(values: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    """Exponential moving average. ``alpha`` is the weight on the newest point."""
    values = np.asarray(values, dtype=float)
    out = np.empty_like(values)
    if values.size == 0:
        return out
    acc = values[0]
    for i, v in enumerate(values):
        if not np.isfinite(v):
            out[i] = acc  # carry forward across gaps; do not corrupt the average with NaN
            continue
        acc = alpha * v + (1.0 - alpha) * acc
        out[i] = acc
    return out


def linear_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Least-squares slope dy/dx; returns NaN if <2 finite points or zero x-spread."""
    x, y = _finite_pairs(x, y)
    if x.size < 2 or np.ptp(x) == 0:
        return math.nan
    a = np.polyfit(x, y, 1)
    return float(a[0])


def recent_slopes(steps: np.ndarray, values: np.ndarray, window: int) -> dict:
    """Slope over the recent window vs step and vs log(step)."""
    steps, values = _finite_pairs(steps, values)
    n = values.size
    if n < 2:
        return {"slope_per_step": math.nan, "slope_per_log_step": math.nan,
                "window_used": int(n)}
    w = min(window, n)
    xs = steps[-w:]
    ys = values[-w:]
    slope_step = linear_slope(xs, ys)
    # log(step) slope: guard non-positive steps.
    pos = xs > 0
    slope_log = linear_slope(np.log(xs[pos]), ys[pos]) if np.count_nonzero(pos) >= 2 else math.nan
    return {"slope_per_step": slope_step, "slope_per_log_step": slope_log, "window_used": int(w)}


def improvement_per_100(steps: np.ndarray, values_smoothed: np.ndarray, window: int) -> float:
    """Estimated validation-loss decrease per 100 steps over the recent window (>=0 = improving)."""
    r = recent_slopes(steps, values_smoothed, window)
    s = r["slope_per_step"]
    if not np.isfinite(s):
        return math.nan
    return float(-s * 100.0)  # positive => loss going down => improving


# --------------------------------------------------------------------------------------- #
# Plateau / instability detection
# --------------------------------------------------------------------------------------- #
@dataclass
class PlateauInfo:
    is_plateau: bool
    best_step: Optional[float]
    best_value: Optional[float]
    evals_since_best: int
    plateau_start_step: Optional[float]
    recent_improvement: float  # improvement (min_delta units) over patience window


def detect_plateau(
    steps: np.ndarray,
    values_smoothed: np.ndarray,
    patience: int = 5,
    min_delta: float = 1e-3,
    min_evals: int = 6,
) -> PlateauInfo:
    """Early-stopping-style plateau detection on smoothed validation loss.

    Plateau = at least ``min_evals`` evaluations exist AND the best smoothed loss has not
    improved by ``min_delta`` for the last ``patience`` evaluations.
    """
    steps, values = _finite_pairs(steps, values_smoothed)
    n = values.size
    if n < max(2, min_evals):
        return PlateauInfo(False, None if n == 0 else float(steps[int(np.argmin(values))]) if n else None,
                           None if n == 0 else float(np.min(values)) if n else None,
                           0, None, math.nan)
    best_idx = int(np.argmin(values))
    best_value = float(values[best_idx])
    best_step = float(steps[best_idx])
    evals_since_best = n - 1 - best_idx

    # Improvement over the last `patience` evals: best-before-window vs best-in-window.
    if n > patience:
        before = float(np.min(values[: n - patience]))
        window_best = float(np.min(values[n - patience:]))
        recent_improvement = before - window_best  # positive => still improving
    else:
        recent_improvement = float(values[0] - best_value)

    is_plateau = (evals_since_best >= patience) or (recent_improvement < min_delta)
    plateau_start = None
    if is_plateau:
        # First step at/after which no >min_delta improvement occurred.
        plateau_start = best_step
    return PlateauInfo(is_plateau, best_step, best_value, int(evals_since_best),
                       plateau_start, float(recent_improvement))


@dataclass
class InstabilityInfo:
    has_nan_inf: bool
    loss_spike_steps: list
    grad_spike_steps: list
    nan_inf_steps: list
    grad_anomaly_threshold: Optional[float]

    @property
    def is_unstable(self) -> bool:
        return self.has_nan_inf or bool(self.loss_spike_steps) or bool(self.grad_spike_steps)


def detect_instability(
    series: MetricsSeries,
    spike_sigma: float = 6.0,
    spike_rel_floor: float = 0.25,
    grad_sigma: float = 6.0,
) -> InstabilityInfo:
    """Detect NaN/Inf, sudden validation-loss spikes, and gradient-norm excursions.

    Spike detection is deliberately conservative (robust MAD scale, high sigma, plus a
    relative floor) so ordinary noise on an improving curve is not flagged.
    """
    # Collect steps where a *reported* value (key present in the record) is NaN or +/-Inf.
    # Missing fields (e.g. val_loss on non-eval rows) are NaN too, so we gate on `reported`.
    nan_inf_steps: list = []
    for name in ("train_loss", "val_loss", "gradient_norm"):
        arr = getattr(series, name)
        reported = series.reported.get(name)
        if reported is None:
            reported = np.ones_like(arr, dtype=bool)
        bad = reported & ~np.isfinite(arr)  # reported but not finite => real NaN/Inf
        if np.any(bad):
            for s in series.step[bad]:
                if np.isfinite(s):
                    nan_inf_steps.append(float(s))
    has_nan_inf = len(nan_inf_steps) > 0

    # Validation-loss spikes on finite points.
    loss_spike_steps: list = []
    st, vl = _finite_pairs(series.step, series.val_loss)
    if vl.size >= 4:
        diffs = np.diff(vl)
        scale = _mad(diffs)
        if scale == 0:
            scale = float(np.std(diffs)) or 1e-9
        loss_range = float(np.ptp(vl)) or 1.0
        for i, d in enumerate(diffs):
            rel = d / max(1e-9, abs(vl[i]))
            if d > spike_sigma * scale and d > spike_rel_floor * loss_range and rel > 0.1:
                loss_spike_steps.append(float(st[i + 1]))

    # Gradient-norm excursions. Robust to the "many identical values + one spike" case where
    # MAD collapses to 0: we combine a robust-scale threshold with a median-multiple rule.
    grad_spike_steps: list = []
    grad_thr: Optional[float] = None
    grad_mult = 5.0
    gs, gv = _finite_pairs(series.step, series.gradient_norm)
    if gv.size >= 4:
        med = float(np.median(gv))
        mad = _mad(gv)
        if mad > 0:
            scale = mad
        else:
            # Trim the top 10% (the suspected outliers) before estimating spread.
            trimmed = np.sort(gv)[: max(1, int(0.9 * gv.size))]
            scale = float(np.std(trimmed))
        scale_thr = (med + grad_sigma * scale) if scale > 0 else math.inf
        mult_thr = grad_mult * max(med, 1e-9)
        grad_thr = float(min(scale_thr, mult_thr))
        for s, v in zip(gs, gv):
            if v > scale_thr or v > mult_thr:
                grad_spike_steps.append(float(s))

    return InstabilityInfo(
        has_nan_inf=has_nan_inf,
        loss_spike_steps=sorted(set(loss_spike_steps)),
        grad_spike_steps=sorted(set(grad_spike_steps)),
        nan_inf_steps=sorted(set(nan_inf_steps)),
        grad_anomaly_threshold=grad_thr,
    )


def _mad(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0
    med = np.median(x)
    return float(1.4826 * np.median(np.abs(x - med)))  # ~std for normal data


# --------------------------------------------------------------------------------------- #
# Curve fitting: L_inf + A t^-alpha ; L_inf + A exp(-k t) ; a + b/sqrt(t)
# --------------------------------------------------------------------------------------- #
@dataclass
class CurveFit:
    name: str
    params: dict
    t_scale: float
    rss: float
    r2: float
    aic: float
    tail_rmse: float
    n_points: int
    asymptote: float

    def predict(self, t) -> np.ndarray:
        return _predict(self.name, self.params, self.t_scale, np.asarray(t, dtype=float))


def _predict(name: str, p: dict, t_scale: float, t: np.ndarray) -> np.ndarray:
    tt = np.asarray(t, dtype=float) / t_scale
    tt = np.where(tt <= 0, np.nan, tt)
    if name == "power":
        return p["L_inf"] + p["A"] * np.power(tt, -p["alpha"])
    if name == "exp":
        return p["L_inf"] + p["A"] * np.exp(-p["k"] * tt)
    if name == "sqrt":
        return p["a"] + p["b"] / np.sqrt(tt)
    raise ValueError(f"unknown model {name}")


def _clean_positive(t: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Keep only finite points with strictly positive step (needed for t^-a and 1/sqrt(t))."""
    m = np.isfinite(t) & np.isfinite(y) & (t > 0)
    return t[m], y[m]


def _fit_linear_basis(y: np.ndarray, basis: np.ndarray):
    """Least squares y ~ basis @ coef; returns (coef, rss) or None if inputs are unusable."""
    if not (np.all(np.isfinite(basis)) and np.all(np.isfinite(y))):
        return None
    try:
        with np.errstate(all="ignore"):
            coef, _, _, _ = np.linalg.lstsq(basis, y, rcond=None)
            if not np.all(np.isfinite(coef)):
                return None
            resid = y - basis @ coef
            rss = float(np.sum(resid ** 2))
    except np.linalg.LinAlgError:
        return None
    return coef, rss


def _fit_power(t: np.ndarray, y: np.ndarray, t_scale: float) -> Optional[dict]:
    t, y = _clean_positive(t, y)
    if t.size < 2:
        return None
    tt = t / t_scale
    best = None
    for alpha in np.linspace(0.05, 2.5, 60):
        basis = np.column_stack([np.ones_like(tt), np.power(tt, -alpha)])
        res = _fit_linear_basis(y, basis)
        if res is None:
            continue
        (L_inf, A), rss = res
        if best is None or rss < best[0]:
            best = (rss, {"L_inf": float(L_inf), "A": float(A), "alpha": float(alpha)})
    return None if best is None else best[1]


def _fit_exp(t: np.ndarray, y: np.ndarray, t_scale: float) -> Optional[dict]:
    t, y = _clean_positive(t, y)
    if t.size < 2:
        return None
    tt = t / t_scale
    best = None
    for k in np.linspace(0.05, 12.0, 80):
        basis = np.column_stack([np.ones_like(tt), np.exp(-k * tt)])
        res = _fit_linear_basis(y, basis)
        if res is None:
            continue
        (L_inf, A), rss = res
        if best is None or rss < best[0]:
            best = (rss, {"L_inf": float(L_inf), "A": float(A), "k": float(k)})
    return None if best is None else best[1]


def _fit_sqrt(t: np.ndarray, y: np.ndarray, t_scale: float) -> Optional[dict]:
    t, y = _clean_positive(t, y)
    if t.size < 2:
        return None
    tt = t / t_scale
    basis = np.column_stack([np.ones_like(tt), 1.0 / np.sqrt(tt)])
    res = _fit_linear_basis(y, basis)
    if res is None:
        return None
    (a, b), _ = res
    return {"a": float(a), "b": float(b)}


def _aic(rss: float, n: int, k_params: int) -> float:
    if n <= 0 or rss <= 0:
        return -math.inf if rss == 0 and n > 0 else math.inf
    return float(n * math.log(rss / n) + 2 * k_params)


def _r2(y: np.ndarray, yhat: np.ndarray) -> float:
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot == 0:
        return 0.0
    return 1.0 - ss_res / ss_tot


_MODEL_NPARAMS = {"power": 3, "exp": 3, "sqrt": 2}


def fit_curves(
    steps: np.ndarray,
    values: np.ndarray,
    min_points: int = 6,
    tail_frac: float = 0.25,
) -> list[CurveFit]:
    """Fit the three candidate models. Returns [] when there are too few finite points.

    Each fit is scored by full-data RSS/R²/AIC and by held-out **tail RMSE** (fit on the
    earlier portion, evaluate on the most-recent tail) — the tail error is the honest test of
    extrapolation quality and is the primary selection criterion.
    """
    steps, values = _finite_pairs(steps, values)
    n = values.size
    if n < min_points or np.ptp(steps) == 0:
        return []
    t_scale = float(np.max(steps)) or 1.0

    n_tail = max(1, int(round(n * tail_frac)))
    n_head = n - n_tail
    fits: list[CurveFit] = []
    fitters = {"power": _fit_power, "exp": _fit_exp, "sqrt": _fit_sqrt}
    for name, fitter in fitters.items():
        params = fitter(steps, values, t_scale)
        if params is None:
            continue
        yhat = _predict(name, params, t_scale, steps)
        if not np.all(np.isfinite(yhat)):
            continue
        rss = float(np.sum((values - yhat) ** 2))
        r2 = _r2(values, yhat)
        aic = _aic(rss, n, _MODEL_NPARAMS[name])

        # Held-out tail error: refit on head, predict tail.
        tail_rmse = math.nan
        if n_head >= min_points - 1 and n_head >= 3:
            head_params = fitter(steps[:n_head], values[:n_head], t_scale)
            if head_params is not None:
                tail_pred = _predict(name, head_params, t_scale, steps[n_head:])
                if np.all(np.isfinite(tail_pred)):
                    tail_rmse = float(np.sqrt(np.mean((values[n_head:] - tail_pred) ** 2)))

        asymptote = params.get("L_inf", params.get("a", math.nan))
        fits.append(CurveFit(name=name, params=params, t_scale=t_scale, rss=rss, r2=r2,
                             aic=aic, tail_rmse=tail_rmse, n_points=n, asymptote=float(asymptote)))
    return fits


def select_best_fit(fits: list[CurveFit]) -> Optional[CurveFit]:
    """Pick the fit with the lowest tail RMSE; fall back to AIC when tail error is unavailable."""
    if not fits:
        return None
    with_tail = [f for f in fits if np.isfinite(f.tail_rmse)]
    if with_tail:
        return min(with_tail, key=lambda f: f.tail_rmse)
    return min(fits, key=lambda f: f.aic)


# --------------------------------------------------------------------------------------- #
# Bootstrap confidence intervals
# --------------------------------------------------------------------------------------- #
def bootstrap_predictions(
    steps: np.ndarray,
    values: np.ndarray,
    model_name: str,
    future_steps: list[float],
    n_boot: int = 200,
    seed: int = 0,
    ci: float = 0.90,
) -> dict:
    """Bootstrap resampling of (step, loss) pairs to get CIs on forecasts and the asymptote.

    Returns per-future-step percentile intervals and a distribution of predictions (useful for
    the probability-like heuristic). Failed refits are skipped; if too few succeed the CI is
    reported as NaN rather than a false-precision number.
    """
    steps, values = _finite_pairs(steps, values)
    n = values.size
    rng = np.random.default_rng(seed)
    t_scale = float(np.max(steps)) or 1.0
    fitter = {"power": _fit_power, "exp": _fit_exp, "sqrt": _fit_sqrt}[model_name]

    preds: dict[float, list] = {ft: [] for ft in future_steps}
    asymptotes: list = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        bs, bv = steps[idx], values[idx]
        if np.ptp(bs) == 0:
            continue
        params = fitter(bs, bv, t_scale)
        if params is None:
            continue
        ok = True
        row = {}
        for ft in future_steps:
            yv = float(_predict(model_name, params, t_scale, np.array([ft]))[0])
            if not np.isfinite(yv):
                ok = False
                break
            row[ft] = yv
        if not ok:
            continue
        for ft in future_steps:
            preds[ft].append(row[ft])
        asymptotes.append(float(params.get("L_inf", params.get("a", math.nan))))

    lo_q = (1.0 - ci) / 2.0
    hi_q = 1.0 - lo_q
    out: dict = {"n_success": len(asymptotes), "ci": ci, "per_step": {}, "samples": {}}
    for ft in future_steps:
        arr = np.asarray(preds[ft], dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size >= max(10, int(0.2 * n_boot)):
            out["per_step"][ft] = {
                "median": float(np.median(arr)),
                "low": float(np.quantile(arr, lo_q)),
                "high": float(np.quantile(arr, hi_q)),
            }
            out["samples"][ft] = arr
        else:
            out["per_step"][ft] = {"median": math.nan, "low": math.nan, "high": math.nan}
            out["samples"][ft] = arr
    asy = np.asarray(asymptotes, dtype=float)
    asy = asy[np.isfinite(asy)]
    if asy.size >= max(10, int(0.2 * n_boot)):
        out["asymptote"] = {"median": float(np.median(asy)),
                            "low": float(np.quantile(asy, lo_q)),
                            "high": float(np.quantile(asy, hi_q))}
    else:
        out["asymptote"] = {"median": math.nan, "low": math.nan, "high": math.nan}
    return out


def prob_beats_target(samples: np.ndarray, target: float) -> float:
    """Heuristic (NOT a calibrated probability): fraction of bootstrap forecasts <= target."""
    samples = np.asarray(samples, dtype=float)
    samples = samples[np.isfinite(samples)]
    if samples.size == 0 or not np.isfinite(target):
        return math.nan
    return float(np.mean(samples <= target))


# --------------------------------------------------------------------------------------- #
# Top-level analysis
# --------------------------------------------------------------------------------------- #
@dataclass
class AnalysisConfig:
    ema_alpha: float = 0.3
    slope_window: int = 8
    patience: int = 5
    min_delta: float = 1e-3
    min_evals: int = 6
    min_fit_points: int = 6
    future_steps: tuple = (500, 1000, 2000)
    target_loss: Optional[float] = None
    negligible_gain_per_100: float = 1e-3  # improvement/100 steps considered economically ~0
    n_boot: int = 200
    ci: float = 0.90
    bootstrap_seed: int = 0
    run_id: str = "run"


@dataclass
class AnalysisResult:
    status: str
    run_id: str
    current_step: Optional[float]
    n_evals: int
    best_step: Optional[float]
    best_val_loss: Optional[float]
    recommended_stop_step: Optional[float]
    recent_slope_per_step: float
    recent_slope_per_log_step: float
    recent_improvement_per_100: float
    plateau: dict
    instability: dict
    chosen_model: Optional[str]
    fit_quality: dict
    all_fits: list
    predicted_val_loss: dict
    predicted_asymptote: dict
    confidence: dict
    prob_beats_target: dict
    target_loss: Optional[float]
    warnings: list
    config: dict
    # Dense forecast grid for plotting: {grid, point, median, low, high}. Empty if no fit.
    forecast: dict
    # Raw series echoed for plotting (structured data only; plotting consumes this).
    series: dict

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


_STATUS = ("CONTINUE", "PLATEAU", "UNSTABLE", "INSUFFICIENT_DATA")


def analyze(records: list[dict], config: Optional[AnalysisConfig] = None) -> AnalysisResult:
    """Run the full conservative analysis and return structured results (no side effects)."""
    cfg = config or AnalysisConfig()
    warnings: list = []
    series = to_series(records)

    st, vl = series.val_points()
    n_evals = int(vl.size)
    current_step = float(series.step[np.isfinite(series.step)][-1]) if np.any(
        np.isfinite(series.step)) else None

    # Smoothed validation loss aligned to the finite val points.
    vl_smoothed = ema(vl, cfg.ema_alpha) if n_evals else np.array([])

    instab = detect_instability(series)
    plateau = detect_plateau(st, vl_smoothed, cfg.patience, cfg.min_delta, cfg.min_evals)
    slopes = recent_slopes(st, vl_smoothed, cfg.slope_window)
    imp100 = improvement_per_100(st, vl_smoothed, cfg.slope_window)

    fits = fit_curves(st, vl, cfg.min_fit_points) if n_evals >= cfg.min_fit_points else []
    best = select_best_fit(fits)

    predicted: dict = {}
    predicted_asymptote: dict = {"point": math.nan}
    confidence: dict = {"per_step": {}, "asymptote": {"median": math.nan, "low": math.nan,
                                                       "high": math.nan}, "n_success": 0,
                        "ci": cfg.ci}
    prob_target: dict = {}
    forecast: dict = {}
    future_steps = list(cfg.future_steps)

    if best is not None:
        # Dense grid spanning observed range through the furthest requested forecast, plus the
        # exact requested future steps, so bootstrap CIs are available for both.
        grid_max = max([float(np.max(st))] + [float(f) for f in future_steps])
        grid = sorted(set(
            list(np.linspace(float(np.min(st)), grid_max, 60))
            + [float(f) for f in future_steps]
        ))
        boot = bootstrap_predictions(st, vl, best.name, grid, cfg.n_boot,
                                     cfg.bootstrap_seed, cfg.ci)
        for ft in future_steps:
            predicted[ft] = float(best.predict(np.array([ft]))[0])
        predicted_asymptote = {"point": float(best.asymptote)}
        confidence = {"n_success": boot["n_success"], "ci": boot["ci"],
                      "asymptote": boot["asymptote"],
                      "per_step": {ft: boot["per_step"][ft] for ft in future_steps}}
        if cfg.target_loss is not None:
            for ft in future_steps:
                prob_target[ft] = prob_beats_target(
                    boot["samples"].get(ft, np.array([])), cfg.target_loss)
        # Build the dense forecast curve for plotting.
        point = _to_list(best.predict(np.array(grid)))
        median = [boot["per_step"][g]["median"] for g in grid]
        low = [boot["per_step"][g]["low"] for g in grid]
        high = [boot["per_step"][g]["high"] for g in grid]
        forecast = {"grid": [float(g) for g in grid], "point": point,
                    "median": [_nan_to_none(v) for v in median],
                    "low": [_nan_to_none(v) for v in low],
                    "high": [_nan_to_none(v) for v in high]}
    else:
        if n_evals < cfg.min_fit_points:
            warnings.append(
                f"Only {n_evals} evaluations (< {cfg.min_fit_points}); no curve fitted.")

    # JSON-safe confidence dict (stringify per-step keys).
    confidence_serializable = {
        "n_success": int(confidence.get("n_success", 0)),
        "ci": confidence.get("ci", cfg.ci),
        "per_step": {str(k): v for k, v in confidence.get("per_step", {}).items()},
        "asymptote": confidence.get("asymptote", {"median": math.nan}),
    }

    status, recommended_stop, extra_warn = _decide_status(
        cfg, n_evals, instab, plateau, imp100, predicted, confidence, current_step)
    warnings.extend(extra_warn)

    fit_quality = {}
    if best is not None:
        fit_quality = {"model": best.name, "r2": best.r2, "aic": best.aic,
                       "tail_rmse": best.tail_rmse, "n_points": best.n_points}

    result = AnalysisResult(
        status=status,
        run_id=cfg.run_id,
        current_step=current_step,
        n_evals=n_evals,
        best_step=plateau.best_step,
        best_val_loss=plateau.best_value,
        recommended_stop_step=recommended_stop,
        recent_slope_per_step=slopes["slope_per_step"],
        recent_slope_per_log_step=slopes["slope_per_log_step"],
        recent_improvement_per_100=imp100,
        plateau=asdict(plateau),
        instability=asdict(instab),
        chosen_model=None if best is None else best.name,
        fit_quality=fit_quality,
        all_fits=[asdict(f) for f in fits],
        predicted_val_loss={str(k): v for k, v in predicted.items()},
        predicted_asymptote=predicted_asymptote,
        confidence=confidence_serializable,
        prob_beats_target={str(k): v for k, v in prob_target.items()},
        target_loss=cfg.target_loss,
        warnings=warnings,
        config=asdict(cfg),
        forecast=forecast,
        series=_series_for_plotting(series, vl_smoothed),
    )
    return result


def _nan_to_none(v):
    try:
        return None if not np.isfinite(v) else float(v)
    except (TypeError, ValueError):
        return None


def _decide_status(cfg, n_evals, instab: InstabilityInfo, plateau: PlateauInfo,
                   imp100: float, predicted: dict, confidence: dict,
                   current_step) -> tuple[str, Optional[float], list]:
    """Conservative status decision. Precedence: NaN/Inf > insufficient > spikes > plateau."""
    warn: list = []
    recommended_stop = None

    if instab.has_nan_inf:
        return "UNSTABLE", None, ["NaN/Inf detected in reported losses/grad norm."]
    if n_evals < cfg.min_evals:
        return "INSUFFICIENT_DATA", None, warn
    if instab.loss_spike_steps or instab.grad_spike_steps:
        return "UNSTABLE", None, ["Loss/gradient spikes detected; forecasts suppressed."]

    negligible = np.isfinite(imp100) and imp100 < cfg.negligible_gain_per_100
    if plateau.is_plateau and (negligible or plateau.evals_since_best >= cfg.patience):
        recommended_stop = current_step
        return "PLATEAU", recommended_stop, warn

    return "CONTINUE", None, warn


def _series_for_plotting(series: MetricsSeries, vl_smoothed: np.ndarray) -> dict:
    """Echo raw + smoothed arrays as plain lists so plotting consumes structured data only."""
    st, vl = series.val_points()
    return {
        "step": _to_list(series.step),
        "tokens_seen": _to_list(series.tokens_seen),
        "train_loss": _to_list(series.train_loss),
        "val_loss": _to_list(series.val_loss),
        "masked_accuracy": _to_list(series.masked_accuracy),
        "learning_rate": _to_list(series.learning_rate),
        "gradient_norm": _to_list(series.gradient_norm),
        "val_step": _to_list(st),
        "val_loss_finite": _to_list(vl),
        "val_loss_smoothed": _to_list(vl_smoothed),
        "present": series.present,
    }


def _to_list(arr: np.ndarray) -> list:
    return [None if not np.isfinite(v) else float(v) for v in np.asarray(arr, dtype=float)]
