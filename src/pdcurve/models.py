"""Power-duration models and a common least-squares fitter.

All models map duration ``t`` [s] to maximal mean power ``P(t)`` [W].
Fitting minimises *relative* residuals ``(P_hat - P) / P`` so that a 1 % miss
at 15 s and a 1 % miss at 60 min weigh the same (and so the loss matches the
MAPE used in the benchmark).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
from scipy.optimize import least_squares

# Fixed constants (documented in the README)
PT_K1 = 30.0     # s, aerobic on-kinetics time constant (Peronnet & Thibault 1989)
PT_K2 = 20.0     # s, anaerobic release time constant (Peronnet & Thibault 1989)
PT_TMAP = 420.0  # s, longest duration sustainable at MAP (Peronnet & Thibault 1989)
OMNI_TCPMAX = 1800.0  # s, duration after which CP starts to decline (Puchowicz et al. 2020)


def cp2(t, cp, wp):
    """2-parameter critical power (Monod & Scherrer 1965): P = CP + W'/t."""
    t = np.asarray(t, float)
    return cp + wp / t


def cp3(t, cp, wp, pmax):
    """3-parameter CP (Morton 1996): t = W'/(P-CP) - W'/(Pmax-CP), solved for P."""
    t = np.asarray(t, float)
    k = wp / np.maximum(pmax - cp, 1e-6)
    return cp + wp / (t + k)


def expo(t, cp, p0, tau):
    """Exponential decay to an asymptote: P = CP + (P0 - CP) exp(-t/tau)."""
    t = np.asarray(t, float)
    return cp + (p0 - cp) * np.exp(-t / tau)


def pt(t, s, mapw, a):
    """Peronnet-Thibault (1989) in mechanical-power form (BMR term omitted).

    P(t) = f(t) * [ MAP * (1 - k1/t * (1 - e^{-t/k1})) + S/t * (1 - e^{-t/k2}) ]
    f(t) = 1                         for t <= T_MAP
    f(t) = 1 - (A/MAP) ln(t/T_MAP)   for t >  T_MAP
    """
    t = np.asarray(t, float)
    aer = mapw * (1.0 - PT_K1 / t * (1.0 - np.exp(-t / PT_K1)))
    ana = s / t * (1.0 - np.exp(-t / PT_K2))
    f = np.where(t > PT_TMAP, 1.0 - (a / mapw) * np.log(np.maximum(t, PT_TMAP) / PT_TMAP), 1.0)
    return f * (aer + ana)


def omni(t, cp, wp, pmax, a):
    """Omni-domain power-duration model (Puchowicz, Baker & Clarke 2020).

    P(t) = W'/t (1 - e^{-t (Pmax-CP)/W'}) + CP - A ln(t/T_CPmax) * 1[t > T_CPmax]
    """
    t = np.asarray(t, float)
    base = wp / t * (1.0 - np.exp(-t * np.maximum(pmax - cp, 1e-6) / wp)) + cp
    return base - a * np.log(np.maximum(t, OMNI_TCPMAX) / OMNI_TCPMAX)


@dataclass(frozen=True)
class Model:
    name: str
    label: str
    func: Callable
    params: tuple[str, ...]
    lower: tuple[float, ...]
    upper: tuple[float, ...]
    init: Callable  # (t, p) -> list of start vectors
    citation: str = ""

    def __call__(self, t, theta):
        return self.func(t, *theta)


def _cp2_guess(t, p):
    # linear regression of work on time: W = CP t + W'
    t = np.asarray(t, float); p = np.asarray(p, float)
    A = np.vstack([t, np.ones_like(t)]).T
    cp, wp = np.linalg.lstsq(A, p * t, rcond=None)[0]
    cp = float(np.clip(cp, 0.3 * p.min(), 0.999 * p.min()))
    wp = float(np.clip(wp, 1e3, 1e5))
    return cp, wp


def _init_cp2(t, p):
    cp, wp = _cp2_guess(t, p)
    return [[cp, wp]]


def _init_cp3(t, p):
    cp, wp = _cp2_guess(t, p)
    return [[cp, wp, m * max(p)] for m in (1.2, 2.0, 4.0)]


def _init_expo(t, p):
    cp, _ = _cp2_guess(t, p)
    return [[cp, m * max(p), tau] for m in (1.2, 2.5) for tau in (30.0, 120.0, 400.0)]


def _init_pt(t, p):
    cp, wp = _cp2_guess(t, p)
    return [[wp, cp * 1.15, a] for a in (5.0, 20.0)] + [[0.5 * wp, cp * 1.05, 10.0]]


def _init_omni(t, p):
    cp, wp = _cp2_guess(t, p)
    return [[cp, wp, m * max(p), a] for m in (1.2, 2.5) for a in (5.0, 25.0)]


MODELS: dict[str, Model] = {
    "cp2": Model("cp2", "CP 2-param", cp2, ("CP", "W'"),
                 (1.0, 10.0), (2000.0, 2e5), _init_cp2, "Monod & Scherrer 1965"),
    "cp3": Model("cp3", "CP 3-param", cp3, ("CP", "W'", "Pmax"),
                 (1.0, 10.0, 50.0), (2000.0, 2e5, 5000.0), _init_cp3, "Morton 1996"),
    "exp": Model("exp", "Exponential", expo, ("CP", "P0", "tau"),
                 (1.0, 50.0, 1.0), (2000.0, 5000.0, 5000.0), _init_expo, "Hopkins et al. 1989"),
    "pt": Model("pt", "Peronnet-Thibault", pt, ("S", "MAP", "A"),
                (10.0, 10.0, 0.0), (2e5, 2500.0, 500.0), _init_pt, "Peronnet & Thibault 1989"),
    "omni": Model("omni", "Omni-domain", omni, ("CP", "W'", "Pmax", "A"),
                  (1.0, 10.0, 50.0, 0.0), (2000.0, 2e5, 5000.0, 500.0), _init_omni,
                  "Puchowicz et al. 2020"),
}


@dataclass
class FitResult:
    model: str
    theta: np.ndarray
    params: dict
    rmse_rel: float          # in-sample RMS relative error
    success: bool
    at_bound: list = field(default_factory=list)
    fixed: dict = field(default_factory=dict)

    def predict(self, t):
        return MODELS[self.model](np.asarray(t, float), self.theta)


def fit(model: str | Model, t: Sequence[float], p: Sequence[float]) -> FitResult:
    """Fit a model by bounded nonlinear least squares on relative residuals.

    Multi-start (a handful of physiologically sensible inits) and keep the best.
    For the omni-domain model the long-duration decline ``A`` is not identifiable
    when no training duration exceeds T_CPmax; in that case A is fixed to 0 and
    reported in ``FitResult.fixed``.
    """
    m = MODELS[model] if isinstance(model, str) else model
    t = np.asarray(t, float); p = np.asarray(p, float)
    ok = np.isfinite(t) & np.isfinite(p) & (t > 0) & (p > 0)
    t, p = t[ok], p[ok]
    if len(t) < len(m.params):
        raise ValueError(f"{m.name}: need >= {len(m.params)} points, got {len(t)}")

    fixed = {}
    func, lower, upper, inits = m.func, np.array(m.lower), np.array(m.upper), m.init(t, p)
    free = np.ones(len(m.params), bool)
    if m.name == "omni" and t.max() <= OMNI_TCPMAX:
        free[3] = False
        fixed["A"] = 0.0

    def full(x):
        th = np.zeros(len(m.params)); th[free] = x
        for k, v in fixed.items():
            th[m.params.index(k)] = v
        return th

    def resid(x):
        return (func(t, *full(x)) - p) / p

    best = None
    for x0 in inits:
        x0 = np.clip(np.array(x0, float)[free], lower[free] * 1.0001 + 1e-9, upper[free] * 0.9999)
        try:
            r = least_squares(resid, x0, bounds=(lower[free], upper[free]), method="trf",
                              x_scale="jac", max_nfev=2000, xtol=1e-12, ftol=1e-12, gtol=1e-12)
        except (ValueError, FloatingPointError):
            continue
        if best is None or r.cost < best.cost:
            best = r
    if best is None:
        th = np.full(len(m.params), np.nan)
        return FitResult(m.name, th, dict(zip(m.params, th)), np.nan, False, fixed=fixed)
    th = full(best.x)
    rel = np.sqrt(np.mean(resid(best.x) ** 2))
    span = upper - lower
    at_bound = [m.params[i] for i in range(len(th)) if free[i] and
                (abs(th[i] - lower[i]) < 1e-6 * span[i] or abs(th[i] - upper[i]) < 1e-6 * span[i])]
    return FitResult(m.name, th, dict(zip(m.params, th)), float(rel), bool(best.success),
                     at_bound, fixed)
