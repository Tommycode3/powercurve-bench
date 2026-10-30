"""Banister fitness-fatigue model as a linear-Gaussian state-space model.

Model (day index t, load w_t, performance y_t observed on some days only)::

    g_t = phi1 g_{t-1} + (1 - phi1) w_{t-1},   phi1 = exp(-1/tau1)    (fitness, "CTL")
    h_t = phi2 h_{t-1} + (1 - phi2) w_{t-1},   phi2 = exp(-1/tau2)    (fatigue, "ATL")
    theta_t = theta_{t-1} + eta_t,   eta_t ~ N(0, Q)                  theta = [p0, a1, a2]
    y_t = p0_t + a1_t g_t - a2_t h_t + eps_t,   eps_t ~ N(0, sigma^2)

Conditional on (tau1, tau2, sigma) the model is linear-Gaussian in theta, so the
Kalman filter gives the exact posterior of theta and the exact marginal
likelihood p(y | tau1, tau2, sigma) by the prediction-error decomposition.
The three nonlinear hyper-parameters are integrated on a grid (exact up to the
grid), so every output is a finite Gaussian mixture.

Normalising the impulse responses by (1 - phi) makes g and h exponentially
weighted *means* of the load (units: load/day, the CTL/ATL convention), so the
prior on the gains a1, a2 [performance units per load/day] does not change
scale with tau. Classic Banister gains are k_i = a_i (1 - phi_i).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import lfilter
from scipy.special import ndtr

# ---------------------------------------------------------------- simulation


def ewma_state(load, tau):
    """g_t = phi g_{t-1} + (1-phi) w_{t-1}, g_0 = 0 (load before day t only)."""
    phi = np.exp(-1.0 / tau)
    return lfilter([0.0, 1.0 - phi], [1.0, -phi], np.asarray(load, float))


def banister_mean(load, p0, a1, a2, tau1, tau2):
    return p0 + a1 * ewma_state(load, tau1) - a2 * ewma_state(load, tau2)


def simulate_load(n_days: int, rng: np.random.Generator, base: float = 60.0):
    """Periodised synthetic load: 3:1 build/recovery blocks, rest days, noise."""
    w = np.zeros(n_days)
    for d in range(n_days):
        block = (d // 7) % 4
        weekly = base * (1.0 + 0.15 * block) if block < 3 else base * 0.55
        weekly *= 1.0 + 0.3 * (d / n_days)            # progressive overload
        dow = d % 7
        if dow in (0, 4):                              # rest days
            continue
        mult = 1.8 if dow == 5 else 1.0                # long ride
        w[d] = max(0.0, rng.normal(weekly * mult, 0.25 * weekly))
    return w


# ---------------------------------------------------------------- utilities


def mixture_quantiles(weights, mu, sd, qs=(0.025, 0.5, 0.975), prune=1e-7, iters=60):
    """Quantiles of sum_k w_k N(mu_k, sd_k^2) by vectorised bisection."""
    w = np.asarray(weights, float); mu = np.asarray(mu, float); sd = np.asarray(sd, float)
    keep = w > prune * w.max()
    w, mu, sd = w[keep] / w[keep].sum(), mu[keep], np.maximum(sd[keep], 1e-12)
    qs = np.asarray(qs, float)
    lo = np.full(qs.shape, np.min(mu - 10 * sd)); hi = np.full(qs.shape, np.max(mu + 10 * sd))
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        cdf = (w[None, :] * ndtr((mid[:, None] - mu[None, :]) / sd[None, :])).sum(1)
        lo = np.where(cdf < qs, mid, lo); hi = np.where(cdf >= qs, mid, hi)
    return 0.5 * (lo + hi)


def weighted_quantiles(values, weights, qs=(0.025, 0.5, 0.975)):
    """Quantiles of a grid posterior, treating each grid value as a log-space cell.

    Grid masses are aggregated per unique value; each value owns the cell between
    the geometric midpoints to its neighbours, and the CDF is linear in log(value)
    inside a cell. This avoids intervals collapsing onto a single grid point.
    """
    v = np.asarray(values, float); w = np.asarray(weights, float)
    u, inv = np.unique(v, return_inverse=True)
    m = np.bincount(inv, weights=w); m /= m.sum()
    if len(u) == 1:
        return np.full(len(qs), u[0])
    lu = np.log(u)
    mid = 0.5 * (lu[1:] + lu[:-1])
    edges = np.r_[lu[0] - (mid[0] - lu[0]), mid, lu[-1] + (lu[-1] - mid[-1])]
    cdf = np.r_[0.0, np.cumsum(m)]
    return np.exp(np.interp(qs, cdf, edges))


# ---------------------------------------------------------------- the filter


@dataclass
class Prior:
    mean: np.ndarray   # [p0, a1, a2]
    sd: np.ndarray


@dataclass
class KFResult:
    # grid
    tau1: np.ndarray; tau2: np.ndarray; sigma: np.ndarray   # (G,) hyper-parameters
    logpost: np.ndarray                                     # (G,) final log posterior (unnorm.)
    m: np.ndarray; P: np.ndarray                            # (G,3), (G,3,3) final filtered theta
    # per-day summaries (mixture over the grid, *one-step-ahead*, i.e. before seeing y_t)
    pred_mean: np.ndarray; pred_lo: np.ndarray; pred_hi: np.ndarray
    # per-day filtered parameter means / sds (mixture moments, after seeing y_t)
    theta_mean: np.ndarray; theta_sd: np.ndarray
    tau1_mean: np.ndarray; tau2_mean: np.ndarray
    n_obs: int

    @property
    def weights(self):
        lp = self.logpost - self.logpost.max()
        w = np.exp(lp)
        return w / w.sum()

    def summary(self, level=0.95):
        """Posterior median and central interval for every parameter."""
        a = (1 - level) / 2
        qs = (a, 0.5, 1 - a)
        w = self.weights
        out = {}
        for i, name in enumerate(("p0", "a1", "a2")):
            out[name] = mixture_quantiles(w, self.m[:, i], np.sqrt(self.P[:, i, i]), qs)
        # classic Banister gains k = a (1 - phi): sample (exact mixture would need a 2-D grid)
        rng = np.random.default_rng(0)
        idx = rng.choice(len(w), size=20000, p=w)
        for name, i, tau in (("k1", 1, self.tau1), ("k2", 2, self.tau2)):
            draws = rng.normal(self.m[idx, i], np.sqrt(self.P[idx, i, i])) * (1 - np.exp(-1 / tau[idx]))
            out[name] = np.quantile(draws, qs)
        for name, v in (("tau1", self.tau1), ("tau2", self.tau2), ("sigma", self.sigma)):
            out[name] = weighted_quantiles(v, w, qs)
        return out


class BanisterKF:
    """Grid-marginalised Kalman filter for the fitness-fatigue model.

    Parameters
    ----------
    tau1_grid, tau2_grid, sigma_grid : 1-D arrays; the product grid (restricted
        to tau1 > tau2) carries a uniform prior (log-uniform if the grids are
        geometric, which is the default in :func:`default_grids`).
    prior : Gaussian prior on theta_0 = [p0, a1, a2].
    q : per-day random-walk standard deviations of theta (0 = static parameters,
        in which case the filter is exact recursive Bayesian linear regression).
    """

    def __init__(self, tau1_grid, tau2_grid, sigma_grid, prior: Prior, q=(0.0, 0.0, 0.0)):
        T1, T2, S = np.meshgrid(tau1_grid, tau2_grid, sigma_grid, indexing="ij")
        ok = T1 > T2
        self.tau1, self.tau2, self.sigma = T1[ok], T2[ok], S[ok]
        self.prior = prior
        self.Q = np.diag(np.asarray(q, float) ** 2)

    def _features(self, load):
        load = np.asarray(load, float)
        u1, i1 = np.unique(self.tau1, return_inverse=True)
        u2, i2 = np.unique(self.tau2, return_inverse=True)
        G1 = np.stack([ewma_state(load, t) for t in u1])
        G2 = np.stack([ewma_state(load, t) for t in u2])
        return G1, i1, G2, i2

    def run(self, load, y, level=0.95, store=False):
        load = np.asarray(load, float); y = np.asarray(y, float)
        T = len(load); G = len(self.tau1)
        G1, i1, G2, i2 = self._features(load)
        m = np.tile(self.prior.mean.astype(float), (G, 1))
        P = np.tile(np.diag(self.prior.sd.astype(float) ** 2), (G, 1, 1))
        R = self.sigma ** 2
        ll = np.zeros(G)
        a = (1 - level) / 2
        pm, plo, phi = (np.full(T, np.nan) for _ in range(3))
        thm, ths = np.full((T, 3), np.nan), np.full((T, 3), np.nan)
        t1m, t2m = np.full(T, np.nan), np.full(T, np.nan)
        hist = {"m": [], "P": [], "mp": [], "Pp": []} if store else None
        n = 0
        for t in range(T):
            P = P + self.Q                                   # predict (random walk)
            if store:
                hist["mp"].append(m.copy()); hist["Pp"].append(P.copy())
            H = np.stack([np.ones(G), G1[i1, t], -G2[i2, t]], axis=1)   # (G,3)
            PH = np.einsum("gij,gj->gi", P, H)
            mu = np.einsum("gi,gi->g", H, m)
            F = np.einsum("gi,gi->g", H, PH) + R
            w = np.exp(ll - ll.max()); w /= w.sum()
            pm[t] = np.sum(w * mu)
            if t == 0 or np.isfinite(y[t]) or t == T - 1 or t % 7 == 0:
                plo[t], phi[t] = mixture_quantiles(w, mu, np.sqrt(F), (a, 1 - a))
            if np.isfinite(y[t]):
                v = y[t] - mu
                ll += -0.5 * (np.log(2 * np.pi * F) + v ** 2 / F)
                K = PH / F[:, None]
                m = m + K * v[:, None]
                P = P - np.einsum("gi,gj->gij", K, PH)
                P = 0.5 * (P + np.transpose(P, (0, 2, 1)))
                n += 1
            if store:
                hist["m"].append(m.copy()); hist["P"].append(P.copy())
            w = np.exp(ll - ll.max()); w /= w.sum()
            thm[t] = w @ m
            ths[t] = np.sqrt(w @ (np.diagonal(P, axis1=1, axis2=2) + m ** 2) - thm[t] ** 2)
            t1m[t] = w @ self.tau1; t2m[t] = w @ self.tau2
        # fill the one-step interval on days we skipped (cosmetic, plots only)
        for arr in (plo, phi):
            ok = np.isfinite(arr)
            arr[~ok] = np.interp(np.where(~ok)[0], np.where(ok)[0], arr[ok])
        res = KFResult(self.tau1, self.tau2, self.sigma, ll, m, P, pm, plo, phi, thm, ths,
                       t1m, t2m, n)
        self._last = (load, y, hist, G1, i1, G2, i2)
        return res

    # ------------------------------------------------------------ smoothing

    def smooth_states(self, load, y, res: KFResult, level=0.95, mass=0.999):
        """RTS-smoothed fitness a1*g_t and fatigue a2*h_t with mixture intervals.

        Only grid points carrying ``mass`` of the posterior are smoothed.
        """
        w = res.weights
        order = np.argsort(w)[::-1]
        k = int(np.searchsorted(np.cumsum(w[order]), mass)) + 1
        sel = order[:k]
        sub = BanisterKF.__new__(BanisterKF)
        sub.tau1, sub.tau2, sub.sigma = self.tau1[sel], self.tau2[sel], self.sigma[sel]
        sub.prior, sub.Q = self.prior, self.Q
        sub.run(load, y, store=True)
        _, _, hist, G1, i1, G2, i2 = sub._last
        T = len(load)
        ms, Ps = hist["m"][-1].copy(), hist["P"][-1].copy()
        sm = [None] * T; sP = [None] * T
        sm[-1], sP[-1] = ms, Ps
        for t in range(T - 2, -1, -1):                     # RTS, random-walk transition
            Pf, Pp = hist["P"][t], hist["Pp"][t + 1]
            C = np.einsum("gij,gjk->gik", Pf, np.linalg.inv(Pp))
            ms = hist["m"][t] + np.einsum("gij,gj->gi", C, ms - hist["mp"][t + 1])
            Ps = Pf + np.einsum("gij,gjk,glk->gil", C, Ps - Pp, C)
            sm[t], sP[t] = ms, Ps
        ws = w[sel] / w[sel].sum()
        a = (1 - level) / 2
        out = {k: np.zeros((T, 3)) for k in ("fitness", "fatigue")}
        for t in range(T):
            g, h = G1[i1, t], G2[i2, t]
            for key, i, f in (("fitness", 1, g), ("fatigue", 2, h)):
                mu = sm[t][:, i] * f
                sd = np.sqrt(np.maximum(sP[t][:, i, i], 0)) * f
                out[key][t] = mixture_quantiles(ws, mu, sd, (a, 0.5, 1 - a))
        return out

    # ------------------------------------------------------------ forecasting

    def forecast(self, res: KFResult, load_hist, load_future, level=0.95):
        """Predictive intervals for y on future days given a planned load."""
        load = np.concatenate([np.asarray(load_hist, float), np.asarray(load_future, float)])
        T0, T = len(load_hist), len(load)
        G1, i1, G2, i2 = self._features(load)
        w = res.weights; a = (1 - level) / 2
        out = np.zeros((T - T0, 3))
        for j, t in enumerate(range(T0, T)):
            H = np.stack([np.ones(len(w)), G1[i1, t], -G2[i2, t]], axis=1)
            P = res.P + (j + 1) * self.Q
            mu = np.einsum("gi,gi->g", H, res.m)
            F = np.einsum("gi,gij,gj->g", H, P, H) + res.sigma ** 2
            out[j] = mixture_quantiles(w, mu, np.sqrt(F), (a, 0.5, 1 - a))
        return out


def default_grids(y, n_tau1=30, n_tau2=30, n_sigma=24):
    y = np.asarray(y, float); yy = y[np.isfinite(y)]
    s0 = np.std(np.diff(yy)) / np.sqrt(2) if len(yy) > 3 else np.std(yy)
    s0 = max(s0, 1e-3)
    return (np.geomspace(8, 120, n_tau1), np.geomspace(1, 30, n_tau2),
            np.geomspace(0.3 * s0, 2.0 * s0, n_sigma))


def default_prior(y, gain_sd=None):
    """Weakly-informative prior: p0 ~ N(median(y), (0.5 median)^2), a_i ~ N(0, s^2).

    ``gain_sd`` defaults to 0.5 * median(y) / 100, i.e. a 100 load/day change in
    CTL may move performance by up to ~half its level a priori. Generous on purpose.
    """
    med = float(np.nanmedian(y))
    s = gain_sd if gain_sd is not None else 0.5 * med / 100.0
    return Prior(np.array([med, 0.0, 0.0]), np.array([0.5 * med, s, s]))
