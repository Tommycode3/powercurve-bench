import numpy as np
from scipy.stats import multivariate_normal, norm

from pdcurve.banister import (BanisterKF, Prior, banister_mean, default_grids, default_prior,
                              ewma_state, mixture_quantiles, simulate_load)


def _data(seed=7, T=300, p_obs=0.4, sigma=4.0, truth=(250.0, 0.9, 1.0, 40.0, 6.0)):
    rng = np.random.default_rng(seed)
    w = simulate_load(T, rng)
    mu = banister_mean(w, *truth)
    y = np.full(T, np.nan); o = rng.random(T) < p_obs
    y[o] = mu[o] + rng.normal(0, sigma, o.sum())
    return w, y


def _design(w, y, tau1, tau2):
    o = np.isfinite(y)
    X = np.column_stack([np.ones(len(w)), ewma_state(w, tau1), -ewma_state(w, tau2)])[o]
    return X, y[o]


def test_ewma_state_definition():
    w = np.array([10.0, 0, 0, 5])
    phi = np.exp(-1 / 3)
    g = ewma_state(w, 3)
    assert g[0] == 0 and np.isclose(g[1], (1 - phi) * 10) and np.isclose(g[3], phi ** 2 * (1 - phi) * 10)


def test_kalman_equals_exact_bayesian_regression_and_marginal_likelihood():
    w, y = _data()
    tau1, tau2, s = 40.0, 6.0, 4.0
    prior = Prior(np.array([240.0, 0.0, 0.0]), np.array([50.0, 2.0, 2.0]))
    kf = BanisterKF([tau1], [tau2], [s], prior)
    res = kf.run(w, y)
    X, yo = _design(w, y, tau1, tau2)
    P0i = np.diag(1 / prior.sd ** 2)
    Pn = np.linalg.inv(X.T @ X / s ** 2 + P0i)
    mn = Pn @ (X.T @ yo / s ** 2 + P0i @ prior.mean)
    np.testing.assert_allclose(res.m[0], mn, rtol=1e-7)
    np.testing.assert_allclose(res.P[0], Pn, rtol=1e-6, atol=1e-12)
    cov = X @ np.diag(prior.sd ** 2) @ X.T + s ** 2 * np.eye(len(yo))
    ll = multivariate_normal(X @ prior.mean, cov).logpdf(yo)
    assert np.isclose(res.logpost[0], ll, rtol=1e-9)


def test_kalman_recovers_banister_parameters():
    truth = dict(p0=250.0, a1=0.9, a2=1.0, tau1=40.0, tau2=6.0)
    w, y = _data(seed=11, T=365)
    kf = BanisterKF(*default_grids(y, 18, 16, 10), default_prior(y))
    s = kf.run(w, y).summary()
    for k, v in truth.items():
        assert s[k][0] <= v <= s[k][2], (k, s[k], v)
    assert abs(s["p0"][1] - 250) < 5
    assert abs(s["tau2"][1] - 6) / 6 < 0.25


def test_one_step_intervals_are_calibrated():
    w, y = _data(seed=3, T=365)
    kf = BanisterKF(*default_grids(y, 14, 12, 8), default_prior(y))
    r = kf.run(w, y)
    o = np.isfinite(y)
    cov = np.mean((y[o] >= r.pred_lo[o]) & (y[o] <= r.pred_hi[o]))
    assert 0.85 <= cov <= 1.0


def test_smoother_static_equals_final_filter():
    w, y = _data(T=120)
    kf = BanisterKF([30.0], [5.0], [4.0], default_prior(y))
    r = kf.run(w, y)
    st = kf.smooth_states(w, y, r)
    g = ewma_state(w, 30.0)
    np.testing.assert_allclose(st["fitness"][:, 1], r.m[0, 1] * g, rtol=1e-6, atol=1e-9)


def test_mixture_quantiles_single_gaussian():
    q = mixture_quantiles([1.0], [3.0], [2.0], (0.025, 0.5, 0.975))
    np.testing.assert_allclose(q, norm(3, 2).ppf([0.025, 0.5, 0.975]), atol=1e-6)
