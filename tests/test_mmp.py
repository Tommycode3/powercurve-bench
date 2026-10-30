import numpy as np

from pdcurve.mmp import mmp, monotone_envelope, normalized_power, resample_1hz, training_load


def test_mmp_matches_brute_force():
    rng = np.random.default_rng(3)
    x = rng.gamma(2.0, 100.0, 700)
    d = np.array([1, 7, 60, 333, 700, 701])
    got = mmp(x, d)
    for i, k in enumerate(d):
        if k > len(x):
            assert np.isnan(got[i])
        else:
            assert np.isclose(got[i], max(x[j:j + k].mean() for j in range(len(x) - k + 1)))
    assert np.all(np.diff(got[:-1]) <= 1e-12)


def test_monotone_envelope():
    raw = np.array([500, 300, np.nan, 320, 250, np.nan])
    env = monotone_envelope(raw)
    np.testing.assert_allclose(env[:5], [500, 320, 320, 320, 250])
    assert np.isnan(env[5])


def test_np_and_load():
    assert np.isclose(normalized_power(np.full(3600, 250.0)), 250.0)
    assert np.isclose(training_load(3600, 250.0, 250.0), 100.0)
    # variable power -> NP > mean
    x = np.tile(np.r_[np.full(60, 400.0), np.full(60, 100.0)], 30)
    assert normalized_power(x) > x.mean()


def test_resample_gaps():
    t = np.r_[np.arange(0, 10), np.arange(40, 50)]
    p = np.full(len(t), 200.0)
    out = resample_1hz(t, p, max_gap_s=10)
    assert len(out) == 50 and out[20] == 0.0 and out[5] == 200.0
