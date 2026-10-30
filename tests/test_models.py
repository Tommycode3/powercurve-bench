import numpy as np
import pytest

from pdcurve.models import MODELS, OMNI_TCPMAX, fit

T = np.array([5, 10, 15, 30, 60, 120, 180, 300, 420, 600, 900, 1200, 1800, 2400, 3600, 5400, 7200.0])
TRUE = {
    "cp2": (260.0, 18000.0),
    "cp3": (250.0, 21000.0, 1100.0),
    "exp": (255.0, 900.0, 95.0),
    "pt": (17000.0, 300.0, 22.0),
    "omni": (265.0, 19000.0, 1150.0, 30.0),
}


@pytest.mark.parametrize("name", list(MODELS))
def test_exact_parameter_recovery(name):
    p = MODELS[name](T, TRUE[name])
    r = fit(name, T, p)
    assert r.success
    np.testing.assert_allclose(r.theta, TRUE[name], rtol=1e-5)
    assert r.rmse_rel < 1e-8


@pytest.mark.parametrize("name", list(MODELS))
def test_recovery_from_noisy_data_is_close(name):
    rng = np.random.default_rng(0)
    p = MODELS[name](T, TRUE[name]) * (1 + rng.normal(0, 0.002, len(T)))
    r = fit(name, T, p)
    assert np.all(np.abs(r.theta - TRUE[name]) / np.abs(TRUE[name]) < 0.1)


def test_omni_fixes_A_when_not_identifiable():
    t = T[T <= OMNI_TCPMAX]
    r = fit("omni", t, MODELS["omni"](t, TRUE["omni"]))
    assert r.fixed == {"A": 0.0} and r.params["A"] == 0.0
    np.testing.assert_allclose(r.theta[:3], TRUE["omni"][:3], rtol=1e-5)


def test_cp3_limits():
    cp, wp, pmax = TRUE["cp3"]
    assert np.isclose(MODELS["cp3"](np.array([1e-9]), TRUE["cp3"])[0], pmax, rtol=1e-6)
    assert np.isclose(MODELS["cp3"](np.array([1e9]), TRUE["cp3"])[0], cp, rtol=1e-4)


def test_models_monotone_decreasing():
    tt = np.geomspace(2, 7200, 300)
    for name, th in TRUE.items():
        assert np.all(np.diff(MODELS[name](tt, th)) <= 1e-9), name
