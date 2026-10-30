import numpy as np

from pdcurve.cli import main
from pdcurve.fitio import read_fit
from pdcurve.mmp import mmp
from pdcurve.models import MODELS

from fitwriter import write_fit


def _effort(cp=250.0, wp=20000.0):
    # back-to-back maximal efforts along a CP2 curve, separated by easy riding
    seg = []
    for d in (180, 240, 300, 420, 600, 720, 900, 1200):
        seg += [MODELS["cp2"](d, (cp, wp))] * d + [100.0] * 600
    return np.array(seg)


def test_read_fit_roundtrip(tmp_path):
    x = _effort()
    f = tmp_path / "ride.fit"
    write_fit(f, x)
    act = read_fit(f)
    assert act.sport == "cycling"
    assert len(act.power_1hz) == len(x)
    np.testing.assert_allclose(act.power_1hz, np.round(x))


def test_cli_fit_recovers_cp(tmp_path, capsys):
    f = tmp_path / "ride.fit"; write_fit(f, _effort())
    main(["fit", str(f), "--models", "cp2", "--min-s", "180", "--max-s", "1200"])
    out = capsys.readouterr().out
    cp = float(out.split("CP=")[1].split()[0])
    assert abs(cp - 250) < 0.5


def test_cli_mmp(tmp_path, capsys):
    f = tmp_path / "ride.fit"; x = _effort(); write_fit(f, x)
    main(["mmp", str(f)])
    rows = capsys.readouterr().out.strip().splitlines()[1:]
    d, p = map(float, rows[-1].split(","))
    assert np.isclose(p, round(mmp(np.round(x), [d])[0], 1))
