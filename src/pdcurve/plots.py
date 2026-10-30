"""Matplotlib figures (static PNG for the README). One palette, fixed order."""
from __future__ import annotations

import numpy as np

# Categorical slots in fixed order (validated palette, light surface)
COLORS = {"cp2": "#2a78d6", "cp3": "#eb6834", "exp": "#1baf7a", "pt": "#eda100", "omni": "#e87ba4"}
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
        "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False,
        "lines.linewidth": 2.0, "figure.dpi": 110, "savefig.dpi": 160, "savefig.bbox": "tight",
    })


def fmt_dur(x, _=None):
    x = float(x)
    if x < 60:
        return f"{x:g}s"
    if x < 3600:
        return f"{x / 60:g}m"
    return f"{x / 3600:g}h"


TICKS = [5, 15, 30, 60, 120, 300, 600, 1200, 1800, 3600, 7200]


def log_time_axis(ax, lo=None, hi=None):
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
    ax.set_xscale("log")
    ticks = [t for t in TICKS if (lo is None or t >= lo) and (hi is None or t <= hi)]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(fmt_dur))


def plot_fits(dur, p, fits, train=None, test=None, title="", ax_top=None, ax_bot=None, path=None):
    """Power-duration data + model curves (top) and relative residuals (bottom)."""
    import matplotlib.pyplot as plt
    from .models import MODELS

    style()
    if ax_top is None:
        fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(7.2, 6), sharex=True,
                                             gridspec_kw={"height_ratios": [3, 1.4]})
    dur = np.asarray(dur, float); p = np.asarray(p, float)
    ok = np.isfinite(p) & (dur >= 5)
    tt = np.geomspace(max(5, dur[ok].min()), dur[ok].max(), 400)
    ax_top.plot(dur[ok], p[ok], "o", ms=3.5, color="#b5b4ae", label="MMP envelope", zorder=1)
    if train is not None:
        ax_top.plot(train, np.interp(train, dur, p), "o", ms=7, mfc=SURF, mec=INK, mew=1.5,
                    label="training points", zorder=3)
    if test is not None:
        ax_top.plot(test, np.interp(test, dur, p), "D", ms=7, color=INK, label="held-out", zorder=3)
    for m, r in fits.items():
        c = COLORS.get(m, INK)
        ax_top.plot(tt, r.predict(tt), color=c, lw=2, label=MODELS[m].label, zorder=2)
        if ax_bot is not None:
            res = 100 * (r.predict(dur[ok]) - p[ok]) / p[ok]
            ax_bot.plot(dur[ok], res, color=c, lw=1.6)
    log_time_axis(ax_top, tt[0], tt[-1]); ax_top.set_ylabel("power [W]")
    ax_top.set_title(title, loc="left")
    ax_top.legend(fontsize=8, ncol=2)
    if ax_bot is not None:
        ax_bot.axhline(0, color=INK2, lw=0.8)
        if train is not None:
            ax_bot.axvspan(min(train), max(train), color="#f1f0ec", zorder=0)
        ax_bot.set_ylabel("residual [%]"); ax_bot.set_xlabel("duration")
        log_time_axis(ax_bot, tt[0], tt[-1])
        lim = np.nanmax(np.abs(ax_bot.get_ylim()))
        ax_bot.set_ylim(-min(lim, 60), min(lim, 60))
    if path:
        ax_top.figure.savefig(path); plt.close(ax_top.figure)
    return ax_top, ax_bot
