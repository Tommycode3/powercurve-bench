# powercurve-bench

[![ci](https://github.com/Tommycode3/powercurve-bench/actions/workflows/ci.yml/badge.svg)](.github/workflows/ci.yml)

![out-of-sample errors](figures/fig2_oos_error.png)

Power-duration models are usually scored on the points they were fitted to. This repo scores five of them on what a coach actually uses them for, predicting efforts that were not done, across 874 cyclists from GoldenCheetah OpenData. Fitted on a classic 2–20 min test window, the 2-parameter CP model over-predicts 60-min power by +17.6 % on average and 30-s power by +28 %. The repo also has the Banister fitness-fatigue model rewritten as a state-space model, so a Kalman filter can update fitness, fatigue and its own parameters online with calibrated intervals.

<!-- TODO(Tommaso): dictate 3-4 lines in your own words -->

## Install and use

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .

# MMP envelope of a block of training (any number of FIT files)
pdcurve mmp rides/*.fit > my_mmp.csv

# fit every model on the 2-20 min part of the envelope, with a figure
pdcurve fit rides/*.fit --min-s 120 --max-s 1200 --plot my_fit.png
# ...or on an existing curve
pdcurve fit --mmp-csv my_mmp.csv --models cp3,pt,omni --min-s 15 --max-s 3600

# daily load with your CP as threshold, then the Kalman fitness-fatigue model
pdcurve load rides/*.fit --cp 265 > load.csv          # date,load
# add a "perf" column (e.g. best 5-min power on test days), then:
pdcurve banister season.csv --forecast-days 14 --plot ff.png
```

FIT files from COROS, Garmin or Wahoo are read with [`fitdecode`](https://github.com/polyvertex/fitdecode), using the native `power` field or a developer field whose name contains "power" (Stryd and similar). Samples go on a 1 Hz grid; gaps ≤ 10 s are interpolated, longer gaps count as 0 W. Running power works the same way, but the benchmark is cycling only.

To reproduce everything:

```bash
pip install -e ".[dev]"
pytest -q                      # 26 synthetic tests, no network
bash scripts/reproduce.sh      # downloads 79 MB, ~10 min on a laptop, single process
```

The scripts in [`scripts/`](scripts/) run in order (download, prepare, benchmark, figures, Kalman simulation, Kalman on real data). CI reruns the full benchmark from the committed envelopes and checks every MAPE against the committed table (|Δ| < 0.2 pp).

## Results

Pooled out-of-sample MAPE (%), N = 874 athletes. Per-duration numbers with bootstrap CIs and bias are in [`results/benchmark_table.csv`](results/benchmark_table.csv).

| protocol | CP 2-param | CP 3-param | Exponential | Peronnet–Thibault | Omni-domain |
|---|---|---|---|---|---|
| P1: fit 2–20 min, predict 30 s, 30 min, 60 min | 18.6 | 13.2 | 17.4 | **10.6** | 14.4 |
| P2: fit 8 points 15 s–80 min, predict 7 in-between | 9.7 | 5.1 | 8.0 | **3.7** | 4.0 |

Peronnet–Thibault beats every other model in both protocols (Wilcoxon signed-rank, Holm-corrected, $p<10^{-11}$, [`results/paired_tests.md`](results/paired_tests.md)). Under P1 it beats CP3 for 73 % of athletes (median 3.4 pp); under P2 its margin over omni-domain is a median 0.38 pp, better for 63 % of athletes. Its advantage in P1 comes mostly from its log-decline starting at 7 min, inside the training window. The omni-domain decline starts at 30 min, so from 2–20 min data it is not estimable and gets fixed to 0. The same switch at $t=1800$ s gives omni-domain a +6 % bias at exactly 30 min under P2.

CP is also not a model-free constant. On identical 2–20 min data the 3-parameter model gives a CP a median 15 W lower than the 2-parameter model (CP2 264 W, IQR 228–292; CP3 238 W, 201–276), and W′ roughly doubles (13.4 kJ vs 28.6 kJ). CP3 $P_{\max}$ hits its 5000 W bound for 155/874 athletes under P1, because 2–20 min data say nothing about sprint power.

## How the benchmark works

The data is the GoldenCheetah OpenData index (Liversedge 2018, CC0). The full archive is about 116 GB; the two index files used here, `activities.csv.zip` (55.1 MB) and `activities_mmp.csv.zip` (24.2 MB), are 79.3 MB in total and cover 1,150 athletes and 642,733 activities (SHA-256 in [`results/dataset_manifest.json`](results/dataset_manifest.json)). They are streamed one athlete at a time, peak RSS ≈ 110 MB. From 295,525 bike rides with Normalized Power, spikes are dropped (1-s > 2500 W or 15-s > 2000 W) and each ride's curve gets a monotone repair $\tilde P(t)=\max_{t'\ge t}P(t')$, since a valid MMP curve cannot increase with duration. Each athlete contributes one curve, the upper envelope of the 90-day window with the most rides (≥ 10), so athletes are independent units. 874 survive all filters, with a median of 57 rides per window (IQR 38–75). The envelopes are committed in [`data/derived/envelopes.csv`](data/derived/envelopes.csv), so the benchmark reruns without downloading.

The five models give maximal mean power $P$ as a function of duration $t$. CP2 (Monod & Scherrer 1965) is $P=\mathrm{CP}+W'/t$. CP3 (Morton 1996) adds a finite $P_{\max}$:

$$P(t)=\mathrm{CP}+\frac{W'}{t+k},\qquad k=\frac{W'}{P_{\max}-\mathrm{CP}}$$

The exponential form (Hopkins et al. 1989) is $P=\mathrm{CP}+(P_0-\mathrm{CP})\,e^{-t/\tau}$. Peronnet–Thibault (1989), in mechanical-power form without the basal-metabolic term, with $k_1=30$ s, $k_2=20$ s, $T_{\mathrm{MAP}}=420$ s fixed and $S, \mathrm{MAP}, A$ free:

$$P(t)=f(t)\left[\mathrm{MAP}\left(1-\frac{k_1}{t}\left(1-e^{-t/k_1}\right)\right)+\frac{S}{t}\left(1-e^{-t/k_2}\right)\right],\qquad
f(t)=\begin{cases}1 & t\le T_{\mathrm{MAP}}\\ 1-\frac{A}{\mathrm{MAP}}\ln\frac{t}{T_{\mathrm{MAP}}} & t>T_{\mathrm{MAP}}\end{cases}$$

Omni-domain (Puchowicz, Baker & Clarke 2020):

$$P(t)=\frac{W'}{t}\left(1-e^{-t\,(P_{\max}-\mathrm{CP})/W'}\right)+\mathrm{CP}-A\ln\!\left(\frac{t}{T_{\mathrm{CP}_{\max}}}\right)\mathbb{1}[t>T_{\mathrm{CP}_{\max}}],\qquad T_{\mathrm{CP}_{\max}}=1800\ \mathrm{s}$$

All are fitted with bounded multi-start least squares on relative residuals, so the loss matches the metric. P1 trains on 2, 3, 4, 5, 7, 10, 12, 15, 20 min and holds out 30 s, 30 min, 60 min. P2 trains on 15 s, 1, 2, 5, 10, 20, 40, 80 min and holds out 30 s, 90 s, 3, 7, 15, 30, 60 min. MAPE CIs are 95 % percentile bootstrap over athletes (2,000 resamples).

## Kalman fitness-fatigue model

Load $w_t$ enters two exponentially weighted means (the CTL/ATL convention), and the gains and baseline are the state:

$$g_t=\phi_1 g_{t-1}+(1-\phi_1)\,w_{t-1},\qquad h_t=\phi_2 h_{t-1}+(1-\phi_2)\,w_{t-1},\qquad \phi_i=e^{-1/\tau_i}$$

$$\theta_t=\begin{bmatrix}p_{0,t}\\ a_{1,t}\\ a_{2,t}\end{bmatrix}=\theta_{t-1}+\eta_t,\qquad y_t=\begin{bmatrix}1 & g_t & -h_t\end{bmatrix}\theta_t+\varepsilon_t,\quad \varepsilon_t\sim\mathcal N(0,\sigma^2)$$

The classic Banister gains are $k_i=a_i(1-\phi_i)$. Normalizing by $(1-\phi_i)$ keeps $g_t$ on the same scale for every $\tau_1$, so one prior on $a_1$ means the same thing across the grid; without it the marginal likelihood picks up a spurious $\tau$-dependent Occam factor. Given $\psi=(\tau_1,\tau_2,\sigma)$ the model is linear-Gaussian, the Kalman filter is exact, and the prediction-error decomposition gives the exact marginal likelihood online:

$$\log p(y_{1:t}\mid\psi)=-\frac12\sum_{s\le t,\ y_s\ \mathrm{obs}}\left(\log 2\pi F_s+\frac{v_s^2}{F_s}\right)$$

$\psi$ is integrated on a 30 × 30 × 24 geometric grid ($\tau_1\in[8,120]$ d, $\tau_2\in[1,30]$ d, $\tau_1>\tau_2$, about 19k filters as one vectorized NumPy recursion), so every prediction is a Gaussian mixture and intervals are exact mixture quantiles. Fitness and fatigue trajectories come from a Rauch–Tung–Striebel smoother on the grid points carrying 99.9 % of the posterior. Kolossa et al. (2017) used a Kalman filter as feedback for this model and Busso (2003) made the gains time-varying; here the time constants and noise are marginalized instead of fixed.

On simulated seasons ($p_0=250$ W, $\tau_1=40$ d, $\tau_2=6$ d, $\sigma=5$ W, ≈128 tests over 365 days), coverage of the 95 % interval over 100 replicates is 0.92–0.96 for all eight parameters and 95.3 % for the one-step predictive interval. A first version with point-mass grid quantiles gave 0.57 coverage for $\sigma$ and 0.72 for $\tau_2$; treating grid values as log-space cells fixed it.

On one real athlete-season (153 test days, CP = 364 W, [`data/demo/athlete_season.csv`](data/demo/athlete_season.csv)) the one-step MAE is 12.1 W static and 11.6 W with a $p_0$ random walk, against 11.9 W for a naive mean of the last 5 tests. The 95-day forecast is worse than that naive baseline (12.0 vs 9.4 W). Both gain intervals include zero, so this season cannot tell fitness from fatigue, but the intervals are calibrated (94–98 %).

## Limitations

- MMP envelopes from training are not maximal tests. Part of the 60-min "error" is submaximal data.
- Cycling only, one 90-day window per athlete. The 1,150 indexed athletes are not a random sample.
- Peronnet–Thibault and exponential are one implementation each of forms with several variants in the literature.
- Banister on field data is weakly identified. One athlete-season is a demonstration, not evidence.
- Grid integration is exact only up to grid resolution.

## References

* Monod H, Scherrer J (1965). The work capacity of a synergic muscular group. *Ergonomics* 8(3):329–338. [10.1080/00140136508930810](https://doi.org/10.1080/00140136508930810)
* Morton RH (1996). A 3-parameter critical power model. *Ergonomics* 39(4):611–619. [10.1080/00140139608964484](https://doi.org/10.1080/00140139608964484)
* Hopkins WG, Edmond IM, Hamilton BH, Macfarlane DJ, Ross BH (1989). Relation between power and endurance for treadmill running of short duration. *Ergonomics* 32(12):1565–1571. [10.1080/00140138908966925](https://doi.org/10.1080/00140138908966925)
* Peronnet F, Thibault G (1989). Mathematical analysis of running performance and world running records. *J Appl Physiol* 67(1):453–465. [10.1152/jappl.1989.67.1.453](https://doi.org/10.1152/jappl.1989.67.1.453)
* Puchowicz MJ, Baker J, Clarke DC (2020). Development and field validation of an omni-domain power-duration model. *J Sports Sci* 38(7):801–813. [10.1080/02640414.2020.1735609](https://doi.org/10.1080/02640414.2020.1735609)
* Calvert TW, Banister EW, Savage MV, Bach T (1976). A systems model of the effects of training on physical performance. *IEEE Trans Syst Man Cybern* SMC-6(2):94–102. [10.1109/TSMC.1976.5409179](https://doi.org/10.1109/TSMC.1976.5409179)
* Busso T (2003). Variable dose-response relationship between exercise training and performance. *Med Sci Sports Exerc* 35(7):1188–1195. [10.1249/01.MSS.0000074465.13621.37](https://doi.org/10.1249/01.MSS.0000074465.13621.37)
* Clarke DC, Skiba PF (2013). Rationale and resources for teaching the mathematical modeling of athletic training and performance. *Adv Physiol Educ* 37(2):134–152. [10.1152/advan.00078.2011](https://doi.org/10.1152/advan.00078.2011)
* Kolossa D, Bin Azhar MA, Rasche C, Endler S, Hanakam F, Ferrauti A, Pfeiffer M (2017). Performance estimation using the fitness-fatigue model with Kalman filter feedback. *Int J Comput Sci Sport* 16(2):117–129. [10.1515/ijcss-2017-0010](https://doi.org/10.1515/ijcss-2017-0010)
* Kalman RE (1960). A new approach to linear filtering and prediction problems. *J Basic Eng* 82(1):35–45. [10.1115/1.3662552](https://doi.org/10.1115/1.3662552)
* Rauch HE, Tung F, Striebel CT (1965). Maximum likelihood estimates of linear dynamic systems. *AIAA J* 3(8):1445–1450. [10.2514/3.3166](https://doi.org/10.2514/3.3166)
* Liversedge M (2018). GoldenCheetah OpenData Project. OSF. [10.17605/OSF.IO/6HFPZ](https://doi.org/10.17605/OSF.IO/6HFPZ), CC0 1.0.

Normalized Power and the TSS formula follow Allen & Coggan, *Training and Racing with a Power Meter* (book, no DOI).

Code: MIT, © 2026 Tommaso Maggiolo. Data: GoldenCheetah OpenData is CC0; derived files in `data/` inherit no additional restrictions.
