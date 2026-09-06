from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.signal import lfilter

try:  # torch is only needed for the Dataset wrappers
    import torch
    from torch.utils.data import Dataset, TensorDataset

    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    Dataset = object
    TensorDataset = object
    _HAS_TORCH = False

Array = np.ndarray
Gen = np.random.Generator

# ===========================================================================
# Registry
# ===========================================================================

GENERATORS: Dict[str, Callable[[Gen, int], Array]] = {}
FAMILY_WEIGHT: Dict[str, float] = {}
FAMILY_GROUP: Dict[str, str] = {}


def register(name: str, group: str = "misc", weight: float = 1.0):
    """Register a generator ``fn(rng, seq_len) -> np.ndarray``."""

    def deco(fn):
        GENERATORS[name] = fn
        FAMILY_WEIGHT[name] = weight
        FAMILY_GROUP[name] = group
        return fn

    return deco


# ===========================================================================
# Primitives
# ===========================================================================


def _t(n: int) -> Array:
    return np.arange(n, dtype=np.float64)


def _u(n: int) -> Array:
    return np.linspace(0.0, 1.0, n)


def _z(x: Array) -> Array:
    s = x.std()
    return (x - x.mean()) / (s + 1e-12)


def _phase(rng: Gen) -> float:
    return float(rng.uniform(0.0, 2 * np.pi))


def _sign(rng: Gen) -> float:
    return 1.0 if rng.random() < 0.5 else -1.0


def _period(rng: Gen, n: int, min_cycles: float = 2.0, pmin: float = 4.0,
            pmax: Optional[float] = None) -> float:
    """Log-uniform period that fits ``min_cycles`` times into the window.

    Log-uniform beats a hand-written list: it covers the whole resolvable
    range without over-representing round numbers like 24 and 96.
    """
    hi = n / float(min_cycles)
    if pmax is not None:
        hi = min(hi, float(pmax))
    hi = max(hi, pmin * 1.01)
    return float(np.exp(rng.uniform(np.log(pmin), np.log(hi))))


def _colored_noise(rng: Gen, n: int, beta: float = 1.0) -> Array:
    """1/f**beta noise by spectral synthesis. beta=0 white, 1 pink, 2 brown."""
    m = max(int(n), 16)
    f = np.fft.rfftfreq(m)
    amp = np.zeros_like(f)
    amp[1:] = f[1:] ** (-beta / 2.0)
    ph = rng.uniform(0.0, 2 * np.pi, f.size)
    x = np.fft.irfft(amp * np.exp(1j * ph), n=m)[:n]
    s = x.std()
    return x / s if s > 1e-12 else x


def _stable_poly(rng: Gen, p: int, rmin: float = 0.2, rmax: float = 0.97) -> Array:
    """Coefficients [1, a1, ..., ap] whose roots all lie inside the unit disk."""
    poles: List[complex] = []
    while len(poles) < p:
        if p - len(poles) >= 2 and rng.random() < 0.75:
            r = rng.uniform(rmin, rmax)
            th = rng.uniform(0.03, np.pi - 0.03)
            poles += [r * np.exp(1j * th), r * np.exp(-1j * th)]
        else:
            poles.append(complex(rng.uniform(-rmax, rmax), 0.0))
    return np.real(np.poly(np.array(poles[:p])))


def _filtered_noise(rng: Gen, n: int, b: Array, a: Array, burn: int = 256) -> Array:
    e = rng.standard_normal(n + burn)
    return np.asarray(lfilter(b, a, e))[burn:]


def _wave(t: Array, period: float, shape: str, duty: float = 0.5,
          smooth: float = 0.0) -> Array:
    """Unified periodic waveform. Replaces ~8 near-duplicate wave families."""
    ph = np.remainder(t, period) / period
    if shape == "sine":
        y = np.sin(2 * np.pi * ph)
    elif shape == "square":
        y = np.where(ph < duty, 1.0, -1.0)
    elif shape == "triangle":
        up = np.clip(ph / max(duty, 1e-3), 0, 1)
        dn = np.clip((1 - ph) / max(1 - duty, 1e-3), 0, 1)
        y = 2 * np.minimum(up, dn) - 1
    elif shape == "sawtooth":
        y = 2 * ph - 1
    elif shape == "pulse":
        y = np.where(ph < duty, 1.0, 0.0)
    elif shape == "trapezoid":
        r = max(duty, 1e-3) / 2
        y = np.clip(np.minimum(ph / r, (1 - ph) / r), 0, 1) * 2 - 1
    else:
        raise ValueError(shape)
    if smooth > 0:  # single-pole lowpass -> rounded edges, like a real actuator
        alpha = float(np.exp(-1.0 / max(smooth * period, 1e-3)))
        y = np.asarray(lfilter([1 - alpha], [1.0, -alpha], y))
    return y


def _warp_time(rng: Gen, x: Array, strength: float = 0.25) -> Array:
    """Monotone random time warp (local speed-ups / slow-downs)."""
    n = x.size
    k = int(rng.integers(2, 6))
    knots = np.concatenate([[0.0], np.sort(rng.uniform(0, 1, k)), [1.0]])
    speeds = np.exp(rng.normal(0.0, strength, k + 2))
    prof = np.interp(_u(n), knots, speeds)
    w = np.cumsum(prof)
    w = (w - w[0]) / (w[-1] - w[0] + 1e-12) * (n - 1)
    return np.interp(w, _t(n), x)


# ===========================================================================
# Trend / growth
# ===========================================================================


@register("linear", "trend", weight=0.5)
def g_linear(rng: Gen, n: int) -> Array:
    # Scale and intercept are removed by normalization, so only sign matters.
    # 20 slope values collapse to 2 signals; weight the family accordingly.
    return _sign(rng) * _t(n)


@register("polynomial", "trend")
def g_polynomial(rng: Gen, n: int) -> Array:
    # Sampled on t in [-1, 1]. Evaluating x**5 on [1000, 1096] (as in the
    # original) gives a monotone curve, never real polynomial structure.
    t = np.linspace(-1.0, 1.0, n)
    deg = int(rng.integers(2, 7))
    basis = np.polynomial.chebyshev.chebvander(t, deg)[:, 1:]
    c = rng.normal(0.0, 1.0, deg) / (1.0 + np.arange(deg))
    return basis @ c


@register("power_trend", "trend")
def g_power(rng: Gen, n: int) -> Array:
    t = _u(n) + 1e-3
    p = float(rng.choice([0.25, 0.4, 0.6, 1.5, 2.0, 2.5, 3.5]))
    return _sign(rng) * t ** p


@register("exp_trend", "trend")
def g_exp(rng: Gen, n: int) -> Array:
    return _sign(rng) * np.exp(rng.uniform(1.0, 6.0) * _u(n))


@register("log_trend", "trend")
def g_log(rng: Gen, n: int) -> Array:
    a = rng.uniform(0.01, 1.0)
    return _sign(rng) * np.log(a + _u(n))


@register("exp_relaxation", "trend")
def g_relax(rng: Gen, n: int) -> Array:
    t = _u(n)
    return -np.exp(-rng.uniform(2.0, 12.0) * t)


@register("logistic_growth", "trend")
def g_logistic(rng: Gen, n: int) -> Array:
    t = _u(n)
    k = rng.uniform(5.0, 30.0)
    c = rng.uniform(0.15, 0.85)
    return 1.0 / (1.0 + np.exp(-k * (t - c)))


@register("gompertz", "trend")
def g_gompertz(rng: Gen, n: int) -> Array:
    t = _u(n)
    b, c = rng.uniform(1.0, 6.0), rng.uniform(2.0, 10.0)
    return np.exp(-b * np.exp(-c * t))


@register("bass_diffusion", "trend")
def g_bass(rng: Gen, n: int) -> Array:
    """Product-adoption curve; cumulative or incremental (bell) form."""
    t = _u(n) * rng.uniform(1.0, 3.0)
    p, q = rng.uniform(0.005, 0.05), rng.uniform(0.2, 0.8)
    e = np.exp(-(p + q) * t * 10)
    F = (1 - e) / (1 + (q / p) * e)
    if rng.random() < 0.5:
        return F
    return np.gradient(F)


@register("piecewise_linear", "trend")
def g_piecewise_linear(rng: Gen, n: int) -> Array:
    """Several changepoints, not just one: real trends break repeatedly."""
    k = int(rng.integers(1, 5))
    cps = np.sort(rng.integers(n // 8, n, k))
    slopes = rng.normal(0.0, 1.0, k + 1)
    t = _t(n)
    y = slopes[0] * t
    for i, cp in enumerate(cps):
        y = y + (slopes[i + 1] - slopes[i]) * np.clip(t - cp, 0, None)
    return y


@register("saturating_ramp", "trend")
def g_saturating_ramp(rng: Gen, n: int) -> Array:
    """Ramp that hits a ceiling and holds: capacity-limited growth."""
    t = _u(n)
    knee = rng.uniform(0.25, 0.8)
    y = np.minimum(t / knee, 1.0)
    if rng.random() < 0.3:  # then decays off the plateau
        y = y * np.exp(-rng.uniform(0.0, 1.5) * np.clip(t - knee, 0, None))
    return y


# ===========================================================================
# Seasonal / periodic
# ===========================================================================


@register("harmonic_sum", "seasonal", weight=1.5)
def g_harmonic(rng: Gen, n: int) -> Array:
    t = _t(n)
    P = _period(rng, n, min_cycles=2.5, pmin=6)
    y = np.zeros(n)
    for h in range(1, int(rng.integers(2, 7))):
        if h * 2 > P:  # stay below Nyquist for the harmonic
            break
        y += (rng.uniform(0.3, 1.5) / h ** rng.uniform(0.5, 2.0)) * np.sin(
            2 * np.pi * h * t / P + _phase(rng)
        )
    return y


@register("multi_seasonal", "seasonal", weight=1.5)
def g_multi_seasonal(rng: Gen, n: int) -> Array:
    """Two or three cycles at different scales, as in hourly/daily/weekly data."""
    t = _t(n)
    y = np.zeros(n)
    long_p = _period(rng, n, min_cycles=1.5, pmin=16)
    for i in range(int(rng.integers(2, 4))):
        P = long_p / float(rng.choice([1, 2, 3, 4, 6, 7, 12, 24])) if i else long_p
        if P < 3:
            continue
        y += rng.uniform(0.3, 1.0) * np.sin(2 * np.pi * t / P + _phase(rng))
    return y


@register("shaped_wave", "seasonal", weight=2.0)
def g_shaped_wave(rng: Gen, n: int) -> Array:
    """One parameterized waveform family covering square/triangle/saw/pulse/
    trapezoid at any duty cycle and edge smoothness."""
    t = _t(n)
    shape = str(rng.choice(["square", "triangle", "sawtooth", "pulse",
                            "trapezoid", "sine"]))
    return _wave(t, _period(rng, n, min_cycles=2.5, pmin=5),
                 shape, duty=float(rng.uniform(0.1, 0.9)),
                 smooth=float(rng.choice([0.0, 0.0, 0.02, 0.05, 0.15])))


@register("wave_mixture", "seasonal")
def g_wave_mixture(rng: Gen, n: int) -> Array:
    """Two waveforms at commensurate periods: the composite repeats exactly."""
    t = _t(n)
    P = _period(rng, n, min_cycles=3.0, pmin=6)
    m = float(rng.choice([2, 3, 4, 6]))
    y = _wave(t, P, str(rng.choice(["square", "triangle", "sine", "sawtooth"])))
    y += rng.uniform(0.2, 0.8) * _wave(
        t, P * m, str(rng.choice(["square", "triangle", "sine", "sawtooth"]))
    )
    return y


@register("repeated_motif", "seasonal", weight=1.5)
def g_repeated_motif(rng: Gen, n: int) -> Array:
    """Arbitrary motif tiled to fill the window. Smooth or rough."""
    P = int(_period(rng, n, min_cycles=2.5, pmin=6))
    if rng.random() < 0.5:
        motif = rng.standard_normal(P)
    else:  # band-limited motif -> smooth but non-sinusoidal shape
        motif = _colored_noise(rng, P, beta=rng.uniform(1.0, 2.5))
    reps = -(-n // P)
    y = np.tile(motif, reps)[:n]
    return y + rng.uniform(-0.5, 0.5) * _t(n) / n * y.std() * 3


@register("drifting_seasonal", "seasonal", weight=1.5)
def g_drifting_seasonal(rng: Gen, n: int) -> Array:
    """Phase-drifting cycle. Real seasonality is never exactly periodic, and a
    model trained only on perfect periods over-commits to the phase."""
    P = _period(rng, n, min_cycles=3.0, pmin=8)
    jitter = _colored_noise(rng, n, beta=2.0) * rng.uniform(0.01, 0.08)
    phase = np.cumsum(1.0 + jitter) * 2 * np.pi / P
    return np.sin(phase + _phase(rng))


@register("amplitude_modulated", "seasonal")
def g_am(rng: Gen, n: int) -> Array:
    t = _t(n)
    Pc = _period(rng, n, min_cycles=6.0, pmin=4)
    Pm = Pc * rng.uniform(4.0, 20.0)
    env = 1.0 + rng.uniform(0.2, 1.0) * np.sin(2 * np.pi * t / Pm + _phase(rng))
    return env * np.sin(2 * np.pi * t / Pc + _phase(rng))


@register("frequency_modulated", "seasonal")
def g_fm(rng: Gen, n: int) -> Array:
    t = _u(n)
    carrier = n / _period(rng, n, min_cycles=4.0, pmin=6)   # cycles per window
    mf = rng.uniform(1.0, 5.0)
    depth = rng.uniform(0.1, 0.5) * carrier
    ph = 2 * np.pi * (carrier * t - depth / (2 * np.pi * mf) * np.cos(2 * np.pi * mf * t))
    return np.sin(ph)


@register("beats", "seasonal")
def g_beats(rng: Gen, n: int) -> Array:
    t = _t(n)
    P = _period(rng, n, min_cycles=8.0, pmin=4)
    P2 = P * (1.0 + rng.uniform(0.03, 0.2))
    return np.sin(2 * np.pi * t / P + _phase(rng)) + np.sin(2 * np.pi * t / P2)


@register("chirp", "seasonal")
def g_chirp(rng: Gen, n: int) -> Array:
    t = _u(n)
    fmax = n / 8.0                                          # >= 8 samples/cycle
    f0 = rng.uniform(1.0, 0.25 * fmax)
    f1 = rng.uniform(f0 + 1.0, fmax)
    if rng.random() < 0.3:
        f0, f1 = f1, f0  # down-chirp
    return np.sin(2 * np.pi * (f0 * t + 0.5 * (f1 - f0) * t ** 2))


@register("damped_oscillation", "seasonal")
def g_damped(rng: Gen, n: int) -> Array:
    t = _u(n)
    P = _period(rng, n, min_cycles=3.0, pmin=4)
    env = np.exp(-rng.uniform(0.5, 6.0) * t)
    if rng.random() < 0.3:
        env = 1.0 / env  # growing instability
    return env * np.sin(2 * np.pi * t * n / P + _phase(rng))


@register("waveform_shaping", "seasonal")
def g_shaping(rng: Gen, n: int) -> Array:
    """Rectified / clipped / dead-zoned sine: sensor and actuator nonlinearity."""
    t = _t(n)
    y = np.sin(2 * np.pi * t / _period(rng, n, min_cycles=3.0, pmin=5) + _phase(rng))
    mode = rng.random()
    if mode < 0.25:
        return np.abs(y)
    if mode < 0.5:
        return np.clip(y, 0, None)
    if mode < 0.75:
        c = rng.uniform(0.2, 0.8)
        return np.clip(y, -c, c)
    dz = rng.uniform(0.1, 0.5)  # dead zone
    return np.sign(y) * np.clip(np.abs(y) - dz, 0, None)


@register("pulse_train_decay", "seasonal")
def g_pulse_decay(rng: Gen, n: int) -> Array:
    """Periodic impulses convolved with a decay kernel (charge/discharge)."""
    P = _period(rng, n, min_cycles=3.0, pmin=6)
    imp = np.zeros(n)
    imp[np.arange(0, n, max(int(P), 2))] = 1.0
    tau = max(P * rng.uniform(0.05, 0.4), 1.0)
    if rng.random() < 0.5:
        a = float(np.exp(-1.0 / tau))
        return np.asarray(lfilter([1.0], [1.0, -a], imp))
    k = np.exp(-_t(int(min(n, 6 * tau))) / tau) * np.sin(
        np.pi * _t(int(min(n, 6 * tau))) / (2 * tau)
    )
    return np.convolve(imp, k)[:n]


@register("staircase", "seasonal")
def g_staircase(rng: Gen, n: int) -> Array:
    P = _period(rng, n, min_cycles=2.5, pmin=max(8.0, n / 40.0))
    y = np.floor(_t(n) / P) * _sign(rng)
    return y + rng.uniform(-1.0, 1.0) * _t(n) / P * 0.5


@register("calendar_profile", "seasonal", weight=1.5)
def g_calendar(rng: Gen, n: int) -> Array:
    """Intraday shape x weekday/weekend factor x holiday spikes: the structure
    that dominates energy, traffic and retail data."""
    t = _t(n)
    day = _period(rng, n, min_cycles=3.0, pmin=8)
    tod = np.remainder(t, day) / day
    peaks = int(rng.integers(1, 3))
    intra = np.zeros(n)
    for _ in range(peaks):
        c, w = rng.uniform(0.2, 0.9), rng.uniform(0.04, 0.15)
        intra += rng.uniform(0.5, 1.0) * np.exp(-((tod - c) ** 2) / (2 * w ** 2))
    dow = np.floor(t / day) % 7
    factor = np.where(dow >= 5, rng.uniform(0.3, 0.9), 1.0)
    y = intra * factor + rng.uniform(0.05, 0.3)
    for _ in range(int(rng.integers(0, 3))):  # holidays / promotions
        s = int(rng.integers(0, n))
        y[s:s + int(day)] *= rng.uniform(1.5, 4.0)
    return y


# ===========================================================================
# Stochastic processes
# ===========================================================================


@register("arma", "stochastic", weight=1.5)
def g_arma(rng: Gen, n: int) -> Array:
    p = int(rng.integers(1, 5))
    q = int(rng.integers(0, 4))
    a = _stable_poly(rng, p)
    b = _stable_poly(rng, q, rmax=0.9) if q else np.array([1.0])
    return _filtered_noise(rng, n, b, a)


@register("seasonal_arma", "stochastic")
def g_sarma(rng: Gen, n: int) -> Array:
    """AR at a seasonal lag: noisy but genuinely periodic autocorrelation."""
    s = int(_period(rng, n, min_cycles=3.0, pmin=4))
    phi_s = rng.uniform(0.5, 0.95)
    a_s = np.zeros(s + 1)
    a_s[0], a_s[s] = 1.0, -phi_s
    a = np.convolve(a_s, _stable_poly(rng, int(rng.integers(1, 3)), rmax=0.8))
    return _filtered_noise(rng, n, np.array([1.0]), a, burn=4 * s + 256)


@register("colored_noise", "stochastic", weight=1.5)
def g_colored(rng: Gen, n: int) -> Array:
    """1/f^beta noise: brown (random walk), pink (long memory), white, blue."""
    return _colored_noise(rng, n, beta=float(rng.uniform(-1.0, 3.0)))


@register("random_walk", "stochastic", weight=1.5)
def g_rw(rng: Gen, n: int) -> Array:
    steps = rng.standard_normal(n)
    if rng.random() < 0.4:  # heavy-tailed jumps
        jump = (rng.random(n) < rng.uniform(0.01, 0.08)) * rng.standard_normal(n) * 8
        steps = steps + jump
    drift = rng.normal(0.0, 0.3)
    return np.cumsum(steps + drift)


@register("ou_process", "stochastic")
def g_ou(rng: Gen, n: int) -> Array:
    """Mean reversion at a controlled timescale (spreads, temperature anomaly)."""
    theta = np.exp(-1.0 / rng.uniform(2.0, n / 4.0))
    return _filtered_noise(rng, n, np.array([1.0]), np.array([1.0, -theta]))


@register("garch", "stochastic")
def g_garch(rng: Gen, n: int) -> Array:
    """Volatility clustering: quiet stretches punctuated by turbulent ones.
    Completely absent from the original bank and ubiquitous in real data."""
    alpha = rng.uniform(0.03, 0.2)
    beta = rng.uniform(0.6, 0.95 - alpha)
    omega = 1.0 - alpha - beta
    eps = rng.standard_normal(n)
    s2, y = omega, np.zeros(n)
    for i in range(n):
        e = eps[i] * math.sqrt(s2)
        y[i] = e
        s2 = omega + alpha * e * e + beta * s2
    return np.cumsum(y) if rng.random() < 0.5 else y


@register("regime_switching", "stochastic", weight=1.2)
def g_regime(rng: Gen, n: int) -> Array:
    """Markov switching between two dynamics: level, volatility and
    persistence all change at unannounced times."""
    stay = rng.uniform(0.97, 0.999)
    mus = rng.normal(0, 2, 2)
    sds = np.exp(rng.normal(0, 0.8, 2))
    phis = rng.uniform(-0.9, 0.95, 2)
    u, eps = rng.random(n), rng.standard_normal(n)
    s, x, y = 0, 0.0, np.zeros(n)
    for i in range(n):
        if u[i] > stay:
            s = 1 - s
        x = phis[s] * x + eps[i] * sds[s]
        y[i] = mus[s] + x
    return y


@register("threshold_ar", "stochastic")
def g_tar(rng: Gen, n: int) -> Array:
    """SETAR: different dynamics above and below a threshold. Produces
    asymmetric cycles (slow build-up, fast collapse) that linear models miss."""
    c = rng.normal(0, 0.5)
    a_lo, a_hi = rng.uniform(0.2, 1.02), rng.uniform(-0.5, 0.9)
    s_lo, s_hi = rng.uniform(0.3, 1.5), rng.uniform(0.3, 1.5)
    eps = rng.standard_normal(n)
    x, y = 0.0, np.zeros(n)
    for i in range(n):
        x = (a_lo * x + eps[i] * s_lo) if x < c else (a_hi * x + eps[i] * s_hi)
        if x > 1e6 or x < -1e6:
            x = math.copysign(1e6, x)
        y[i] = x
    return y


@register("local_linear_trend", "stochastic", weight=1.5)
def g_llt(rng: Gen, n: int) -> Array:
    """Structural / ETS model: stochastic level + stochastic slope (+ optional
    stochastic seasonal). The generative story behind most business series."""
    sl = rng.uniform(0.0, 0.5)
    ss = rng.uniform(0.0, 0.05)
    slope = np.cumsum(rng.standard_normal(n) * ss) + rng.normal(0, 0.2)
    level = np.cumsum(slope + rng.standard_normal(n) * sl)
    y = level
    if rng.random() < 0.6:
        P = int(_period(rng, n, min_cycles=2.5, pmin=4))
        seas = rng.standard_normal(P) * rng.uniform(0.5, 3.0) * (1 + level.std())
        drift = np.cumsum(rng.standard_normal((-(-n // P), P)) *
                          rng.uniform(0.0, 0.2), axis=0)
        y = y + (seas + drift).reshape(-1)[:n]
    return y + rng.standard_normal(n) * rng.uniform(0.0, 0.5) * (1 + level.std())


@register("mean_reverting_seasonal", "stochastic")
def g_mrs(rng: Gen, n: int) -> Array:
    """Deterministic cycle plus persistent coloured residual: the realistic
    'seasonality is right, the detail is not' regime."""
    t = _t(n)
    P = _period(rng, n, min_cycles=2.5, pmin=6)
    sig = np.sin(2 * np.pi * t / P + _phase(rng))
    if rng.random() < 0.5:
        sig = sig + 0.4 * np.sin(4 * np.pi * t / P + _phase(rng))
    sig = sig + rng.normal(0, 0.5) * t / n
    resid = _colored_noise(rng, n, beta=rng.uniform(0.5, 2.0))
    return sig + rng.uniform(0.05, 0.8) * resid


# ===========================================================================
# Counts, events, sparse
# ===========================================================================


@register("intermittent_demand", "events", weight=1.2)
def g_intermittent(rng: Gen, n: int) -> Array:
    """Mostly zeros with occasional lumps (Croston regime). Extremely common
    in retail/spare-parts data and impossible for a model that has never
    seen a zero floor."""
    p = rng.uniform(0.03, 0.35)
    if rng.random() < 0.4:  # demand probability itself is seasonal
        P = _period(rng, n, min_cycles=2.5, pmin=6)
        p = np.clip(p * (1 + 0.9 * np.sin(2 * np.pi * _t(n) / P + _phase(rng))), 0, 1)
    occ = rng.random(n) < p
    size = np.exp(rng.normal(0.0, rng.uniform(0.3, 1.2), n))
    y = occ * size
    return np.round(y * rng.uniform(1, 20)) if rng.random() < 0.5 else y


@register("hawkes_counts", "events")
def g_hawkes(rng: Gen, n: int) -> Array:
    """Self-exciting events: bursts beget bursts (outages, clicks, quakes)."""
    decay = rng.uniform(0.5, 0.95)
    branch = rng.uniform(0.2, 0.85)
    alpha = branch * (1 - decay)
    mu = rng.uniform(0.05, 2.0)
    lam, y = mu, np.zeros(n)
    for i in range(n):
        k = rng.poisson(min(lam, 1e4))
        y[i] = k
        lam = mu + decay * (lam - mu) + alpha * k
    return y


@register("queue_length", "events")
def g_queue(rng: Gen, n: int) -> Array:
    """Reflected birth-death process: non-negative, integer, bursty."""
    lam = rng.uniform(0.5, 4.0)
    mu = lam * rng.uniform(1.0, 1.6)
    arr, srv = rng.poisson(lam, n), rng.poisson(mu, n)
    x, y = 0.0, np.zeros(n)
    for i in range(n):
        x = max(0.0, x + arr[i] - srv[i])
        y[i] = x
    return y


@register("renewal_pulses", "events")
def g_renewal(rng: Gen, n: int) -> Array:
    """Pulses at random inter-arrival times with a shaped response."""
    shape = rng.uniform(1.0, 20.0)  # high shape -> nearly periodic
    mean_gap = _period(rng, n, min_cycles=2.0, pmin=5)
    gaps = rng.gamma(shape, mean_gap / shape, size=int(n / mean_gap) + 4)
    idx = np.cumsum(gaps).astype(int)
    idx = idx[idx < n]
    imp = np.zeros(n)
    if idx.size:
        imp[idx] = rng.uniform(0.5, 1.5, idx.size)
    w = max(int(mean_gap * rng.uniform(0.05, 0.3)), 1)
    k = np.exp(-_t(6 * w) / w)
    return np.convolve(imp, k)[:n]


@register("counter_with_resets", "events")
def g_counter(rng: Gen, n: int) -> Array:
    """Monotone accumulation reset periodically (meters, cumulative counters).
    Predictable slope, predictable reset, but a discontinuous jump."""
    P = _period(rng, n, min_cycles=2.0, pmin=8)
    rate = np.abs(rng.standard_normal(n)) + rng.uniform(0.1, 1.0)
    if rng.random() < 0.5:
        rate *= 1 + 0.6 * np.sin(2 * np.pi * _t(n) / (P / rng.uniform(1, 4)))
    y = np.zeros(n)
    acc, nxt = 0.0, rng.uniform(0, P)
    for i in range(n):
        if i >= nxt:
            acc, nxt = 0.0, i + P * rng.uniform(0.9, 1.1)
        acc += rate[i]
        y[i] = acc
    return y


@register("on_off_schedule", "events")
def g_schedule(rng: Gen, n: int) -> Array:
    """Thermostat / lighting: a duty cycle whose switch times jitter."""
    P = _period(rng, n, min_cycles=2.5, pmin=8)
    duty = rng.uniform(0.15, 0.7)
    y = np.zeros(n)
    i = rng.uniform(0, P)
    while i < n:
        on = int(i + P * duty * rng.uniform(0.8, 1.2))
        y[int(i):max(on, int(i) + 1)] = 1.0
        i += P * rng.uniform(0.9, 1.1)
    tau = rng.uniform(0.5, 8.0)  # thermal lag
    a = float(np.exp(-1.0 / tau))
    return np.asarray(lfilter([1 - a], [1.0, -a], y))


# ===========================================================================
# Chaotic / nonlinear dynamics (deterministic but hard: good curriculum)
# ===========================================================================


@register("logistic_map", "chaos", weight=0.6)
def g_logistic_map(rng: Gen, n: int) -> Array:
    r = float(rng.uniform(3.4, 4.0))
    x = float(rng.uniform(0.1, 0.9))
    for _ in range(200):
        x = r * x * (1 - x)
    y = np.empty(n)
    for i in range(n):
        x = r * x * (1 - x)
        y[i] = x
    return y


@register("mackey_glass", "chaos", weight=0.6)
def g_mackey_glass(rng: Gen, n: int) -> Array:
    tau = int(rng.integers(15, 30))
    sub = int(rng.integers(1, 5))
    total = n * sub + 1000
    x = np.empty(total + tau)
    x[:tau] = 0.5 + rng.standard_normal(tau) * 0.01
    beta, gamma, p = 0.2, 0.1, 10
    for i in range(tau, total + tau):
        xd = x[i - tau]
        x[i] = x[i - 1] + beta * xd / (1 + xd ** p) - gamma * x[i - 1]
    return x[tau + 1000::sub][:n]


@register("lorenz", "chaos", weight=0.5)
def g_lorenz(rng: Gen, n: int) -> Array:
    dt = 0.01
    sub = int(rng.integers(4, 14))
    sg, r, b = 10.0, float(rng.uniform(24.0, 30.0)), 8.0 / 3.0
    x, y, z = [float(v) for v in rng.uniform(-10, 10, 3)]
    burn, comp = 1000, int(rng.integers(0, 3))
    out = np.empty(n)
    for i in range(burn + n * sub):
        x, y, z = (x + dt * sg * (y - x),
                   y + dt * (x * (r - z) - y),
                   z + dt * (x * y - b * z))
        j = i - burn
        if j >= 0 and j % sub == 0 and j // sub < n:
            out[j // sub] = x if comp == 0 else (y if comp == 1 else z)
    return out


@register("van_der_pol", "chaos", weight=0.5)
def g_vdp(rng: Gen, n: int) -> Array:
    """Relaxation oscillator: periodic but sharply non-sinusoidal."""
    mu = rng.uniform(0.5, 6.0)
    dt = 0.01
    sub = int(rng.integers(5, 20))
    x, v = float(rng.uniform(-2, 2)), float(rng.uniform(-2, 2))
    burn = 2000
    out = np.empty(n)
    for i in range(burn + n * sub):
        v += dt * (mu * (1 - x * x) * v - x)
        x += dt * v
        j = i - burn
        if j >= 0 and j % sub == 0 and j // sub < n:
            out[j // sub] = x
    return out


# ===========================================================================
# Composites
# ===========================================================================


@register("trend_seasonal_noise", "composite", weight=2.0)
def g_tsn(rng: Gen, n: int) -> Array:
    """The classic additive decomposition, with a multiplicative variant."""
    t = _t(n)
    trend = rng.normal(0, 1) * t / n
    if rng.random() < 0.3:
        trend = trend + rng.normal(0, 1) * (t / n) ** 2
    seas = np.zeros(n)
    for _ in range(int(rng.integers(1, 4))):
        P = _period(rng, n, min_cycles=2.0, pmin=4)
        seas += rng.uniform(0.2, 1.0) * np.sin(2 * np.pi * t / P + _phase(rng))
    noise = _colored_noise(rng, n, beta=rng.uniform(0.0, 2.0)) * rng.uniform(0.02, 0.4)
    if rng.random() < 0.35:  # multiplicative: amplitude scales with level
        level = trend - trend.min() + rng.uniform(0.3, 2.0) * (1 + abs(trend).max())
        return level * (1 + rng.uniform(0.1, 0.5) * seas + noise)
    return trend * (1 + abs(seas).max()) + seas + noise


@register("hierarchical_sum", "composite")
def g_hierarchical(rng: Gen, n: int) -> Array:
    """Aggregate of several independent sub-series, as in a total across
    stores or sensors. Aggregation smooths noise and superposes periods."""
    names = [k for k, g in FAMILY_GROUP.items()
             if g in ("seasonal", "trend", "stochastic", "events")]
    out = np.zeros(n)
    for _ in range(int(rng.integers(2, 6))):
        f = str(rng.choice(names))
        out += _z(np.nan_to_num(GENERATORS[f](rng, n))) * rng.uniform(0.3, 1.0)
    return out


@register("segment_mixture", "composite")
def g_segments(rng: Gen, n: int) -> Array:
    """Alternating waveform regimes. Tiled (so the motif cycle is inferable)
    most of the time; free-running occasionally for robustness."""
    shapes = list(rng.choice(["square", "triangle", "sine", "sawtooth", "flat"],
                             size=int(rng.integers(2, 5)), replace=False))
    P = _period(rng, n, min_cycles=6.0, pmin=4)
    amp_shared = rng.random() < 0.5

    def piece(shape: str, m: int) -> Array:
        a = 1.0 if amp_shared else float(np.exp(rng.normal(0, 0.6)))
        if shape == "flat":
            return np.full(m, rng.uniform(-a, a))
        return a * _wave(_t(m), P, shape, duty=float(rng.uniform(0.2, 0.8)))

    if rng.random() < 0.6:  # tiled -> transitions are periodic
        seg = max(int(n / (2 * len(shapes))), 4)
        block = np.concatenate([piece(s, int(rng.integers(seg // 2 + 1, seg + 1)))
                                for s in shapes])
        return np.tile(block, -(-n // block.size))[:n]
    pieces, tot, i = [], 0, 0
    while tot < n:
        m = int(rng.integers(max(n // 10, 6), max(n // 3, 12)))
        pieces.append(piece(shapes[i % len(shapes)], m))
        tot += m
        i += 1
    return np.concatenate(pieces)[:n]


# ===========================================================================
# Augmentation / corruption pipeline
# ===========================================================================


@dataclass
class AugmentConfig:
    """Probabilities for each corruption. These are what turn ~45 families into
    a genuinely broad distribution, and they are what real data looks like."""

    p_noise: float = 0.85
    snr_db: Tuple[float, float] = (5.0, 45.0)
    p_colored_noise: float = 0.4

    p_mix: float = 0.15          # TSMixup across the whole bank
    p_flip_sign: float = 0.5
    p_reverse: float = 0.10
    p_warp: float = 0.08         # nonlinear time warp

    p_outliers: float = 0.10     # additive spikes
    p_dropout: float = 0.05      # sensor drops to zero
    p_freeze: float = 0.05       # stuck sensor repeats a value
    p_level_shift: float = 0.06
    p_clip: float = 0.05         # sensor saturation
    p_quantize: float = 0.06
    p_positive: float = 0.20     # push to a positive, skewed, zero-floored scale


def _apply_noise(rng: Gen, x: Array, cfg: AugmentConfig) -> Array:
    if rng.random() >= cfg.p_noise:
        return x
    snr = rng.uniform(*cfg.snr_db)
    sigma = x.std() * 10 ** (-snr / 20.0)
    beta = rng.uniform(0.0, 2.0) if rng.random() < cfg.p_colored_noise else 0.0
    e = _colored_noise(rng, x.size, beta) if beta > 0 else rng.standard_normal(x.size)
    return x + sigma * e


def augment(rng: Gen, x: Array, cfg: AugmentConfig,
            draw: Optional[Callable[[Gen, int], Array]] = None) -> Array:
    n = x.size
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    if draw is not None and rng.random() < cfg.p_mix:
        k = int(rng.integers(2, 4))
        w = rng.dirichlet([1.5] * k)
        x = w[0] * _z(x) + sum(w[i] * _z(np.nan_to_num(draw(rng, n)))
                               for i in range(1, k))

    if rng.random() < cfg.p_reverse:
        x = x[::-1].copy()
    if rng.random() < cfg.p_flip_sign:
        x = -x
    if rng.random() < cfg.p_warp:
        x = _warp_time(rng, x, strength=rng.uniform(0.1, 0.4))

    x = _apply_noise(rng, x, cfg)
    s = x.std() + 1e-12

    if rng.random() < cfg.p_outliers:
        m = rng.random(n) < rng.uniform(0.002, 0.03)
        x = x + m * rng.standard_normal(n) * s * rng.uniform(3.0, 12.0)
    if rng.random() < cfg.p_level_shift:
        for _ in range(int(rng.integers(1, 3))):
            cp = int(rng.integers(n // 8, n))
            x = x + np.where(_t(n) >= cp, rng.normal(0, 2.0) * s, 0.0)
    if rng.random() < cfg.p_freeze:
        st = int(rng.integers(0, max(n - 4, 1)))
        ln = int(rng.integers(3, max(n // 6, 4)))
        x[st:st + ln] = x[st]
    if rng.random() < cfg.p_dropout:
        st = int(rng.integers(0, max(n - 2, 1)))
        ln = int(rng.integers(1, max(n // 8, 2)))
        x[st:st + ln] = 0.0
    if rng.random() < cfg.p_clip:
        lo, hi = np.quantile(x, [rng.uniform(0.0, 0.1), rng.uniform(0.9, 1.0)])
        x = np.clip(x, lo, hi)
    if rng.random() < cfg.p_quantize:
        levels = int(rng.integers(3, 40))
        lo, hi = x.min(), x.max()
        if hi > lo:
            x = np.round((x - lo) / (hi - lo) * levels) / levels * (hi - lo) + lo
    if rng.random() < cfg.p_positive:
        # Strictly positive, right-skewed, hard zero floor: the shape of most
        # demand/traffic/energy series. Nothing in the original bank had this.
        z = _z(x)
        if rng.random() < 0.5:
            x = np.exp(np.clip(z * rng.uniform(0.2, 0.7), -6.0, 6.0))
        else:
            x = np.clip(z * rng.uniform(0.5, 2.0) + rng.uniform(-0.5, 2.0), 0, None)
    return x


# ===========================================================================
# Validity filter and normalization
# ===========================================================================


def is_valid(x: Array, min_rel_std: float = 1e-6) -> bool:
    """Reject the degenerate windows the original silently emits as all-zeros:
    constant segments (period longer than the window), overflow, NaN."""
    if x.size == 0 or not np.all(np.isfinite(x)):
        return False
    scale = float(np.max(np.abs(x)))
    if scale > 1e15 or scale < 1e-30:
        return False
    if float(x.std()) < min_rel_std * max(scale, 1e-12):
        return False
    if np.unique(x).size < 2:  # constant window (e.g. period > seq_len)
        return False
    return True


def normalize(x: Array) -> Array:
    loc, scale = x.mean(), x.std()
    return (x - loc) / (scale + 1e-8)


# ===========================================================================
# Datasets
# ===========================================================================


def default_weights() -> Dict[str, float]:
    return dict(FAMILY_WEIGHT)


class SignalDataset(Dataset):
    """Map-style dataset of synthetic windows, generated on demand.

    ``size`` is a *virtual* length: nothing is stored. Each index maps to a
    deterministic RNG seed, so

      * ``ds[i]`` is reproducible and identical across workers, machines and
        restarts (important for resuming, and for DistributedSampler);
      * every worker derives its randomness from the index it was handed, so
        ``num_workers > 1`` does not duplicate data (the usual failure mode of
        on-the-fly map datasets that seed from module state);
      * shuffling, samplers, ``len()`` and epoch bookkeeping all behave
        normally.

    Call :meth:`set_epoch` to draw a fresh set of signals each epoch, or leave
    it alone to treat the dataset as a fixed corpus of ``size`` windows.
    """

    def __init__(
        self,
        seq_len: int = 512,
        size: int = 1_000_000,
        return_label: bool = False,
        seed: int = 0,
        epoch: int = 0,
        max_abs: float = 200.0,
        max_retries: int = 25,
    ):
        self.seq_len = seq_len
        self.size = int(size)
        self.cfg = AugmentConfig()
        self.return_label = return_label
        self.seed = seed
        self.epoch = epoch
        self.max_abs = max_abs
        self.max_retries = max_retries

        w = default_weights()
        self.names = sorted(w)
        p = np.array([w[k] for k in self.names], dtype=np.float64)
        self.probs = p / p.sum()

    # -- epoch control ----------------------------------------------------
    def set_epoch(self, epoch: int) -> None:
        """Re-roll the whole dataset. Call once per epoch, before the loader
        is iterated, if you want fresh signals rather than a fixed corpus."""
        self.epoch = int(epoch)

    # -- sampling ---------------------------------------------------------
    def _draw_raw(self, rng: Gen, n: int) -> Array:
        name = self.names[int(rng.choice(len(self.names), p=self.probs))]
        return GENERATORS[name](rng, n)

    def sample(self, rng: Gen) -> Tuple[Array, str]:
        """Draw one normalized window plus its family name."""
        n = self.seq_len
        for _ in range(self.max_retries):
            name = self.names[int(rng.choice(len(self.names), p=self.probs))]
            try:
                x = np.asarray(GENERATORS[name](rng, n), dtype=np.float64)
                if x.size != n:
                    continue
                x = augment(rng, x, self.cfg, draw=self._draw_raw)
                if not is_valid(x):
                    continue
                x = normalize(x)
                if not np.all(np.isfinite(x)):
                    continue
                if self.max_abs and np.abs(x).max() > self.max_abs:
                    continue  # runaway scale (usually a steep exponential)
                return x, name
            except (FloatingPointError, ValueError, ZeroDivisionError):
                continue
        return normalize(_colored_noise(rng, n, 1.0)), "fallback"

    def rng_for(self, idx: int) -> Gen:
        return np.random.default_rng([self.seed, self.epoch, int(idx)])

    # -- Dataset protocol -------------------------------------------------
    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int):
        if idx < 0:
            idx += self.size
        if not 0 <= idx < self.size:
            raise IndexError(idx)
        x, name = self.sample(self.rng_for(idx))
        out = torch.from_numpy(x).float() if _HAS_TORCH else x.astype(np.float32)
        return (out, name) if self.return_label else out

    # -- optional freezing ------------------------------------------------
    def materialize(self, n: Optional[int] = None, verbose: bool = False):
        """Generate ``n`` windows once and return them as a plain tensor
        (or a ``TensorDataset``). Use for eval sets or if you would rather pay
        the generation cost up front than in the dataloader."""
        n = self.size if n is None else int(n)
        xs = np.empty((n, self.seq_len), dtype=np.float32)
        labels: List[str] = []
        for i in range(n):
            x, name = self.sample(self.rng_for(i))
            xs[i] = x
            labels.append(name)
            if verbose and i % 10000 == 0:
                print(f"  {i}/{n}", flush=True)
        if _HAS_TORCH:
            return torch.from_numpy(xs), labels
        return xs, labels