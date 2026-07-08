import os
import random

import numpy as np
import torch
from scipy.signal import lfilter
from torch.utils.data import Dataset


class SyntheticTimeSeriesDataset(Dataset):
    def __init__(self, seq_len=96, noise=True, n_samples=20):
        self.seq_len = seq_len
        self.samples = []
        self.noise = noise
        self.n_samples = n_samples

        # set random seed for reproducibility
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)

        # --- Utility functions ---
        def add(label, ctx):
            self.samples.append((ctx, label))

        def get_xs(seq_len, start=None):
            if start is None:
                start = random.uniform(0, 1000)
            xx = torch.linspace(start, start + seq_len, seq_len)
            return xx

        # === Patterns ===

        for _ in range(n_samples):
            abscisse = 0
            slope = random.sample(
                [
                    -100,
                    -50,
                    -10,
                    -5,
                    -3,
                    -1,
                    -0.1,
                    -0.05,
                    -0.01,
                    0.01,
                    0.05,
                    0.1,
                    1,
                    3,
                    5,
                    10,
                    50,
                    100,
                ],
                1,
            )[0]
            x_ctx = get_xs(self.seq_len)
            ctx = abscisse + slope * x_ctx
            add("linear", ctx)

        for _ in range(n_samples):
            abscisse = 0
            degree = random.choice([2, 3, 4, 5])
            coeffs = [random.uniform(-1, 1) for _ in range(degree)]
            x_ctx = get_xs(self.seq_len)
            ctx = abscisse + sum(c * x_ctx**i for i, c in enumerate(coeffs, start=1))
            add("multipolynomial", ctx)

        for _ in range(n_samples):
            x_ctx = get_xs(self.seq_len)
            alpha = random.uniform(-10, 10)
            delta = random.uniform(-100, 100)
            sin_scale = random.uniform(1000, 1000000)
            sin_freq = random.uniform(0.01, 0.1)
            factor = random.randint(1, 100)
            ctx = (
                alpha * x_ctx**2 + delta + sin_scale * torch.sin(sin_freq * x_ctx)
            ) / factor
            add("poly_sin", ctx)

        for _ in range(n_samples):
            abscisse = 0
            slope = random.sample(
                [
                    -100,
                    -50,
                    -10,
                    -5,
                    -3,
                    -1,
                    -0.1,
                    -0.05,
                    -0.01,
                    0.01,
                    0.05,
                    0.1,
                    1,
                    3,
                    5,
                    10,
                    50,
                    100,
                ],
                1,
            )[0]
            amp = random.sample([1, 2, 4, 8, 16, 32, 64, 128], 1)[0]
            x_ctx = get_xs(self.seq_len)
            # vary the oscillation frequency (was fixed at 0.3) for diversity
            freq = random.uniform(0.1, 0.5)
            ctx = abscisse + amp * torch.sin(x_ctx * freq) + slope * x_ctx
            add("linear_sin", ctx)

        for _ in range(n_samples):
            abscisse = 0
            x_ctx = get_xs(self.seq_len)
            freq = random.uniform(0.01, 0.1)
            amp = random.uniform(1, 100)
            ctx = abscisse + amp * torch.sin(x_ctx * freq)
            add("vary_sin", ctx)

        for _ in range(n_samples):
            # longer periods only when the context can still show >=4 cycles
            period = random.choice(
                [8, 16, 32, 64]
                + [p for p in [128, 256] if p <= self.seq_len // 4]
            )
            step_height = random.choice([5, 10, 20, 50, 100, 1000])
            abscisse = 0
            x_ctx = get_xs(self.seq_len)
            ctx = abscisse + step_height * torch.floor(x_ctx / period)
            add("step", ctx)

        for _ in range(n_samples):
            abscisse = 0
            amp = random.sample([10, 20, 50, 100, 200], 1)[0]
            every = random.sample([20, 50, 100, 200, 400], 1)[0]
            x_ctx = get_xs(self.seq_len)
            burst_width = random.randint(2, 5)
            full_len = len(x_ctx)
            full = abscisse + torch.zeros(full_len)
            for i in range(0, full_len - burst_width, every):
                full[i : i + burst_width] += amp
            ctx = full[: len(x_ctx)]
            add("burst_repeat", ctx)

        for _ in range(n_samples):
            abscisse = 0
            freq = random.sample(
                [
                    1 / 500,
                    1 / 100,
                    1 / 75,
                    1 / 50,
                    1 / 25,
                    1 / 10,
                    1 / 5,
                    1,
                    5,
                    10,
                    25,
                    50,
                    75,
                    100,
                    500,
                ],
                2,
            )
            amp = random.sample(
                [
                    1000,
                    500,
                    100,
                    50,
                    20,
                    10,
                    5,
                    1,
                    0.1,
                    0.05,
                    0.01,
                    0.005,
                    0.001,
                    -0.1,
                    -0.5,
                    -1,
                    -5,
                    -10,
                    -20,
                    -50,
                    -100,
                    -500,
                    -1000,
                ],
                2,
            )
            freq1, freq2 = freq[0], freq[1]
            amp1, amp2 = amp[0], amp[1]
            x_ctx = get_xs(self.seq_len)
            ctx = (
                abscisse
                + amp1 * torch.sin(x_ctx * freq1)
                + amp2 * torch.sin(x_ctx * freq2)
            )
            add("complex_sin", ctx)

        for _ in range(n_samples):
            abscisse = 0
            freq1 = random.choice(
                [1 / 50, 1 / 10, 1 / 5, 1 / 4, 1 / 3, 1 / 2, 1, 2, 3, 4, 5, 10, 50]
            )
            freq2 = random.choice(
                [1 / 50, 1 / 10, 1 / 5, 1 / 4, 1 / 3, 1 / 2, 1, 2, 3, 4, 5, 10, 50]
            )
            x_ctx = get_xs(self.seq_len)
            ctx = abscisse + torch.sin(freq1 * x_ctx) + torch.cos(freq2 * x_ctx)
            add("sin_cos", ctx)

        for _ in range(n_samples):
            abscisse = 0
            x_ctx = get_xs(self.seq_len)
            xx = x_ctx
            freq = random.uniform(50, 500)
            delay = random.uniform(-100, 100)
            slope = random.uniform(-20, 20)
            yy = abscisse + slope * torch.abs(torch.remainder(xx, freq) - delay)
            ctx = yy[: self.seq_len]
            add("sawtooth", ctx)

        for _ in range(n_samples):
            x = get_xs(self.seq_len)

            # generate a sawtooth wave with flat zone between each peak
            y = torch.zeros_like(x)
            abscisse = 0
            space = torch.randint(10, 200, (1,)).item()
            height = torch.randint(1, 10, (1,)).item()
            for i in range(len(x)):
                if i % (space * 2) < space:
                    y[i] = height
                else:
                    y[i] = -height
            y = y + abscisse
            ctx = y[: self.seq_len]
            add("sawtooth_flat", ctx)

        for _ in range(n_samples):
            abscisse = 0
            x_ctx = get_xs(self.seq_len)
            xx = x_ctx
            freq = random.uniform(50, 500)
            slope = random.uniform(-20, 20)
            flat_ratio = random.uniform(0.2, 0.8)
            yy = abscisse + triangle_with_flat(
                xx, freq=freq, slope=slope, flat_ratio=flat_ratio
            )
            ctx = yy[: self.seq_len]
            add("triangle_with_flat", ctx)

        for _ in range(n_samples):
            abscisse = 0
            x_ctx = get_xs(self.seq_len)
            xx = x_ctx
            freq = random.uniform(50, 500)
            amp = random.uniform(1, 100)
            yy = abscisse + rectangular_wave(xx, freq=freq, amp=amp)
            ctx = yy[: self.seq_len]
            add("rectangular", ctx)

        for _ in range(n_samples):
            abscisse = 0
            x_ctx = get_xs(self.seq_len)
            xx = x_ctx
            freq = random.uniform(30, 300)
            amp = random.uniform(1, 100)
            slope = random.uniform(-20, 20)
            yy = abscisse + rectangular_triangle_sequence(
                xx, freq=freq, amp=amp, slope=slope
            )
            ctx = yy[: self.seq_len]
            add("rectangular_triangle_sequence", ctx)

        for _ in range(n_samples):
            abscisse = 0
            x_ctx = get_xs(self.seq_len)
            xx = x_ctx
            freq = random.uniform(30, 300)
            amp = random.uniform(1, 100)
            slope = random.uniform(-20, 20)
            yy = abscisse + triangle_rectangular_sequence(
                xx, freq=freq, amp=amp, slope=slope
            )
            ctx = yy[: self.seq_len]
            add("triangle_rectangular_sequence", ctx)

        for _ in range(n_samples):
            abscisse = 0
            x_ctx = get_xs(self.seq_len)
            xx = x_ctx
            segment_freq = random.uniform(20, 150)
            amp = random.uniform(1, 100)
            slope = random.uniform(-20, 20)
            yy = abscisse + multi_shape_sequence(
                xx, segment_freq=segment_freq, amp=amp, slope=slope
            )
            ctx = yy[: self.seq_len]
            add("multi_shape_sequence", ctx)

        for _ in range(n_samples):
            t = torch.linspace(0, 1, self.seq_len)
            sign = random.choice([-1, 1])
            rate = random.uniform(1, 6)
            scale = random.uniform(1, 100)
            ctx = sign * scale * torch.exp(rate * t)
            add("exponential_trend", ctx)

        for _ in range(n_samples):
            t = torch.linspace(0, 1, self.seq_len)
            amp = random.uniform(1, 100)
            decay = random.uniform(1, 8)
            freq = random.uniform(10, 100)
            ctx = amp * torch.exp(-decay * t) * torch.sin(freq * t)
            add("damped_sin", ctx)

        for _ in range(n_samples):
            t = torch.linspace(0, 1, self.seq_len)
            amp = random.uniform(1, 100)
            # frequencies are in cycles per window: keep the sweep below
            # Nyquist (seq_len / 2) or the signal aliases into pseudo-noise
            nyquist = self.seq_len / 2
            f0 = random.uniform(2, 0.2 * nyquist)
            f1 = random.uniform(f0 + 2, 0.8 * nyquist)
            ctx = amp * torch.sin(2 * torch.pi * (f0 + (f1 - f0) * t / 2) * t)
            add("chirp", ctx)

        for _ in range(n_samples):
            t = torch.linspace(0, 1, self.seq_len)
            # carrier + modulation sidebands must stay below Nyquist
            nyquist = self.seq_len / 2
            carrier_freq = random.uniform(10, 0.7 * nyquist)
            mod_freq = random.uniform(1, min(10.0, carrier_freq / 3))
            amp = random.uniform(1, 100)
            mod_depth = random.uniform(0.2, 1.0)
            envelope = 1 + mod_depth * torch.sin(2 * torch.pi * mod_freq * t)
            ctx = amp * envelope * torch.sin(2 * torch.pi * carrier_freq * t)
            add("amplitude_modulated_sin", ctx)

        for _ in range(n_samples):
            t = torch.linspace(0, 1, self.seq_len)
            amp = random.choice([-100, -50, -10, -5, -1, 1, 5, 10, 50, 100])
            steepness = random.uniform(5, 50)
            center = random.uniform(0.2, 0.8)
            ctx = amp / (1 + torch.exp(-steepness * (t - center)))
            add("sigmoid", ctx)

        for _ in range(n_samples):
            scale = random.uniform(0.1, 10)
            steps = torch.randn(self.seq_len) * scale
            drift = random.uniform(-1, 1) * scale
            ctx = torch.cumsum(steps + drift, dim=0)
            add("random_walk", ctx)

        for _ in range(n_samples):
            phi = random.uniform(-0.99, 0.99)
            scale = random.uniform(0.1, 10)
            eps = np.random.randn(self.seq_len) * scale
            ctx = torch.from_numpy(
                lfilter([1.0], [1.0, -phi], eps).astype(np.float32)
            )
            add("ar1", ctx)

        # Event timing in these next families is not learnable when it falls
        # in the forecast horizon, so keep them at half share.
        for _ in range(max(1, n_samples // 2)):
            t = torch.arange(self.seq_len, dtype=torch.float32)
            changepoint = random.randint(self.seq_len // 4, 3 * self.seq_len // 4)
            slope1 = random.uniform(-5, 5)
            slope2 = random.uniform(-5, 5)
            ctx = torch.where(
                t < changepoint,
                slope1 * t,
                slope1 * changepoint + slope2 * (t - changepoint),
            )
            add("trend_change", ctx)

        for _ in range(max(1, n_samples // 2)):
            t = torch.arange(self.seq_len, dtype=torch.float32)
            changepoint = random.randint(self.seq_len // 4, 3 * self.seq_len // 4)
            shift = random.choice([-100, -50, -10, 10, 50, 100])
            freq = random.uniform(0.05, 0.5)
            amp = random.uniform(1, 20)
            base = amp * torch.sin(freq * t)
            ctx = base + torch.where(t < changepoint, 0.0, float(shift))
            add("level_shift", ctx)

        for _ in range(max(1, n_samples // 2)):
            t = torch.arange(self.seq_len, dtype=torch.float32)
            n_bumps = random.randint(1, 4)
            ctx = torch.zeros(self.seq_len)
            for _ in range(n_bumps):
                center = random.uniform(0, self.seq_len)
                width = random.uniform(self.seq_len / 50, self.seq_len / 8)
                height = random.uniform(-100, 100)
                ctx += height * torch.exp(-((t - center) ** 2) / (2 * width**2))
            add("gaussian_bumps", ctx)

        for _ in range(n_samples):
            t = torch.arange(self.seq_len, dtype=torch.float32)
            period = random.choice(
                [8, 16, 32, 64, 128]
                + [p for p in [256, 512] if p <= self.seq_len // 4]
            )
            duty = random.uniform(0.1, 0.9)
            amp = random.uniform(1, 100)
            mod = torch.remainder(t, period)
            ctx = torch.where(mod < duty * period, amp, -amp)
            add("duty_cycle_square", ctx)

        for _ in range(n_samples):
            t = torch.arange(self.seq_len, dtype=torch.float32)
            slope = random.uniform(-2, 2)
            period = random.choice(
                [12, 24, 48, 96]
                + [p for p in [192, 384] if p <= self.seq_len // 4]
            )
            amp = random.uniform(1, 50)
            growth = random.uniform(0.0, 2.0)
            envelope = 1 + growth * t / self.seq_len
            ctx = slope * t + amp * envelope * torch.sin(2 * torch.pi * t / period)
            add("trend_seasonal_growing", ctx)

        # Tile half of the segment mixtures so the motif cycle repeats and
        # the transitions are learnable; keep the rest free for robustness.
        for _ in range(n_samples):
            ctx = alternating_shape_segments(
                self.seq_len,
                shapes=["square", "triangle"],
                tile=random.random() < 0.5,
            )
            add("square_triangle_segments", ctx)

        for _ in range(n_samples):
            bank = ["square", "triangle", "sine", "sawtooth", "flat"]
            shapes = random.sample(bank, random.randint(2, 4))
            ctx = alternating_shape_segments(
                self.seq_len, shapes=shapes, tile=random.random() < 0.5
            )
            add("mixed_shape_segments", ctx)

        for _ in range(n_samples):
            bank = ["square", "triangle", "sine", "sawtooth"]
            motifs = random.sample(bank, 3)
            shapes = []
            for motif in motifs:
                shapes += [motif, "flat"]
            ctx = alternating_shape_segments(
                self.seq_len, shapes=shapes, tile=random.random() < 0.5
            )
            add("three_motifs_with_flat", ctx)

        # ============================================================
        # Added signal families (extra predictable, diverse patterns).
        # ============================================================

        # Fourier series: a fundamental plus integer harmonics with decaying
        # amplitude. Strictly periodic, so the whole waveform is predictable.
        for _ in range(n_samples):
            t = torch.arange(self.seq_len, dtype=torch.float32)
            period = random.choice(
                [16, 24, 32, 48, 64, 96]
                + [p for p in [128, 192, 256] if p <= self.seq_len // 4]
            )
            n_harm = random.randint(2, 5)
            ctx = torch.zeros(self.seq_len)
            for h in range(1, n_harm + 1):
                amp = random.uniform(1, 20) / h
                phase = random.uniform(0, 2 * torch.pi)
                ctx = ctx + amp * torch.sin(2 * torch.pi * h * t / period + phase)
            add("harmonic_sum", ctx)

        # Seasonal-naive: repeat a random short motif to fill the window. Pure
        # periodicity with an arbitrary shape - the model must copy the cycle.
        for _ in range(n_samples):
            period = random.choice(
                [8, 12, 16, 24, 32, 48]
                + [p for p in [64, 96, 128, 192, 256] if p <= self.seq_len // 4]
            )
            motif = torch.randn(period) * random.uniform(1, 50)
            reps = -(-self.seq_len // period)
            ctx = motif.repeat(reps)[: self.seq_len]
            add("repeated_motif", ctx)

        # Classic decomposable series: linear trend + one to three additive
        # seasonal components (e.g. daily/weekly), like real demand data.
        for _ in range(n_samples):
            t = torch.arange(self.seq_len, dtype=torch.float32)
            slope = random.uniform(-2, 2)
            ctx = random.uniform(-50, 50) + slope * t
            for _ in range(random.randint(1, 3)):
                period = random.choice(
                    [7, 12, 24, 48, 96]
                    + [p for p in [168, 192, 256] if p <= self.seq_len // 4]
                )
                amp = random.uniform(1, 30)
                phase = random.uniform(0, 2 * torch.pi)
                ctx = ctx + amp * torch.sin(2 * torch.pi * t / period + phase)
            add("trend_multi_seasonal", ctx)

        # Multiplicative seasonality: oscillation amplitude grows with the
        # level, so peaks and troughs widen along a rising trend.
        for _ in range(n_samples):
            t = torch.arange(self.seq_len, dtype=torch.float32)
            base = random.uniform(10, 100)
            growth = random.uniform(0.0, 3.0)
            level = base + growth * base * t / self.seq_len
            period = random.choice(
                [8, 12, 24, 48, 96]
                + [p for p in [128, 192, 256] if p <= self.seq_len // 4]
            )
            rel_amp = random.uniform(0.1, 0.5)
            phase = random.uniform(0, 2 * torch.pi)
            ctx = level * (1 + rel_amp * torch.sin(2 * torch.pi * t / period + phase))
            add("multiplicative_seasonal", ctx)

        # Waveform shaping I - rectified sine (half- or full-wave), the shape a
        # diode/absolute-value nonlinearity produces. Periodic, hence learnable.
        for _ in range(n_samples):
            t = torch.arange(self.seq_len, dtype=torch.float32)
            period = random.choice([8, 12, 16, 24, 32, 48, 64])
            amp = random.uniform(1, 100)
            wave = amp * torch.sin(2 * torch.pi * t / period)
            ctx = torch.abs(wave) if random.random() < 0.5 else torch.clamp(wave, min=0)
            add("rectified_sine", ctx)

        # Waveform shaping II - a sine clipped (saturated) at a threshold, giving
        # flattened peaks between smooth transitions.
        for _ in range(n_samples):
            t = torch.arange(self.seq_len, dtype=torch.float32)
            period = random.choice([8, 12, 16, 24, 32, 48, 64])
            amp = random.uniform(5, 100)
            clip = random.uniform(0.3, 0.8) * amp
            ctx = torch.clamp(amp * torch.sin(2 * torch.pi * t / period), -clip, clip)
            add("clipped_sine", ctx)

        # Beats: two nearly-equal frequencies interfere into a slow amplitude
        # envelope. Both carrier and envelope are periodic and predictable.
        for _ in range(n_samples):
            t = torch.arange(self.seq_len, dtype=torch.float32)
            nyquist = self.seq_len / 2
            f = random.uniform(5, 0.4 * nyquist)  # cycles per window
            delta = random.uniform(1, 3)  # small offset -> slow beat
            amp = random.uniform(1, 100)
            ctx = amp * (
                torch.sin(2 * torch.pi * f * t / self.seq_len)
                + torch.sin(2 * torch.pi * (f + delta) * t / self.seq_len)
            )
            add("beats", ctx)

        # AR(2) with complex-conjugate roots: a stable stochastic recursion that
        # produces quasi-periodic, damped oscillations with learnable dynamics.
        for _ in range(n_samples):
            r = random.uniform(0.7, 0.99)  # pole magnitude (stability)
            period = random.uniform(4, 40)
            theta = 2 * np.pi / period
            a1 = 2 * r * np.cos(theta)
            a2 = -(r**2)
            eps = np.random.randn(self.seq_len) * random.uniform(0.1, 5)
            ctx = torch.from_numpy(
                lfilter([1.0], [1.0, -a1, -a2], eps).astype(np.float32)
            )
            add("ar2", ctx)

        # Regularly spaced Gaussian pulses: like gaussian_bumps but on a fixed
        # period, so the pulse train can be extrapolated.
        for _ in range(n_samples):
            t = torch.arange(self.seq_len, dtype=torch.float32)
            period = random.choice(
                [16, 24, 32, 48, 64]
                + [p for p in [96, 128, 192, 256] if p <= self.seq_len // 4]
            )
            width = random.uniform(period / 12, period / 4)
            height = random.uniform(1, 100) * random.choice([-1, 1])
            phase = random.uniform(0, period)
            d = torch.remainder(t - phase, period)
            d = torch.minimum(d, period - d)  # distance to nearest pulse center
            ctx = height * torch.exp(-(d**2) / (2 * width**2))
            add("periodic_gaussian_pulses", ctx)

        # First-order relaxation: exponential approach from a start value to a
        # steady level (the step response of an RC-type system).
        for _ in range(n_samples):
            t = torch.linspace(0, 1, self.seq_len)
            start = random.uniform(-100, 100)
            end = random.uniform(-100, 100)
            rate = random.uniform(3, 12)
            ctx = end + (start - end) * torch.exp(-rate * t)
            add("exp_relaxation", ctx)

        # Ringing step: an underdamped second-order response that oscillates
        # while decaying onto a final level.
        for _ in range(n_samples):
            t = torch.linspace(0, 1, self.seq_len)
            level = random.uniform(-50, 50)
            amp = random.uniform(1, 100)
            decay = random.uniform(1, 6)
            freq = random.uniform(10, 60)
            ctx = level + amp * torch.exp(-decay * t) * torch.cos(freq * t)
            add("ringing_step", ctx)

        # Frequency-modulated sine: the instantaneous frequency itself varies
        # sinusoidally. Peak frequency is kept below Nyquist to stay resolvable.
        for _ in range(n_samples):
            t = torch.linspace(0, 1, self.seq_len)
            nyquist = self.seq_len / 2
            carrier = random.uniform(5, 0.5 * nyquist)
            mod_freq = random.uniform(1, 5)
            mod_depth = random.uniform(0.1, 0.4) * carrier
            amp = random.uniform(1, 100)
            phase = 2 * torch.pi * (
                carrier * t
                - (mod_depth / (2 * torch.pi * mod_freq))
                * torch.cos(2 * torch.pi * mod_freq * t)
            )
            ctx = amp * torch.sin(phase)
            add("frequency_modulated_sin", ctx)

        # Power-law trend: sub-linear roots and super-linear powers, covering
        # curvature that plain polynomials with integer degree miss.
        for _ in range(n_samples):
            t = torch.linspace(0, 1, self.seq_len)
            sign = random.choice([-1, 1])
            p = random.choice([0.3, 0.5, 1.5, 2.5, 3.0])
            scale = random.uniform(1, 100)
            ctx = sign * scale * t**p
            add("power_trend", ctx)

        # Staircase riding a linear trend: discrete level jumps superimposed on
        # a slow drift.
        for _ in range(n_samples):
            t = torch.arange(self.seq_len, dtype=torch.float32)
            period = random.choice(
                [8, 16, 24, 32, 48]
                + [p for p in [64, 96, 128] if p <= self.seq_len // 4]
            )
            step_height = random.uniform(1, 30) * random.choice([-1, 1])
            slope = random.uniform(-1, 1)
            ctx = slope * t + step_height * torch.floor(t / period)
            add("staircase_trend", ctx)

        # Realistic composite: trend + seasonal terms + AR(1) coloured noise,
        # the additive decomposition most real series roughly follow.
        for _ in range(n_samples):
            t = torch.arange(self.seq_len, dtype=torch.float32)
            slope = random.uniform(-1.5, 1.5)
            ctx = random.uniform(-20, 20) + slope * t
            for _ in range(random.randint(1, 2)):
                period = random.choice([7, 12, 24, 48])
                phase = random.uniform(0, 2 * torch.pi)
                ctx = ctx + random.uniform(1, 20) * torch.sin(
                    2 * torch.pi * t / period + phase
                )
            phi = random.uniform(0.3, 0.9)
            eps = np.random.randn(self.seq_len) * random.uniform(0.5, 3)
            ar = lfilter([1.0], [1.0, -phi], eps).astype(np.float32)
            ctx = ctx + torch.from_numpy(ar)
            add("trend_seasonal_ar_noise", ctx)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ctx, _ = self.samples[idx]
        if self.noise:
            std = torch.std(ctx) * 0.1
            ctx = ctx + torch.randn_like(ctx) * std
        mean = ctx.mean()
        std = ctx.std() + 1e-6
        ctx = (ctx - mean) / std
        return ctx.float()


def plot_all_families(
    seq_len=1024,
    n_examples=3,
    noise=False,
    normalize=True,
    ncols=4,
    save_path="signal_families.png",
    show=False,
):
    """Render every SyntheticTimeSeriesDataset family in a grid, one panel per
    family with a few example draws overlaid.

    Args:
        seq_len: length of the generated signals.
        n_examples: number of example curves overlaid per family.
        noise: apply the same observation noise the training pipeline adds.
        normalize: z-normalize each curve (what the model actually sees).
        ncols: number of grid columns.
        save_path: where the PNG is written (None to skip saving).
        show: call plt.show() at the end (for interactive sessions).
    """
    import math

    import matplotlib.pyplot as plt

    # Generate a small dataset; n_examples per family is enough for a preview.
    ds = SyntheticTimeSeriesDataset(
        seq_len=seq_len, noise=noise, n_samples=n_examples
    )

    # Group sample indices by family label, preserving first-seen order.
    families = {}
    for idx, (_, label) in enumerate(ds.samples):
        families.setdefault(label, []).append(idx)

    n_fam = len(families)
    nrows = math.ceil(n_fam / ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.6 * ncols, 2.0 * nrows), squeeze=False
    )
    axes = axes.flatten()

    for ax, (label, idxs) in zip(axes, families.items()):
        for j in idxs[:n_examples]:
            # __getitem__ applies noise + normalization; use it when both are
            # wanted, otherwise pull the raw stored signal.
            if noise and normalize:
                y = ds[j]
            else:
                y = ds.samples[j][0].clone().float()
                if normalize:
                    y = (y - y.mean()) / (y.std() + 1e-6)
            ax.plot(y.numpy(), lw=0.8, alpha=0.85)
        ax.set_title(label, fontsize=8)
        ax.tick_params(labelsize=6)
        ax.margins(x=0.01)

    # Blank any unused cells in the last row.
    for ax in axes[n_fam:]:
        ax.axis("off")

    fig.suptitle(
        f"Synthetic signal families  (seq_len={seq_len}, "
        f"{'normalized' if normalize else 'raw'})",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.985))

    if save_path is not None:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"Saved {n_fam} families to {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return save_path


def triangle_with_flat(xx, freq, slope, flat_ratio):
    period = freq
    tri_len = (1 - flat_ratio) * period
    half_tri = tri_len / 2

    mod = torch.remainder(xx, period)

    tri_wave = torch.where(
        mod < half_tri,
        slope * mod,  # rising edge
        torch.where(
            mod < tri_len,
            slope * (tri_len - mod),  # falling edge
            torch.tensor(0.0),  # flat zone
        ),
    )
    return tri_wave


def rectangular_wave(xx, freq, amp):
    period = freq
    mod = torch.remainder(xx, period)
    half_period = period / 2
    return torch.where(mod < half_period, amp, -amp)


def rectangular_triangle_sequence(xx, freq, amp, slope):
    period = freq * 2
    mod = torch.remainder(xx, period)
    half_period = period / 2

    rect_wave = torch.where(mod < half_period, amp, -amp)

    tri_mod = torch.remainder(xx, period)
    half_tri = half_period / 2
    tri_wave = torch.where(
        tri_mod < half_tri,
        slope * tri_mod,
        torch.where(
            tri_mod < half_period,
            slope * (half_period - tri_mod),
            torch.tensor(0.0),
        ),
    )

    first_half = torch.where(mod < half_period, rect_wave, tri_wave)
    return first_half


def triangle_rectangular_sequence(xx, freq, amp, slope):
    period = freq * 2
    mod = torch.remainder(xx, period)
    half_period = period / 2

    tri_mod = torch.remainder(xx, period)
    half_tri = half_period / 2
    tri_wave = torch.where(
        tri_mod < half_tri,
        slope * tri_mod,
        torch.where(
            tri_mod < half_period,
            slope * (half_period - tri_mod),
            torch.tensor(0.0),
        ),
    )

    rect_wave = torch.where(mod < half_period, amp, -amp)

    first_half = torch.where(mod < half_period, tri_wave, rect_wave)
    return first_half


def multi_shape_sequence(xx, segment_freq, amp, slope):
    period = segment_freq * 3
    mod = torch.remainder(xx, period)
    third_period = period / 3

    rect_part = torch.where(mod < third_period / 2, amp, -amp)

    tri_mod = torch.remainder(mod, third_period)
    half_tri = third_period / 4
    tri_part = torch.where(
        tri_mod < half_tri,
        slope * tri_mod,
        torch.where(
            tri_mod < third_period / 2,
            slope * (third_period / 2 - tri_mod),
            torch.tensor(0.0),
        ),
    )

    saw_mod = torch.remainder(mod, third_period)
    saw_part = slope * saw_mod

    result = torch.where(
        mod < third_period,
        rect_part,
        torch.where(
            mod < 2 * third_period,
            tri_part,
            saw_part,
        ),
    )
    return result


def alternating_shape_segments(seq_len, shapes, tile=False):
    # Half the time, keep amplitude/period consistent across segments so the
    # alternation is clean; otherwise each segment gets its own scale.
    # Periods must fit inside a single segment, hence the seq_len // 8 cap.
    period_bank = [8, 16, 32, 64] + [p for p in [128] if p <= seq_len // 8]
    shared_amp = random.uniform(1, 100) if random.random() < 0.5 else None
    shared_period = random.choice(period_bank) if random.random() < 0.5 else None

    def make_piece(shape, seg_len):
        amp = shared_amp if shared_amp is not None else random.uniform(1, 100)
        period = (
            shared_period
            if shared_period is not None
            else random.choice(period_bank)
        )
        t = torch.arange(seg_len, dtype=torch.float32)
        mod = torch.remainder(t, period)
        if shape == "square":
            return torch.where(mod < period / 2, amp, -amp)
        if shape == "triangle":
            return 4 * amp / period * torch.abs(mod - period / 2) - amp
        if shape == "sine":
            return amp * torch.sin(2 * torch.pi * t / period)
        if shape == "sawtooth":
            return amp * (2 * mod / period - 1)
        return torch.full((seg_len,), random.uniform(-amp, amp))  # flat

    if tile:
        # One segment per shape, then repeat the whole block: motif
        # transitions become periodic, hence predictable from context.
        max_seg = max(seq_len // (2 * len(shapes)), 8)
        block = torch.cat(
            [make_piece(shape, random.randint(8, max_seg)) for shape in shapes]
        )
        reps = -(-seq_len // len(block))
        return block.repeat(reps)[:seq_len]

    pieces = []
    total = 0
    shape_idx = random.randrange(len(shapes))
    while total < seq_len:
        seg_len = random.randint(max(seq_len // 10, 8), max(seq_len // 3, 16))
        pieces.append(make_piece(shapes[shape_idx % len(shapes)], seg_len))
        total += seg_len
        shape_idx += 1

    return torch.cat(pieces)[:seq_len]


class TSMixUp(Dataset):
    def __init__(
        self,
        seq_len=96,
        total_samples=200_000,
        K=4,
        alpha=1.5,
        samples_per_class=1000,
    ):

        def get_xs(seq_len, start=None):
            if start is None:
                start = random.uniform(0, 1000)
            xx = torch.linspace(start, start + seq_len, seq_len)
            return xx[:seq_len]

        self.sinusoidal = []
        for _ in range(samples_per_class):
            abscisse = 0
            freq = random.choice(
                [
                    1 / 1000,
                    1 / 500,
                    1 / 100,
                    1 / 75,
                    1 / 50,
                    1 / 25,
                    1 / 10,
                    1 / 5,
                    1,
                    5,
                    10,
                    25,
                    50,
                    75,
                    100,
                    500,
                    1000,
                ]
            )
            amp = random.choice(
                [
                    1000,
                    500,
                    100,
                    50,
                    20,
                    10,
                    5,
                    1,
                    0.1,
                    0.05,
                    0.01,
                    0.005,
                    0.001,
                    -0.1,
                    -0.5,
                    -1,
                    -5,
                    -10,
                    -20,
                    -50,
                    -100,
                    -500,
                ]
            )
            x_ctx = get_xs(seq_len=seq_len)
            ctx = abscisse + amp * torch.sin(x_ctx * freq)
            self.sinusoidal.append(ctx.float())

        self.linear = []
        for _ in range(samples_per_class):
            abscisse = 0
            slope = random.choice(
                [
                    100,
                    50,
                    10,
                    5,
                    3,
                    1,
                    0.1,
                    0.05,
                    0.01,
                    -0.1,
                    -0.5,
                    -1,
                    -3,
                    -5,
                    -10,
                    -50,
                    -100,
                ]
            )
            x_ctx = get_xs(seq_len=seq_len)
            ctx = abscisse + x_ctx * slope
            self.linear.append(ctx.float())

        self.cosinusoidal = []
        for _ in range(samples_per_class):
            abscisse = 0
            freq = random.choice(
                [
                    1 / 1000,
                    1 / 500,
                    1 / 100,
                    1 / 75,
                    1 / 50,
                    1 / 25,
                    1 / 10,
                    1 / 5,
                    1,
                    5,
                    10,
                    25,
                    50,
                    75,
                    100,
                    500,
                    1000,
                ]
            )
            amp = random.choice(
                [
                    1000,
                    500,
                    100,
                    50,
                    20,
                    10,
                    5,
                    1,
                    0.1,
                    0.05,
                    0.01,
                    0.005,
                    0.001,
                    -0.1,
                    -0.5,
                    -1,
                    -5,
                    -10,
                    -20,
                    -50,
                    -100,
                    -500,
                ]
            )
            x_ctx = get_xs(seq_len=seq_len)
            ctx = abscisse + amp * torch.cos(x_ctx * freq)
            self.cosinusoidal.append(ctx.float())

        self.polynomial = []
        for _ in range(samples_per_class):
            abscisse = 0
            sign = random.choice([-1, 1])
            x_ctx = get_xs(seq_len=seq_len)
            ctx = abscisse + sign * x_ctx**2
            self.polynomial.append(ctx.float())

        self.logarithmic = []
        for _ in range(samples_per_class):
            abscisse = 0
            scale = random.choice([1, 2, 5, 10, 50])
            sign = random.choice([-1, 1])
            x_ctx = get_xs(seq_len=seq_len, start=1)  # éviter log(0)
            ctx = abscisse + sign * torch.log(x_ctx * scale)
            self.logarithmic.append(ctx.float())

        self.mix_signals = []
        for _ in range(total_samples):
            k = random.randint(2, K)
            datasets = random.choices(
                [
                    self.sinusoidal,
                    self.linear,
                    self.cosinusoidal,
                    self.polynomial,
                    self.logarithmic,
                ],
                k=k,
            )
            signals = []
            for dataset in datasets:
                ctx = random.choice(dataset)
                xx = ctx / torch.mean(torch.abs(ctx))
                signals.append(xx)
            lambdas = np.random.dirichlet(alpha=[alpha] * k)
            new_signal = sum(
                lambdas[i] * signals[i] for i in range(k)
            ) * random.randint(1, 100)
            ctx = new_signal[:seq_len]
            self.mix_signals.append(ctx)

        self.all_signals = (
            self.sinusoidal
            + self.linear
            + self.cosinusoidal
            + self.polynomial
            + self.logarithmic
            + self.mix_signals
        )
        random.shuffle(self.all_signals)

    def __len__(self):
        return len(self.all_signals)

    def __getitem__(self, idx):
        ctx = self.all_signals[idx]
        mean = ctx.mean()
        std = ctx.std() + 1e-6
        ctx = (ctx - mean) / std
        return ctx.float()


class SyntheticGPTimeSeriesDataset(Dataset):
    def __init__(self, file_path="synthetic_timeseries_gp.npy", seq_len=1024):

        if not os.path.exists(file_path):
            generate_gp_dataset(size=seq_len)

        self.data = np.load(file_path)
        self.data = torch.tensor(self.data, dtype=torch.float32)
        self.seq_len = seq_len

        assert (
            self.data.shape[1] >= self.seq_len
        ), "Input data must have at least seq_len + target_len features"
        self.data = self.data[:, : self.seq_len]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        ctx = sample[: self.seq_len]
        mean = ctx.mean()
        std = ctx.std() + 1e-6
        ctx = (ctx - mean) / std
        return ctx.float()


def artificial_dataset(
    seq_len=256,
    noise=True,
    file_path="data/synthetic_timeseries_full.npy",
):
    tsmixup_dataset = TSMixUp(
        seq_len=seq_len,
        total_samples=150_000,
        K=4,
        alpha=1.5,
        samples_per_class=1000,
    )
    artificial_dataset = SyntheticTimeSeriesDataset(
        seq_len=seq_len, noise=noise, n_samples=10000
    )
    gpdataset = SyntheticGPTimeSeriesDataset(file_path=file_path, seq_len=seq_len)
    return torch.utils.data.ConcatDataset(
        [tsmixup_dataset, artificial_dataset, gpdataset]
    )


#### Helpers for GP dataset generation ####


def generate_gp_dataset(size=1056):

    import random
    from multiprocessing import cpu_count

    import numpy as np
    from joblib import Parallel, delayed, parallel_backend
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import (
        RBF,
        ConstantKernel,
        DotProduct,
        ExpSineSquared,
        RationalQuadratic,
        WhiteKernel,
    )
    from tqdm import tqdm

    def get_xs(seq_len, target_len, start=None):
        if start is None:
            start = random.uniform(0, 1000)
        xx = np.linspace(start, start + seq_len + target_len - 1, seq_len + target_len)
        return xx

    def sample_kernel_from_bank():
        kernel_choices = []

        kernel_choices.append(ConstantKernel(constant_value=1.0))
        sigma_n = np.random.choice([0.1, 1])
        kernel_choices.append(WhiteKernel(noise_level=sigma_n))
        sigma_0 = np.random.choice([0, 1, 10])
        kernel_choices.append(DotProduct(sigma_0=sigma_0))
        length_scale = np.random.choice([0.1, 1, 10])
        kernel_choices.append(RBF(length_scale=length_scale))
        alpha = np.random.choice([0.1, 1, 10])
        kernel_choices.append(RationalQuadratic(length_scale=1.0, alpha=alpha))

        p_choices = [
            24,
            48,
            96,
            168,
            336,
            672,
            7,
            14,
            30,
            60,
            365,
            730,
            4,
            26,
            52,
            6,
            12,
            40,
            10,
        ]
        p = np.random.choice(p_choices)
        kernel_choices.append(ExpSineSquared(length_scale=1.0, periodicity=p))

        return random.choice(kernel_choices)

    def compose_kernels(k1, k2, op):
        return k1 + k2 if op == "+" else k1 * k2

    def generate_synthetic_timeseries(size):
        lsyn = size
        J = 5

        j = np.random.randint(1, J + 1)
        kernels = [sample_kernel_from_bank() for _ in range(j)]
        kernel_star = kernels[0]

        for i in range(1, j):
            op = random.choice(["+", "*"])
            kernel_star = compose_kernels(kernel_star, kernels[i], op)

        X = np.linspace(0, 1, lsyn).reshape(-1, 1)
        gp = GaussianProcessRegressor(kernel=kernel_star, alpha=1e-6, normalize_y=True)
        y = gp.sample_y(X, n_samples=1).flatten()

        return y

    # Gaussian Process Time Series Generation

    total_samples = 160000
    n_jobs = cpu_count() - 4  # Leave some core free

    with parallel_backend("loky"):  # 'loky' is the default and best for sklearn
        results = Parallel(n_jobs=n_jobs)(
            delayed(generate_synthetic_timeseries)(size)
            for i in tqdm(range(total_samples))
        )

    np_array = np.array(results, dtype=np.float32)  # shape: (total_samples, 1056)
    np.save("../data/synthetic_timeseries_gp.npy", np_array)

    print(f"Finished Gaussian Process Time Series Generation.")