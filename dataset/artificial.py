import os
import random

import numpy as np
import torch
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
            ctx = abscisse + amp * torch.sin(x_ctx * 0.3) + slope * x_ctx
            add("linear_sin", ctx)

        for _ in range(n_samples):
            abscisse = 0
            x_ctx = get_xs(self.seq_len)
            freq = random.uniform(0.01, 0.1)
            amp = random.uniform(1, 100)
            ctx = abscisse + amp * torch.sin(x_ctx * freq)
            add("vary_sin", ctx)

        for _ in range(n_samples):
            period = random.choice([8, 16, 32, 64])
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

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ctx, _ = self.samples[idx]
        if self.noise:
            std = torch.std(ctx) * 0.1
            ctx += torch.randn_like(ctx) * std
        mean = ctx.mean()
        std = ctx.std() + 1e-6
        ctx = (ctx - mean) / std
        return ctx.float()


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
    from sklearn.gaussian_process.kernels import (RBF, ConstantKernel, DotProduct,
                                                ExpSineSquared,
                                                RationalQuadratic, WhiteKernel)
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