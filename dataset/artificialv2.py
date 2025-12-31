import torch 
from torch.utils.data import Dataset
import random
import numpy as np
import os

class SyntheticTimeSeriesDataset(Dataset):
    def __init__(self, seq_len=96, target_len=96, noise=True, n_samples=20):
        self.seq_len = seq_len
        self.target_len = target_len
        self.samples = []
        self.noise = noise
        self.n_samples = n_samples

        # set random seed for reproducibility
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)


        # --- Utility functions ---
        def add(label, ctx, target):
            self.samples.append((ctx, target, label))

        def get_xs(seq_len, target_len, start=None):
            if start is None:
                start = random.uniform(0, 1000)
            xx = torch.linspace(start, start + seq_len + target_len - 1, seq_len + target_len)
            return xx[:seq_len], xx[seq_len:]

        # === Patterns ===

        for _ in range(n_samples):
            abscisse = random.sample(range(-1000, 1000, 10), 1)[0]
            slope = random.sample([-100, -50, -10, -5, -3, -1, -0.1, -0.05, -0.01, 0.01, 0.05, 0.1, 1, 3, 5, 10, 50, 100], 1)[0]
            x_ctx, x_target = get_xs(self.seq_len, self.target_len)
            ctx = abscisse + slope * x_ctx
            target = abscisse + slope * x_target
            add("linear", ctx, target)

        for _ in range(n_samples):
            abscisse = random.sample(range(-1000, 1000, 10), 1)[0]
            degree = random.choice([2, 3, 4, 5])
            coeffs = [random.uniform(-1, 1) for _ in range(degree)]
            x_ctx, x_target = get_xs(self.seq_len, self.target_len)
            ctx = abscisse + sum(c * x_ctx**i for i, c in enumerate(coeffs, start=1))
            target = abscisse + sum(c * x_target**i for i, c in enumerate(coeffs, start=1))
            add("multipolynomial", ctx, target)

        
        for _ in range(n_samples):
            x_ctx, x_target = get_xs(self.seq_len, self.target_len)
            alpha = random.uniform(-10, 10)
            delta = random.uniform(-100, 100)
            sin_scale = random.uniform(1000, 1000000)
            sin_freq = random.uniform(0.01, 0.1)
            factor = random.randint(1, 100)
            ctx = (alpha * x_ctx**2 + delta + sin_scale * torch.sin(sin_freq*x_ctx)) / factor
            target = (alpha * x_target**2 + delta + sin_scale * torch.sin(sin_freq*x_target)) / factor
            add("poly_sin", ctx, target)
        
        for _ in range(n_samples):
            abscisse = random.sample(np.arange(-1000, 1000, 10).tolist(), 1)[0]
            slope= random.sample([-100, -50, -10, -5, -3, -1, -0.1, -0.05, -0.01, 0.01, 0.05, 0.1, 1, 3, 5, 10, 50, 100], 1)[0]
            amp = random.sample([1, 2, 4, 8, 16, 32, 64, 128], 1)[0]
            x_ctx, x_target = get_xs(self.seq_len, self.target_len)
            ctx = abscisse + amp * torch.sin(x_ctx * 0.3) + slope * x_ctx
            target = abscisse + amp * torch.sin(x_target * 0.3) + slope * x_target
            add("linear_sin", ctx, target)
            
        
        for _ in range(n_samples):
            abscisse = random.sample(range(-1000, 1000, 10), 1)[0]
            x_ctx, x_target = get_xs(self.seq_len, self.target_len)
            freq = random.uniform(0.01, 0.1)
            amp = random.uniform(1, 100)
            ctx = abscisse + amp * torch.sin(x_ctx * freq)
            target = abscisse + amp * torch.sin(x_target * freq)
            add("vary_sin", ctx, target)

        for _ in range(n_samples):
            period = random.choice([8, 16, 32, 64])
            step_height = random.choice([5, 10, 20, 50, 100, 1000])
            abscisse = random.sample(np.arange(-1000, 1000, 50).tolist(), 1)[0]
            x_ctx, x_target = get_xs(self.seq_len, self.target_len)
            ctx = abscisse + step_height * torch.floor(x_ctx / period)
            target = abscisse + step_height * torch.floor(x_target / period)
            add("step", ctx, target)
        
        for _ in range(n_samples):
            abscisse = random.sample(np.arange(-500, 500, 50).tolist(), 1)[0]
            amp = random.sample([10, 20, 50, 100, 200], 1)[0]
            every = random.sample([20, 50, 100, 200, 400], 1)[0]
            x_ctx, x_target = get_xs(self.seq_len, self.target_len)
            burst_width = random.randint(2, 5)
            full_len = len(x_ctx) + len(x_target)
            full = abscisse + torch.zeros(full_len)
            for i in range(0, full_len - burst_width, every):
                full[i:i+burst_width] += amp
            ctx = full[:len(x_ctx)]
            target = full[len(x_ctx):]
            add("burst_repeat", ctx, target)

        for _ in range(n_samples):
            abscisse = random.sample(range(0, 1000, 10), 1)[0]
            freq = random.sample([ 1/500, 1/100, 1/75, 1/50, 1/25, 1/10, 1/5, 1, 5, 10, 25, 50, 75, 100, 500], 2)
            amp = random.sample([1000, 500, 100, 50, 20, 10, 5, 1, 0.1, 0.05, 0.01, 0.005, 0.001, -0.1, -0.5, -1, -5, -10, -20, -50, -100, -500, -1000], 2)
            freq1, freq2 = freq[0], freq[1]
            amp1, amp2 = amp[0], amp[1]
            x_ctx, x_target = get_xs(self.seq_len, self.target_len)
            ctx = abscisse + amp1 * torch.sin(x_ctx * freq1) + amp2 * torch.sin(x_ctx * freq2)
            target = abscisse + amp1 * torch.sin(x_target * freq1) + amp2 * torch.sin(x_target * freq2)
            add("complex_sin", ctx, target)

        for _ in range(n_samples):
            abscisse = random.sample(range(0, 1000, 10), 1)[0]
            freq1 = random.choice([1/50, 1/10, 1/5, 1/4, 1/3, 1/2, 1, 2, 3, 4, 5, 10, 50])
            freq2 = random.choice([1/50, 1/10, 1/5, 1/4, 1/3, 1/2, 1, 2, 3, 4, 5, 10, 50])
            x_ctx, x_target = get_xs(self.seq_len, self.target_len)
            ctx = abscisse + torch.sin(freq1*x_ctx) + torch.cos(freq2*x_ctx)
            target = abscisse + torch.sin(freq1*x_target) + torch.cos(freq2*x_target)
            add("sin_cos", ctx, target)

        for _ in range(n_samples):
            abscisse = random.sample(range(0, 1000, 10), 1)[0]
            x_ctx, x_target = get_xs(self.seq_len, self.target_len)
            xx = torch.cat([x_ctx, x_target])
            freq = random.uniform(50, 500)
            delay = random.uniform(-100, 100)
            slope = random.uniform(-20, 20)
            yy = abscisse + slope * torch.abs(torch.remainder(xx, freq) - delay)
            ctx = yy[:self.seq_len]
            target = yy[self.seq_len:]
            add("sawtooth", ctx, target)

        for _ in range(n_samples):
            x, y = get_xs(self.seq_len, self.target_len)
            x = torch.cat([x, y])
            # generate a sawtooth wave with flat zone between each peak
            y = torch.zeros_like(x)
            abscisse = torch.randint(-1000, 1000, (1,)).item()
            space = torch.randint(10, 200, (1,)).item()
            height = torch.randint(1, 10, (1,)).item()
            for i in range(len(x)):
                if i % (space * 2) < space:
                    y[i] = height
                else:
                    y[i] = -height
            y = y + abscisse
            ctx = y[:self.seq_len]
            target = y[self.seq_len:]
            add("sawtooth_flat", ctx, target)

        for _ in range(n_samples):
            abscisse = random.uniform(-500, 500)
            x_ctx, x_target = get_xs(self.seq_len, self.target_len)
            xx = torch.cat([x_ctx, x_target])
            freq = random.uniform(50, 500)
            slope = random.uniform(-20, 20)
            flat_ratio = random.uniform(0.2, 0.8)
            yy = abscisse + triangle_with_flat(xx, freq=freq, slope=slope, flat_ratio=flat_ratio)
            ctx = yy[:self.seq_len]
            target = yy[self.seq_len:]
            add("triangle_with_flat", ctx, target)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ctx, target, label = self.samples[idx]
        if self.noise:
            ctx += torch.randn_like(ctx) * 0.1
            target += torch.randn_like(target) * 0.1
        return ctx.float(), target.float()
    
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
            torch.tensor(0.0)  # flat zone
        )
    )
    return tri_wave

class TSMixUp(Dataset):
    def __init__(self, seq_len=96, target_len=96, total_samples=200_000, K=4, alpha=1.5, samples_per_class=1000):

        def get_xs(seq_len, target_len, start=None):
            if start is None:
                start = random.uniform(0, 1000)
            xx = torch.linspace(start, start + seq_len + target_len - 1, seq_len + target_len)
            return xx[:seq_len], xx[seq_len:]
        
        self.sinusoidal = []
        for _ in range(samples_per_class):
            abscisse = random.uniform(-1000, 1000)
            freq = random.choice([1/1000, 1/500, 1/100, 1/75, 1/50, 1/25, 1/10, 1/5, 1, 5, 10, 25, 50, 75, 100, 500, 1000])
            amp = random.choice([1000, 500, 100, 50, 20, 10, 5, 1, 0.1, 0.05, 0.01, 0.005, 0.001, -0.1, -0.5, -1, -5, -10, -20, -50, -100, -500])
            x_ctx, x_target = get_xs(seq_len=seq_len, target_len=target_len)
            ctx = abscisse + amp * torch.sin(x_ctx * freq) 
            target = abscisse + amp * torch.sin(x_target * freq) 
            self.sinusoidal.append((ctx.float(), target.float()))

        self.linear = []
        for _ in range(samples_per_class):
            abscisse = random.uniform(-1000, 1000)
            slope = random.choice([100, 50, 10, 5, 3, 1, 0.1, 0.05, 0.01, -0.1, -0.5, -1, -3, -5, -10, -50, -100])
            x_ctx, x_target = get_xs(seq_len=seq_len, target_len=target_len)
            ctx = abscisse + x_ctx * slope 
            target = abscisse + x_target * slope 
            self.linear.append((ctx.float(), target.float()))

        self.cosinusoidal = []
        for _ in range(samples_per_class):
            abscisse = random.uniform(-1000, 1000)
            freq = random.choice([1/1000, 1/500, 1/100, 1/75, 1/50, 1/25, 1/10, 1/5, 1, 5, 10, 25, 50, 75, 100, 500, 1000])
            amp = random.choice([1000, 500, 100, 50, 20, 10, 5, 1, 0.1, 0.05, 0.01, 0.005, 0.001, -0.1, -0.5, -1, -5, -10, -20, -50, -100, -500])
            x_ctx, x_target = get_xs(seq_len=seq_len, target_len=target_len)
            ctx = abscisse + amp * torch.cos(x_ctx * freq) 
            target = abscisse + amp * torch.cos(x_target * freq) 
            self.cosinusoidal.append((ctx.float(), target.float()))

        self.polynomial = []
        for _ in range(samples_per_class):
            abscisse = random.uniform(-1000, 1000)
            sign = random.choice([-1, 1])
            x_ctx, x_target = get_xs(seq_len=seq_len, target_len=target_len)
            ctx = abscisse + sign * x_ctx**2
            target = abscisse + sign * x_target**2
            self.polynomial.append((ctx.float(), target.float()))

        self.logarithmic = []
        for _ in range(samples_per_class):
            abscisse = random.uniform(-1000, 1000)
            scale = random.choice([1, 2, 5, 10, 50])
            sign = random.choice([-1, 1])
            x_ctx, x_target = get_xs(seq_len=seq_len, target_len=target_len, start=1)  # éviter log(0)
            ctx = abscisse + sign * torch.log(x_ctx * scale)
            target = abscisse + sign * torch.log(x_target * scale)
            self.logarithmic.append((ctx.float(), target.float()))

        self.mix_signals = []
        for _ in range(total_samples):
            k = random.randint(2, K)
            datasets = random.choices(
                [self.sinusoidal, self.linear, self.cosinusoidal, self.polynomial, self.logarithmic],
                k=k
            )
            signals = []
            for dataset in datasets:
                ctx, target = random.choice(dataset)
                xx = torch.cat([ctx, target], dim=0)
                xx = xx / torch.mean(torch.abs(xx))
                signals.append(xx)
            lambdas = np.random.dirichlet(alpha=[alpha] * k)
            new_signal = sum(lambdas[i] * signals[i] for i in range(k)) * random.randint(1, 100)
            ctx, target = new_signal[:seq_len], new_signal[seq_len:]
            self.mix_signals.append((ctx, target))

        self.all_signals = self.sinusoidal + self.linear + self.cosinusoidal + self.polynomial + self.logarithmic + self.mix_signals
        random.shuffle(self.all_signals)

    def __len__(self):
        return len(self.all_signals)
    
    def __getitem__(self, idx):
        ctx, target = self.all_signals[idx]
        return ctx.float(), target.float()

class SyntheticGPTimeSeriesDataset(Dataset):
    def __init__(self, file_path="synthetic_timeseries_gp.npy", seq_len=1024, target_len=32):

        if not os.path.exists(file_path):
            print(f"File {file_path} not found. Generate the dataset first.")
            from dataset import generate_gp_dataset
            generate_gp_dataset(size=seq_len+target_len)

        self.data = np.load(file_path)
        self.data = torch.tensor(self.data, dtype=torch.float32)
        self.seq_len = seq_len
        self.target_len = target_len

        assert self.data.shape[1] >= self.seq_len + self.target_len, "Input data must have at least seq_len + target_len features"
        self.data = self.data[:, :self.seq_len + self.target_len]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        ctx = sample[:self.seq_len]
        target = sample[self.seq_len:self.seq_len + self.target_len]
        return ctx, target

def artificial_dataset(seq_len=256, target_len=96, noise=True, file_path="data/synthetic_timeseries_full.npy"):
    tsmixup_dataset = TSMixUp(seq_len=seq_len, target_len=target_len, total_samples=150_000, K=4, alpha=1.5, samples_per_class=1000)
    artificial_dataset = SyntheticTimeSeriesDataset(seq_len=seq_len, target_len=target_len, noise=noise, n_samples=10000)
    gpdataset = SyntheticGPTimeSeriesDataset(file_path=file_path, seq_len=seq_len, target_len=target_len)
    return torch.utils.data.ConcatDataset([tsmixup_dataset, artificial_dataset, gpdataset])
