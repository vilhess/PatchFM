import datasets
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import random

# subset name can be one of the following:
# 'UTSD-1G', 'UTSD-2G', 'UTSD-4G', 'UTSD-12G',

class UTSDataset(Dataset):
    def __init__(self, subset_name=r'UTSD-1G', flag='train', split=0.9,
                 input_len=None, output_len=None, scale=False, stride=1):
        self.input_len = input_len
        self.output_len = output_len
        self.seq_len = input_len + output_len
        assert flag in ['train', 'val']
        assert split >= 0 and split <=1.0
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.flag = flag
        self.scale = scale
        self.split = split
        self.stride = stride

        self.data_list = []
        self.n_window_list = []

        self.subset_name = subset_name
        self.__read_data__()

    def __read_data__(self):
        dataset = datasets.load_dataset("thuml/UTSD", self.subset_name, split='train')
        print('Indexing dataset...')
        for item in tqdm(dataset):
            self.scaler = StandardScaler()
            data = item['target']
            data = np.array(data).reshape(-1, 1)
            num_train = int(len(data) * self.split)
            border1s = [0, num_train - self.seq_len]
            border2s = [num_train, len(data)]

            border1 = border1s[self.set_type]
            border2 = border2s[self.set_type]

            if self.scale:
                train_data = data[border1s[0]:border2s[0]]
                self.scaler.fit(train_data)
                data = self.scaler.transform(data)

            data = data[border1:border2]
            n_window = (len(data) - self.seq_len) // self.stride + 1
            if n_window < 1:
                continue

            self.data_list.append(data)
            self.n_window_list.append(n_window if len(self.n_window_list) == 0 else self.n_window_list[-1] + n_window)


    def __getitem__(self, index):
        # you can wirte your own processing code here
        dataset_index = 0
        while index >= self.n_window_list[dataset_index]:
            dataset_index += 1

        index = index - self.n_window_list[dataset_index - 1] if dataset_index > 0 else index
        n_timepoint = (len(self.data_list[dataset_index]) - self.seq_len) // self.stride + 1

        s_begin = index % n_timepoint
        s_begin = self.stride * s_begin
        s_end = s_begin + self.seq_len
        seq_x = self.data_list[dataset_index][s_begin:s_end, :]

        ctx = seq_x[:self.input_len, :]
        target = seq_x[self.input_len:self.seq_len, :]

        return torch.from_numpy(ctx).float().squeeze(-1), torch.from_numpy(target).float().squeeze(-1)

    def __len__(self):
        return self.n_window_list[-1]

class SyntheticTimeSeriesDataset(Dataset):
    def __init__(self, seq_len=96, target_len=96, noise=True):
        self.seq_len = seq_len
        self.target_len = target_len
        self.samples = []
        self.noise = noise

        # --- Utility functions ---
        def add(label, ctx, target):
            self.samples.append((ctx, target, label))

        def get_xs(seq_len, target_len, start=None):
            if start is None:
                start = random.uniform(0, 1000)
            xx = torch.linspace(start, start + seq_len + target_len - 1, seq_len + target_len)
            return xx[:seq_len], xx[seq_len:]

        # === Patterns ===
        
        for abscisse in np.arange(-100, 100, 5):
            for _ in range(500):
                x_ctx, x_target = get_xs(self.seq_len, self.target_len)
                alpha = random.uniform(-10, 10)
                delta = random.uniform(-100, 100)
                sin_scale = random.uniform(1000, 1000000)
                sin_freq = random.uniform(0.01, 0.1)
                factor = random.randint(1, 100)
                ctx = (alpha * x_ctx**2 + delta + sin_scale * torch.sin(sin_freq*x_ctx)) / factor
                target = (alpha * x_target**2 + delta + sin_scale * torch.sin(sin_freq*x_target)) / factor
                add("poly_sin", ctx, target)
        
        for abscisse in np.arange(-1000, 1000, 10):
            for slope in [100, 50, 10, 5, 3, 1, 0.1, 0.05, 0.01, -0.1, -0.5, -1, -3, -5, -10, -50, -100]:
                for amp in [1, 2, 4, 8, 16, 32, 64, 128]:
                        for _ in range(1):
                                x_ctx, x_target = get_xs(self.seq_len, self.target_len)
                                ctx = abscisse + amp * torch.sin(x_ctx * 0.3) + slope * x_ctx
                                target = abscisse + amp * torch.sin(x_target * 0.3) + slope * x_target
                                add("linear_sin", ctx, target)
                        
        
        for abscisse in np.arange(-1000, 1000, 10):
            for _ in range(300):
                x_ctx, x_target = get_xs(self.seq_len, self.target_len)
                freq = random.uniform(0.01, 0.1)
                amp = random.uniform(1, 100)
                ctx = abscisse + amp * torch.sin(x_ctx * freq)
                target = abscisse + amp * torch.sin(x_target * freq)
                add("vary_sin", ctx, target)
        
        for abscisse in np.arange(-1000, 1000, 50):
            for step_height in [5, 10, 20, 50, 100, 1000]:
                for period in [8, 16, 32, 64]:
                    for _ in range(50):
                        x_ctx, x_target = get_xs(self.seq_len, self.target_len)
                        ctx = abscisse + step_height * torch.floor(x_ctx / period)
                        target = abscisse + step_height * torch.floor(x_target / period)
                        add("step", ctx, target)
        
        for abscisse in np.arange(-500, 500, 50):
            for amp in [10, 20, 50, 100, 200]:
                for every in [20, 50, 100, 200, 400]:
                    for _ in range(50):
                        x_ctx, x_target = get_xs(self.seq_len, self.target_len)
                        burst_width = random.randint(2, 5)
                        full_len = len(x_ctx) + len(x_target)
                        full = abscisse + torch.zeros(full_len)
                        for i in range(0, full_len - burst_width, every):
                            full[i:i+burst_width] += amp
                        ctx = full[:len(x_ctx)]
                        target = full[len(x_ctx):]
                        add("burst_repeat", ctx, target)

        for _ in range(80000):
            abscisse = random.sample(range(0, 1000, 10), 1)[0]
            freq = random.sample([ 1/500, 1/100, 1/75, 1/50, 1/25, 1/10, 1/5, 1, 5, 10, 25, 50, 75, 100, 500], 2)
            amp = random.sample([1000, 500, 100, 50, 20, 10, 5, 1, 0.1, 0.05, 0.01, 0.005, 0.001, -0.1, -0.5, -1, -5, -10, -20, -50, -100, -500, -1000], 2)
            freq1, freq2 = freq[0], freq[1]
            amp1, amp2 = amp[0], amp[1]
            x_ctx, x_target = get_xs(self.seq_len, self.target_len)
            ctx = abscisse + amp1 * torch.sin(x_ctx * freq1) + amp2 * torch.sin(x_ctx * freq2)
            target = abscisse + amp1 * torch.sin(x_target * freq1) + amp2 * torch.sin(x_target * freq2)
            add("complex_sin", ctx, target)

        for _ in range(150000):
            abscisse = random.sample(range(0, 1000, 10), 1)[0]
            freq1 = random.choice([1/50, 1/10, 1/5, 1/4, 1/3, 1/2, 1, 2, 3, 4, 5, 10, 50])
            freq2 = random.choice([1/50, 1/10, 1/5, 1/4, 1/3, 1/2, 1, 2, 3, 4, 5, 10, 50])
            x_ctx, x_target = get_xs(self.seq_len, self.target_len)
            ctx = abscisse + torch.sin(freq1*x_ctx) + torch.cos(freq2*x_ctx)
            target = abscisse + torch.sin(freq1*x_target) + torch.cos(freq2*x_target)
            add("sin_cos", ctx, target)

        for _ in range(50000):
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

        for _ in range(50000):
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

        for _ in range(50000):
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
    
class TSMixUp(Dataset):
    def __init__(self, seq_len=96, target_len=96, K=3, alpha=1.5):

        def get_xs(seq_len, target_len, start=None):
            if start is None:
                start = random.uniform(0, 1000)
            xx = torch.linspace(start, start + seq_len + target_len - 1, seq_len + target_len)
            return xx[:seq_len], xx[seq_len:]
        
        self.sinusoidal = []
        for abscisse in np.arange(-1000, 1000, 10):
            for freq in [1/1000, 1/500, 1/100, 1/75, 1/50, 1/25, 1/10, 1/5, 1, 5, 10, 25, 50, 75, 100, 500, 1000]:
                for amp in [1000, 500, 100, 50, 20, 10, 5, 1, 0.1, 0.05, 0.01, 0.005, 0.001, -0.1, -0.5, -1, -5, -10, -20, -50, -100, -500]:
                    for _ in range(1):
                        x_ctx, x_target = get_xs(seq_len=seq_len, target_len=target_len)
                        ctx = abscisse + amp * torch.sin(x_ctx * freq) 
                        target = abscisse + amp * torch.sin(x_target * freq) 
                        self.sinusoidal.append((ctx.float(), target.float()))

        self.linear = []
        for abscisse in np.arange(-1000, 1000, 5):
            for slope in [100, 50, 10, 5, 3, 1, 0.1, 0.05, 0.01, -0.1, -0.5, -1, -3, -5, -10, -50, -100]: 
                for _ in range(6):
                    x_ctx, x_target = get_xs(seq_len=seq_len, target_len=target_len)
                    ctx = abscisse + x_ctx * slope 
                    target = abscisse + x_target * slope 
                    self.linear.append((ctx.float(), target.float()))

        self.cosinusoidal = []
        for abscisse in np.arange(-1000, 1000, 10):
            for freq in [1/1000, 1/500, 1/100, 1/75, 1/50, 1/25, 1/10, 1/5, 1, 5, 10, 25, 50, 75, 100, 500, 1000]:
                for amp in [1000, 500, 100, 50, 20, 10, 5, 1, 0.1, 0.05, 0.01, 0.005, 0.001, -0.1, -0.5, -1, -5, -10, -20, -50, -100, -500]:
                    for _ in range(1):
                        x_ctx, x_target = get_xs(seq_len=seq_len, target_len=target_len)
                        ctx = abscisse + amp * torch.cos(x_ctx * freq) 
                        target = abscisse + amp * torch.cos(x_target * freq) 
                        self.cosinusoidal.append((ctx.float(), target.float()))

        self.polynomial = []
        for abscisse in np.arange(-1000, 1000, 10):
            for sign in [-1, 1]:
                for _ in range(100):
                    x_ctx, x_target = get_xs(seq_len=seq_len, target_len=target_len)
                    ctx = abscisse + sign * x_ctx**2
                    target = abscisse + sign * x_target**2
                    self.polynomial.append((ctx.float(), target.float()))

        self.logarithmic = []
        for abscisse in np.arange(-100, 100, 1):
            for scale in [1, 2, 5, 10, 50]:
                for sign in [-1, 1]:
                    for _ in range(10):
                        x_ctx, x_target = get_xs(seq_len=seq_len, target_len=target_len, start=1)  # éviter log(0)
                        ctx = abscisse + sign * torch.log(x_ctx * scale)
                        target = abscisse + sign * torch.log(x_target * scale)
                        self.logarithmic.append((ctx.float(), target.float()))

        self.mix_signals = []
        for _ in range(150000):
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
    
def artificial_dataset(seq_len=256, target_len=96, K=3, alpha=1.5, noise=True, file_path="/lustre/fswork/projects/rech/ulm/uww31rp/patchfm/data/synthetic_timeseries_gp.npy"):
    tsmixup_dataset = TSMixUp(seq_len=seq_len, target_len=target_len, K=K, alpha=alpha)
    artificial_dataset = SyntheticTimeSeriesDataset(seq_len=seq_len, target_len=target_len, noise=noise)
    gpdataset = SyntheticGPTimeSeriesDataset(file_path=file_path, seq_len=seq_len, target_len=target_len)
    return torch.utils.data.ConcatDataset([tsmixup_dataset, artificial_dataset, gpdataset])
    
def get_dataset(seq_len, target_len, utsd_name='UTSD-1G', noise=True, scale=False):
    ds = SyntheticTimeSeriesDataset(seq_len=seq_len, target_len=target_len, noise=noise)
    utsd = UTSDataset(subset_name=utsd_name, input_len=seq_len, output_len=target_len, scale=scale, flag='train', stride=1)
    return torch.utils.data.ConcatDataset([ds, utsd])

### utils functions for dataset

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
