import torch 
import random
import numpy as np

class InnerMixUP(torch.utils.data.Dataset):
    def __init__(self, dataset, K=4, alpha=1.5, n_samples=200_000):
        self.dataset = dataset
        self.K = K
        self.alpha = alpha
        self.n_samples = n_samples
        self.len_ds = len(dataset)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        k = random.randint(2, self.K)
        indices = random.choices(range(self.len_ds), k=k)
        signals = [self.dataset[i] for i in indices]
        signals = [s / (torch.mean(torch.abs(s)) + 1e-8) for s in signals]
        lambdas = np.random.dirichlet(alpha=[self.alpha] * k)
        new_signal = sum(lambdas[i] * signals[i] for i in range(k))
        new_signal = (new_signal - new_signal.mean()) / (new_signal.std() + 1e-8)
        return new_signal.float()

class InterMixup(torch.utils.data.Dataset):
    def __init__(self, datasets, K=4, alpha=1.5, n_samples=200_000):
        self.datasets = datasets
        self.K = K
        self.alpha = alpha
        self.n_samples = n_samples
        self.len_ds = [len(ds) for ds in datasets]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        k = random.randint(2, self.K)
        chosen_datasets = random.choices(self.datasets, k=k)
        signals = []
        for ds in chosen_datasets:
            i = random.randint(0, len(ds) - 1)
            s = ds[i]
            s = s / (torch.mean(torch.abs(s)) + 1e-8)
            signals.append(s)
        lambdas = np.random.dirichlet(alpha=[self.alpha] * k)
        new_signal = sum(lambdas[i] * signals[i] for i in range(k))
        new_signal = (new_signal - new_signal.mean()) / (new_signal.std() + 1e-8)
        return new_signal.float()