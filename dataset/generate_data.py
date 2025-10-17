import numpy as np
import random
from sklearn.gaussian_process.kernels import (
    ConstantKernel, WhiteKernel, RBF, DotProduct,
    RationalQuadratic, ExpSineSquared
)
from sklearn.gaussian_process import GaussianProcessRegressor
from joblib import Parallel, delayed, parallel_backend
from multiprocessing import cpu_count
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

    p_choices = [24, 48, 96, 168, 336, 672, 7, 14, 30, 60, 365, 730,
                 4, 26, 52, 6, 12, 40, 10]
    p = np.random.choice(p_choices)
    kernel_choices.append(ExpSineSquared(length_scale=1.0, periodicity=p))

    return random.choice(kernel_choices)


def compose_kernels(k1, k2, op):
    return k1 + k2 if op == '+' else k1 * k2


def generate_synthetic_timeseries(_):
    lsyn = 1024 + 32
    J = 5

    j = np.random.randint(1, J + 1)
    kernels = [sample_kernel_from_bank() for _ in range(j)]
    kernel_star = kernels[0]

    for i in range(1, j):
        op = random.choice(['+', '*'])
        kernel_star = compose_kernels(kernel_star, kernels[i], op)

    X = np.linspace(0, 1, lsyn).reshape(-1, 1)
    gp = GaussianProcessRegressor(kernel=kernel_star, alpha=1e-6, normalize_y=True)
    y = gp.sample_y(X, n_samples=1).flatten()

    return y


def generate_gp_dataset():

    # Gaussian Process Time Series Generation
    
    total_samples = 160000
    n_jobs = cpu_count() - 4  # Leave some core free

    with parallel_backend("loky"):  # 'loky' is the default and best for sklearn
        results = Parallel(n_jobs=n_jobs)(
            delayed(generate_synthetic_timeseries)(i) for i in tqdm(range(total_samples))
        )

    np_array = np.array(results, dtype=np.float32)  # shape: (total_samples, 1056)
    np.save("../data/synthetic_timeseries_gp.npy", np_array)

    print(f"Finished Gaussian Process Time Series Generation.")

if __name__ == "__main__":
    generate_gp_dataset()
