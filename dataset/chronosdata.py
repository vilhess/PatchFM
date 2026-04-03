import os

import numpy as np
import torch
from torch.utils.data import Dataset


class ChronosDataset(Dataset):
    def __init__(self, file_path):
        # 'training_corpus_kernel_synth_1m.npz'
        assert (
            "training_corpus_kernel_synth_1m.npz" in file_path
        ), "File name must contain 'training_corpus_kernel_synth_1m.npz'"

        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"File {file_path} not found. Please ensure the file exists."
            )

        self.data = np.load(file_path)["data"]
        self.data = torch.tensor(self.data, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        mean = sample.mean()
        std = sample.std() + 1e-6
        sample = (sample - mean) / std
        return sample.float()


class ChronosDataset_mmap(Dataset):
    def __init__(self, file_path, file_shape_path):
        # 'training_corpus_tsmixup_10m'

        assert (
            "training_corpus_tsmixup_10m.npy" in file_path
        ), "File name must contain 'training_corpus_tsmixup_10m.npy'"

        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"File {file_path} not found. Please ensure the file exists."
            )
        if not os.path.exists(file_shape_path):
            raise FileNotFoundError(
                f"File {file_shape_path} not found. Please ensure the file exists."
            )

        n_samples, seq_len = np.load(file_shape_path)

        data = np.memmap(
            file_path, dtype=np.float32, mode="r", shape=(n_samples, seq_len)
        )
        self.data = torch.tensor(data, dtype=torch.float32)
        nan_idx = torch.isnan(self.data).any(dim=1)
        if nan_idx.any():
            print(
                f"Warning: Found {nan_idx.sum().item()} samples with NaN values. These samples will be removed."
            )
            self.data = self.data[~nan_idx]
        self.n_samples = self.data.shape[0]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        sample = self.data[idx]
        mean = sample.mean()
        std = sample.std() + 1e-6
        sample = (sample - mean) / std
        return sample.float()


def download_chronos_datasets_and_process(
    hf_folder="data/chronos", np_folder="data/chronos_numpy"
):
    import os
    import subprocess

    print(f"Downloading Chronos datasets to {hf_folder}...")

    os.makedirs(hf_folder, exist_ok=True)
    os.makedirs(np_folder, exist_ok=True)
    import datasets
    import numpy as np
    from tqdm import tqdm

    print("Downloading tsmixup_10m dataset...")
    subprocess.run(
        [
            "huggingface-cli",
            "download",
            "autogluon/chronos_datasets",
            '--include "training_corpus/tsmixup_10m/*"' "--repo-type=dataset",
            "--local-dir",
            f"{hf_folder}/{'tsmixup_10m'}",
            "--cache-dir",
            hf_folder,
        ],
        check=True,
    )
    print("Processing tsmixup_10m dataset...")
    ds = datasets.load_dataset(
        "parquet",
        data_files=f"{hf_folder}/tsmixup_10m/*.parquet",
        split="train",
        streaming=True,
    ).select_columns(["target"])
    max_len = 1024
    count = 0
    for item in ds:
        s = item["target"]
        if len(s) >= max_len:
            if not np.isnan(s).any():
                count += 1

    print("Total valid samples:", count)
    data = np.memmap(
        f"{np_folder}/training_corpus_tsmixup_10m.npy",
        dtype=np.float32,
        mode="w+",
        shape=(count, max_len),
    )

    ds = datasets.load_dataset(
        "parquet",
        data_files=f"{hf_folder}/tsmixup_10m/*.parquet",
        split="train",
        streaming=True,
    ).select_columns(["target"])

    idx = 0
    for item in tqdm(ds, total=count):
        c = item["target"]
        if len(c) < max_len:
            continue

        s = c[:max_len]
        if np.isnan(s).any():
            continue
        data[idx] = np.asarray(s, dtype=np.float32)
        idx += 1
    # --- Flush to disk ---
    data.flush()
    np.save(
        f"{np_folder}/training_corpus_tsmixup_10m_shape.npy", np.array([count, max_len])
    )

    print("Downloading kernel_synth_1m dataset...")
    subprocess.run(
        [
            "huggingface-cli",
            "download",
            "autogluon/chronos_datasets",
            '--include "training_corpus/kernel_synth_1m/*"' "--repo-type=dataset",
            "--local-dir",
            f"{hf_folder}/{'kernel_synth_1m'}",
            "--cache-dir",
            hf_folder,
        ],
        check=True,
    )
    ds = datasets.load_dataset(
        "autogluon/chronos_datasets",
        f"{hf_folder}/kernel_synth_1m",
        streaming=False,
        split="train",
    ).select_columns(["target"])

    # --- Detect shape ---
    n = len(ds)
    first = ds[0]["target"]
    d = len(first)

    # Preallocate (MUCH better than list)
    np_array = np.empty((n, d), dtype=np.float32)  # float32 saves memory

    # --- Fill with batching ---
    batch_size = 1000

    idx = 0
    for start in tqdm(range(0, n, batch_size)):
        batch = ds[start : start + batch_size]["target"]  # FAST batch access
        batch_np = np.asarray(batch, dtype=np.float32)

        end = idx + len(batch_np)
        np_array[idx:end] = batch_np
        idx = end

    # --- Save ---
    np.savez_compressed(
        f"{np_folder}/training_corpus_kernel_synth_1m.npz", data=np_array
    )


if __name__ == "__main__":
    DOWNLOAD = False
    if DOWNLOAD:
        download_chronos_datasets_and_process(
            hf_folder="data/chronos", np_folder="data/chronos_numpy"
        )
