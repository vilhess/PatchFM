import os
import subprocess

import numpy as np
import torch
from torch.utils.data import Dataset

# ── Dataset classes ───────────────────────────────────────────────────────────


class ChronosDataset(Dataset):
    """In-memory dataset loaded from a compressed .npz file."""

    EXPECTED_FILENAME = "training_corpus_kernel_synth_1m.npz"

    def __init__(self, file_path: str):
        if self.EXPECTED_FILENAME not in file_path:
            raise ValueError(f"File path must contain '{self.EXPECTED_FILENAME}'.")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        data = np.load(file_path)["data"]
        self.data = torch.tensor(data, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return _normalize(self.data[idx])


class ChronosDataset_mmap(Dataset):
    """Memory-mapped dataset for large .npy files."""

    EXPECTED_FILENAME = "training_corpus_tsmixup_10m.npy"

    def __init__(self, file_path: str, file_shape_path: str):
        if self.EXPECTED_FILENAME not in file_path:
            raise ValueError(f"File path must contain '{self.EXPECTED_FILENAME}'.")
        for path in (file_path, file_shape_path):
            if not os.path.exists(path):
                raise FileNotFoundError(f"File not found: {path}")

        n_samples, seq_len = np.load(file_shape_path)
        raw = np.memmap(
            file_path, dtype=np.float32, mode="r", shape=(n_samples, seq_len)
        )
        data = torch.tensor(raw, dtype=torch.float32)

        nan_mask = torch.isnan(data).any(dim=1)
        if nan_mask.any():
            print(
                f"Warning: Dropping {nan_mask.sum().item()} samples containing NaN values."
            )
            data = data[~nan_mask]

        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return _normalize(self.data[idx])


# ── Helpers ───────────────────────────────────────────────────────────────────


def _normalize(sample: torch.Tensor) -> torch.Tensor:
    """Z-score normalize a 1-D tensor."""
    return (sample - sample.mean()) / (sample.std() + 1e-6)


# ── Data download & processing ────────────────────────────────────────────────


def download_chronos_datasets_and_process(
    hf_folder: str = "data/chronos",
    np_folder: str = "data/chronos_numpy",
) -> None:
    """Download the Chronos datasets from HuggingFace and convert them to numpy arrays."""
    import datasets
    from tqdm import tqdm

    os.makedirs(hf_folder, exist_ok=True)
    os.makedirs(np_folder, exist_ok=True)

    _download_tsmixup(hf_folder, np_folder, tqdm)
    _download_kernel_synth(hf_folder, np_folder, tqdm)


def _hf_download(
    repo_id: str, include_glob: str, local_dir: str, cache_dir: str
) -> None:
    subprocess.run(
        [
            "huggingface-cli",
            "download",
            repo_id,
            f'--include "{include_glob}"',
            "--repo-type=dataset",
            "--local-dir",
            local_dir,
            "--cache-dir",
            cache_dir,
        ],
        check=True,
    )


def _download_tsmixup(hf_folder: str, np_folder: str, tqdm) -> None:
    import datasets

    MAX_LEN = 1024
    local_dir = f"{hf_folder}/tsmixup_10m"

    print("Downloading tsmixup_10m dataset...")
    _hf_download(
        "autogluon/chronos_datasets",
        "training_corpus/tsmixup_10m/*",
        local_dir,
        hf_folder,
    )

    def _stream_valid(folder: str):
        """Yield valid (no NaN, length ≥ MAX_LEN) samples from parquet files."""
        ds = datasets.load_dataset(
            "parquet",
            data_files=f"{folder}/*.parquet",
            split="train",
            streaming=True,
        ).select_columns(["target"])
        for item in ds:
            s = item["target"]
            if len(s) >= MAX_LEN and not np.isnan(s).any():
                yield s

    print("Counting valid tsmixup_10m samples...")
    count = sum(1 for _ in _stream_valid(local_dir))
    print(f"Valid samples: {count}")

    out_path = f"{np_folder}/training_corpus_tsmixup_10m.npy"
    mmap = np.memmap(out_path, dtype=np.float32, mode="w+", shape=(count, MAX_LEN))

    print("Writing tsmixup_10m samples...")
    for idx, s in enumerate(tqdm(_stream_valid(local_dir), total=count)):
        mmap[idx] = np.asarray(s[:MAX_LEN], dtype=np.float32)

    mmap.flush()
    np.save(
        f"{np_folder}/training_corpus_tsmixup_10m_shape.npy", np.array([count, MAX_LEN])
    )
    print("tsmixup_10m done.")


def _download_kernel_synth(hf_folder: str, np_folder: str, tqdm) -> None:
    import datasets

    BATCH_SIZE = 1000
    local_dir = f"{hf_folder}/kernel_synth_1m"

    print("Downloading kernel_synth_1m dataset...")
    _hf_download(
        "autogluon/chronos_datasets",
        "training_corpus/kernel_synth_1m/*",
        local_dir,
        hf_folder,
    )

    ds = datasets.load_dataset(
        "autogluon/chronos_datasets",
        local_dir,
        streaming=False,
        split="train",
    ).select_columns(["target"])

    n = len(ds)
    seq_len = len(ds[0]["target"])
    np_array = np.empty((n, seq_len), dtype=np.float32)

    print("Processing kernel_synth_1m samples...")
    idx = 0
    for start in tqdm(range(0, n, BATCH_SIZE)):
        batch = np.asarray(ds[start : start + BATCH_SIZE]["target"], dtype=np.float32)
        np_array[idx : idx + len(batch)] = batch
        idx += len(batch)

    np.savez_compressed(
        f"{np_folder}/training_corpus_kernel_synth_1m.npz", data=np_array
    )
    print("kernel_synth_1m done.")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    DOWNLOAD = False
    if DOWNLOAD:
        download_chronos_datasets_and_process()
