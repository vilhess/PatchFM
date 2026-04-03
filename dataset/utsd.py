import datasets
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class UTSDataset(Dataset):
    def __init__(
        self,
        subset_name=r"UTSD-12G",
        flag="train",
        split=0.9,
        input_len=None,
        min_stride=32,
        cache_dir=None,
        max_samples=1000,
    ):
        self.input_len = input_len
        self.seq_len = input_len
        assert flag in ["train", "val"]
        assert 0 <= split <= 1.0
        type_map = {"train": 0, "val": 1, "test": 2}
        self.set_type = type_map[flag]
        self.flag = flag
        self.split = split
        self.min_stride = min_stride
        self.cache_dir = cache_dir
        self.max_samples = max_samples
        self.data_list = []
        self.stride_list = []  # per-signal adaptive stride
        self.n_window_list = []  # cumulative window counts
        self.subset_name = subset_name
        self.__read_data__()

    def __read_data__(self):
        dataset = datasets.load_dataset("thuml/UTSD", self.subset_name, split="train")

        print("Indexing dataset...")
        for item in tqdm(dataset):
            data = np.array(item["target"]).reshape(-1, 1)

            num_train = int(len(data) * self.split)
            border1s = [0, num_train - self.seq_len]
            border2s = [num_train, len(data)]
            border1 = border1s[self.set_type]
            border2 = border2s[self.set_type]
            data = data[border1:border2]

            max_windows = len(data) - self.seq_len  # denominator space
            if max_windows < 1:
                continue

            # --- adaptive stride -------------------------------------------
            # n_window = max_windows // stride + 1  =>  we want n_window <= max_samples
            # => stride >= max_windows / (max_samples - 1)   (when max_samples > 1)
            if self.max_samples > 1:
                required_stride = max_windows / (self.max_samples - 1)
            else:
                required_stride = max_windows  # only 1 window allowed

            stride = max(self.min_stride, int(np.ceil(required_stride)))
            # ----------------------------------------------------------------

            n_window = max_windows // stride + 1  # guaranteed <= max_samples

            self.data_list.append(data)
            self.stride_list.append(stride)
            cumulative = (
                n_window
                if len(self.n_window_list) == 0
                else self.n_window_list[-1] + n_window
            )
            self.n_window_list.append(cumulative)

    def __getitem__(self, index):
        # --- find which signal this index belongs to -----------------------
        dataset_index = 0
        while index >= self.n_window_list[dataset_index]:
            dataset_index += 1

        local_index = (
            index - self.n_window_list[dataset_index - 1]
            if dataset_index > 0
            else index
        )

        # --- pick the right stride for this signal -------------------------
        stride = self.stride_list[dataset_index]

        s_begin = local_index * stride
        s_end = s_begin + self.seq_len

        seq_x = self.data_list[dataset_index][s_begin:s_end, 0]
        ctx = seq_x[: self.input_len]
        # normalise
        mean = ctx.mean()
        std = ctx.std() + 1e-6
        ctx = (ctx - mean) / std

        return (torch.from_numpy(ctx).float(),)

    def __len__(self):
        return self.n_window_list[-1]
