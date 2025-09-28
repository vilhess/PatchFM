import datasets
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

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