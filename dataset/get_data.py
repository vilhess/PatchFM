import torch
from dataset.utsd import UTSDataset
from dataset.artificial import artificial_dataset

def get_dataset(seq_len, target_len):
    art_trainset = artificial_dataset(seq_len=seq_len, target_len=target_len, noise=True)
    utsd_trainset = UTSDataset(input_len=seq_len, output_len=target_len)
    return torch.utils.data.ConcatDataset([art_trainset, utsd_trainset])