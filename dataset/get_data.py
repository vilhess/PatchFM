import torch

from dataset.artificial import artificial_dataset
from dataset.gift import GiftEvalPretrain
from dataset.utsd import UTSDataset
from dataset.chronosdata import ChronosDataset, ChronosDataset_mmap


def get_dataset(seq_len=1024):
    art_trainset = artificial_dataset(
        seq_len=seq_len, noise=True
    )
    utsd_trainset = UTSDataset(input_len=seq_len, flag="train", split=0.9, min_stride=32, max_samples=1000)
    gift_trainset = GiftEvalPretrain(
        path="data/giftpretrain/", input_len=seq_len, min_stride=32, max_samples=1000
    )
    kernel_synth = ChronosDataset(file_path="data/training_corpus_kernel_synth_1m.npz")
    tsmixup = ChronosDataset_mmap(file_path="data/training_corpus_tsmixup_1m.npy", file_shape_path="data/training_corpus_tsmixup_1m_shape.npy")
    return torch.utils.data.ConcatDataset([art_trainset,  gift_trainset, kernel_synth, tsmixup, utsd_trainset])