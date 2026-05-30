import torch

from dataset.artificial import artificial_dataset
from dataset.chronosdata import ChronosDataset, ChronosDataset_mmap
from dataset.gift import GiftEvalDataset
from dataset.boom import BoomDataset
from dataset.mixup import InnerMixUP, InterMixup


def get_dataset(seq_len=1024):
    art_trainset = artificial_dataset(
        seq_len=seq_len, noise=True
    )
    gift_pretrain = GiftEvalDataset(
        input_len=seq_len, min_stride=32, max_samples=1000, path="path/to/gift/pretrain",
    )
    kernel_synth = ChronosDataset(file_path="path/to/chronos_numpy/training_corpus_kernel_synth_1m.npz")

    mixup = ChronosDataset_mmap("path/to/chronos_numpy/training_corpus_tsmixup_10m_clean.npy", "path/to/chronos_numpy/training_corpus_tsmixup_10m_clean_shape.npy" )
    mixup_1 = InnerMixUP(kernel_synth, K=4, alpha=1.5, n_samples=200_000)
    mixup_2 = InterMixup([gift_pretrain, kernel_synth], K=4, alpha=1.5, n_samples=200_000)
    mixup_3 = InnerMixUP(gift_pretrain, K=4, alpha=1.5, n_samples=200_000)
    mixup_5 = InterMixup([art_trainset, gift_pretrain], K=4, alpha=1.5, n_samples=200_000)
    return torch.utils.data.ConcatDataset([art_trainset,  gift_pretrain, kernel_synth, mixup, mixup_1, mixup_2, mixup_3, mixup_5])

def get_dataset_leakage(seq_len=1024):
    art_trainset = artificial_dataset(
        seq_len=seq_len, noise=True
    )
    gift_pretrain = GiftEvalDataset(
        input_len=seq_len, min_stride=32, max_samples=1000, path="path/to/gift/pretrain",
    )
    gift_eval = GiftEvalDataset(
        input_len=seq_len, min_stride=32, max_samples=1000, path="path/to/gift/eval",
    )
    boom = BoomDataset(input_len=seq_len, min_stride=32, max_samples=1000, path="path/to/boom")
    kernel_synth = ChronosDataset(file_path="path/to/chronos_numpy/training_corpus_kernel_synth_1m.npz")

    mixup = ChronosDataset_mmap("path/to/chronos_numpy/training_corpus_tsmixup_10m_clean.npy", "path/to/chronos_numpy/training_corpus_tsmixup_10m_clean_shape.npy" )

    mixup_1 = InnerMixUP(kernel_synth, K=4, alpha=1.5, n_samples=200_000)
    mixup_2 = InterMixup([gift_pretrain, kernel_synth], K=4, alpha=1.5, n_samples=200_000)
    mixup_3 = InnerMixUP(gift_pretrain, K=4, alpha=1.5, n_samples=200_000)
    mixup_5 = InterMixup([art_trainset, gift_pretrain], K=4, alpha=1.5, n_samples=200_000)
    mixup_6 = InnerMixUP(gift_eval, K=4, alpha=1.5, n_samples=200_000)
    mixup_7 = InterMixup([art_trainset, gift_eval], K=4, alpha=1.5, n_samples=200_000)
    mixup_8 = InterMixup([gift_pretrain, gift_eval], K=2, alpha=1.5, n_samples=200_000)
    mixup_9 = InterMixup([kernel_synth, gift_eval], K=2, alpha=1.5, n_samples=200_000)
    return torch.utils.data.ConcatDataset([art_trainset,  gift_pretrain, gift_eval, boom, kernel_synth, mixup, mixup_1, mixup_2, mixup_3, mixup_5, mixup_6, mixup_7, mixup_8, mixup_9])
