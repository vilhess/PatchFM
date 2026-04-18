import torch

from dataset.artificial import artificial_dataset
from dataset.chronosdata import ChronosDataset, ChronosDataset_mmap
from dataset.gift import GiftEvalPretrain
from dataset.utsd import UTSDataset
from dataset.mixup import InnerMixUP, InterMixup


def get_dataset(seq_len=1024):
    art_trainset = artificial_dataset(seq_len=seq_len, noise=True)
    utsd_trainset = UTSDataset(
        input_len=seq_len, flag="train", split=0.9, min_stride=32, max_samples=1000
    )
    gift_trainset = GiftEvalPretrain(
        path="data/giftpretrain/", input_len=seq_len, min_stride=32, max_samples=1000
    )
    kernel_synth = ChronosDataset(file_path="data/training_corpus_kernel_synth_1m.npz")
    tsmixup = ChronosDataset_mmap(
        file_path="data/training_corpus_tsmixup_1m.npy",
        file_shape_path="data/training_corpus_tsmixup_1m_shape.npy",
    )
    mixup_1 = InnerMixUP(kernel_synth, K=4, alpha=1.5, n_samples=200_000)
    mixups_2 = InnerMixUP(utsd_trainset, K=4, alpha=1.5, n_samples=200_000)
    mixup_3 = InnerMixUP(gift_trainset, K=4, alpha=1.5, n_samples=200_000)
    mixup_4 = InterMixup([art_trainset, utsd_trainset], K=4, alpha=1.5, n_samples=200_000)
    mixup_5 = InterMixup([art_trainset, gift_trainset], K=4, alpha=1.5, n_samples=200_000)

    return torch.utils.data.ConcatDataset(
        [art_trainset, gift_trainset, kernel_synth, tsmixup, utsd_trainset, mixup_1, mixups_2, mixup_3, mixup_4, mixup_5]
    )
