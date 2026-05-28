import torch

from dataset.artificial import artificial_dataset
from dataset.chronosdata import ChronosDataset, ChronosDataset_mmap
from dataset.gift import GiftEvalDataset
from dataset.mixup import InnerMixUP, InterMixup


def get_dataset(seq_len=1024):
    art_trainset = artificial_dataset(seq_len=seq_len, noise=True)
    gift_trainset = GiftEvalDataset(
        path="data/giftpretrain/", input_len=seq_len, min_stride=32, max_samples=1000
    )
    kernel_synth = ChronosDataset(file_path="data/training_corpus_kernel_synth_1m.npz")
    tsmixup = ChronosDataset_mmap(
        file_path="data/training_corpus_tsmixup_1m.npy",
        file_shape_path="data/training_corpus_tsmixup_1m_shape.npy",
    )
    mixup_1 = InnerMixUP(kernel_synth, K=4, alpha=1.5, n_samples=200_000)
    mixup_2 = InnerMixUP(gift_trainset, K=4, alpha=1.5, n_samples=200_000)
    mixup_3 = InterMixup(
        [art_trainset, gift_trainset], K=4, alpha=1.5, n_samples=200_000
    )
    mixup_4 = InterMixup(
        [tsmixup, gift_trainset], K=4, alpha=1.5, n_samples=200_000
    )
    mixup_5 = InterMixup(
        [kernel_synth, gift_trainset], K=4, alpha=1.5, n_samples=200_000
    )
    mixup_6 = InterMixup(
        [tsmixup, kernel_synth], K=4, alpha=1.5, n_samples=200_000
    )
    mixup_7 = InterMixup(
        [tsmixup, gift_trainset], K=4, alpha=1.5, n_samples=200_000
    ) 


    return torch.utils.data.ConcatDataset(
        [
            art_trainset,
            gift_trainset,
            kernel_synth,
            tsmixup,
            mixup_1,
            mixup_2,
            mixup_3,
            mixup_4,
            mixup_5,
            mixup_6,
            mixup_7,
        ]
    )
