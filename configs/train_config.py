from dataclasses import dataclass, asdict

@dataclass
class TrainConfig:

    seq_len: int = 1024
    epochs: int = 150
    start_lr: float = 1e-5
    max_lr: float = 5e-4
    lower_lr: float = 1e-5
    reach_max: int = 10_000
    iter_cycle: int = 100_000
    dropout: float = 0.1
    batch_size: int = 256

    checkpoint_path: str = "./ckpts/"
    num_workers: int = 21
    pin_memory: bool = True
    gpus: int = 1
    num_nodes: int = 1
    strategy: str = "auto"

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def to_dict(self):
        return asdict(self)