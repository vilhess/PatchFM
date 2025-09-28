from dataclasses import dataclass

@dataclass
class TrainConfig:

    seq_len: int = 1024
    epochs: int = 150
    lr: float = 1e-4
    dropout: float = 0.15
    batch_size: int = 256

    checkpoint_path: str = "./ckpts/"
    num_workers: int = 21
    pin_memory: bool = True
    gpus: int = 1
    num_nodes: int = 1
    strategy: str = None