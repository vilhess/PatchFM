from dataclasses import dataclass, field, asdict

@dataclass
class PatchFMConfig:
    patch_len: int = 32
    d_model: int = 2048
    n_heads: int = 64
    n_layers_encoder: int = 6
    quantiles: list[float] = field(default_factory=lambda: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

    # for inference
    load_from_hub: bool = True
    ckpt_path: str = "./ckpts/huge_v7.pth"
    compile: bool = True

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def to_dict(self):
        return asdict(self)