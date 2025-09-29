from dataclasses import dataclass, field, asdict

@dataclass
class PatchFMConfig:
    patch_len: int = 32
    d_model: int = 2048
    n_heads: int = 64
    n_layers_encoder: int = 6
    quantiles: list[float] = field(default_factory=lambda: [0.1 * i for i in range(1, 10)])

    # for inference
    load_from_hub: bool = False
    ckpt_path: str = "./ckpts/huge_v3.pth"
    compile: bool = False

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def to_dict(self):
        return asdict(self)