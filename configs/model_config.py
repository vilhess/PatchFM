from dataclasses import dataclass, field

@dataclass
class PatchFMConfig:
    patch_len: int = 32
    d_model: int = 2048
    n_heads: int = 64
    n_layers_encoder: int = 6
    quantiles: list[float] = field(default_factory=lambda: [0.1 * i for i in range(1, 10)])

    # for inference
    ckpt_path: str = "../ckpts/huge_v3.pth"
    compile: bool = True # the first time compilation takes a while but speeds up subsequent inferences