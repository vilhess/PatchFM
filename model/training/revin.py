import torch
import torch.nn as nn

class RevIN(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.cached_mean = None
        self.cached_std = None

    def forward(self, x, mode: str):
        assert x.dim() == 3, "Input tensor must be (batch, n_patches, patch_len)"

        if mode == "norm":
            mean, std = self._get_statistics(x)
            self.cached_mean, self.cached_std = mean.detach(), std.detach()
            out = (x - mean) / std
            out = torch.asinh(out)

        elif mode == "denorm":
            assert self.cached_mean is not None and self.cached_std is not None, \
                "Call forward(..., 'norm') before 'denorm'"
            out = torch.sinh(x) * self.cached_std + self.cached_mean
            
        else:
            raise NotImplementedError(f"Mode '{mode}' not implemented.")
        return out

    def _get_statistics(self, x):
        mean = x.mean(dim=(-1, -2), keepdim=True)
        std = x.std(dim=(-1, -2), keepdim=True) + self.eps
        return mean, std