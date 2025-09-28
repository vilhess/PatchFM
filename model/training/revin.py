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

        # Cast to float64 for stable statistics computation
        x64 = x.double()

        if mode == "norm":
            mean, std = self._get_statistics(x64)
            self.cached_mean, self.cached_std = mean.detach(), std.detach()
            out = (x64 - mean) / std

        elif mode == "denorm":
            assert self.cached_mean is not None and self.cached_std is not None, \
                "Call forward(..., 'norm') before 'denorm'"
            out = x64 * self.cached_std + self.cached_mean

        else:
            raise NotImplementedError(f"Mode '{mode}' not implemented.")

        # Convert back to float32 for compatibility with main model
        return out.float()

    def _get_statistics(self, x):
        """
        Numerically stable mean and variance computation using 
        incremental mean and variance along the patch dimension.
        x: (B, P, L) float64
        Returns: mean, std (both (B, P, 1))
        """
        B, P, L = x.shape
        counts = torch.arange(1, P+1, device=x.device).view(1, P, 1) * L

        # Incrementally compute mean
        cumsum_x = torch.cumsum(x.sum(dim=-1, keepdim=True), dim=1)
        mean = cumsum_x / counts

        # Variance: mean of squared deviations from the mean
        # Efficient incremental formula:
        # var_i = (sum(x^2) - 2*mean*sum(x) + count*mean^2)/count
        cumsum_x2 = torch.cumsum((x**2).sum(dim=-1, keepdim=True), dim=1)
        var = (cumsum_x2 - 2 * mean * cumsum_x + counts * mean**2) / counts
        std = torch.sqrt(var + self.eps)

        return mean, std