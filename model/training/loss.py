import torch
import torch.nn as nn

class MultiQuantileLoss(nn.Module):
    def __init__(self, quantiles):
        super().__init__()

        if not isinstance(quantiles, torch.Tensor):
            quantiles = torch.tensor(quantiles)

        assert all(0 < q < 1 for q in quantiles), "Quantiles must be in the range (0, 1)"
        self.quantiles = quantiles

    def forward(self, pred, target):
        assert pred.shape[-1] == len(self.quantiles)
        assert target.shape[1] == pred.shape[1] # n_patches
        assert target.shape[2] == pred.shape[2] # patch_len
        self.quantiles = self.quantiles.to(pred.device)
        target = target.unsqueeze(-1) 
        errors = target - pred
        losses = torch.max((self.quantiles - 1) * errors, self.quantiles * errors)
        return losses.mean()