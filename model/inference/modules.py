# Modules efficient for inference with caching

import torch
import torch.nn as nn
from einops import rearrange
from rotary_embedding_torch import RotaryEmbedding
from huggingface_hub import PyTorchModelHubMixin
import numpy as np

class SeqTypeConverter:
    def __init__(self):
        self.init_type = None
    
    def convert(self, seq):
        if isinstance(seq, torch.Tensor):
            self.init_type = 'torch'
            return seq
        
        elif isinstance(seq, np.ndarray):
            self.init_type = 'numpy'
            return torch.from_numpy(seq)
        
        elif isinstance(seq, list):
            if all(isinstance(x, torch.Tensor) for x in seq):
                self.init_type = 'list_of_tensors'
                try:
                    return torch.stack(seq)
                except Exception:
                    raise ValueError("All tensors in the list must have the same shape to stack.")
            else:
                self.init_type = 'list'
                return torch.tensor(seq)
        
        else:
            raise ValueError(f"Unsupported type: {type(seq)}")

    def deconvert(self, seq, quantiles):
        seq = seq.detach().cpu()
        quantiles = quantiles.detach().cpu()
        
        if self.init_type == 'torch':
            return self._ensure_torch(seq), self._ensure_torch(quantiles)
        
        elif self.init_type == 'numpy':
            return self._ensure_numpy(seq), self._ensure_numpy(quantiles)
        
        elif self.init_type == 'list':
            return seq.tolist(), quantiles.tolist()
        
        elif self.init_type == 'list_of_tensors':
            seqs = list(seq.unbind(0))
            quants = list(quantiles.unbind(0))
            return seqs, quants
        
        else:
            raise ValueError(f"Unsupported type: {self.init_type}")
    
    def _ensure_torch(self, x):
        return x if isinstance(x, torch.Tensor) else torch.tensor(x)

    def _ensure_numpy(self, x):
        return x if isinstance(x, np.ndarray) else np.array(x)

def fill_nan_with_last_observed(x):
    bs, pn, pl = x.size()
    x = rearrange(x, "b pn pl -> (b pn) pl")
    valid_mask = ~torch.isnan(x)
    x_temp = torch.where(valid_mask, x, torch.zeros_like(x))
    seq_indices = torch.arange(x.size(-1), device=x.device).unsqueeze(0)
    valid_indices = torch.where(valid_mask, seq_indices, torch.tensor(-1, device=x.device))
    last_valid_idx = torch.cummax(valid_indices, dim=-1)[0]
    x = x_temp.gather(-1, torch.clamp(last_valid_idx, min=0))
    x = rearrange(x, "(b pn) pl -> b pn pl", b=bs)
    return x

def nanstd(o, dim, keepdim=False):
    m = torch.nanmean(o, dim=dim, keepdim=True)
    sq = (o - m) ** 2
    n = torch.sum(~torch.isnan(o), dim=dim, keepdim=True).float()
    n_safe = torch.clamp(n - 1, min=1.0)
    var = torch.nansum(sq, dim=dim, keepdim=True) / n_safe
    std = torch.sqrt(var)
    if not keepdim:
        std = std.squeeze(dim)
    return std

class CausalRevIN(nn.Module):
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
            out = torch.asinh(out)

        elif mode == "denorm":
            assert self.cached_mean is not None and self.cached_std is not None, \
                "Call forward(..., 'norm') before 'denorm'"
            out = torch.sinh(x64) * self.cached_std[:, -1:, :] + self.cached_mean[:, -1:, :]

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
    
    def clear_cache(self):
        self.cached_mean = None
        self.cached_std = None

class ResidualBlock(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super().__init__()
        self.hidden_layer = nn.Linear(in_dim, hid_dim)
        self.output_layer = nn.Linear(hid_dim, out_dim)
        self.residual_layer = nn.Linear(in_dim, out_dim)
        self.act = nn.ReLU()

    def forward(self, x):
        hid = self.act(self.hidden_layer(x))
        out = self.output_layer(hid)
        res = self.residual_layer(x)
        out = out+res
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, last=False):
        super().__init__()
        assert d_model%n_heads==0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"

        self.WQ = nn.Linear(d_model, d_model)
        self.WK = nn.Linear(d_model, d_model)
        self.WV = nn.Linear(d_model, d_model)

        self.out_proj = nn.Linear(d_model, d_model)

        self.head_dim = d_model//n_heads
        self.n_heads = n_heads

        self.rope = RotaryEmbedding(dim=self.head_dim//2)

        self.last = last
    
    def forward(self, q):
        bs, context, dim = q.size()
        offset = 0
        is_causal = True

        k = q
        v = q

        if self.last:
            q = q[:, -1:, :]
            is_causal = False
            offset += (context - 1)

        q = self.WQ(q).reshape(bs, -1, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.WK(k).reshape(bs, -1, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.WV(v).reshape(bs, -1, self.n_heads, self.head_dim).transpose(1, 2)

        q = self.rope.rotate_queries_or_keys(q, offset=offset)
        k = self.rope.rotate_queries_or_keys(k)

        values = nn.functional.scaled_dot_product_attention(q, k, v, is_causal=is_causal)

        values = values.transpose(1, 2).reshape(bs, -1, dim)
        values = self.out_proj(values)
        return values
    
class FeedForward(nn.Module):
    def __init__(self, d_model, multiple_of=256):
        super().__init__()

        hidden_dim = d_model*4
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, d_model, bias=False)
        self.w3 = nn.Linear(d_model, hidden_dim, bias=False)

        self.act = nn.SiLU()

    def forward(self, x):
        x = self.w2(self.act(self.w1(x)) * self.w3(x))
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, last=False):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model=d_model, n_heads=n_heads, last=last)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model=d_model)
    
    def forward(self, x):
        out_attn = self.attn(self.ln1((x)))
        x = x + out_attn
        out = x + self.ff(self.ln2(x))
        return out
    
class TransformerEncoder(nn.Module):
    def __init__(self, d_model, n_heads, n_layers):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(d_model=d_model, n_heads=n_heads)
                for _ in range(n_layers-1)
            ]
        )
        self.layers.append(TransformerEncoderLayer(d_model=d_model, n_heads=n_heads, last=True))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)
    
class PatchFM(nn.Module, PyTorchModelHubMixin): 
    def __init__(self, config):
        super().__init__()

        # Store config
        self.patch_len = config["patch_len"]
        self.d_model = config["d_model"]
        self.n_heads = config["n_heads"]
        self.n_layers_encoder = config["n_layers_encoder"]
        self.quantiles = config["quantiles"]
        self.n_quantiles = len(self.quantiles)

        # Components
        self.revin = CausalRevIN()
        self.proj_embedding = ResidualBlock(
            in_dim=self.patch_len, 
            hid_dim=2 * self.patch_len, 
            out_dim=self.d_model
        )
        self.transformer_encoder = TransformerEncoder(
            d_model=self.d_model, 
            n_heads=self.n_heads, 
            n_layers=self.n_layers_encoder
        )
        self.proj_output = ResidualBlock(
            in_dim=self.d_model, 
            hid_dim=2 * self.d_model, 
            out_dim=self.patch_len * self.n_quantiles
        )