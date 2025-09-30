# Modules efficient for inference with caching

import torch
import torch.nn as nn
from einops import rearrange
from rotary_embedding_torch import RotaryEmbedding
from huggingface_hub import PyTorchModelHubMixin

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

class RevIN(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.cached_mean = None
        self.cached_std = None

        self.cached_cumsum_x = None
        self.cached_cumsum_x2 = None
        self.cached_counts = None

    def forward(self, x, mode):
        assert x.dim() == 3, "Input tensor must be (batch, n_patches, patch_len)"

        x64 = x.double()

        if mode == "norm":
            mean, std = self._get_statistics(x64)
            self.cached_mean, self.cached_std = mean[:, -1:].detach(), std[:, -1:].detach()
            out = (x64 - mean) / std
            
            nan_idx = out.isnan()
            if nan_idx.any():
                out = fill_nan_with_last_observed(out)

        elif mode == "denorm_last":
            assert self.cached_mean is not None and self.cached_std is not None, \
                "Call forward(..., 'norm') before 'denorm'"
            out = x64 * self.cached_std + self.cached_mean

        else:
            raise NotImplementedError(f"Mode '{mode}' not implemented.")

        return out.float()

    def _get_statistics(self, x):
        """
        Numerically stable mean and variance computation using 
        incremental mean and variance along the patch dimension.
        x: (B, P, L) float64
        Returns: mean, std (both (B, P, 1))
        """
        B, P, L = x.shape

        nan_counts = torch.isnan(x).sum(-1, keepdim=True)
        nan_counts = torch.cumsum(nan_counts, dim=1)

        counts = torch.arange(1, P+1, device=x.device).view(1, P, 1).repeat(B, 1, 1) * L
        counts = counts - nan_counts
    
        if self.cached_counts is not None:
            counts += self.cached_counts
        self.cached_counts = counts[:, -1:, :]

        cumsum_x = torch.cumsum(x.nansum(dim=-1, keepdim=True), dim=1)
        if self.cached_cumsum_x is not None:
            cumsum_x += self.cached_cumsum_x
        self.cached_cumsum_x = cumsum_x[:, -1:, :]

        mean = cumsum_x / counts

        cumsum_x2 = torch.cumsum((x**2).nansum(dim=-1, keepdim=True), dim=1)
        if self.cached_cumsum_x2 is not None:
            cumsum_x2 += self.cached_cumsum_x2
        self.cached_cumsum_x2 = cumsum_x2[:, -1:, :]

        var = (cumsum_x2 - 2 * mean * cumsum_x + counts * mean**2) / counts
        std = torch.sqrt(var + 1e-5)

        return mean, std
    
    def clear_cache(self):
        self.cached_cumsum_x = None
        self.cached_cumsum_x2 = None
        self.cached_counts = None


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

        self.k_cache = None
        self.v_cache = None

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

        if self.k_cache is not None and self.v_cache is not None:
            offset += self.k_cache.size(2)
            is_causal = False
            k = torch.cat([self.k_cache, k], dim=2)
            v = torch.cat([self.v_cache, v], dim=2)

        self.k_cache = k
        self.v_cache = v

        q = self.rope.rotate_queries_or_keys(q, offset=offset)
        k = self.rope.rotate_queries_or_keys(k)

        values = nn.functional.scaled_dot_product_attention(q, k, v, is_causal=is_causal)

        values = values.transpose(1, 2).reshape(bs, -1, dim)
        values = self.out_proj(values)
        return values
    
    def clear_cache(self):
        self.k_cache = None
        self.v_cache = None
    
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
        self.revin = RevIN()
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