import torch
import torch.nn as nn
from einops import rearrange
import lightning as L
import torch.optim as optim
from rotary_embedding_torch import RotaryEmbedding

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

        elif mode == "denorm_last":
            assert self.cached_mean is not None and self.cached_std is not None, \
                "Call forward(..., 'norm') before 'denorm'"
            out = x64 * self.cached_std[:, -1:] + self.cached_mean[:, -1:]

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

class ResidualBlock(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout=0.):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
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
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model%n_heads==0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"

        self.WQ = nn.Linear(d_model, d_model)
        self.WK = nn.Linear(d_model, d_model)
        self.WV = nn.Linear(d_model, d_model)

        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = dropout

        self.head_dim = d_model//n_heads
        self.n_heads = n_heads

        self.rope = RotaryEmbedding(dim=self.head_dim//2)
    
    def forward(self, q):
        bs, context, dim = q.size()

        k = q
        v = q

        q = self.WQ(q).reshape(bs, -1, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.WK(k).reshape(bs, -1, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.WV(v).reshape(bs, -1, self.n_heads, self.head_dim).transpose(1, 2)

        q  = self.rope.rotate_queries_or_keys(q)
        k = self.rope.rotate_queries_or_keys(k)

        values = nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )

        values = values.transpose(1, 2).reshape(bs, -1, dim)
        values = self.out_proj(values)
        return values
    
class FeedForward(nn.Module):
    def __init__(self, d_model, dropout=0.1, multiple_of=256):
        super().__init__()

        hidden_dim = d_model*4
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, d_model, bias=False)
        self.w3 = nn.Linear(d_model, hidden_dim, bias=False)

        self.act = nn.SiLU()
        self.dp = nn.Dropout(dropout)

    def forward(self, x):
        x = self.w2(self.act(self.w1(x)) * self.w3(x))
        return self.dp(x)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model=d_model, n_heads=n_heads, dropout=dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model=d_model, dropout=dropout)
    
    def forward(self, x):
        out_attn = self.attn(self.ln1((x)))
        x = x + out_attn
        out = x + self.ff(self.ln2(x))
        return out
    
class TransformerEncoder(nn.Module):
    def __init__(self, d_model, n_heads, n_layers, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(d_model=d_model, n_heads=n_heads, dropout=dropout)
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

class PatchFM(nn.Module): 
    def __init__(self, seq_len, patch_len, d_model, n_heads, n_layers_encoder, dropout=0.1, quantiles=None):
        super().__init__()
        assert seq_len%patch_len==0, f"seq_len ({seq_len}) should be divisible by patch_len ({patch_len})"
        
        self.seq_len = seq_len
        self.patch_len = patch_len

        self.quantiles = quantiles if quantiles is not None else [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        self.n_quantiles = len(self.quantiles)

        self.revin = RevIN()

        self.proj_embedding = ResidualBlock(in_dim=patch_len, hid_dim=2*patch_len, out_dim=d_model, dropout=dropout)
        self.dp = nn.Dropout(dropout)
        self.transformer_encoder = TransformerEncoder(d_model=d_model, n_heads=n_heads, n_layers=n_layers_encoder, dropout=dropout)

        self.proj_output = ResidualBlock(in_dim=d_model, hid_dim=2*d_model, out_dim=patch_len * self.n_quantiles, dropout=dropout)
        #self.proj_output = ResidualBlock(in_dim=d_model, hid_dim=2*d_model, out_dim=patch_len, dropout=dropout)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                m.bias.data.fill_(0.0)
                m.weight.data.fill_(1.0)
    
    def forward(self, x): 
        bs, ws = x.size()

        x = rearrange(x, "b (pn pl) -> b pn pl", pl=self.patch_len)  # Reshape to (bs, patch_num, patch_len)
        if self.training:
            x_patch = x[:, 1:, :].clone().detach()
        x = self.revin(x, mode="norm")

        x = self.proj_embedding(x) # bs, pn, d_model
        x = self.dp(x)
        x = self.transformer_encoder(x) # bs, pn, d_model

        forecasting = self.proj_output(x)  # bs, pn, patch_len * n_quantiles

        forecasting = self.revin(forecasting, mode="denorm")

        forecasting = rearrange(forecasting, "b pn (pl q) -> b pn pl q", pl=self.patch_len, q=self.n_quantiles)  # Reshape to (bs, patch_len, n_quantiles)

        if self.training:
            return forecasting, x_patch
        else:
            return forecasting

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

class PatchFMLit(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        
        assert config.epochs >= config.epochs_warmups, "epochs must be greater than epochs_warmups"
        assert config.n_warmups > 0, "n_warmups must be greater than 0"
        assert config.epochs_warmups % config.n_warmups == 0, "number of warmups epochs must be divisible by n_warmups"

        self.model = PatchFM(
            seq_len=config.ws, 
            patch_len=config.patch_len, 
            d_model=config.d_model, 
            n_heads=config.n_heads, 
            n_layers_encoder=config.n_layers_encoder, 
            dropout=config.dropout, 
            quantiles=config.quantiles
        )

        self.criterion = MultiQuantileLoss(self.model.quantiles)
        
        self.ctx = [config.ws // config.n_warmups * i for i in range(1, config.n_warmups + 1)]
        self.n_epochs = config.epochs_warmups // config.n_warmups

        self.save_hyperparameters(config)

    def training_step(self, batch, batch_idx):
        x, y = batch

        current_epoch = self.current_epoch
        ctx = self.ctx[current_epoch // self.n_epochs if current_epoch < len(self.ctx) * self.n_epochs else -1]
        x = x[:, -ctx:]

        prediction, x_patch = self.model(x)
        y = y.unsqueeze(1)
        y = torch.cat([x_patch, y], dim=1)    
        loss = self.criterion(prediction, y)
        self.log("train_loss", loss, sync_dist=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=1e-6)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.epochs, eta_min=0)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
