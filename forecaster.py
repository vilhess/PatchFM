from dataclasses import dataclass, field
import torch
import torch.nn as nn
from einops import rearrange

from model import RevIN, ResidualBlock, TransformerEncoder

# --- Configuration ---
@dataclass
class PatchFMConfig:
    patch_len: int = 32
    d_model: int = 1024
    n_heads: int = 32
    n_layers_encoder: int = 6
    quantiles: list[float] = field(default_factory=lambda: [0.1 * i for i in range(1, 10)])
    ckpt_path: str = "../ckpts/pretrained_patchfm.pth"


# --- Forecaster Model ---
class Forecaster(nn.Module): 
    def __init__(self, config: PatchFMConfig):
        super().__init__()

        # Store config
        self.patch_len = config.patch_len
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.n_layers_encoder = config.n_layers_encoder
        self.quantiles = config.quantiles
        self.n_quantiles = len(config.quantiles)

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

        # Load pretrained weights if available
        self.load_state_dict(
            torch.load(config.ckpt_path, weights_only=True), 
            strict=False
        )
        self.eval()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
    
    @torch.no_grad()
    def forecast(self, x: torch.Tensor, forecast_horizon: int | None = None, quantiles: list[float] | None = None) -> torch.Tensor: 
        x = x.to(self.device)
        # Ensure input shape (bs, length)
        if x.ndim != 2:
            x = x.unsqueeze(0)
        bs, ws = x.size()

        # Pad so length is divisible by patch_len
        pad = (self.patch_len - ws % self.patch_len) % self.patch_len
        if pad > 0:
            x = torch.cat([x[:, :1].repeat(1, pad), x], dim=1)

        # Default horizon = patch_len
        forecast_horizon = forecast_horizon or self.patch_len

        # Reshape into patches
        x = rearrange(x, "b (pn pl) -> b pn pl", pl=self.patch_len)  

        rollouts = -(-forecast_horizon // self.patch_len)  # ceil division
        predictions = []

        for _ in range(rollouts):
            x_input = x.clone()

            # Forward pass
            x = self.revin(x, mode="norm")
            x = self.proj_embedding(x)
            x = self.transformer_encoder(x)
            x = x[:, -1:, :]  # Keep only the last patch for autoregressive forecasting
            forecasting = self.proj_output(x)
            forecasting = self.revin(forecasting, mode="denorm_last")

            # Reshape to (bs, patch_num, patch_len, n_quantiles)
            forecasting = rearrange(
                forecasting, "b 1 (pl q) -> b 1 pl q", 
                pl=self.patch_len, q=self.n_quantiles
            )
            
            # Take median quantile (index 4)
            patch_median = forecasting[:, -1:, :, 4].detach()
            predictions.append(forecasting[:, -1, :, :])

            # Append median patch for next rollout
            x = torch.cat([x_input, patch_median], dim=1)
        
        pred_quantiles = torch.cat(predictions, dim=1)
        pred_quantiles = pred_quantiles[:, :forecast_horizon, :]
        pred_median = pred_quantiles[:, :, 4]

        pred_quantiles = pred_quantiles[..., [self.quantiles.index(q) for q in quantiles]] if quantiles is not None else pred_quantiles

        return pred_median, pred_quantiles

    def __call__(self, context: torch.Tensor, forecast_horizon: int | None = None, quantiles: list[float] | None = None) -> torch.Tensor:
        return self.forecast(context, forecast_horizon, quantiles)
    
    def anomaly_scorer(self, x):
        if x.ndim == 1:
            x = x.unsqueeze(0)
        ctx = x[:, :-self.patch_len]
        target = x[:, -self.patch_len:]

        median, _ = self(ctx, forecast_horizon=self.patch_len, quantiles=[0.5])
        median = median.detach().cpu()
        loss = torch.nn.functional.mse_loss(median, target).detach().cpu()
        return loss.numpy()
    

# --- Plotting Utility ---
import matplotlib.pyplot as plt
def plot_forecast(context, median, quantiles=None, target_pred=None, context_plot_limit=100):

    if median.ndim>1:
        assert median.shape[0]==1
        median = median[0]
    if quantiles is not None and quantiles.ndim>2:
        assert quantiles.shape[0]==1
        quantiles = quantiles[0]
    if target_pred is not None and target_pred.ndim>1:
        assert target_pred.shape[0]==1
        target_pred = target_pred[0]
    if context.ndim>1:
        assert context.shape[0]==1
        context = context[0]
    
    if quantiles is not None:
        assert quantiles.shape[1]==3, "Quantiles should be of shape (L, 3)"
        print(f"Warning: Currently quantiles works only for 3 quantiles (lower, median, upper), got {quantiles.shape[1]}")

    if median.device != torch.device('cpu'):
        median = median.cpu()
    if quantiles is not None and quantiles.device != torch.device('cpu'):
        quantiles = quantiles.cpu()
    if target_pred is not None and target_pred.device != torch.device('cpu'):
        target_pred = target_pred.cpu()
    if context.device != torch.device('cpu'):
        context = context.cpu()

    plt.figure(figsize=(12, 6))
    plt.plot(range(len(context))[-context_plot_limit:], context[-context_plot_limit:], label='Context', color='blue')
    plt.plot(range(len(context), len(context) + len(median)), median, label='Median Forecast', color='orange')
    if quantiles is not None:
        plt.fill_between(range(len(context), len(context) + len(median)), quantiles[:, 0], quantiles[:, 2], color='orange', alpha=0.3, label='Quantiles')
    if target_pred is not None:
        plt.plot(range(len(context), len(context) + len(target_pred)), target_pred, label='Target', color='green', linestyle='--')
    plt.axvline(x=len(context)-1, color='black', linestyle='--', label='Forecast Start')
    plt.legend()
    plt.title('Forecasting with PatchFM')
    plt.xlabel('Time Steps')
    plt.ylabel('Value')
    plt.show()