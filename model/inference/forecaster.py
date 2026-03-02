import torch
import torch.nn as nn
from einops import rearrange
from model.inference.modules import CausalRevIN, ResidualBlock, TransformerEncoder, PatchFM, SeqTypeConverter
from model.inference.utils import flip_last_dim


# --- Forecaster Model ---
class Forecaster(nn.Module): 
    def __init__(self, config):
        super().__init__()

        # Store config
        self.max_seq_len = config["max_seq_len"]
        self.patch_len = config["patch_len"]
        self.d_model = config["d_model"]
        self.n_heads = config["n_heads"]
        self.n_layers_encoder = config["n_layers_encoder"]
        self.quantiles = config["quantiles"]
        self.n_quantiles = len(self.quantiles)
        self.max_patches = self.max_seq_len // self.patch_len

        assert config["load_from_hub"] or config["ckpt_path"] is not None, (
            "Either load_from_hub must be True or ckpt_path must be provided."
        )

        # Load weights either from HF Hub or local checkpoint
        if config["load_from_hub"]:
            print("Loading base model from HuggingFace Hub...")
            base_model = PatchFM.from_pretrained("vilhess/PatchFM-CausalRevIN-asinh")
            self._init_from_base(base_model)
        else:
            print(f"Loading weights from local ckpt: {config['ckpt_path']}")
            self._init_components()
            state = torch.load(config["ckpt_path"], weights_only=True)
            self.load_state_dict(state, strict=False)

        self.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

        self.converter = SeqTypeConverter()

        if config["compile"]:
            self = torch.compile(self)

    def _init_components(self):
        """Initialize modules from scratch."""
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

    def _init_from_base(self, base_model):
        """Initialize modules by reusing a pretrained PatchFM model."""
        self.revin = base_model.revin
        self.proj_embedding = base_model.proj_embedding
        self.transformer_encoder = base_model.transformer_encoder
        self.proj_output = base_model.proj_output
    
    @torch.inference_mode()
    def auto_regressive_quantile_decoding(self, x: torch.Tensor, forecast_horizon: int | None = None, quantiles: list[float] | None = None) -> torch.Tensor: 

        q = torch.tensor(self.quantiles, device=self.device)

        # Default horizon = patch_len
        forecast_horizon = forecast_horizon or self.patch_len

        x = self.converter.convert(x)
        assert x.ndim in (1, 2), f"Input dimension must be 1D (time) or 2D (batch, time), got {x.ndim}D."

        batch_dim=True
        if x.ndim != 2:
            x = x.unsqueeze(0)
            batch_dim=False
        bs, ws = x.size()

        x = x.to(self.device)

        if ws > self.max_seq_len:
            print(f"Warning: Input length {ws} exceeds max_seq_len {self.max_seq_len}. Truncating input.")
            x = x[:, -self.max_seq_len:]
            ws = self.max_seq_len

        # Pad so length is divisible by patch_len
        pad = (self.patch_len - ws % self.patch_len) % self.patch_len
        if pad > 0:
            x = torch.cat([x[:, :1].repeat(1, pad), x], dim=1)

        # Reshape into patches
        x = rearrange(x, "b (pn pl) -> b pn pl", pl=self.patch_len) 

        rollouts = -(-forecast_horizon // self.patch_len)  # ceil division
        predictions = []

        # 1st Forward pass
        x = self.revin(x, mode="norm")
        x = self.proj_embedding(x)
        x = self.transformer_encoder(x)
        x = x[:, -1:, :]  # Keep only the last patch for autoregressive forecasting

        forecasting = self.proj_output(x)
        forecasting = self.revin(forecasting, mode="denorm")

        # Reshape to (bs, patch_num, patch_len, n_quantiles)
        forecasting = rearrange(
            forecasting, "b 1 (pl q) -> b 1 pl q", 
            pl=self.patch_len, q=self.n_quantiles
        )     
        x = forecasting.permute(0, 3, 1, 2).reshape(forecasting.size(0)*self.n_quantiles, 1, self.patch_len)

        predictions.append(forecasting[:, -1, :, :].detach())

        for _ in range(rollouts-1):
                
            # Forward pass
            x = self.revin(x, mode="norm")
            x = self.proj_embedding(x)
            x = self.transformer_encoder(x)
            x = x[:, -1:, :]  # Keep only the last patch for autoregressive forecasting
            forecasting = self.proj_output(x)
            forecasting = self.revin(forecasting, mode="denorm")

            # Reshape to (bs*n_quantiles, patch_num, patch_len, n_quantiles)
            forecasting = rearrange(
                forecasting, "b 1 (pl q) -> b 1 pl q", 
                pl=self.patch_len, q=self.n_quantiles
            )

            forecasting = rearrange(
                forecasting, "(b q) 1 pl h -> b q 1 pl h", 
                q=self.n_quantiles
            )
            forecasting = forecasting.permute(0, 2, 3, 1, 4).flatten(start_dim=-2)  # batch x 1 x patch_len x n_quantiles**2
            forecasting = torch.quantile(forecasting, q, dim=-1) # n_quantiles x batch x 1 x patch_len
            
            x = forecasting.permute(1, 0, 2, 3).reshape(-1, 1, self.patch_len)
            predictions.append(forecasting.permute(1, 2, 3, 0)[:, 0].detach())

        self.clear_cache()

        pred_quantiles = torch.cat(predictions, dim=1)
        pred_quantiles = pred_quantiles[:, :forecast_horizon, :]
        pred_median = pred_quantiles[:, :, 4]

        pred_quantiles = pred_quantiles[..., [self.quantiles.index(q) for q in quantiles]] if quantiles is not None else pred_quantiles

        if torch.any(torch.isnan(pred_median)) or torch.any(torch.isinf(pred_median)):
            print("Warning: NaN or Inf values detected in predictions. Returning zeros.")
            pred_median = torch.zeros_like(pred_median)
            pred_quantiles = torch.zeros_like(pred_quantiles)
        
        if not batch_dim:
            pred_median = pred_median.squeeze(0)
            pred_quantiles = pred_quantiles.squeeze(0)   

        pred_median, pred_quantiles = self.converter.deconvert(pred_median, pred_quantiles)
        return pred_median, pred_quantiles

    def __call__(self, context: torch.Tensor, forecast_horizon: int | None = None, quantiles: list[float] | None = None, flip_equivariance: bool = False) -> torch.Tensor:
        if flip_equivariance:
            print("Flip equivariance enabled: forecast = (f(x) - f(-x)) / 2. This requires multiplying by 2 the batch size (Reverso: Efficient Time Series Foundation Models for Zero-shot Forecasting).")
            bs = context.size(0)
            context_flipped = -context
            concat_context = torch.cat([context, context_flipped], dim=0)
            pred_median_full, pred_quantiles_full = self.auto_regressive_quantile_decoding(concat_context, forecast_horizon, quantiles)
            pred_median, pred_quantiles = pred_median_full[:bs], pred_quantiles_full[:bs]
            pred_median2, pred_quantiles2 = pred_median_full[bs:], pred_quantiles_full[bs:]
            pred_median = (pred_median - pred_median2) / 2
            pred_quantiles = (pred_quantiles - flip_last_dim(pred_quantiles2)) / 2
        else:
            pred_median, pred_quantiles = self.auto_regressive_quantile_decoding(context, forecast_horizon, quantiles)
        return pred_median, pred_quantiles
    
    def clear_cache(self):
        self.revin.clear_cache()    
        for layer in self.transformer_encoder.layers:
            layer.attn.clear_cache()  