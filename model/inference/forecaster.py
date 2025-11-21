import torch
import torch.nn as nn
from einops import rearrange
from model.inference.modules import RevIN, ResidualBlock, TransformerEncoder, PatchFM, SeqTypeConverter


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
            base_model = PatchFM.from_pretrained("vilhess/PatchFM")
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

    def _init_from_base(self, base_model):
        """Initialize modules by reusing a pretrained PatchFM model."""
        self.revin = base_model.revin
        self.proj_embedding = base_model.proj_embedding
        self.transformer_encoder = base_model.transformer_encoder
        self.proj_output = base_model.proj_output
    
    @torch.inference_mode()
    def forecast(self, x: torch.Tensor, forecast_horizon: int | None = None, quantiles: list[float] | None = None) -> torch.Tensor: 
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

        # Default horizon = patch_len
        forecast_horizon = forecast_horizon or self.patch_len

        # Reshape into patches
        x = rearrange(x, "b (pn pl) -> b pn pl", pl=self.patch_len)  

        rollouts = -(-forecast_horizon // self.patch_len)  # ceil division
        predictions = []

        for _ in range(rollouts):

            if x.size(1) > self.max_patches:
                x = x[:, -self.max_patches:, :]
                
            init_x = x.clone()
            # Forward pass
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
            
            # Take median quantile (index 4)
            patch_median = forecasting[:, -1:, :, 4].detach()
            predictions.append(forecasting[:, -1, :, :])

            # Append median patch for next rollout
            x = patch_median.clone()
            x = torch.cat([init_x, x], dim=1)

        
        pred_quantiles = torch.cat(predictions, dim=1)
        pred_quantiles = pred_quantiles[:, :forecast_horizon, :]
        pred_median = pred_quantiles[:, :, 4]

        pred_quantiles = pred_quantiles[..., [self.quantiles.index(q) for q in quantiles]] if quantiles is not None else pred_quantiles

        self.clear_cache()

        if torch.any(torch.isnan(pred_median)) or torch.any(torch.isinf(pred_median)):
            print("Warning: NaN or Inf values detected in predictions. Returning zeros.")
            pred_median = torch.zeros_like(pred_median)
            pred_quantiles = torch.zeros_like(pred_quantiles)
        
        if not batch_dim:
            pred_median = pred_median.squeeze(0)
            pred_quantiles = pred_quantiles.squeeze(0)   

        pred_median, pred_quantiles = self.converter.deconvert(pred_median, pred_quantiles)
        return pred_median, pred_quantiles

    def __call__(self, context: torch.Tensor, forecast_horizon: int | None = None, quantiles: list[float] | None = None) -> torch.Tensor:
        return self.forecast(context, forecast_horizon, quantiles)
    
    def clear_cache(self):
        self.revin.clear_cache()    