# A tutorial on how to build a Foundation Model for Univariate Time Series Forecasting

[Huggingface Model Card](https://huggingface.co/vilhess/PatchFM)

A concise, reproducible recipe for training a transformer-based, patch-to-patch forecasting model for univariate time series. The approach mirrors Large Language Model (LLM) practices (next-token → next-patch) while remaining lightweight compared to a classic LLM and practical.

## Highlights
- Next-patch prediction objective (autoregressive, causal)
- Patch-based representation of time series (tokens ↔ patches)
- Causal masking self-attention with RoPE (relative positions)
- RevIN (Reversible Instance Normalization)
- SwiGLU feed-forward networks
- Multi-quantile outputs (median + uncertainty bands)

## Quick Start

### from source code

1. Clone the repository and install dependencies
```bash
git clone https://github.com/vilhess/PatchFM
cd PatchFM
pip install -r requirements.txt
```
2. Run inference with a pretrained model from Huggingface Hub

```python 
import torch
from model import Forecaster
from configs import PatchFMConfig

# --- Instantiate model ---
config = PatchFMConfig(load_from_hub=True)
model = Forecaster(config)

# --- Inference ---
forecast_horizon = 64
seq = torch.randn(1, 1024)  # (batch, time)
pred_median, pred_quantiles = model(seq, forecast_horizon=forecast_horizon, quantiles=[0.1, 0.5, 0.9])  #  (batch, time), (batch, time, quantiles)
```

### from pip package

1. Install the package from PyPI
```bash
pip install patchfm
```
2. Run inference with a pretrained model from Huggingface Hub

```python 
import torch
from patchfm import PatchFMConfig, Forecaster

# same as above
```

We provide an extended quick start example in [notebooks/tutorial.ipynb](./notebooks/tutorial.ipynb).
If you dont have suitable hardware you can run the the extended quick start example example also in Google Colab:

<a target="_blank" href="https://colab.research.google.com/drive/17sdf-7luCkv5TaeLj3Z6kIaTDkwkz3VR?usp=share_link">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open Quick Start In Colab"/> 
</a>

## Method (TL;DR)
- Patching: Split a context signal of length $w$ into $P_{num} = w / P_{len}$ patches of length $P_{len}$.
- Causal RevIN: Normalize input signal and denormalize outputs to the original scale without statistics leakage.
- Architecture: Input residual MLP → stacked Transformer blocks (MHA + SwiGLU FFN, pre-norm, residual) → $|\mathcal{Q}|$ output heads mapping back to patch space.
- Positional encoding: Rotary Position Embeddings (RoPE) applied to queries/keys.
- Training: Multi-quantile (pinball) loss across positions, elements, and quantiles $\mathcal{Q}$.
- Inference: Predict next patch; roll out autoregressively for long horizons.
- KV-cache: during inference, cache keys/values to avoid redundant computations.

## Problem Formulation
Given context patches $x_{p_1}, \ldots, x_{p_n}$, predict the next patch $x_{p_{i+1}}$ for each position $i$ using only past patches (causality). The model outputs quantiles $\{\hat{x}_{p_{i+1}}^{(q)}: q \in \mathcal{Q}\}$ with median (q=0.5) as the point forecast.

## Loss: Multi-Quantile (Pinball)
For residual $u = x - \hat{x}^{(q)}$:
$$\rho_q(u) = \begin{cases} q\,u, & u \ge 0,\\ (q-1)\,u, & u < 0. \end{cases}$$
Aggregate over positions, patch elements, and quantiles.

## Architecture
- Input MLP: $\mathbb{R}^{P_{len}} \to \mathbb{R}^{dim}$ residual 2-layer MLP (ReLU)
- Multi-Head Attention: causal mask, RoPE; queries/keys/values per head
- FFN: SwiGLU (SiLU-gated), pre-norm + residual
- Output heads: |Q| linear maps $\mathbb{R}^{dim} \to \mathbb{R}^{P_{len}}$ (one per quantile)

### Model Details
- Patch size: 32
- Max context: 32 patches (1024 steps)
- Forecast horizon: 32 steps per forward pass
- Quantiles $\mathcal{Q}$: {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9}
- Layers: 6
- Attention heads: 64 (head dim 32)
- Model dim: 2048
- Parameters: ~300M

## Inference
- Single step: predict next patch ($P_{len}$ values)
- Long-horizon: append prediction to context and repeat (optionally drop oldest patch to keep window fixed)

## Datasets
- UTSD (Unified Time Series Dataset) [UTSD]: seven domains (Energy, IoT, Nature, Web, Health, Transport, Environment). We work with UTSD-12G (~18M series after preprocessing).
- GIFT-Eval pretraining dataset [GIFT]: aligned with the GIFT-Eval dataset but without data leakage issue with the benchmark. The dataset contains approximately 71 univariate and 17 multivariate time series datasets from various
domains and various frequencies. After preprocessing, this yields approximately 600K univariate series. 
- Artificial: ~1M synthetic series (sinusoidal, linear, polynomial, logarithmic) plus mixtures via TSMixup [Chronos]; Gaussian Process samples via KernelSynth (mixtures of RBF/periodic/linear kernels with swept hyperparameters).

## Repository Layout

- `model/training/` — main PatchFM model class

  - `modules.py` - core modules (Residual Layers, MHA, SwiGLU, RoPE, Transformer Encoder, ...)
  - `revin.py` — causal RevIN
  - `loss.py` — multi-quantile (pinball) loss
  - `trainer.py` — PyTorch Lightning trainer class

- `model/inference/` — main PatchFM model class for inference 
  - `modules.py` — core modules with caching support
  - `forecaster.py` — Forecasting model and rollout logic

- `dataset/` — data loading and preprocessing
  - `artificial.py` — synthetic dataset : artificial signals + TSMixup + KernelSynth
  - `utsd.py` — Unified Time Series Dataset (UTSD) loading and preprocessing
  - `gift.py` — GIFT-Eval pretraining dataset loading and preprocessing
  - `get_data.py` — utility to fetch and preprocess datasets
  - `generate_data.py` — utility to generate and save the KernelSynth dataset (long to generate)

- `configs/` — model and training configurations
- `notebooks/inference` — how to load a trained model and generate forecasts
- `training.py` — training script using PyTorch Lightning

## Acknowledgements
We thank the authors of the following repositories for inspiration and code snippets:
- [TiRex](https://github.com/NX-AI/tirex)

## Incoming Works

- [ ] Improve performance: extend training duration, tune schedules, and explore larger effective batch sizes.  
- [ ] Data scaling: train on larger corpora (e.g., UTSD-12G) and expand synthetic generators to broaden dynamics and scales.  
- [ ] Benchmarking: evaluate on standard SOTA datasets with common metrics (e.g., MAE/MSE and quantile coverage) to compare against baselines.  
- [ ] Ablations: gradually increase context length, RevIN (on/off and causal variants).  
- [ ] Mixture of Experts: test sparse MoE in FFNs with routing.  
- [ ] Implement LoRA for finetuning.  


## Citation
If you use this work, please cite the paper ...