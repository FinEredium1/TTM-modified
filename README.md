# Tiny Time Mixer on FRED-MD (Pretraining + Architecture Change)

This repo contains experiments that evaluate and adapt **Tiny Time Mixers (TTM)** for **multivariate macro time series forecasting** on **FRED-MD**. The main focus is:

- Benchmarking multiple time series forecasting models on FRED-MD
- Pretraining a TTM model on FRED-MD and comparing it against random initialization
- Modifying the TTM architecture by increasing hidden dimensions (64 → 128) and measuring the impact

The work is motivated by the Tiny Time Mixer paper, which proposes a compact, efficient forecasting model built on mixer-style blocks (no transformer self-attention) and supports multivariate plus exogenous modeling via modular fine-tuning components. :contentReference[oaicite:1]{index=1}

---

## What’s inside

### 1) Model benchmarks (3 models)
We benchmarked three time series forecasting models using AutoGluon:

- Chronos (amazon/chronos-bolt-tiny)
- PatchTST
- DeepAR

Experiment setup:
- Dataset: `FRED-MD_2024m12.csv`
- Context window: 512 months (1981-04-01 to 2023-11-01)
- Forecast horizon: 12 months (to 2024-11-01)
- Forecasting is done for all **126** series simultaneously
- Data standard scaled, then converted to PyTorch tensors
- Metrics: RMSE, MAE, MASE

Reported RMSE (lower is better):
- Chronos RMSE: 0.91
- DeepAR RMSE: 0.54
- PatchTST RMSE: 1.08
- WeightedEnsemble RMSE: 0.91 (Chronos in this case)

DeepAR performed best on RMSE in this run. :contentReference[oaicite:2]{index=2}

---

### 2) Pretraining TTM vs random init, plus hidden-dim change
We trained TTM directly on FRED-MD for multivariate forecasting and compared:

- Pretrained TTM
- Random-init TTM (no pretraining)
- Pretrained TTM with architecture change: hidden dims increased from 64 to 128

Experiment setup:
- Dataset: `FRED-MD_2024m12.csv`
- Context window: 60 months (2018-12-01 to 2023-11-01)
- Forecast horizon: 12 months (to 2024-11-01)
- Multivariate forecasting for all **126** series
- Standard scaling
- Config:
  - `input_size = 126`
  - `context_length = 60`
  - `prediction_length = 12`
- Loss: MSE
- Optimizer: Adam
- Training: 50 epochs

Reported results (MSE):
- Pretrained TTM: 0.555197
- Random-init TTM: 1.8264
- Pretrained TTM with hidden dims 128: 0.314932

This shows a large gain from pretraining, and an additional gain from increasing model capacity via hidden dimensions. :contentReference[oaicite:3]{index=3}

---

## Why TTM
TTM is designed to be lightweight and efficient while still transferring well in zero-shot or few-shot settings. It is based on mixer blocks (simple matrix ops) rather than attention, and introduces pretraining-friendly changes like adaptive patching, diverse resolution sampling, and resolution prefix tuning. :contentReference[oaicite:4]{index=4}

Even though the original paper evaluates across standard benchmarks, this repo explores how well the idea works when forecasting **all 126 FRED-MD series simultaneously**, and what happens when we adjust architecture capacity. :contentReference[oaicite:5]{index=5}

---

## Setup

### Requirements
- Python 3.10+ recommended
- PyTorch
- AutoGluon (for DeepAR and PatchTST benchmarks)
- Transformers / HF stack (if loading Chronos)
- Standard DS stack: numpy, pandas, scikit-learn, matplotlib

Example install:
```bash
pip install -r requirements.txt
