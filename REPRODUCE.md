# Reproducibility guide

Everything needed to rebuild the experiment from scratch with the same
numbers. One command does the whole cycle:

```powershell
.\reproduce.ps1          # all 5 tickers: NASDAQ SPY500 WIG20 INVESCO GLD
.\reproduce.ps1 SPY500   # single ticker
```

The script: builds and starts the stack (`docker compose up -d --build`),
waits for the ML service, retrains each instrument end-to-end
(Optuna + final training), and restarts the serving layer so the fresh
artifacts are picked up. ~10 minutes per ticker.

## What makes the results reproducible

| Element | Mechanism | Where |
|---|---|---|
| Global seed | `RANDOM_SEED = 42`, `set_seed()` (Python `random`, NumPy, Torch, CUDA) | `engineer-backend-llm/src/config.py`, called at the start of `run_pipeline.main()` |
| Optuna search | `TPESampler(seed=42)` for both the NN and the RF studies | `src/tune_nn.py`, `src/tune_rf.py` |
| Library versions | pinned in `requirements.txt` (fastapi, numpy, pandas, scikit-learn, joblib, optuna, ...) | `engineer-backend-llm/requirements.txt` |
| PyTorch | pinned CPU-only build `torch==2.15.0+cpu` | `engineer-backend-llm/Dockerfile` |
| Frontend stack | pinned (streamlit, pandas, numpy, plotly, requests) | `frontend/requirements.txt` |
| Base image | `python:3.12-slim` | `engineer-backend-llm/Dockerfile` |

## Sanity checks for the statistics

```powershell
docker compose exec frontend python /app/test_statistical_tests.py
```

Monte-Carlo validation (4000 simulations, seed 42): under H0 both the
Diebold-Mariano test and the one-sided binomial DA test reject at ~5%
(measured: 5.25% and 3.7-4.1%), a clearly worse predictor is rejected,
and edge cases (identical predictions, k = 0, k = n, p0 = 1) behave as
documented.

## Serving contract

- Every `/evaluate` and `/predict` call must name the ticker; the legacy
  shared-artifact fallback is disabled (missing per-ticker artifacts fail
  loudly with HTTP 503 instead of silently serving wrong weights).
- Kalman hyperparameters (Q, R) are estimated from training data only:
  R = variance of training returns, Q by one-step-ahead grid search on
  the last 20% of the training segment. Chosen values are printed by the
  pipeline and returned in the `/evaluate` response (`kalman_params`).

## Data

Training CSVs are committed under `engineer-backend-llm/data/` (one file
per ticker). The Postgres volume that feeds the dashboard is NOT wiped
by `reproduce.ps1`; to refresh it from Tiingo, call the Java backend's
sync endpoint for each symbol.
