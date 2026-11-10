# -*- coding: utf-8 -*-
"""
ETF Prediction System - Proof of Concept (MVP)
Streamlit frontend for a bachelor's (engineering) thesis project.

How to run:
    streamlit run frontend/app.py

Current state of the code:
    - REAL INTEGRATION: prices come from the Quarkus backend (Tiingo data),
      model predictions come from the FastAPI ML service (POST /evaluate runs
      the rolling validation with LSTM/GRU/RF + baselines on real trained
      artifacts from engineer-backend-llm/data/articles).
    - MOCK FALLBACK: when either backend is unreachable, the app switches to
      generated mock data and shows a warning banner, so the demo always works.
    - Metrics (MAE/RMSE/DA) and the Diebold-Mariano test are computed on the
      frontend from the returned prediction arrays (same formulas as the
      training pipeline engineer-backend-llm/src/run_pipeline.py).

Methodology (consistent with the backend):
    - Rolling validation with 3 folds (like TimeSeriesSplit).
    - Models: LSTM, GRU, Random Forest + 2 baselines:
        * Zero Return (naive: tomorrow's price equals today's),
        * Mean Return (prediction = today's price * (1 + mean historical return)).
    - Dollar metrics: MAE ($), RMSE ($) and Directional Accuracy (%).
    - Diebold-Mariano test (squared errors): RF vs LSTM and RF vs GRU.
"""

import datetime
import math
import os
import zlib  # for a process-stable hash of ticker names (str hash is randomized per interpreter)
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# =====================================================================
# BACKEND CONNECTION CONFIGURATION (Quarkus / FastAPI)
# =====================================================================
# Inside Docker, compose sets both URLs to the container service names:
#   JAVA_API_BASE_URL=http://java-backend:8080  (Quarkus - price history, Tiingo)
#   ML_API_BASE_URL=http://python-ml:8000       (FastAPI - model evaluation)
JAVA_API_BASE_URL = os.environ.get("JAVA_API_BASE_URL", "http://localhost:8080")
ML_API_BASE_URL = os.environ.get("ML_API_BASE_URL", "http://localhost:8000")

# HTTP timeouts in seconds.
REQUEST_TIMEOUT = 10       # simple calls (history, health checks)
ML_EVALUATE_TIMEOUT = 240  # full rolling validation may take a while on CPU

# =====================================================================
# ETF LIST - full human-readable names + data source CSVs
# (selectbox label format: "Full name (SYMBOL)", e.g. "S&P 500 (SPY500)")
#
# IMPORTANT: the Tiingo ticker mapping (SPY500 -> SPY, NASDAQ -> QQQ,
# WIG20 -> EPOL, INVESCO -> RSP) lives on the Quarkus side
# (engineer-backend-api/src/main/java/org/acme/etf/service/TickerMapping.java),
# so this frontend always passes its own display symbols to the backend.
# =====================================================================
AVAILABLE_ETFS: List[Dict[str, str]] = [
    {"symbol": "NASDAQ", "file": "NASDAQ.csv", "name": "Nasdaq-100"},
    {"symbol": "SPY500", "file": "SPY500.csv", "name": "S&P 500"},
    {"symbol": "WIG20", "file": "WIG20.csv", "name": "WIG20"},
    {"symbol": "INVESCO", "file": "INVESCO.csv", "name": "Invesco S&P 500 Equal Weight"},
    {"symbol": "GLD", "file": "GLD.csv", "name": "SPDR Gold Shares"},
]

# ML models available in the system - must match the backend.
ML_MODELS: List[str] = ["LSTM", "GRU", "Random Forest"]

# Baseline models - reference points for the ML models.
BASELINE_MODELS: List[str] = ["Zero Return", "Mean Return"]

ALL_MODELS: List[str] = ML_MODELS + BASELINE_MODELS

N_FOLDS = 3           # number of rolling-validation folds (as in the backend)
TEST_DAYS = 60        # test-window length of a single fold (mocked data)
HISTORY_DAYS = 500    # length of the generated price history

# "Character" of the mocked ML models:
#   correlation - how strongly the model captures the actual daily return
#                 (0 = pure guessing),
#   bias        - systematic drift of predictions (e.g. RF slightly understates),
#   noise       - random prediction noise.
MODEL_PROFILES: Dict[str, Dict[str, float]] = {
    "Random Forest": {"correlation": 0.55, "bias": -0.0002, "noise": 0.004},
    "LSTM":          {"correlation": 0.70, "bias": 0.0001, "noise": 0.002},
    "GRU":           {"correlation": 0.63, "bias": 0.0,    "noise": 0.003},
}

# Chart colors per model.
MODEL_COLORS: Dict[str, str] = {
    "LSTM": "#2ca02c",
    "GRU": "#ff7f0e",
    "Random Forest": "#d62728",
    "Zero Return": "#7f7f7f",
    "Mean Return": "#9467bd",
}


# =====================================================================
# DATA LAYER: MOCKS (currently) / API (target state)
# =====================================================================
def fetch_history_mock(symbol: str, n_days: int = HISTORY_DAYS) -> pd.DataFrame:
    """
    MOCK - generates historical closing prices for the given ETF.

    Simulates n_days trading sessions with a geometric random walk,
    which produces a realistic-looking chart.
    Target state: replace with a backend API call.
    """
    # zlib.crc32 keeps the seed stable across process restarts (unlike hash()).
    rng = np.random.default_rng(seed=zlib.crc32(symbol.encode()) % (2**32))
    daily_return = rng.normal(loc=0.0003, scale=0.012, size=n_days)
    price = 400 * np.exp(np.cumsum(daily_return))  # start ~400 USD
    dates = pd.bdate_range(end=datetime.date.today(), periods=n_days)  # business days only
    return pd.DataFrame({"date": dates, "close": price})


def build_folds(history: pd.DataFrame, n_folds: int = N_FOLDS, test_days: int = TEST_DAYS) -> List[Dict]:
    """
    Splits the history into n_folds rolling-validation folds, analogous to
    TimeSeriesSplit in the backend: test windows follow one another in time
    and training includes all data preceding a given fold.
    """
    folds: List[Dict] = []
    for fold in range(n_folds):
        test_start = len(history) - (n_folds - fold) * test_days
        folds.append({
            "fold": fold + 1,
            "train_idx": np.arange(0, test_start),
            "test_idx": np.arange(test_start, test_start + test_days),
        })
    return folds


def fetch_predictions_mock(history: pd.DataFrame, fold_cfg: Dict) -> Dict[str, pd.DataFrame]:
    """
    MOCK - generates predictions of all 5 variants for the test fold.

    Prediction expressed as a price: pred_t = close_{t-1} * (1 + a*r_t + bias + noise),
    where r_t is the actual daily return. Baselines are computed deterministically:
      - Zero Return: pred_t = close_{t-1},
      - Mean Return: pred_t = close_{t-1} * (1 + mean training return).
    """
    test_df = history.iloc[fold_cfg["test_idx"]].reset_index(drop=True)
    actual = test_df["close"].values
    prev_close = history["close"].values[fold_cfg["test_idx"] - 1]  # previous day's close
    true_return = actual / prev_close - 1.0

    train_close = history["close"].values[fold_cfg["train_idx"]]
    mean_train_return = float(np.mean(train_close[1:] / train_close[:-1] - 1.0))

    # --- Baseline models (deterministic) --------------------------------
    raw_predictions: Dict[str, np.ndarray] = {}
    raw_predictions["Zero Return"] = prev_close.copy()
    raw_predictions["Mean Return"] = prev_close * (1.0 + mean_train_return)

    # --- ML models -------------------------------------------------------
    rng = np.random.default_rng(seed=98765 + fold_cfg["fold"])
    for model, params in MODEL_PROFILES.items():
        noise = rng.normal(0, params["noise"], size=len(test_df))
        raw_predictions[model] = prev_close * (
            1.0 + params["correlation"] * true_return + params["bias"] + noise
        )

    # Per model, return a frame: date, actual close, previous close, prediction.
    return {
        model: pd.DataFrame({
            "date": test_df["date"].values,
            "actual": actual,
            "prev_close": prev_close,
            "prediction": preds,
        })
        for model, preds in raw_predictions.items()
    }


# ---------------------------------------------------------------------
# METRICS (shared with the backend) - computed for real
# ---------------------------------------------------------------------
def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Directional Accuracy in %: how often the price direction is guessed correctly."""
    return float(np.mean(np.sign(y_true) == np.sign(y_pred)) * 100)


def compute_metrics(actual: np.ndarray, prediction: np.ndarray, prev_close: np.ndarray) -> Dict[str, float]:
    """
    Computes metrics in DOLLARS:
      - MAE ($)  - mean absolute error (average price miss),
      - RMSE ($) - root mean squared error (punishes large errors),
      - DA (%)   - Directional Accuracy (up/down vs the previous day).
    """
    err = prediction - actual
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    da = directional_accuracy(np.sign(actual - prev_close), np.sign(prediction - prev_close))
    return {"MAE ($)": mae, "RMSE ($)": rmse, "DA (%)": da}


def dm_test(actuals: np.ndarray, pred1: np.ndarray, pred2: np.ndarray) -> Tuple[float, float]:
    """
    Diebold-Mariano test on squared errors (MSE).
    H0: the expected difference in losses between the two models is 0.
    Same methodology as the backend (run_pipeline.py);
    the normal CDF is computed via math.erf (no scipy dependency).
    """
    e1 = actuals - pred1
    e2 = actuals - pred2
    d = e1**2 - e2**2
    mean_d = np.mean(d)
    var_d = np.var(d, ddof=1)

    if var_d == 0:
        return 0.0, 1.0

    stat = mean_d / np.sqrt(var_d / len(d))
    # Two-sided test: use |stat| in the normal CDF, so p-value stays in [0, 1]
    # even when the second model strongly outperforms the first (stat << 0).
    phi = 0.5 * (1.0 + math.erf(abs(stat) / math.sqrt(2.0)))
    return float(stat), float(2 * (1 - phi))


def evaluate_fold(preds_by_model: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Builds the metrics table (rows = models) for a single fold."""
    rows = []
    for model, pred_df in preds_by_model.items():
        metrics = compute_metrics(
            pred_df["actual"].values,
            pred_df["prediction"].values,
            pred_df["prev_close"].values,
        )
        rows.append({"Model": model, **metrics})
    return pd.DataFrame(rows).set_index("Model")


def evaluate_all_folds(results: Dict[int, Dict[str, pd.DataFrame]]) -> Dict[int, pd.DataFrame]:
    """Metrics per fold: {1: DataFrame, 2: ..., 3: ...}."""
    return {fold: evaluate_fold(preds) for fold, preds in results.items()}


def compute_dm_pvalues(results: Dict[int, Dict[str, pd.DataFrame]]) -> pd.DataFrame:
    """
    Diebold-Mariano test for each pair and fold + mean p-value
    (averaging over folds as a simple summary).

    Pairs include ML-vs-ML comparisons and ML-vs-naive-baseline ones -
    the latter are the key academic proof that the ML models beat the
    naive "price unchanged" assumption.
    """
    pairs = [
        ("Random Forest", "LSTM"),
        ("Random Forest", "GRU"),
        ("Zero Return", "LSTM"),
        ("Zero Return", "Random Forest"),
    ]
    rows = []
    for model1, model2 in pairs:
        stats, pvals = [], []
        for fold, preds in results.items():
            pred1 = preds[model1]["prediction"].values
            pred2 = preds[model2]["prediction"].values
            actual = preds[model1]["actual"].values
            stat, pval = dm_test(actual, pred1, pred2)
            stats.append(stat)
            pvals.append(pval)
        rows.append({
            "Porównanie": f"{model1} vs {model2}",  # displayed on screen - kept in Polish
            "Fold 1": pvals[0],
            "Fold 2": pvals[1],
            "Fold 3": pvals[2],
            "p-value (śr.)": float(np.mean(pvals)),
        })
    return pd.DataFrame(rows).set_index("Porównanie")


# ---------------------------------------------------------------------
# REAL API LAYER (Quarkus history + FastAPI evaluation) with mock fallback
# ---------------------------------------------------------------------
def fetch_history(symbol: str) -> Tuple[Optional[pd.DataFrame], bool]:
    """
    Fetches price history from the Quarkus backend:
        GET {JAVA_API_BASE_URL}/api/etfs/{symbol}/history
    (data populated from Tiingo by POST /api/etfs/{symbol}/sync; the
    frontend symbol -> Tiingo ticker mapping lives in the Java backend).

    Rows arrive newest-first, so they are reversed to chronological order.
    Returns (dataframe, came_from_api); on a failure returns (None, False)
    and the caller falls back to the mock generator.
    """
    try:
        response = requests.get(
            f"{JAVA_API_BASE_URL}/api/etfs/{symbol.upper()}/history",
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        rows = response.json()
        if not rows:
            return None, False
        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        df = df[["date", "closePrice", "volume"]].rename(columns={"closePrice": "close"})
        return df, True
    except Exception:
        return None, False


def evaluate_from_api(history_df: pd.DataFrame) -> Tuple[Dict[int, Dict[str, pd.DataFrame]], Optional[Dict]]:
    """
    Calls the ML backend rolling validation:
        POST {ML_API_BASE_URL}/evaluate
    with the history payload. The response {fold -> {model -> arrays}}
    is converted to the same structure the mock generator produces:
        {fold: {model: DataFrame(date, actual, prev_close, prediction)}}
    Metrics and the DM test are computed on the frontend side.

    Also returns the backend's "next_day" forward forecast
    ({date, prev_close, predictions: {model: price}}) or None when absent.
    """
    payload_rows = [
        # itertuples field: row.date is a pandas Timestamp -> ISO "YYYY-MM-DD"
        {"date": f"{row.date.date()}", "closePrice": float(row.close), "volume": int(row.volume)}
        for row in history_df.itertuples(index=False)
    ]
    payload_rows = payload_rows[-HISTORY_DAYS:]  # folds need ~ (n_folds+1)*TEST_DAYS rows

    response = requests.post(
        f"{ML_API_BASE_URL}/evaluate",
        json={"history": payload_rows, "n_folds": N_FOLDS, "test_days": TEST_DAYS},
        timeout=ML_EVALUATE_TIMEOUT,
    )
    response.raise_for_status()
    data = response.json()

    results: Dict[int, Dict[str, pd.DataFrame]] = {}
    for fold in data["folds"]:
        dates = pd.to_datetime(fold["dates"])
        results[int(fold["fold"])] = {
            model: pd.DataFrame({
                "date": dates,
                "actual": np.array(fold["actual"], dtype=float),
                "prev_close": np.array(fold["prev_close"], dtype=float),
                "prediction": np.array(arr, dtype=float),
            })
            for model, arr in fold["predictions"].items()
        }

    next_day = None
    if isinstance(data.get("next_day"), dict):
        next_day = {
            "date": data["next_day"]["date"],
            "prev_close": float(data["next_day"]["prev_close"]),
            "predictions": {
                model: float(price)
                for model, price in data["next_day"]["predictions"].items()
            },
        }
    return results, next_day


def fetch_next_day_mock(history: pd.DataFrame) -> Dict:
    """
    MOCK forward forecast for the next trading day: anchored at the last
    mock close, each model drifts by its own seeded noise so the shape
    resembles the real ML service output.
    """
    last_close = float(history["close"].iloc[-1])
    next_session = pd.bdate_range(
        start=history["date"].iloc[-1] + pd.Timedelta(days=1), periods=1
    )[0]
    rng = np.random.default_rng(seed=zlib.crc32(b"next-day") % (2**32))
    predictions: Dict[str, float] = {
        "Zero Return": last_close,
    }
    train_close = history["close"].values
    mean_return = float(np.mean(train_close[1:] / train_close[:-1] - 1.0))
    predictions["Mean Return"] = last_close * (1.0 + mean_return)
    for model, params in MODEL_PROFILES.items():
        drift = rng.normal(0, params["noise"])
        predictions[model] = last_close * (
            1.0 + params["correlation"] * drift + params["bias"]
        )
    return {"date": str(pd.Timestamp(next_session).date()), "prev_close": last_close, "predictions": predictions}


# =====================================================================
# PRESENTATION LAYER (UI)
# =====================================================================
st.set_page_config(
    page_title="System Predykcji ETF",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

etf_labels = {etf["symbol"]: f"{etf['name']} ({etf['symbol']})" for etf in AVAILABLE_ETFS}

# ---------------------------------------------------------------------
# SIDEBAR - analysis parameters
# ---------------------------------------------------------------------
with st.sidebar:
    st.title("📈 System Predykcji ETF")
    st.caption("Proof of Concept - praca inżynierska")

    st.divider()

    selected_symbol = st.selectbox(
        "Wybierz aktywo (ETF):",
        options=list(etf_labels.keys()),
        format_func=lambda s: etf_labels[s],
        help="Dane historyczne wczytywane z plików CSV (np. SPY500.csv).",
    )
    selected_file = next(e["file"] for e in AVAILABLE_ETFS if e["symbol"] == selected_symbol)
    st.caption(f"Dane: Tiingo przez Quarkus (fallback: `{selected_file}`)")

    selected_models = st.multiselect(
        "Modele do porównania:",
        options=ALL_MODELS,
        default=ALL_MODELS,
        help="Wybór wpływa na wykres główny oraz tabele. "
             "Modele bazowe (Zero Return, Mean Return) pokazywane są tylko w tabelach i słupkach.",
    )

    selected_fold = st.selectbox(
        "Okres testowy (fold):",
        options=list(range(1, N_FOLDS + 1)),
        index=N_FOLDS - 1,
        help="Walidacja krocząca - wykres cen i błędy w czasie pokazywane dla wybranego foldu.",
    )

    # Action button - defines when predictions are recomputed.
    run_prediction = st.button("▶️ Wygeneruj predykcję", type="primary", use_container_width=True)

    st.divider()
    # Health dot for the ML service (simple liveness check).
    try:
        ml_ok = requests.get(f"{ML_API_BASE_URL}/health", timeout=REQUEST_TIMEOUT).ok
    except Exception:
        ml_ok = False
    st.caption(f"{'🟢' if ml_ok else '🔴'} ML service: `{ML_API_BASE_URL}`")
    st.caption(f"Quarkus API: `{JAVA_API_BASE_URL}`")

# Nothing to show without at least one selected model.
if not selected_models:
    st.warning("Wybierz co najmniej jeden model w panelu bocznym.")
    st.stop()


# Peer evaluation cache per ETF (all folds are computed anyway). Returns
# (results, next_day, mock_mode) where False means everything came from the
# real backends (Quarkus + FastAPI) and True means the app fell back to mocks.
# ttl keeps results fresh after the backend scheduled sync picks up new data.
@st.cache_data(show_spinner="Pobieranie danych historycznych i ocena modeli ML...", ttl=600)
def run_evaluation(symbol: str) -> Tuple[Dict[int, Dict[str, pd.DataFrame]], Optional[Dict], bool]:
    history, came_from_api = fetch_history(symbol)
    if came_from_api:
        try:
            results, next_day = evaluate_from_api(history)
            return results, next_day, False
        except Exception:
            pass  # fall through to the mock layer below
    mock_history = fetch_history_mock(symbol)
    return (
        {cfg["fold"]: fetch_predictions_mock(mock_history, cfg) for cfg in build_folds(mock_history)},
        fetch_next_day_mock(mock_history),
        True,
    )


# Cached history for the header panel (always renders, even in mock mode).
# ttl keeps the "last trading day" numbers fresh after scheduled backend sync.
@st.cache_data(show_spinner=False, ttl=600)
def get_history(symbol: str) -> Tuple[pd.DataFrame, bool]:
    df, came_from_api = fetch_history(symbol)
    if df is None:
        df, came_from_api = fetch_history_mock(symbol), False
    return df, came_from_api


# Recompute on button press: a fresh click forces a new run.
if run_prediction or "last_run" not in st.session_state:
    results, next_day, mock_mode = run_evaluation(selected_symbol)
    st.session_state["last_run"] = selected_symbol
    st.session_state["results"] = results
    st.session_state["next_day"] = next_day
    st.session_state["mock_mode"] = mock_mode

results = st.session_state["results"]
next_day = st.session_state.get("next_day")
mock_mode = st.session_state.get("mock_mode", False)
metrics_by_fold = evaluate_all_folds(results)

# ---------------------------------------------------------------------
# MAIN PANEL - results
# ---------------------------------------------------------------------
active_symbol = st.session_state["last_run"]
history_df, history_from_api = get_history(active_symbol)
last_price = history_df["close"].iloc[-1]
first_price = history_df["close"].iloc[0]
first_date = history_df["date"].iloc[0]
last_date = history_df["date"].iloc[-1]

# Returns over different horizons - full history vs last 365 calendar days.
full_period_return = (last_price / first_price - 1) * 100
history_1y = history_df[history_df["date"] >= last_date - pd.Timedelta(days=365)]
one_year_return = (last_price / history_1y["close"].iloc[0] - 1) * 100

# Result provenance banner: green when the real backends answered.
if mock_mode:
    st.warning(
        "🟡 **Tryb MOCK** - brak połączenia z backendem (Tiingo/ML). "
        "Poniższe dane i predykcje są wygenerowane SYNTEZYCZNIE."
    )
else:
    st.success("🟢 Dane z backendu: historia - Quarkus (Tiingo), predykcje - inferencja modeli ML.")

st.header(f"Analiza: {etf_labels[active_symbol]}")
st.caption(
    "ℹ️ Wszystkie ceny to ceny JEDNEJ AKCJI ETF (dane Tiingo), a nie poziomy indeksów giełdowych: "
    "ETF SPY kosztuje ok. 1/10 poziomu indeksu S&P 500. WIG20 jest aproksymowany ETF-em EPOL "
    "(iShares MSCI Poland) - GPW nie ma pokrycia w Tiingo."
)

info_col1, info_col2, info_col3, info_col4, info_col5 = st.columns(5)
info_col1.metric("Pełna nazwa", etf_labels[active_symbol].rsplit(" (", 1)[0])
info_col2.metric(
    "Aktualna cena",
    f"{last_price:,.2f} USD",
    help="Cena JEDNEJ AKCJI ETF z ostatniej zamkniętej sesji (Tiingo), nie poziom indeksu. "
         "Np. SPY jest zaprojektowany tak, że jego cena ≈ 1/10 poziomu indeksu S&P 500 "
         "(indeks ~7 500 pkt -> SPY ~750 USD). To akcję ETF prognozują modele.",
)
info_col3.metric(
    "Zwrot - ostatnie 12 mies.",
    f"{one_year_return:+.2f}%",
    help=f"Zmiana ceny od {history_1y['date'].iloc[0]:%Y-%m-%d} do {last_date:%Y-%m-%d}.",
)
info_col4.metric(
    f"Zwrot od {first_date:%Y-%m-%d}",
    f"{full_period_return:+.2f}%",
    help="Zmiana ceny na CAŁEJ dostępnej historii z Tiingo, "
         "dlatego dla długich ETF-ów (np. GLD z 2004 r.) są to wysokie wartości.",
)
if next_day:
    ml_forecasts = [next_day["predictions"][m] for m in ML_MODELS if m in next_day["predictions"]]
    forecast_detail = "; ".join(
        f"{m}: {next_day['predictions'].get(m, float('nan')):,.2f} USD" for m in ML_MODELS
    )
    info_col5.metric(
        f"Prognoza na {next_day['date']}",
        f"{np.mean(ml_forecasts):,.2f} USD",
        delta=f"{(np.mean(ml_forecasts) / next_day['prev_close'] - 1) * 100:+.2f}% vs zamknięcie",
        help=f"Średnia prognoza modeli ML na najbliższą sesję. Szczegóły: {forecast_detail}. "
             "Punkty diamentowe na wykresie głównym pokazują prognozę każdego modelu osobno.",
    )

st.divider()

# ============ Section 2: price vs. predictions chart =================
st.subheader(f"Cena rzeczywista vs. predykcje modeli (fold {selected_fold})")
st.caption(
    "Na wykresach pokazujemy wyłącznie modele ML (LSTM, GRU, Random Forest). "
    "Modele bazowe (Zero Return, Mean Return) porównujemy w tabelach metryk "
    "i na wykresach słupkowych - nie wnoszą wartości wizualnej. "
    + (
        f"Progi prognozy na następną sesję ({next_day['date']}) widoczne są dla ostatniego foldu "
        "jako przerywane przedłużenia z diamentem."
        if next_day else ""
    )
)

# Chart-only model selection (kept in st.session_state via the widget key,
# so the choice survives reruns and does not affect tables/metrics).
with st.expander("⚙️ Wybierz, co zestawić z ceną rzeczywistą", expanded=False):
    st.caption(
        "Domyślnie pokazywana jest cena rzeczywista bez predykcji. "
        "Zaznacz modele, których predykcje mają być nałożone na wykres."
    )
    chart_models = st.multiselect(
        "Predykcje na wykresie:",
        options=ML_MODELS,
        default=[],
        help="Wybór dotyczy wyłącznie tego wykresu - tabele metryk "
             "zawsze pokazują komplet wyników.",
    )
    chart_show_predictions = st.toggle(
        "Pokaż wszystkie predykcje naraz",
        value=False,
        help="Włącz, aby narysować wszystkie modele ML jednocześnie.",
    )

fold_data = results[selected_fold]
fig = go.Figure()

actual_df = fold_data[selected_models[0]]  # actual price is shared across models
fig.add_trace(
    go.Scatter(
        x=actual_df["date"],
        y=actual_df["actual"],
        mode="lines",
        name="Cena rzeczywista",
        line=dict(color="#1f77b4", width=2),
        hovertemplate="Data: %{x|%Y-%m-%d}<br>Cena: %{y:.2f} USD<extra>Rzeczywista</extra>",
    )
)

# Models to draw: either all ML models (toggle) or the manual selection.
if chart_show_predictions:
    drawn_models = ML_MODELS
else:
    drawn_models = chart_models

for model in drawn_models:
    pred_df = fold_data[model]
    fig.add_trace(
        go.Scatter(
            x=pred_df["date"],
            y=pred_df["prediction"],
            mode="lines",
            name=f"Predykcja: {model}",
            line=dict(
                color=MODEL_COLORS.get(model, "#9467bd"),
                width=2,
            ),
            hovertemplate=f"Model: {model}<br>Data: %{{x|%Y-%m-%d}}<br>Predykcja: %{{y:.2f}} USD<extra></extra>",
        )
    )

# Forward forecast (next trading day) drawn ONLY on the last fold - it is
# anchored at that fold's final prediction, so stitching it to older folds
# would draw a misleading jump across the test-window gap.
if next_day and selected_fold == N_FOLDS:
    fwd_ts = pd.Timestamp(next_day["date"])
    for model in drawn_models:
        if model not in next_day["predictions"]:
            continue
        pred_df = fold_data[model]
        fig.add_trace(
            go.Scatter(
                x=[pred_df["date"].iloc[-1], fwd_ts],
                y=[pred_df["prediction"].iloc[-1], next_day["predictions"][model]],
                mode="lines+markers",
                marker=dict(size=10, symbol="diamond"),
                line=dict(dash="dot", width=2, color=MODEL_COLORS.get(model, "#9467bd")),
                name=f"Prognoza {next_day['date']}: {model}",
                hovertemplate=(
                    f"Prognoza na {next_day['date']}<br>Model: {model}"
                    "<br>Cena: %{y:.2f} USD<extra></extra>"
                ),
            )
        )

if not drawn_models:
    fig.add_annotation(
        text="Zaznacz w sekcji 'Wybierz, co zestawić z ceną rzeczywistą' predykcje do pokazania.",
        showarrow=False,
        font=dict(size=14, color="#666666"),
    )

fig.update_layout(
    height=520,
    template="plotly_white",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    margin=dict(l=20, r=20, t=40, b=20),
    xaxis=dict(title="Data", rangeslider=dict(visible=True)),
    yaxis=dict(title="Cena zamknięcia (USD)"),
    hovermode="x unified",
)
st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True})

st.divider()

# ================== Section 3: ERROR CHARTS ==========================
st.subheader(f"Wykresy błędów (fold {selected_fold})")

# --- Row 1: bar-chart model comparison (MAE / RMSE / DA) -------------
def metric_bar_chart(metric: str, unit_hint: str) -> go.Figure:
    """Bar chart: one metric, x = model, colors follow MODEL_COLORS."""
    rows = [{"Model": model, metric: metrics_by_fold[selected_fold].loc[model, metric]}
            for model in selected_models]
    df = pd.DataFrame(rows)
    bar_fig = px.bar(
        df,
        x="Model",
        y=metric,
        color="Model",
        color_discrete_map=MODEL_COLORS,
        text_auto=".3f",
        title=f"{metric} per model (fold {selected_fold})",
    )
    bar_fig.update_layout(
        height=340,
        template="plotly_white",
        showlegend=False,
        margin=dict(l=20, r=20, t=50, b=20),
        yaxis_title=unit_hint if metric != "DA (%)" else "Skuteczność kierunku (%)",
        xaxis_title="",
    )
    return bar_fig


bar_col1, bar_col2, bar_col3 = st.columns(3)
with bar_col1:
    mae_chart = metric_bar_chart("MAE ($)", "USD")
    st.plotly_chart(mae_chart, use_container_width=True)
with bar_col2:
    rmse_chart = metric_bar_chart("RMSE ($)", "USD")
    st.plotly_chart(rmse_chart, use_container_width=True)
with bar_col3:
    da_chart = metric_bar_chart("DA (%)", "%")
    st.plotly_chart(da_chart, use_container_width=True)

# --- Row 2: error over time for the ML models (baselines excluded - they
#     dominate the plot visually without adding information) -------------
st.markdown("###### Błąd predykcji w czasie (rzeczywista − predykcja)")

fig_err_time = go.Figure()
for model in ML_MODELS:
    pred_df = fold_data[model]
    err_series = pred_df["actual"].values - pred_df["prediction"].values
    fig_err_time.add_trace(
        go.Scatter(
            x=pred_df["date"],
            y=err_series,
            mode="lines",
            name=model,
            line=dict(color=MODEL_COLORS.get(model, "#9467bd"), width=1.6),
            hovertemplate=f"Model: {model}<br>Data: %{{x|%Y-%m-%d}}<br>Błąd: %{{y:.2f}} USD<extra></extra>",
        )
    )
fig_err_time.add_hline(y=0, line=dict(color="black", width=1, dash="dot"))
fig_err_time.update_layout(
    height=360,
    template="plotly_white",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    margin=dict(l=20, r=20, t=30, b=20),
    xaxis=dict(title="Data", rangeslider=dict(visible=True)),
    yaxis=dict(title="Błąd (USD)"),
    hovermode="x unified",
)
st.plotly_chart(fig_err_time, use_container_width=True, config={"scrollZoom": True})

# --- Row 3: error distribution histogram (ML models only - the naive
#     baselines create huge symmetric saturating blocks) ------------------
st.markdown("###### Rozkład błędów (histogram)")

fig_hist = go.Figure()
for model in ML_MODELS:
    err = fold_data[model]["actual"].values - fold_data[model]["prediction"].values
    fig_hist.add_trace(
        go.Histogram(
            x=err,
            name=model,
            opacity=0.65,
            marker_color=MODEL_COLORS.get(model, "#9467bd"),
            hovertemplate="Zakres błędu: %{x}<br>Liczba obserwacji: %{y}<extra></extra>",
            nbinsx=25,
        )
    )
fig_hist.add_vline(x=0, line=dict(color="black", width=1, dash="dot"))
fig_hist.update_layout(
    height=360,
    template="plotly_white",
    barmode="overlay",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    margin=dict(l=20, r=20, t=30, b=20),
    xaxis=dict(title="Błąd predykcji (USD)"),
    yaxis=dict(title="Liczba dni"),
)
st.plotly_chart(fig_hist, use_container_width=True)

st.divider()

# =========== Section 4: METRICS TABLE (per fold + summary) ===========
st.subheader(f"Metryki błędu z walidacji (fold {selected_fold})")

fold_metrics = metrics_by_fold[selected_fold].loc[selected_models].copy()
fold_metrics["DA (%)"] = fold_metrics["DA (%)"].map(lambda v: f"{v:.2f}%")
fold_metrics["MAE ($)"] = fold_metrics["MAE ($)"].map(lambda v: f"{v:,.2f}")
fold_metrics["RMSE ($)"] = fold_metrics["RMSE ($)"].map(lambda v: f"{v:,.2f}")
st.dataframe(fold_metrics, use_container_width=True)

st.caption(
    "MAE ($) - średni błąd bezwzględny ceny | RMSE ($) - pierwiastek błędu średniokwadratowego, "
    "silniej karze duże błędy | DA (%) - trafność kierunku (wzrost/spadek) względem dnia poprzedniego. "
    "Uwaga: **Zero Return** zawsze przewiduje brak zmiany, więc nie obstawia żadnego kierunku - "
    "stąd jego DA = 0% (konwencja zgodna z backendem)."
)

# Summary - mean across all folds (rolling validation).
summary_df = pd.concat(metrics_by_fold.values()).groupby(level="Model").mean().loc[selected_models]
summary_df["DA (%)"] = summary_df["DA (%)"].map(lambda v: f"{v:.2f}%")
summary_df["MAE ($)"] = summary_df["MAE ($)"].map(lambda v: f"{v:,.2f}")
summary_df["RMSE ($)"] = summary_df["RMSE ($)"].map(lambda v: f"{v:,.2f}")
summary_df = summary_df.rename(columns={
    "MAE ($)": "MAE - średnia ($) (3 foldy)",
    "RMSE ($)": "RMSE - średnia ($) (3 foldy)",
    "DA (%)": "DA - średnia (%) (3 foldy)",
})
st.markdown("###### Podsumowanie - średnia ze wszystkich foldów")
st.dataframe(summary_df, use_container_width=True)

st.divider()

# ========= Section 5: STATISTICAL TESTS (Diebold-Mariano) ============
st.subheader("🧪 Testy statystyczne - Diebold-Mariano")
st.caption(
    "Test porównuje jakość predykcji obu modeli (błędy kwadratowe). "
    "**p-value < 0.05** oznacza, że różnica między modelami jest istotna "
    "statystycznie, a nie przypadkowa. Pary z baseline'em *Zero Return* to "
    "ostateczny dowód, że modele ML istotnie przewyższają naiwne "
    "założenie niezmienności ceny."
)

dm_table = compute_dm_pvalues(results)

for pair_name, row in dm_table.iterrows():
    p_avg = row["p-value (śr.)"]
    significant = p_avg < 0.05
    with st.container(border=True):
        c1, c2, c3 = st.columns([3, 1, 2])
        with c1:
            st.markdown(f"**Porównanie: {pair_name}**")
        with c2:
            st.metric("p-value (śr.)", f"{p_avg:.5f}")
        with c3:
            if significant:
                st.success("✅ Różnica istotna statystycznie (p < 0.05)")
            else:
                st.info("ℹ️ Brak istotności statystycznej (p ≥ 0.05)")

        # Per-fold breakdown shown as a small helper table.
        fold_cols = ["Fold 1", "Fold 2", "Fold 3"]
        fold_view = pd.DataFrame({col: [f"{row[col]:.5f}"] for col in fold_cols},
                                 index=["p-value"])
        st.dataframe(fold_view, use_container_width=True)

if mock_mode:
    st.caption(
        "Uwaga: na danych MOCKOWANYCH p-value służą wyłącznie demonstracji działania testu. "
        "Po podpięciu prawdziwego backendu wyniki będą pochodziły z treningu modeli."
    )
