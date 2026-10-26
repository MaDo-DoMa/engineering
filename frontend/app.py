# -*- coding: utf-8 -*-
"""
System Predykcji ETF - Proof of Concept (MVP)
Frontend w Streamlit dla pracy inżynierskiej.

Uruchomienie:
    streamlit run frontend/app.py

Stan obecnego kodu (MVP):
    - Dane są MOCKOWANE (generowane losowo przez pandas/numpy).
    - Struktura kodu jest już przygotowana pod integrację z REST API
      backendu Quarkus (patrz funkcje fetch_* i stała API_BASE_URL).

Do zrobienia (integracja):
    1. Ustawić API_BASE_URL na prawdziwy adres mikroserwisu/Quarkusa.
    2. Dostosować endpointy i format JSON do kontraktu z backendem.
"""

import datetime
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests  # zaległy import - użyty dopiero przy integracji z API
import streamlit as st

# =====================================================================
# KONFIGURACJA POŁĄCZENIA Z BACKENDEM (Quarkus / FastAPI)
# =====================================================================
# TODO: PODPIĄĆ PRAWDZIWY ADRES IP BACKENDU QUARKUS <<<
# Docelowo np.: "http://192.168.0.15:8080/api" albo "http://localhost:8080/api"
API_BASE_URL = "http://localhost:8080/api"

# Timeouty dla zapytań HTTP (sekundy) - używane przy prawdziwym API.
REQUEST_TIMEOUT = 10

# Lista obsługiwanych ETF-ów (musi zgadzać się z listą w backendzie Quarkus).
AVAILABLE_TICKERS: List[str] = ["SPY", "QQQ", "DIA", "IWM", "VTI", "VOO"]

# Modele ML dostępne w systemie - muszą zgadzać się z backendem.
AVAILABLE_MODELS: List[str] = ["Random Forest", "LSTM", "GRU"]


# =====================================================================
# WARSTWA DANYCH: MOCKI (obecnie) / API (docelowo)
# =====================================================================
def fetch_history_mock(ticker: str, n_days: int = 500) -> pd.DataFrame:
    """
    MOCK - generuje historyczne ceny zamknięcia danego ETF.

    Symuluje ~500 sesji giełdowych z logarytmiczno-normalnym ruchem cen
    (tzw. random walk geometryczny), co daje realistyczny wygląd wykresu.
    Docelowo: podmienić na wywołanie API backendu.
    """
    rng = np.random.default_rng(seed=hash(ticker) % (2**32))  # deterministyczny seed per ticker
    # Parametry ruchu ceny: tendencja wzrostowa + zmienność typowa dla ETF.
    daily_return = rng.normal(loc=0.0003, scale=0.012, size=n_days)
    price = 400 * np.exp(np.cumsum(daily_return))  # start ~400 USD
    dates = pd.bdate_range(end=datetime.date.today(), periods=n_days)  # tylko dni robocze
    df = pd.DataFrame({"date": dates, "close": price})
    return df


def fetch_predictions_mock(
    history: pd.DataFrame,
    models: List[str],
    n_test: int = 60,
) -> Dict[str, pd.DataFrame]:
    """
    MOCK - generuje predykcje dla wybranych modeli w okresie testowym.

    Predykcja = rzeczywista cena + szum, przeskalowany per model tak,
    aby każdy model miał inny charakter błędu (RF zaniża szczyty,
    LSTM "wyprzedza" cenę, GRU jest gdzieś pomiędzy).
    """
    rng = np.random.default_rng(seed=12345)
    test_slice = history.tail(n_test).copy()
    predictions: Dict[str, pd.DataFrame] = {}

    noise_std = {
        "Random Forest": 0.004,
        "LSTM": 0.002,
        "GRU": 0.003,
    }
    offset = {
        "Random Forest": -0.002,   # lekkie zaniżenie
        "LSTM": 0.0015,            # lekkie wyprzedzenie
        "GRU": 0.0,
    }
    for model in models:
        # Prognoza opóźniona o 1 dzień + szum - "plausible-looking".
        lagged = test_slice["close"].shift(1).bfill()
        noise = rng.normal(0, noise_std.get(model, 0.003), size=len(test_slice))
        pred = lagged * (1 + noise) + offset.get(model, 0.0) * lagged
        predictions[model] = pd.DataFrame(
            {"date": test_slice["date"].values, "prediction": pred.values}
        )
    return predictions


def compute_metrics_mock(model: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    MOCK - liczy metryki błędu (MAE, RMSE, MAPE) z walidacji.
    Docelowo: backend Quarkus powinien zwracać te metryki z API,
    a frontend tylko je wyświetlać (wtedy usunąć tę funkcję).
    """
    err = y_true - y_pred
    mae = np.mean(np.abs(err))
    rmse = np.sqrt(np.mean(err**2))
    mape = np.mean(np.abs(err / y_true)) * 100  # w procentach
    return {"Model": model, "MAE": round(mae, 2), "RMSE": round(rmse, 2), "MAPE (%)": round(mape, 2)}


# ---------------------------------------------------------------------
# FUNKCJE DO INTEGRACJI Z PRAWDZIWYM BACKENDEM (Quarkus)
# ---------------------------------------------------------------------
#  TODO: PODPIĄĆ PRAWDZIWY ADRES IP - użyj funkcji poniżej zamiast mocków <<<
def fetch_history_from_api(ticker: str) -> pd.DataFrame:
    """
    Docelowa funkcja: pobiera historię cen z backendu Quarkus.
    Endpoint (przykładowy): GET {API_BASE_URL}/etf/{ticker}/history
    Oczekiwany JSON: [{"date": "2024-01-02", "close": 401.2}, ...]
    """
    response = requests.get(f"{API_BASE_URL}/etf/{ticker}/history", timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    df = pd.DataFrame(response.json())
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def fetch_predictions_from_api(ticker: str, model: str) -> pd.DataFrame:
    """
    Docelowa funkcja: pobiera predykcje modelu ML z backendu Quarkus.
    Endpoint (przykładowy): GET {API_BASE_URL}/predict?ticker={t}&model={m}
    Oczekiwany JSON: [{"date": "2026-01-05", "prediction": 402.5}, ...]
    """
    response = requests.get(
        f"{API_BASE_URL}/predict",
        params={"ticker": ticker, "model": model},
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    df = pd.DataFrame(response.json())
    df["date"] = pd.to_datetime(df["date"])
    return df


def fetch_metrics_from_api(ticker: str, model: str) -> dict:
    """
    Docelowa funkcja: pobiera metryki (MAE, RMSE, MAPE) z walidacji
    danego modelu, liczone po stronie backendu.
    Endpoint (przykładowy): GET {API_BASE_URL}/metrics?ticker={t}&model={m}
    Oczekiwany JSON: {"model": "LSTM", "mae": 2.31, "rmse": 3.10, "mape": 0.84}
    """
    response = requests.get(
        f"{API_BASE_URL}/metrics",
        params={"ticker": ticker, "model": model},
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    data = response.json()
    return {"Model": data["model"], "MAE": data["mae"], "RMSE": data["rmse"], "MAPE (%)": data["mape"]}


# =====================================================================
# WARSTWA PREZENTACJI (UI)
# =====================================================================
st.set_page_config(
    page_title="System Predykcji ETF",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------
# PASEK BOCZNY (Sidebar) - parametryzacja predykcji
# ---------------------------------------------------------------------
with st.sidebar:
    st.title("📈 System Predykcji ETF")
    st.caption("Proof of Concept - praca inżynierska")

    st.divider()

    selected_ticker = st.selectbox(
        "Wybierz aktywo (ETF):",
        options=AVAILABLE_TICKERS,
        index=0,
        help="Ticker wybranego funduszu ETF przetwarzanego przez backend.",
    )

    model_choice = st.selectbox(
        "Wybierz model ML:",
        options=["Wszystkie"] + AVAILABLE_MODELS,
        index=0,
        help='Wybierz "Wszystkie", aby porównać predykcje Random Forest, LSTM i GRU na jednym wykresie.',
    )

    # Przycisk akcji - definiuje moment ponownego obliczenia predykcji.
    run_prediction = st.button("▶️ Wygeneruj predykcję", type="primary", use_container_width=True)

    st.divider()
    st.caption(f"Backend Quarkus: `{API_BASE_URL}`")
    #  TODO: tutaj wyświetlić status połączenia z prawdziwym backendem,
    # np. przez requests.get(f"{API_BASE_URL}/health") i pokazać zieloną/czerwoną kropkę. <<<

# Rozstrzygnięcie, które modele pokazać na wykresie.
if model_choice == "Wszystkie":
    selected_models = AVAILABLE_MODELS
else:
    selected_models = [model_choice]

# ---------------------------------------------------------------------
# GŁÓWNY PANEL - wyniki
# ---------------------------------------------------------------------
st.header(f"Analiza: {selected_ticker}")

# --- Sekcja 1: podstawowe informacje o wybranym ETF -------------------
history_df = fetch_history_mock(selected_ticker)
last_price = history_df["close"].iloc[-1]
first_price = history_df["close"].iloc[0]
period_return = (last_price / first_price - 1) * 100

info_col1, info_col2, info_col3, info_col4 = st.columns(4)
info_col1.metric("Ticker", selected_ticker)
info_col2.metric("Aktualna cena (mock)", f"{last_price:,.2f} USD")
info_col3.metric("Zwrot w okresie", f"{period_return:+.2f}%")
info_col4.metric(
    "Liczba sesji w danych",
    len(history_df),
    help="Liczba sesji giełdowych pobranych z backendu.",
)

st.divider()

# --- Sekcja 2: wykres predykcje vs. rzeczywistość ---------------------
st.subheader("Cena rzeczywista vs. predykcje modeli (okres testowy)")

if run_prediction or st.session_state.get("last_run") is not None:
    if run_prediction:
        # Blokada zapisu stanu: przeliczamy tylko po kliknięciu przycisku.
        st.session_state["last_run"] = {
            "ticker": selected_ticker,
            "models": list(selected_models),
        }

    run_config = st.session_state["last_run"]
    run_history = fetch_history_mock(run_config["ticker"])
    predictions = fetch_predictions_mock(
        run_history, run_config["models"]
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=run_history["date"],
            y=run_history["close"],
            mode="lines",
            name="Cena rzeczywista",
            line=dict(color="#1f77b4", width=2),
            # Czytelny hover: data + cena w USD.
            hovertemplate="Data: %{x|%Y-%m-%d}<br>Cena: %{y:.2f} USD<extra>Rzeczywista</extra>",
        )
    )

    # Kolory predykcji: czerwony / zielony / pomarańczowy.
    model_colors = {
        "Random Forest": "#d62728",
        "LSTM": "#2ca02c",
        "GRU": "#ff7f0e",
    }
    for model in run_config["models"]:
        pred_df = predictions[model]
        fig.add_trace(
            go.Scatter(
                x=pred_df["date"],
                y=pred_df["prediction"],
                mode="lines",
                name=f"Predykcja: {model}",
                line=dict(color=model_colors.get(model, "#9467bd"), width=2, dash="dash"),
                hovertemplate=f"Model: {model}<br>Data: %{{x|%Y-%m-%d}}<br>Predykcja: %{{y:.2f}} USD<extra></extra>",
            )
        )

    fig.update_layout(
        height=520,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(
            title="Data",
            # Włączony zoom i przesuwanie wykresu myszką.
            rangeslider=dict(visible=True),
        ),
        yaxis=dict(title="Cena zamknięcia (USD)"),
        hovermode="x unified",
    )

    st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True})

    # --- Sekcja 3: tabela metryk walidacji ----------------------------
    st.subheader("Metryki błędu z walidacji")
    # MOCK: docelowo metryki przyjdą z API (fetch_metrics_from_api),
    # a poniższa pętla zostanie zastąpiona listą odpowiedzi z backendu.
    metrics_rows = []
    for model in run_config["models"]:
        pred = predictions[model][predictions[model]["date"].isin(run_history.tail(60)["date"])]
        true_vals = run_history[run_history["date"].isin(pred["date"])]["close"].values
        metrics_rows.append(compute_metrics_mock(model, true_vals, pred["prediction"].values))

    metrics_df = pd.DataFrame(metrics_rows).set_index("Model")
    st.dataframe(metrics_df, use_container_width=True)

    st.caption(
        "MAE - średni błąd bezwzględny | RMSE - pierwiastek błędu średniokwadratowego | "
        "MAPE - błąd procentowy średni bezwzględny."
    )
else:
    st.info(
        "Ustaw parametry w panelu bocznym i kliknij **▶️ Wygeneruj predykcję**, "
        "aby zobaczyć wykres i metryki."
    )
