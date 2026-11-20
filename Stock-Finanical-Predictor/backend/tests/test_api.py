"""API smoke tests with yfinance replaced by synthetic data (no network, no TensorFlow)."""

from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from app import data, main
from app.data import TickerNotFound
from app.features import build_dataset
from app.models import registry, stacking
from app.modes import MODES

from .conftest import make_prices


@pytest.fixture
def client(monkeypatch, tmp_path):
    def fake_prices(ticker, mode):
        ticker = data.normalize_ticker(ticker)
        if ticker == "NOPE":
            raise TickerNotFound("No price data found for 'NOPE'")
        if ticker == "NEW":
            return make_prices(250)
        return make_prices(78 * 30, freq="5min") if mode.interval == "5m" else make_prices(2600)

    monkeypatch.setattr(main.data, "get_prices", fake_prices)
    monkeypatch.setattr("app.fundamentals.get_info", lambda t: {
        "trailingPE": 12.0, "debtToEquity": 200.0, "profitMargins": 0.2, "beta": 1.1, "sector": "Tech",
    })
    monkeypatch.setattr(registry, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(registry, "get_lstm", lambda *a: ("training", None, None, True))
    registry._memory.clear()
    return TestClient(main.app)


@pytest.mark.parametrize("mode", ["long", "swing", "day"])
def test_history(client, mode):
    r = client.get(f"/api/history/TEST?mode={mode}")
    assert r.status_code == 200
    body = r.json()
    assert body["bars"] and {"time", "open", "high", "low", "close", "volume"} <= body["bars"][0].keys()
    assert body["overlays"] and all(o["points"] for o in body["overlays"])
    assert body["rsi"] and body["macd"]["hist"]


@pytest.mark.parametrize("mode", ["long", "swing", "day"])
def test_predict_without_lstm(client, mode):
    r = client.get(f"/api/predict/TEST?mode={mode}")
    assert r.status_code == 200
    body = r.json()
    assert body["signal"] in {"Buy", "Hold", "Sell"}
    assert body["provisional"] is True
    assert 0 <= body["gbm"]["prob_up"] <= 1
    assert {"accuracy", "baseline", "beats_baseline"} <= body["gbm"]["metrics"].keys()
    assert body["gbm"]["equity_curve"] and body["gbm"]["importances"]
    assert body["lstm"]["status"] == "training"


def test_predict_with_stacked_meta_model(client, monkeypatch):
    """Once the LSTM is ready, the meta-model makes the call, but only if it earned trust."""
    # Reuse the real gradient-boosting run's out-of-fold predictions; fake the LSTM's.
    client.get("/api/predict/TEST?mode=swing")
    g = next(b for k, b in registry._memory.items() if k[2] == "gbm")
    lstm_oof = g["oof_prob"] - 0.5
    X, fwd_ret = build_dataset(make_prices(2600), MODES["swing"])
    stack = stacking.train(g["oof_prob"], lstm_oof, X, fwd_ret, MODES["swing"])
    assert stack is not None

    metrics = {**g["metrics"], "rmse": 0.01, "correlation": 0.1, "epochs": 3}
    bundle = {"metrics": metrics, "trained_at": datetime.now(timezone.utc), "stack": stack}
    monkeypatch.setattr(registry, "get_lstm", lambda *a: ("ready", bundle, None, False))
    monkeypatch.setattr(main.lstm, "predict", lambda *a: 0.01)

    # Random-walk prices have nothing to learn, so the meta-model is shown but not used.
    stack["earned_trust"] = False
    body = client.get("/api/predict/TEST?mode=swing").json()
    assert body["method"] == "agreement"
    assert body["stack"]["earned_trust"] is False

    stack["earned_trust"] = True
    body = client.get("/api/predict/TEST?mode=swing").json()
    assert body["method"] == "stacked"
    assert body["provisional"] is False
    assert 0 <= body["stack"]["prob_up"] <= 1
    assert body["stack"]["weights"].keys() == {"gbm", "lstm"}
    assert {"gbm", "lstm", "stacked", "naive", "samples"} <= body["stack"]["comparison"].keys()


def test_fundamentals_scores(client):
    body = client.get("/api/fundamentals/TEST").json()
    ratings = {m["key"]: m["rating"] for m in body["metrics"]}
    assert ratings == {"trailingPE": "good", "debtToEquity": "poor", "profitMargins": "good", "beta": "info"}
    assert body["summary"] == "Strong"


def test_errors(client):
    assert client.get("/api/predict/NOPE?mode=swing").status_code == 404
    assert client.get("/api/history/bad$ticker").status_code == 404
    assert client.get("/api/predict/TEST?mode=yearly").status_code == 400
    r = client.get("/api/predict/NEW?mode=long")
    assert r.status_code == 422 and "history" in r.json()["detail"]
