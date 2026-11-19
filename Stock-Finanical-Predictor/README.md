# Stock Financial Predictor

A stock dashboard that predicts price direction for three investing styles:

| Mode | Data | Predicts | Chart overlays | Extras |
|---|---|---|---|---|
| **Long-term** | Daily bars, 10 years | ~6 months ahead | SMA 50/200, 52-week high/low | Fundamentals scorecard |
| **Swing** | Daily bars, 5 years | ~2 weeks ahead | EMA 20/50, Bollinger Bands | |
| **Day trade** | 5-minute bars, 60 days | ~1 hour ahead | VWAP, EMA 9/21 | Auto-refreshes every minute |

Each prediction combines two models:

- **Gradient boosting** (scikit-learn `HistGradientBoostingClassifier`) on ~20 technical indicators gives the **probability the price is higher** after the horizon.
- **LSTM neural network** (Keras) over the last 48–60 bars forecasts the **return over the horizon**, which gives a price target.
- **Signal:** **Buy** or **Sell** only when both models agree. Otherwise **Hold**.

Every model is **backtested walk-forward**: it is trained on the past and scored on later data it never saw. The UI compares its accuracy with a naive "always guess the common direction" baseline. Most of the time the models do *not* beat that baseline, and the app says so. Stock prices are mostly noise, and the backtest is there to keep the predictions honest.

> Educational project, not financial advice.

## Running it

You need Python 3.11+ and Node 20.19+ (or 22.12+). The app needs two terminals.

**1. Backend** (FastAPI on port 8000):

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

**2. Frontend** (Vite on port 5173, which forwards `/api` to the backend):

```bash
cd stock-financial-app
npm install
npm run dev
```

Open http://localhost:5173. You don't need any API keys: prices come from Yahoo Finance via `yfinance`.

The first prediction for a ticker and mode takes about 5–10 seconds while gradient boosting trains. The LSTM then trains in the background (roughly 10–60 seconds), and the page updates when it's done. Trained models are saved in `backend/model_cache/`. They are retrained after 1 hour (day trade), 1 day (swing) or 1 week (long-term).

## Tests

```bash
cd backend && .venv/bin/python -m pytest
cd stock-financial-app && npm run lint && npm run build
```

The backend tests check:

- the indicators against hand-calculated values
- that features never use future prices (look-ahead leakage)
- every API route, using synthetic data so the tests need no network

## How it fits together

```
backend/app/
  main.py          API routes: /api/search, /quote, /history, /fundamentals, /predict
  modes.py         per-mode settings (bar size, history, horizon, retrain age)
  data.py          yfinance downloads + in-memory cache
  indicators.py    RSI, MACD, Bollinger, ATR, OBV, VWAP… in plain pandas
  features.py      model inputs + labels (forward returns), no look-ahead
  fundamentals.py  P/E, PEG, margins, growth… scored good / neutral / weak
  signals.py       combines both models into Buy / Hold / Sell + confidence
  models/gbm.py    gradient boosting + walk-forward backtest + equity curve
  models/lstm.py   LSTM + held-out backtest
  models/registry.py  model caching and background LSTM training

stock-financial-app/src/
  api/             typed fetch + React Query hooks (polls while the LSTM trains)
  components/      PriceChart, PredictionCard, BacktestCard, TopDrivers, FundamentalsCard…
  hooks/           URL state (?t=AAPL&m=swing), chart theme colors
```

## Caveats

- yfinance is unofficial and delayed (about 15 minutes intraday). Yahoo can change or rate-limit it at any time.
- Fundamentals reflect today's values only, so they appear in the scorecard but are **not** used to train the models. Using them would leak future information into the backtest.
- Day-trade labels never cross the market close. The model predicts moves within the same session only.
- The strategy curve in the backtest ignores trading costs and slippage.
