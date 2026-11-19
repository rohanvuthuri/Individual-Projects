export function Disclaimer() {
  return (
    <footer className="disclaimer">
      <p>
        <b>Educational project, not financial advice.</b> Predictions come from models trained on past prices, which do not guarantee
        future results. Most stock-prediction models barely beat chance; check the backtest before trusting any signal. Data from Yahoo
        Finance via yfinance, delayed and unofficial.
      </p>
      <p className="muted">
        Charts by{' '}
        <a href="https://www.tradingview.com/" target="_blank" rel="noreferrer">
          TradingView Lightweight Charts™
        </a>
        .
      </p>
    </footer>
  )
}
