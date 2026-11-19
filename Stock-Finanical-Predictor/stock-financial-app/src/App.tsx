import { useFundamentals, useHistory, usePrediction, useQuote } from './api/client'
import { BacktestCard } from './components/BacktestCard'
import { Disclaimer } from './components/Disclaimer'
import { ErrorBoundary } from './components/ErrorBoundary'
import { FundamentalsCard } from './components/FundamentalsCard'
import { ModeTabs } from './components/ModeTabs'
import { PredictionCard } from './components/PredictionCard'
import { PriceChart } from './components/PriceChart'
import { QuoteHeader } from './components/QuoteHeader'
import { TickerSearch } from './components/TickerSearch'
import { TopDrivers } from './components/TopDrivers'
import { useChartTheme } from './hooks/useChartTheme'
import { useUrlState } from './hooks/useUrlState'
import './App.css'

const POPULAR = ['AAPL', 'MSFT', 'NVDA', 'TSLA', 'AMZN', 'SPY']

function Status({ title, children, loading = false }: { title?: string; children: React.ReactNode; loading?: boolean }) {
  return (
    <div className={`card status ${loading ? 'status--loading' : 'status--error'}`} role={loading ? 'status' : 'alert'}>
      {loading && <span className="spinner" aria-hidden="true" />}
      <div>
        {title && <b>{title}</b>}
        <p>{children}</p>
      </div>
    </div>
  )
}

function App() {
  const { ticker, mode, update } = useUrlState()
  const theme = useChartTheme()
  const quote = useQuote(ticker, mode)
  const history = useHistory(ticker, mode)
  const prediction = usePrediction(ticker, mode)
  const fundamentals = useFundamentals(ticker, mode === 'long')

  const notFound = quote.error && 'status' in quote.error && quote.error.status === 404

  return (
    <div className="app">
      <header className="topbar">
        <span className="brand">
          <span className="brand__mark" aria-hidden="true">◆</span> Stock Predictor
        </span>
        <TickerSearch onSelect={(t) => update({ ticker: t })} />
        <nav className="popular" aria-label="Popular tickers">
          {POPULAR.map((t) => (
            <button key={t} type="button" className={t === ticker ? 'is-active' : undefined} onClick={() => update({ ticker: t })}>
              {t}
            </button>
          ))}
        </nav>
      </header>

      <main>
        {notFound ? (
          <Status title={`Couldn't find “${ticker}”`}>Check the symbol, or search by company name above.</Status>
        ) : (
          <>
            <div className="overview">
              {quote.data ? (
                <QuoteHeader quote={quote.data} />
              ) : quote.error ? (
                <Status>{quote.error.message}</Status>
              ) : (
                <div className="quote-header quote-header--loading">
                  <h1>{ticker}</h1>
                </div>
              )}
              <ModeTabs mode={mode} onChange={(m) => update({ mode: m })} />
            </div>

            <div className="layout">
              <div className="layout__main">
                <section className="card" aria-label="Price chart">
                  {history.data ? (
                    <ErrorBoundary key={`${ticker}-${mode}`} label="price chart">
                      <PriceChart history={history.data} theme={theme} />
                    </ErrorBoundary>
                  ) : history.error ? (
                    <Status>{history.error.message}</Status>
                  ) : (
                    <Status loading>Loading prices…</Status>
                  )}
                </section>
                {prediction.data && (
                  <ErrorBoundary key={`backtest-${ticker}-${mode}`} label="backtest">
                    <BacktestCard prediction={prediction.data} theme={theme} />
                  </ErrorBoundary>
                )}
              </div>

              <aside className="layout__side">
                {prediction.data ? (
                  <>
                    <PredictionCard prediction={prediction.data} />
                    <TopDrivers importances={prediction.data.gbm.importances} />
                  </>
                ) : prediction.error ? (
                  <Status title="No prediction">{prediction.error.message}</Status>
                ) : (
                  <Status loading>Training the gradient-boosting model on {ticker}'s history… this takes about 5–10 seconds.</Status>
                )}
                {mode === 'long' &&
                  (fundamentals.data ? (
                    <FundamentalsCard data={fundamentals.data} />
                  ) : fundamentals.isLoading ? (
                    <Status loading>Loading fundamentals…</Status>
                  ) : null)}
              </aside>
            </div>
          </>
        )}
      </main>

      <Disclaimer />
    </div>
  )
}

export default App
