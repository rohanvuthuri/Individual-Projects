import type { BacktestMetrics, Prediction } from '../api/types'
import type { ChartTheme } from '../hooks/useChartTheme'
import { formatPct } from '../lib/format'
import { EquityChart } from './EquityChart'

function ModelRow({ name, metrics, extra }: { name: string; metrics: BacktestMetrics; extra: string }) {
  // The bar to beat: always guessing whichever direction happened more often.
  const naive = Math.max(metrics.baseline, 1 - metrics.baseline)
  return (
    <div className="backtest-row">
      <div className="backtest-row__title">
        <b>{name}</b>
        {metrics.beats_baseline ? (
          <span className="good-text">✓ Beats a naive guess</span>
        ) : (
          <span className="warning-text">⚠ No better than a naive guess</span>
        )}
      </div>
      <div className="bullet" role="img" aria-label={`Accuracy ${formatPct(metrics.accuracy)} versus naive guess ${formatPct(naive)}`}>
        <div className="bullet__fill" style={{ width: `${metrics.accuracy * 100}%` }} />
        <div className="bullet__target" style={{ left: `${naive * 100}%` }} />
      </div>
      <div className="backtest-row__stats num">
        <span>
          Accuracy <b>{formatPct(metrics.accuracy)}</b>
        </span>
        <span>
          Naive guess <b>{formatPct(naive)}</b>
        </span>
        <span className="muted">{extra}</span>
      </div>
      <p className="muted backtest-row__period">
        Tested on {metrics.test_samples.toLocaleString()} unseen periods, {metrics.test_start} → {metrics.test_end}. Price went up{' '}
        {formatPct(metrics.baseline, { digits: 0 })} of the time.
      </p>
    </div>
  )
}

export function BacktestCard({ prediction: p, theme }: { prediction: Prediction; theme: ChartTheme }) {
  const { gbm, lstm } = p
  return (
    <section className="card" aria-labelledby="backtest-title">
      <header className="card__header">
        <h2 id="backtest-title">How accurate were the models?</h2>
      </header>
      <p className="muted card__intro">
        Each model was trained only on the past and tested on later data it had never seen. The bar shows its accuracy; the tick marks
        what you'd get by always guessing the more common direction.
      </p>

      <ModelRow
        name="Gradient boosting"
        metrics={gbm.metrics}
        extra={gbm.metrics.auc !== null ? `AUC ${gbm.metrics.auc.toFixed(2)} (0.5 = coin flip)` : ''}
      />
      {lstm.metrics ? (
        <ModelRow
          name="LSTM"
          metrics={lstm.metrics}
          extra={`Correlation with actual moves ${lstm.metrics.correlation.toFixed(2)}`}
        />
      ) : (
        <p className="muted">LSTM backtest appears once training finishes.</p>
      )}

      <h3 className="card__subhead">Gradient boosting strategy vs. buy &amp; hold</h3>
      <p className="muted card__intro">
        Holds the stock only while the model says “up” (before trading costs), across the same test periods.
      </p>
      <EquityChart points={gbm.equity_curve} theme={theme} intraday={p.mode === 'day'} />
    </section>
  )
}
