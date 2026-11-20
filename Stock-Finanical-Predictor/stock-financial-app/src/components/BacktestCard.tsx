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

function SamePeriodComparison({ comparison: c }: { comparison: NonNullable<Prediction['stack']>['comparison'] }) {
  const rows = [
    { label: 'Gradient boosting', value: c.gbm },
    { label: 'LSTM', value: c.lstm },
    { label: 'Stacked meta-model', value: c.stacked },
  ]
  const best = Math.max(...rows.map((r) => r.value))
  return (
    <>
      <h3 className="card__subhead">Head to head on the same {c.samples.toLocaleString()} test periods</h3>
      <p className="muted card__intro">
        Each base model's numbers above cover slightly different periods. This scores all three on identical ones, the fair way to
        see whether stacking helped. The tick is the naive guess ({formatPct(c.naive)}).
      </p>
      <ul className="comparison">
        {rows.map((r) => (
          <li key={r.label}>
            <span className={r.value === best ? 'comparison__label is-best' : 'comparison__label'}>{r.label}</span>
            <div className="bullet" aria-hidden="true">
              <div className="bullet__fill" style={{ width: `${r.value * 100}%` }} />
              <div className="bullet__target" style={{ left: `${c.naive * 100}%` }} />
            </div>
            <span className="num comparison__value">{formatPct(r.value)}</span>
          </li>
        ))}
      </ul>
    </>
  )
}

export function BacktestCard({ prediction: p, theme }: { prediction: Prediction; theme: ChartTheme }) {
  const { gbm, lstm, stack } = p
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
      {stack && (
        <ModelRow
          name="Stacked meta-model"
          metrics={stack.metrics}
          extra={stack.metrics.auc !== null ? `AUC ${stack.metrics.auc.toFixed(2)} (needs > 0.52 to be used)` : ''}
        />
      )}

      {stack && <SamePeriodComparison comparison={stack.comparison} />}

      <h3 className="card__subhead">Gradient boosting strategy vs. buy &amp; hold</h3>
      <p className="muted card__intro">
        Holds the stock only while the model says “up” (before trading costs), across the same test periods.
      </p>
      <EquityChart points={gbm.equity_curve} theme={theme} intraday={p.mode === 'day'} />
    </section>
  )
}
