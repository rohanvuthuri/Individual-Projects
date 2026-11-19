import type { Prediction, Signal } from '../api/types'
import { formatDateTime, formatPct, formatPrice } from '../lib/format'

const SIGNAL_ICON: Record<Signal, string> = { Buy: '▲', Hold: '■', Sell: '▼' }

const AGREEMENT_TEXT = {
  agree: 'Both models point the same way.',
  disagree: 'The models disagree, so the call is Hold.',
  neutral: 'Gradient boosting has no strong view, so the call is Hold.',
}

export function PredictionCard({ prediction: p }: { prediction: Prediction }) {
  const intraday = p.mode === 'day'
  const lstmReady = p.lstm.status === 'ready' && p.lstm.target_price !== undefined
  const lstmReturn = p.lstm.predicted_return ?? 0

  return (
    <section className="card prediction" aria-labelledby="prediction-title">
      <header className="card__header">
        <h2 id="prediction-title">Prediction · {p.horizon_label}</h2>
        <span className="muted">as of {formatDateTime(p.as_of, intraday)}</span>
      </header>

      <div className="prediction__headline">
        <span className={`signal signal--${p.signal.toLowerCase()}`}>
          <span aria-hidden="true">{SIGNAL_ICON[p.signal]}</span> {p.signal}
        </span>
        <div className="prediction__confidence">
          <span className="muted">Confidence</span>
          <meter min={0} max={1} value={p.confidence} aria-label="Confidence" />
          <span className="num">{Math.round(p.confidence * 100)}%</span>
        </div>
      </div>

      <p className="prediction__note">
        {p.provisional ? (
          <>
            <span className="spinner" aria-hidden="true" /> Provisional: the LSTM is still training, so this call is based on
            gradient boosting alone.
          </>
        ) : (
          p.agreement && AGREEMENT_TEXT[p.agreement]
        )}
      </p>

      <div className="model-tiles">
        <div className="model-tile">
          <span className="model-tile__label">Gradient boosting</span>
          <span className="model-tile__value num">{formatPct(p.gbm.prob_up, { digits: 0 })}</span>
          <span className="muted">chance the price is higher in {p.horizon_label.replace('~', '')}</span>
        </div>
        <div className="model-tile">
          <span className="model-tile__label">LSTM neural network</span>
          {lstmReady ? (
            <>
              <span className="model-tile__value num">{formatPrice(p.lstm.target_price!)}</span>
              <span className={lstmReturn >= 0 ? 'good-text' : 'critical-text'}>
                {lstmReturn >= 0 ? '▲' : '▼'} {formatPct(lstmReturn, { signed: true, digits: 2 })} price target
              </span>
            </>
          ) : p.lstm.status === 'error' ? (
            <span className="critical-text">Training failed: {p.lstm.error}</span>
          ) : (
            <>
              <span className="model-tile__value muted">Training…</span>
              <span className="muted">usually 10–60 seconds the first time</span>
            </>
          )}
        </div>
      </div>

      <p className="prediction__range muted">
        A typical {p.horizon_label.replace('~', '')} move for this stock is ±{formatPct(p.typical_move)}, about{' '}
        <span className="num">
          {formatPrice(p.expected_range.low)} – {formatPrice(p.expected_range.high)}
        </span>{' '}
        from {formatPrice(p.last_price)}.
      </p>
    </section>
  )
}
