import type { Prediction, Signal } from '../api/types'
import { formatDateTime, formatPct, formatPrice } from '../lib/format'

const SIGNAL_ICON: Record<Signal, string> = { Buy: '▲', Hold: '■', Sell: '▼' }

const AGREEMENT_TEXT = {
  agree: 'Both models point the same way.',
  disagree: 'The models disagree, so the call is Hold.',
  neutral: 'Gradient boosting has no strong view, so the call is Hold.',
}

function MethodNote({ p }: { p: Prediction }) {
  if (p.method === 'gbm_only') {
    return (
      <>
        <span className="spinner" aria-hidden="true" /> Provisional: the LSTM is still training, so this call is based on gradient
        boosting alone.
      </>
    )
  }
  if (p.method === 'stacked') {
    return <>Call made by the stacking meta-model, which learned from the backtest how much to trust each model.</>
  }
  const agreement = p.agreement ? AGREEMENT_TEXT[p.agreement] : ''
  if (p.stack && !p.stack.earned_trust) {
    return (
      <>
        The meta-model didn't beat a naive guess in its backtest, so the call uses the simpler rule (Buy/Sell only when both models
        agree). {agreement}
      </>
    )
  }
  return <>{agreement}</>
}

function MetaTile({ stack, horizon }: { stack: NonNullable<Prediction['stack']>; horizon: string }) {
  const g = Math.max(stack.weights.gbm, 0)
  const l = Math.max(stack.weights.lstm, 0)
  const total = g + l
  const gbmShare = total > 0 ? g / total : 0.5
  // Below this the meta-model is effectively ignoring both models.
  const ignoresBoth = total < 0.02

  return (
    <div className={`model-tile model-tile--meta${stack.earned_trust ? '' : ' is-unused'}`}>
      <div className="model-tile__row">
        <span className="model-tile__label">Stacking meta-model</span>
        {stack.earned_trust ? (
          <span className="tag tag--good">✓ Used for the call</span>
        ) : (
          <span className="tag">Not used: didn't beat a naive guess</span>
        )}
      </div>
      <div className="model-tile__row">
        <span className="model-tile__value num">{formatPct(stack.prob_up, { digits: 0 })}</span>
        <span className="muted">chance the price is higher in {horizon}</span>
      </div>
      {ignoresBoth ? (
        <span className="muted">Learned to ignore both models; it is predicting the historical up-rate.</span>
      ) : (
        <div className="trust" aria-label={`Trusts gradient boosting ${formatPct(gbmShare, { digits: 0 })}, LSTM ${formatPct(1 - gbmShare, { digits: 0 })}`}>
          <span className="trust__label">Trust right now</span>
          <div className="trust__bar" aria-hidden="true">
            <span className="trust__gbm" style={{ width: `${gbmShare * 100}%` }} />
            <span className="trust__lstm" style={{ width: `${(1 - gbmShare) * 100}%` }} />
          </div>
          <div className="trust__legend">
            <span>
              <span className="swatch trust__swatch--gbm" /> Gradient boosting {formatPct(gbmShare, { digits: 0 })}
            </span>
            <span>
              <span className="swatch trust__swatch--lstm" /> LSTM {formatPct(1 - gbmShare, { digits: 0 })}
            </span>
          </div>
        </div>
      )}
    </div>
  )
}

export function PredictionCard({ prediction: p }: { prediction: Prediction }) {
  const intraday = p.mode === 'day'
  const horizon = p.horizon_label.replace('~', '')
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
        <MethodNote p={p} />
      </p>

      <div className="model-tiles">
        {p.stack && <MetaTile stack={p.stack} horizon={horizon} />}
        <div className="model-tile">
          <span className="model-tile__label">Gradient boosting</span>
          <span className="model-tile__value num">{formatPct(p.gbm.prob_up, { digits: 0 })}</span>
          <span className="muted">chance the price is higher in {horizon}</span>
        </div>
        <div className="model-tile">
          <span className="model-tile__label">LSTM neural network</span>
          {lstmReady ? (
            <>
              <span className="model-tile__value num">{formatPrice(p.lstm.target_price!)}</span>
              <span className={lstmReturn >= 0 ? 'good-text' : 'critical-text'}>
                {lstmReturn >= 0 ? '▲' : '▼'} {formatPct(lstmReturn, { signed: true, digits: 2 })} price target
              </span>
              {p.lstm.retraining && (
                <span className="muted">
                  <span className="spinner" aria-hidden="true" /> Updating with newer data…
                </span>
              )}
            </>
          ) : p.lstm.status === 'error' ? (
            <span className="critical-text">Training failed: {p.lstm.error}</span>
          ) : (
            <>
              <span className="model-tile__value muted">Training…</span>
              <span className="muted">usually 30–60 seconds the first time; the meta-model trains right after</span>
            </>
          )}
        </div>
      </div>

      <p className="prediction__range muted">
        A typical {horizon} move for this stock is ±{formatPct(p.typical_move)}, about{' '}
        <span className="num">
          {formatPrice(p.expected_range.low)} – {formatPrice(p.expected_range.high)}
        </span>{' '}
        from {formatPrice(p.last_price)}.
      </p>
    </section>
  )
}
