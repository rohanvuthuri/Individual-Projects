import type { Quote } from '../api/types'
import { formatPrice } from '../lib/format'

const MARKET_STATE: Record<string, string> = {
  REGULAR: 'Market open',
  PRE: 'Pre-market',
  PREPRE: 'Pre-market',
  POST: 'After hours',
  POSTPOST: 'Market closed',
  CLOSED: 'Market closed',
}

export function QuoteHeader({ quote }: { quote: Quote }) {
  const up = quote.change >= 0
  return (
    <div className="quote-header">
      <div>
        <h1>
          {quote.name} <span className="muted">{quote.ticker}</span>
        </h1>
        <p className="muted">
          {[quote.exchange, quote.market_state && MARKET_STATE[quote.market_state]].filter(Boolean).join(' · ')}
        </p>
      </div>
      <div className="quote-header__price">
        <span className="num">{formatPrice(quote.price, quote.currency)}</span>
        <span className={`num ${up ? 'good-text' : 'critical-text'}`}>
          {up ? '▲' : '▼'} {formatPrice(Math.abs(quote.change), quote.currency)} ({up ? '+' : ''}
          {quote.change_pct.toFixed(2)}%)
        </span>
      </div>
    </div>
  )
}
