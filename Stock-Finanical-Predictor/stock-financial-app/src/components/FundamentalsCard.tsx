import type { FundamentalMetric, Fundamentals, Rating } from '../api/types'
import { formatCompact, formatPrice } from '../lib/format'

const RATING: Record<Rating, { icon: string; label: string; className: string }> = {
  good: { icon: '✓', label: 'Good', className: 'good-text' },
  neutral: { icon: '–', label: 'Neutral', className: 'muted' },
  poor: { icon: '✗', label: 'Weak', className: 'critical-text' },
  info: { icon: 'i', label: 'Info', className: 'muted' },
}

function formatMetric(m: FundamentalMetric): string {
  if (m.format === 'pct') return `${(m.value * 100).toFixed(1)}%`
  if (m.format === 'pct_raw') return `${m.value.toFixed(m.value < 10 ? 2 : 0)}%`
  return m.value.toFixed(2)
}

export function FundamentalsCard({ data }: { data: Fundamentals }) {
  const { profile } = data
  return (
    <section className="card" aria-labelledby="fundamentals-title">
      <header className="card__header">
        <h2 id="fundamentals-title">Company fundamentals</h2>
        {data.summary && <span className={`pill pill--${data.summary.toLowerCase()}`}>{data.summary}</span>}
      </header>
      <p className="muted card__intro">
        Today's financial health, scored with simple rules of thumb. Shown alongside the models; not used to train them.
      </p>

      {!data.available ? (
        <p className="muted">No fundamentals available for this symbol (common for ETFs, indexes and crypto).</p>
      ) : (
        <table className="fundamentals">
          <tbody>
            {data.metrics.map((m) => {
              const r = RATING[m.rating]
              return (
                <tr key={m.key} title={m.explanation}>
                  <th scope="row">{m.label}</th>
                  <td className="num">{formatMetric(m)}</td>
                  <td className={r.className}>
                    <span aria-hidden="true">{r.icon}</span> {r.label}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      )}

      <dl className="profile">
        {profile.sector && (
          <>
            <dt>Sector</dt>
            <dd>{[profile.sector, profile.industry].filter(Boolean).join(' · ')}</dd>
          </>
        )}
        {profile.market_cap && (
          <>
            <dt>Market cap</dt>
            <dd className="num">${formatCompact(profile.market_cap)}</dd>
          </>
        )}
        {profile.week52_low && profile.week52_high && (
          <>
            <dt>52-week range</dt>
            <dd className="num">
              {formatPrice(profile.week52_low)} – {formatPrice(profile.week52_high)}
            </dd>
          </>
        )}
        {profile.analyst_target && (
          <>
            <dt>Analyst target</dt>
            <dd className="num">
              {formatPrice(profile.analyst_target)}
              {profile.analyst_rating && ` · ${profile.analyst_rating.replace('_', ' ')}`}
              {profile.analyst_count && <span className="muted"> ({profile.analyst_count} analysts)</span>}
            </dd>
          </>
        )}
      </dl>
    </section>
  )
}
