export function formatPrice(value: number, currency = 'USD'): string {
  try {
    return new Intl.NumberFormat('en-US', { style: 'currency', currency, maximumFractionDigits: 2 }).format(value)
  } catch {
    return value.toFixed(2)
  }
}

/** 0.0312 → "+3.12%" */
export function formatPct(fraction: number, { signed = false, digits = 1 } = {}): string {
  const text = `${(fraction * 100).toFixed(digits)}%`
  return signed && fraction > 0 ? `+${text}` : text
}

export function formatCompact(value: number): string {
  return new Intl.NumberFormat('en-US', { notation: 'compact', maximumFractionDigits: 2 }).format(value)
}

/** Formats the exchange's own wall-clock time from an ISO string like
 * "2026-11-18T15:55:00-05:00", without converting to the viewer's time zone. */
export function formatDateTime(iso: string, intraday: boolean): string {
  const wallClock = new Date(`${iso.slice(0, 16)}Z`)
  const opts: Intl.DateTimeFormatOptions = intraday
    ? { month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit', timeZone: 'UTC' }
    : { month: 'short', day: 'numeric', year: 'numeric', timeZone: 'UTC' }
  const text = wallClock.toLocaleString('en-US', opts)
  return intraday ? `${text} exchange time` : text
}
