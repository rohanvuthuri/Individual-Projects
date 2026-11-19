import { useEffect, useState } from 'react'
import type { ModeKey } from '../api/types'

const MODES: ModeKey[] = ['long', 'swing', 'day']
const DEFAULT_TICKER = 'AAPL'
const DEFAULT_MODE: ModeKey = 'swing'

const isMode = (value: string | null): value is ModeKey => MODES.includes(value as ModeKey)

/** Ticker and mode live in the URL (?t=AAPL&m=swing) so a view can be shared or reloaded. */
export function useUrlState() {
  const [params, setParams] = useState(() => new URLSearchParams(window.location.search))

  useEffect(() => {
    const onPop = () => setParams(new URLSearchParams(window.location.search))
    window.addEventListener('popstate', onPop)
    return () => window.removeEventListener('popstate', onPop)
  }, [])

  const ticker = params.get('t')?.trim().toUpperCase() || DEFAULT_TICKER
  const modeParam = params.get('m')
  const mode = isMode(modeParam) ? modeParam : DEFAULT_MODE

  const update = (next: { ticker?: string; mode?: ModeKey }) => {
    const p = new URLSearchParams(window.location.search)
    if (next.ticker) p.set('t', next.ticker.toUpperCase())
    if (next.mode) p.set('m', next.mode)
    window.history.pushState(null, '', `?${p}`)
    setParams(p)
  }

  return { ticker, mode, update }
}
