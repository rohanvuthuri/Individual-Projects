import { useEffect, useRef, useState } from 'react'
import { LineSeries, createChart, type UTCTimestamp } from 'lightweight-charts'
import type { EquityPoint } from '../api/types'
import type { ChartTheme } from '../hooks/useChartTheme'
import { formatPct } from '../lib/format'

interface Props {
  points: EquityPoint[]
  theme: ChartTheme
  intraday: boolean
}

/** Growth of $1: following the model vs. just holding the stock, over the backtest period. */
export function EquityChart({ points, theme, intraday }: Props) {
  const containerRef = useRef<HTMLDivElement>(null)
  const [hoverIndex, setHoverIndex] = useState<number | null>(null)

  useEffect(() => {
    const container = containerRef.current
    if (!container || points.length === 0) return
    const chart = createChart(container, {
      autoSize: true,
      layout: { background: { color: theme.surface }, textColor: theme.muted, attributionLogo: false },
      grid: { vertLines: { visible: false }, horzLines: { color: theme.grid } },
      rightPriceScale: { borderColor: theme.axis },
      timeScale: { borderColor: theme.axis, timeVisible: intraday },
      handleScroll: false,
      handleScale: false,
      crosshair: {
        vertLine: { color: theme.muted, labelBackgroundColor: theme.axis },
        horzLine: { visible: false, labelVisible: false },
      },
    })
    const opts = { lineWidth: 2 as const, priceLineVisible: false, lastValueVisible: false }
    const strategy = chart.addSeries(LineSeries, { ...opts, color: theme.series[0] })
    const buyHold = chart.addSeries(LineSeries, { ...opts, color: theme.series[1] })
    strategy.setData(points.map((p) => ({ time: p.time as UTCTimestamp, value: p.strategy })))
    buyHold.setData(points.map((p) => ({ time: p.time as UTCTimestamp, value: p.buy_hold })))
    chart.timeScale().fitContent()
    chart.subscribeCrosshairMove((param) => {
      setHoverIndex(param.time === undefined ? null : points.findIndex((p) => p.time === param.time))
    })
    return () => chart.remove()
  }, [points, theme, intraday])

  const p = points[hoverIndex ?? points.length - 1]
  if (!p) return null

  return (
    <div>
      <ul className="equity-legend">
        <li>
          <span className="swatch" style={{ background: theme.series[0] }} />
          Following the model <b className="num">{formatPct(p.strategy - 1, { signed: true })}</b>
        </li>
        <li>
          <span className="swatch" style={{ background: theme.series[1] }} />
          Buy &amp; hold <b className="num">{formatPct(p.buy_hold - 1, { signed: true })}</b>
        </li>
      </ul>
      <div className="equity-chart" ref={containerRef} role="img" aria-label="Backtest growth: model strategy versus buy and hold" />
    </div>
  )
}
