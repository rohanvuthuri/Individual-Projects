import { useEffect, useRef, useState } from 'react'
import {
  CandlestickSeries,
  HistogramSeries,
  LineSeries,
  LineStyle,
  createChart,
  type IChartApi,
  type ISeriesApi,
  type UTCTimestamp,
} from 'lightweight-charts'
import type { Bar, History, Point } from '../api/types'
import type { ChartTheme } from '../hooks/useChartTheme'
import { formatCompact } from '../lib/format'
import './PriceChart.css'

interface Props {
  history: History
  theme: ChartTheme
}

interface SeriesRefs {
  chart: IChartApi
  candles: ISeriesApi<'Candlestick'>
  volume: ISeriesApi<'Histogram'>
  overlays: ISeriesApi<'Line'>[]
  rsi: ISeriesApi<'Line'>
  macd: ISeriesApi<'Line'>
  signal: ISeriesApi<'Line'>
  hist: ISeriesApi<'Histogram'>
}

const ts = (time: number) => time as UTCTimestamp
const line = (points: Point[]) => points.map((p) => ({ time: ts(p.time), value: p.value }))

/** Candles + the mode's overlays + volume, with RSI and MACD panes below on the same time axis. */
export function PriceChart({ history, theme }: Props) {
  const containerRef = useRef<HTMLDivElement>(null)
  const seriesRef = useRef<SeriesRefs | null>(null)
  const [hoverTime, setHoverTime] = useState<number | null>(null)
  const intraday = history.interval.endsWith('m')
  const overlayCount = history.overlays.length

  // Build the chart (re-built only when the theme changes; the parent remounts
  // this component when the ticker or mode changes).
  useEffect(() => {
    const container = containerRef.current
    if (!container) return
    const chart = createChart(container, {
      autoSize: true,
      layout: {
        background: { color: theme.surface },
        textColor: theme.muted,
        fontFamily: getComputedStyle(document.body).fontFamily,
        panes: { separatorColor: theme.grid, separatorHoverColor: theme.axis },
      },
      grid: { vertLines: { visible: false }, horzLines: { color: theme.grid } },
      rightPriceScale: { borderColor: theme.axis },
      timeScale: { borderColor: theme.axis, timeVisible: intraday, secondsVisible: false, rightOffset: 4 },
      crosshair: {
        vertLine: { color: theme.muted, labelBackgroundColor: theme.axis },
        horzLine: { color: theme.muted, labelBackgroundColor: theme.axis },
      },
    })

    const candles = chart.addSeries(CandlestickSeries, {
      upColor: theme.up,
      downColor: theme.down,
      borderVisible: false,
      wickUpColor: theme.up,
      wickDownColor: theme.down,
      priceLineVisible: false,
    })
    const volume = chart.addSeries(HistogramSeries, {
      color: theme.volume,
      priceScaleId: 'volume',
      priceFormat: { type: 'volume' },
      lastValueVisible: false,
      priceLineVisible: false,
    })
    volume.priceScale().applyOptions({ scaleMargins: { top: 0.82, bottom: 0 } })
    candles.priceScale().applyOptions({ scaleMargins: { top: 0.06, bottom: 0.2 } })

    const overlays = Array.from({ length: overlayCount }, (_, i) =>
      chart.addSeries(LineSeries, {
        color: theme.series[i % theme.series.length],
        lineWidth: 2,
        priceLineVisible: false,
        lastValueVisible: false,
        crosshairMarkerVisible: false,
      }),
    )

    const rsi = chart.addSeries(
      LineSeries,
      {
        color: theme.series[0],
        lineWidth: 2,
        priceLineVisible: false,
        // RSI always lives on 0–100; a fixed scale keeps 70/30 in the same place.
        autoscaleInfoProvider: () => ({ priceRange: { minValue: 0, maxValue: 100 } }),
      },
      1,
    )
    for (const level of [70, 30]) {
      rsi.createPriceLine({ price: level, color: theme.axis, lineWidth: 1, lineStyle: LineStyle.Solid, axisLabelVisible: false, title: '' })
    }

    const hist = chart.addSeries(HistogramSeries, { priceLineVisible: false, lastValueVisible: false }, 2)
    const macd = chart.addSeries(LineSeries, { color: theme.series[0], lineWidth: 2, priceLineVisible: false, lastValueVisible: false }, 2)
    const signal = chart.addSeries(LineSeries, { color: theme.series[1], lineWidth: 2, priceLineVisible: false, lastValueVisible: false }, 2)

    const panes = chart.panes()
    panes[0].setStretchFactor(3.2)
    panes[1].setStretchFactor(1)
    panes[2].setStretchFactor(1)

    chart.subscribeCrosshairMove((param) => {
      setHoverTime(param.time === undefined ? null : (param.time as number))
    })

    seriesRef.current = { chart, candles, volume, overlays, rsi, macd, signal, hist }
    return () => {
      seriesRef.current = null
      chart.remove()
    }
  }, [theme, intraday, overlayCount])

  // Push data in whenever it arrives or refreshes (keeps the user's zoom on refresh).
  useEffect(() => {
    const s = seriesRef.current
    if (!s) return
    const firstLoad = s.candles.data().length === 0
    s.candles.setData(history.bars.map((b) => ({ time: ts(b.time), open: b.open, high: b.high, low: b.low, close: b.close })))
    s.volume.setData(history.bars.map((b) => ({ time: ts(b.time), value: b.volume })))
    history.overlays.forEach((o, i) => s.overlays[i]?.setData(line(o.points)))
    s.rsi.setData(line(history.rsi))
    s.macd.setData(line(history.macd.macd))
    s.signal.setData(line(history.macd.signal))
    s.hist.setData(
      history.macd.hist.map((p) => ({ time: ts(p.time), value: p.value, color: p.value >= 0 ? theme.upFaint : theme.downFaint })),
    )
    if (firstLoad) s.chart.timeScale().fitContent()
  }, [history, theme])

  // Legend values: the hovered bar, or the latest one.
  const barIndex = hoverTime === null ? history.bars.length - 1 : history.bars.findIndex((b) => b.time === hoverTime)
  const bar: Bar | undefined = history.bars[barIndex] ?? history.bars.at(-1)
  const valueAt = (points: Point[]) => (bar ? points.find((p) => p.time === bar.time)?.value : undefined)
  const change = bar && barIndex > 0 ? bar.close / history.bars[barIndex - 1].close - 1 : null

  return (
    <div className="price-chart">
      <div className="price-chart__legend" aria-live="off">
        {bar && (
          <div className="price-chart__ohlc num">
            <span>O <b>{bar.open.toFixed(2)}</b></span>
            <span>H <b>{bar.high.toFixed(2)}</b></span>
            <span>L <b>{bar.low.toFixed(2)}</b></span>
            <span>C <b>{bar.close.toFixed(2)}</b></span>
            {change !== null && (
              <span className={change >= 0 ? 'good-text' : 'critical-text'}>
                {change >= 0 ? '▲' : '▼'} {(change * 100).toFixed(2)}%
              </span>
            )}
            <span>Vol <b>{formatCompact(bar.volume)}</b></span>
          </div>
        )}
        <ul className="price-chart__keys">
          {history.overlays.map((o, i) => {
            const v = valueAt(o.points)
            return (
              <li key={o.label}>
                <span className="swatch" style={{ background: theme.series[i % theme.series.length] }} />
                {o.label} <span className="num muted">{v?.toFixed(2) ?? '—'}</span>
              </li>
            )
          })}
        </ul>
      </div>
      <div className="price-chart__canvas" ref={containerRef} role="img" aria-label={`${history.ticker} price chart with RSI and MACD`} />
      <div className="price-chart__pane-labels" aria-hidden="true">
        <span>RSI (14) <span className="num muted">{valueAt(history.rsi)?.toFixed(1) ?? '—'}</span> · 70 overbought / 30 oversold</span>
        <span>
          MACD <span className="swatch" style={{ background: theme.series[0] }} /> line
          {' '}<span className="swatch" style={{ background: theme.series[1] }} /> signal
        </span>
      </div>
    </div>
  )
}
