import { useEffect, useState } from 'react'

export interface ChartTheme {
  surface: string
  text: string
  muted: string
  grid: string
  axis: string
  up: string
  down: string
  upFaint: string
  downFaint: string
  volume: string
  series: string[]
}

// Charts draw on a canvas, so they can't use CSS variables directly; read the
// current values from :root (index.css) and re-read when light/dark changes.
function readTheme(): ChartTheme {
  const style = getComputedStyle(document.documentElement)
  const v = (name: string) => style.getPropertyValue(name).trim()
  return {
    surface: v('--surface-1'),
    text: v('--text-secondary'),
    muted: v('--text-muted'),
    grid: v('--grid'),
    axis: v('--axis'),
    up: v('--up'),
    down: v('--down'),
    upFaint: v('--up-faint'),
    downFaint: v('--down-faint'),
    volume: v('--volume'),
    series: [v('--series-1'), v('--series-2'), v('--series-3'), v('--series-4')],
  }
}

export function useChartTheme(): ChartTheme {
  const [theme, setTheme] = useState(readTheme)

  useEffect(() => {
    const media = window.matchMedia('(prefers-color-scheme: dark)')
    const onChange = () => setTheme(readTheme())
    media.addEventListener('change', onChange)
    return () => media.removeEventListener('change', onChange)
  }, [])

  return theme
}
