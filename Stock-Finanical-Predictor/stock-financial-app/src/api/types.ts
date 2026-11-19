// Shapes returned by the FastAPI backend (backend/app/main.py).

export type ModeKey = 'long' | 'swing' | 'day'

export interface SearchResult {
  symbol: string
  name: string
  exchange: string | null
  type: string | null
}

export interface Quote {
  ticker: string
  name: string
  exchange: string | null
  currency: string
  price: number
  previous_close: number
  change: number
  change_pct: number
  market_state: string | null
}

export interface Point {
  time: number
  value: number
}

export interface Bar {
  time: number
  open: number
  high: number
  low: number
  close: number
  volume: number
}

export interface History {
  ticker: string
  mode: ModeKey
  interval: string
  bars: Bar[]
  overlays: { label: string; points: Point[] }[]
  rsi: Point[]
  macd: { macd: Point[]; signal: Point[]; hist: Point[] }
}

export interface BacktestMetrics {
  accuracy: number
  baseline: number // share of test periods where the price went up
  beats_baseline: boolean
  test_samples: number
  test_start: string
  test_end: string
}

export interface GbmMetrics extends BacktestMetrics {
  auc: number | null
}

export interface LstmMetrics extends BacktestMetrics {
  rmse: number
  correlation: number
  epochs: number
}

export interface Importance {
  feature: string
  label: string
  importance: number
}

export interface EquityPoint {
  time: number
  strategy: number
  buy_hold: number
}

export type Signal = 'Buy' | 'Hold' | 'Sell'

export interface Prediction {
  ticker: string
  mode: ModeKey
  horizon_label: string
  horizon_bars: number
  as_of: string
  last_price: number
  typical_move: number
  expected_range: { low: number; high: number }
  signal: Signal
  confidence: number
  provisional: boolean
  agreement: 'agree' | 'disagree' | 'neutral' | null
  gbm: {
    prob_up: number
    metrics: GbmMetrics
    importances: Importance[]
    equity_curve: EquityPoint[]
    trained_at: string
  }
  lstm: {
    status: 'ready' | 'training' | 'error'
    error: string | null
    predicted_return?: number
    target_price?: number
    metrics?: LstmMetrics
    trained_at?: string
  }
}

export type Rating = 'good' | 'neutral' | 'poor' | 'info'

export interface FundamentalMetric {
  key: string
  label: string
  value: number
  format: 'ratio' | 'pct' | 'pct_raw'
  rating: Rating
  explanation: string
}

export interface Fundamentals {
  available: boolean
  summary: 'Strong' | 'Mixed' | 'Weak' | null
  score: number
  metrics: FundamentalMetric[]
  profile: {
    sector: string | null
    industry: string | null
    market_cap: number | null
    week52_low: number | null
    week52_high: number | null
    analyst_target: number | null
    analyst_rating: string | null
    analyst_count: number | null
  }
}
