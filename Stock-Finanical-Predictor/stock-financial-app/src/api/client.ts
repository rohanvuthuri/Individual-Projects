import { useQuery } from '@tanstack/react-query'
import type { Fundamentals, History, ModeKey, Prediction, Quote, SearchResult } from './types'

export class ApiError extends Error {
  status: number

  constructor(status: number, message: string) {
    super(message)
    this.status = status
  }
}

async function get<T>(path: string): Promise<T> {
  let res: Response
  try {
    res = await fetch(`/api${path}`)
  } catch {
    throw new ApiError(0, 'Could not reach the backend. Is it running on port 8000?')
  }
  if (!res.ok) {
    let detail: string | undefined
    try {
      detail = (await res.json()).detail
    } catch {
      // Not JSON: the Vite proxy couldn't reach the backend, or it crashed.
    }
    if (!detail && res.status >= 500) {
      detail = 'The backend is not responding. Check that it is running on port 8000 and look at its terminal for errors.'
    }
    throw new ApiError(res.status, detail ?? res.statusText)
  }
  return res.json() as Promise<T>
}

// Client errors (bad ticker, not enough history) won't fix themselves on retry.
const retry = (count: number, error: Error) =>
  !(error instanceof ApiError && error.status >= 400 && error.status < 500) && count < 2

const INTRADAY_REFRESH_MS = 60_000
const TRAINING_POLL_MS = 5_000

const t = encodeURIComponent

export function useSearch(query: string) {
  const q = query.trim()
  return useQuery({
    queryKey: ['search', q.toLowerCase()],
    queryFn: () => get<SearchResult[]>(`/search?q=${t(q)}`),
    enabled: q.length > 0,
    staleTime: 60 * 60_000,
    retry,
  })
}

export function useQuote(ticker: string, mode: ModeKey) {
  return useQuery({
    queryKey: ['quote', ticker],
    queryFn: () => get<Quote>(`/quote/${t(ticker)}`),
    refetchInterval: mode === 'day' ? INTRADAY_REFRESH_MS : false,
    staleTime: 30_000,
    retry,
  })
}

export function useHistory(ticker: string, mode: ModeKey) {
  return useQuery({
    queryKey: ['history', ticker, mode],
    queryFn: () => get<History>(`/history/${t(ticker)}?mode=${mode}`),
    refetchInterval: mode === 'day' ? INTRADAY_REFRESH_MS : false,
    staleTime: mode === 'day' ? 30_000 : 10 * 60_000,
    retry,
  })
}

export function usePrediction(ticker: string, mode: ModeKey) {
  return useQuery({
    queryKey: ['predict', ticker, mode],
    queryFn: () => get<Prediction>(`/predict/${t(ticker)}?mode=${mode}`),
    // Keep asking while the LSTM trains in the background on the server.
    refetchInterval: (query) =>
      query.state.data?.lstm.status === 'training'
        ? TRAINING_POLL_MS
        : mode === 'day'
          ? INTRADAY_REFRESH_MS
          : false,
    staleTime: mode === 'day' ? 30_000 : 10 * 60_000,
    retry,
  })
}

export function useFundamentals(ticker: string, enabled: boolean) {
  return useQuery({
    queryKey: ['fundamentals', ticker],
    queryFn: () => get<Fundamentals>(`/fundamentals/${t(ticker)}`),
    enabled,
    staleTime: 60 * 60_000,
    retry,
  })
}
