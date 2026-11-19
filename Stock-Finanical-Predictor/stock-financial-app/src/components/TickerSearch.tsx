import { useEffect, useId, useState } from 'react'
import { useSearch } from '../api/client'
import './TickerSearch.css'

interface Props {
  onSelect: (ticker: string) => void
}

const DEBOUNCE_MS = 250

/** Search box with company-name autocomplete. Enter on raw text jumps straight to that ticker. */
export function TickerSearch({ onSelect }: Props) {
  const [text, setText] = useState('')
  const [query, setQuery] = useState('')
  const [open, setOpen] = useState(false)
  const [active, setActive] = useState(-1)
  const listId = useId()

  useEffect(() => {
    const id = setTimeout(() => setQuery(text), DEBOUNCE_MS)
    return () => clearTimeout(id)
  }, [text])

  const { data: results = [], isFetching } = useSearch(query)
  const showList = open && text.trim().length > 0 && (results.length > 0 || isFetching)

  const choose = (ticker: string) => {
    const t = ticker.trim().toUpperCase()
    if (!t) return
    onSelect(t)
    setText('')
    setQuery('')
    setOpen(false)
    setActive(-1)
  }

  const onKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'ArrowDown') {
      e.preventDefault()
      setOpen(true)
      setActive((i) => Math.min(i + 1, results.length - 1))
    } else if (e.key === 'ArrowUp') {
      e.preventDefault()
      setActive((i) => Math.max(i - 1, -1))
    } else if (e.key === 'Enter') {
      e.preventDefault()
      choose(active >= 0 && results[active] ? results[active].symbol : text)
    } else if (e.key === 'Escape') {
      setOpen(false)
    }
  }

  return (
    <div className="ticker-search">
      <input
        type="search"
        role="combobox"
        aria-expanded={showList}
        aria-controls={listId}
        aria-activedescendant={active >= 0 ? `${listId}-${active}` : undefined}
        aria-label="Search for a stock by ticker or company name"
        placeholder="Search ticker or company (e.g. NVDA, Tesla)"
        value={text}
        onChange={(e) => {
          setText(e.target.value)
          setOpen(true)
          setActive(-1)
        }}
        onFocus={() => setOpen(true)}
        onBlur={() => setTimeout(() => setOpen(false), 150)}
        onKeyDown={onKeyDown}
        autoComplete="off"
        spellCheck={false}
      />
      {showList && (
        <ul className="ticker-search__list" id={listId} role="listbox">
          {results.length === 0 && <li className="ticker-search__empty">Searching…</li>}
          {results.map((r, i) => (
            <li
              key={r.symbol}
              id={`${listId}-${i}`}
              role="option"
              aria-selected={i === active}
              className={i === active ? 'is-active' : undefined}
              onMouseDown={(e) => {
                e.preventDefault()
                choose(r.symbol)
              }}
              onMouseEnter={() => setActive(i)}
            >
              <span className="ticker-search__symbol">{r.symbol}</span>
              <span className="ticker-search__name">{r.name}</span>
              <span className="ticker-search__meta">{[r.type, r.exchange].filter(Boolean).join(' · ')}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
