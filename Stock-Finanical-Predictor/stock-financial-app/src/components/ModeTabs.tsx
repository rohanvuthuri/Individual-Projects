import type { ModeKey } from '../api/types'

const TABS: { key: ModeKey; label: string; hint: string }[] = [
  { key: 'long', label: 'Long-term', hint: 'Daily bars · predicts ~6 months ahead' },
  { key: 'swing', label: 'Swing', hint: 'Daily bars · predicts ~2 weeks ahead' },
  { key: 'day', label: 'Day trade', hint: '5-minute bars · predicts ~1 hour ahead' },
]

interface Props {
  mode: ModeKey
  onChange: (mode: ModeKey) => void
}

export function ModeTabs({ mode, onChange }: Props) {
  const current = TABS.find((t) => t.key === mode)
  return (
    <div className="mode-tabs">
      <div role="tablist" aria-label="Investing style">
        {TABS.map((tab) => (
          <button
            key={tab.key}
            role="tab"
            type="button"
            aria-selected={tab.key === mode}
            className={tab.key === mode ? 'is-active' : undefined}
            onClick={() => onChange(tab.key)}
          >
            {tab.label}
          </button>
        ))}
      </div>
      <span className="mode-tabs__hint muted">{current?.hint}</span>
    </div>
  )
}
