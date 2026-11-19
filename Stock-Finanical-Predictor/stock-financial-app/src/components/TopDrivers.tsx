import type { Importance } from '../api/types'

/** Which inputs the gradient-boosting model leaned on most (permutation importance). */
export function TopDrivers({ importances }: { importances: Importance[] }) {
  const useful = importances.filter((i) => i.importance > 0)
  const max = Math.max(...useful.map((i) => i.importance), 1e-9)

  return (
    <section className="card" aria-labelledby="drivers-title">
      <header className="card__header">
        <h2 id="drivers-title">What the model looks at</h2>
      </header>
      <p className="muted card__intro">How much the model's test score drops when each input is scrambled. Longer = more important.</p>
      {useful.length === 0 ? (
        <p className="muted">No single input helped on the test data; the model is mostly noise for this stock and horizon.</p>
      ) : (
        <ul className="drivers">
          {useful.map((i) => (
            <li key={i.feature} title={`AUC drop: ${i.importance.toFixed(4)}`}>
              <span className="drivers__label">{i.label}</span>
              <span className="drivers__track">
                <span className="drivers__bar" style={{ width: `${(i.importance / max) * 100}%` }} />
              </span>
            </li>
          ))}
        </ul>
      )}
    </section>
  )
}
