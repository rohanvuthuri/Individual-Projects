import { Component, type ReactNode } from 'react'

interface Props {
  label: string
  children: ReactNode
}

/** Keeps one broken card (e.g. a chart given bad data) from blanking the whole page. */
export class ErrorBoundary extends Component<Props, { error: Error | null }> {
  state = { error: null as Error | null }

  static getDerivedStateFromError(error: Error) {
    return { error }
  }

  render() {
    if (this.state.error) {
      return (
        <div className="card status status--error" role="alert">
          <div>
            <b>The {this.props.label} failed to render.</b>
            <p className="muted">{this.state.error.message}</p>
          </div>
        </div>
      )
    }
    return this.props.children
  }
}
