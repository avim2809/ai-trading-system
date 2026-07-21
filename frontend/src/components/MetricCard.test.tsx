import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import MetricCard from './MetricCard'

describe('MetricCard', () => {
  it('renders label and formatted currency value', () => {
    render(<MetricCard label="Equity" value={1001712.88} format="currency" />)
    expect(screen.getByText('Equity')).toBeInTheDocument()
    expect(screen.getByText('$1,001,713')).toBeInTheDocument()
  })

  it('renders a percentage with sign-based coloring', () => {
    render(<MetricCard label="Return" value={0.15} format="pct" />)
    expect(screen.getByText('15.00%')).toBeInTheDocument()
  })

  it('renders a plain string value unformatted', () => {
    render(<MetricCard label="Broker" value="ibkr_paper" />)
    expect(screen.getByText('ibkr_paper')).toBeInTheDocument()
  })

  it('shows an em dash for NaN/missing numeric values', () => {
    render(<MetricCard label="Sharpe" value={NaN} format="ratio" />)
    expect(screen.getByText('—')).toBeInTheDocument()
  })
})
