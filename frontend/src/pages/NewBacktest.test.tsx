import { describe, it, expect } from 'vitest'
import { screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { renderWithProviders } from '../test/utils'
import NewBacktest from './NewBacktest'

describe('NewBacktest', () => {
  it('renders strategy checkboxes once loaded', async () => {
    renderWithProviders(<NewBacktest />)
    await waitFor(() => expect(screen.getByText('momentum')).toBeInTheDocument())
    expect(screen.getByText('multi_factor')).toBeInTheDocument()
  })

  it('lets a user select a strategy', async () => {
    const user = userEvent.setup()
    renderWithProviders(<NewBacktest />)
    await waitFor(() => expect(screen.getByText('momentum')).toBeInTheDocument())

    const checkbox = screen.getAllByRole('checkbox')[0]!
    expect(checkbox).not.toBeChecked()
    await user.click(checkbox)
    expect(checkbox).toBeChecked()
  })

  it('shows an error and blocks the walk-forward request on invalid parameter grid JSON', async () => {
    const user = userEvent.setup()
    renderWithProviders(<NewBacktest />)
    await waitFor(() => expect(screen.getByText('momentum')).toBeInTheDocument())

    await user.click(screen.getAllByRole('checkbox')[0]!)
    await user.type(screen.getByPlaceholderText(/^e\.g\./), 'not valid json')
    await user.click(screen.getByRole('button', { name: /Run Walk-Forward/i }))

    await waitFor(() =>
      expect(screen.getAllByText(/Invalid parameter grid JSON/).length).toBeGreaterThan(0),
    )
  })

  it('expands the risk overrides section with all 11 RiskConfig fields', async () => {
    const user = userEvent.setup()
    renderWithProviders(<NewBacktest />)
    await waitFor(() => expect(screen.getByText('momentum')).toBeInTheDocument())

    await user.click(screen.getByText('Risk Overrides (optional)'))
    for (const field of [
      'max_position_pct', 'max_gross_exposure', 'max_net_exposure', 'max_sector_pct',
      'vol_target', 'max_drawdown_pct', 'max_participation_pct', 'adv_lookback_days',
      'correlation_threshold', 'max_correlated_pair_pct', 'correlation_lookback_days',
    ]) {
      expect(screen.getByText(field)).toBeInTheDocument()
    }
  })
})
