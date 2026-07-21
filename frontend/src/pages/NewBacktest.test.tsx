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

  it('expands the risk overrides section with all 6 RiskConfig fields', async () => {
    const user = userEvent.setup()
    renderWithProviders(<NewBacktest />)
    await waitFor(() => expect(screen.getByText('momentum')).toBeInTheDocument())

    await user.click(screen.getByText('Risk Overrides (optional)'))
    for (const field of ['max_position_pct', 'max_gross_exposure', 'max_net_exposure', 'max_sector_pct', 'vol_target', 'max_drawdown_pct']) {
      expect(screen.getByText(field)).toBeInTheDocument()
    }
  })
})
