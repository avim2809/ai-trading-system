import { describe, it, expect } from 'vitest'
import { screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { http, HttpResponse } from 'msw'
import { server } from '../test/server'
import { renderWithProviders } from '../test/utils'
import { mockLiveAttributionHistory } from '../test/mockData'
import AttributionHistory from './AttributionHistory'

describe('AttributionHistory', () => {
  it('shows an empty state when no strategy has any history yet', async () => {
    server.use(http.get('http://localhost/api/live/attribution/history', () => HttpResponse.json({})))
    renderWithProviders(<AttributionHistory />)
    expect(await screen.findByText(/No performance history yet/)).toBeInTheDocument()
  })

  it('renders the WTD/MTD headline table with both strategies from the default mock series', async () => {
    renderWithProviders(<AttributionHistory />)
    await waitFor(() => expect(screen.getByText('Week-to-Date / Month-to-Date')).toBeInTheDocument())
    expect(screen.getAllByText('momentum').length).toBeGreaterThan(0)
    expect(screen.getAllByText('trend').length).toBeGreaterThan(0)

    // momentum's last week (2024-01-15/16/17) returns are 0.007, 0.002,
    // 0.004 -> compounded (1.007*1.002*1.004 - 1) rounds to 1.31%, pinning
    // the WTD figure actually flows end-to-end through periodReturns.ts
    // rather than just rendering a table shape.
    expect(screen.getByText('1.31%')).toBeInTheDocument()
  })

  it('switches to the weekly breakdown and shows a week-of row for each strategy', async () => {
    const user = userEvent.setup()
    renderWithProviders(<AttributionHistory />)
    await screen.findByText('Week-to-Date / Month-to-Date')

    await user.click(screen.getByText('Weekly'))
    expect(await screen.findByText('Weekly Breakdown')).toBeInTheDocument()
    // trend's series only starts 2024-01-08 (a week after momentum's), so
    // its earliest week ("Week of 2024-01-08") is where the union of
    // bucket keys across both strategies is exercised.
    expect(screen.getByText('Week of 2024-01-15')).toBeInTheDocument()
    expect(screen.getByText('Week of 2024-01-08')).toBeInTheDocument()
  })

  it('computes a custom date-range return after Apply is clicked', async () => {
    const user = userEvent.setup()
    renderWithProviders(<AttributionHistory />)
    await screen.findByText('Week-to-Date / Month-to-Date')

    const [startInput, endInput] = screen.getAllByDisplayValue('') as HTMLInputElement[]
    await user.type(startInput!, '2024-01-15')
    await user.type(endInput!, '2024-01-17')
    await user.click(screen.getByText('Apply'))

    expect(await screen.findByText('Return (2024-01-15 → 2024-01-17)')).toBeInTheDocument()
    // Same window as the WTD assertion above -- same expected value, so
    // "1.31%" now renders twice (WTD headline + this custom-range table).
    expect(screen.getAllByText('1.31%').length).toBeGreaterThanOrEqual(2)
  })

  it('links back to the Live Dashboard', async () => {
    renderWithProviders(<AttributionHistory />)
    await screen.findByText('Week-to-Date / Month-to-Date')
    expect(screen.getByText('← Back to Live Dashboard').closest('a')).toHaveAttribute('href', '/live')
  })

  it('sorts the headline table by WTD, defaulting to descending and toggling on a second click', async () => {
    const user = userEvent.setup()
    const { container } = renderWithProviders(<AttributionHistory />)
    await screen.findByText('Week-to-Date / Month-to-Date')

    const strategyCellsInOrder = () =>
      Array.from(container.querySelectorAll('table')[0]!.querySelectorAll('tbody tr td:first-child')).map(
        (td) => td.textContent,
      )

    // Natural (unsorted) order is alphabetical: momentum, trend -- and
    // momentum's WTD (1.31%) already exceeds trend's WTD (~0.05%), so this
    // first assertion alone wouldn't distinguish "sorted descending" from
    // "just alphabetical". The second click (ascending) below is what
    // actually proves the sort is wired up, since it must reverse the order.
    await user.click(screen.getByText('WTD'))
    expect(strategyCellsInOrder()).toEqual(['momentum', 'trend'])

    await user.click(screen.getByText('WTD'))
    expect(strategyCellsInOrder()).toEqual(['trend', 'momentum'])
  })

  it('sorts the headline table by strategy name independently of the WTD/MTD sort', async () => {
    const user = userEvent.setup()
    const { container } = renderWithProviders(<AttributionHistory />)
    await screen.findByText('Week-to-Date / Month-to-Date')

    const strategyCellsInOrder = () =>
      Array.from(container.querySelectorAll('table')[0]!.querySelectorAll('tbody tr td:first-child')).map(
        (td) => td.textContent,
      )

    // Descending by name: trend before momentum (reverse alphabetical).
    await user.click(screen.getByText('Strategy'))
    expect(strategyCellsInOrder()).toEqual(['trend', 'momentum'])

    // Clicking a different column (MTD) starts that column fresh at
    // descending rather than continuing the Strategy column's toggle state.
    await user.click(screen.getByText('MTD'))
    const afterMtdClick = strategyCellsInOrder()
    await user.click(screen.getByText('MTD'))
    const afterSecondMtdClick = strategyCellsInOrder()
    expect(afterMtdClick).not.toEqual(afterSecondMtdClick)
  })

  it('sorts the period breakdown table by a strategy column', async () => {
    const user = userEvent.setup()
    const { container } = renderWithProviders(<AttributionHistory />)
    await screen.findByText('Week-to-Date / Month-to-Date')
    await user.click(screen.getByText('Weekly'))
    await screen.findByText('Weekly Breakdown')

    const breakdownTable = () =>
      Array.from(container.querySelectorAll('table')).find((t) => t.textContent?.includes('Period'))!
    const periodCellsInOrder = () =>
      Array.from(breakdownTable().querySelectorAll('tbody tr td:first-child')).map((td) => td.textContent)

    // rowKeys is already naturally ordered most-recent-first (descending by
    // key), so the first click (descending) shouldn't visibly reorder it --
    // the second click (ascending) is what proves the sort is wired up.
    const naturalOrder = periodCellsInOrder()
    await user.click(screen.getByText('Period'))
    expect(periodCellsInOrder()).toEqual(naturalOrder)

    await user.click(screen.getByText('Period'))
    expect(periodCellsInOrder()).toEqual([...naturalOrder].reverse())
  })
})

// Sanity check that the fixture used above is what this file's assertions
// assume -- guards against a future edit to mockLiveAttributionHistory
// silently invalidating the hand-computed 1.31% expectation.
describe('mockLiveAttributionHistory fixture', () => {
  it('has both strategies ending on the same last date', () => {
    const dates = Object.values(mockLiveAttributionHistory).map((s) => s.dates[s.dates.length - 1])
    expect(new Set(dates).size).toBe(1)
  })
})
