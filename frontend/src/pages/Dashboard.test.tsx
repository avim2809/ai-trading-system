import { describe, it, expect } from 'vitest'
import { screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { http, HttpResponse } from 'msw'
import { server } from '../test/server'
import { renderWithProviders } from '../test/utils'
import Dashboard from './Dashboard'

describe('Dashboard', () => {
  it('renders the run count and table once data loads', async () => {
    renderWithProviders(<Dashboard />)
    await waitFor(() => expect(screen.getByText('1 backtest run')).toBeInTheDocument())
    expect(screen.getByText('New Backtest')).toBeInTheDocument()
  })

  it('shows an empty state with no runs', async () => {
    server.use(http.get('http://localhost/api/runs', () => HttpResponse.json([])))
    renderWithProviders(<Dashboard />)
    await waitFor(() => expect(screen.getByText('No backtest runs yet.')).toBeInTheDocument())
  })

  it('shows an error state if the request fails', async () => {
    server.use(http.get('http://localhost/api/runs', () => HttpResponse.text('boom', { status: 500 })))
    renderWithProviders(<Dashboard />)
    await waitFor(() => expect(screen.getByText('Failed to load runs')).toBeInTheDocument())
  })

  it('enables Compare Selected only once 2+ runs are checked', async () => {
    server.use(http.get('http://localhost/api/runs', () => HttpResponse.json([
      { run_id: 'run-1', status: 'completed', start_time: '2026-07-01T00:00:00', end_time: null, notes: '', metrics: {} },
      { run_id: 'run-2', status: 'completed', start_time: '2026-07-01T00:00:00', end_time: null, notes: '', metrics: {} },
    ])))
    const user = userEvent.setup()
    renderWithProviders(<Dashboard />)
    await waitFor(() => expect(screen.getByText('2 backtest runs')).toBeInTheDocument())

    const compareButton = screen.getByText(/Compare Selected/)
    expect(compareButton).toBeDisabled()

    const checkboxes = screen.getAllByRole('checkbox')
    await user.click(checkboxes[0]!)
    await user.click(checkboxes[1]!)
    expect(compareButton).toBeEnabled()
  })
})
