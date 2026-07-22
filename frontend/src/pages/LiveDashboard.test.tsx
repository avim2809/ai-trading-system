import { describe, it, expect } from 'vitest'
import { screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { http, HttpResponse } from 'msw'
import { server } from '../test/server'
import { renderWithProviders } from '../test/utils'
import LiveDashboard from './LiveDashboard'

describe('LiveDashboard', () => {
  it('shows the stopped state with a Start Engine button', async () => {
    renderWithProviders(<LiveDashboard />)
    await waitFor(() => expect(screen.getByText('Engine is stopped.')).toBeInTheDocument())
    expect(screen.getByText('Start Engine')).toBeInTheDocument()
  })

  it('shows running state, account, positions, and the alert banner', async () => {
    server.use(
      http.get('http://localhost/api/live/status', () => HttpResponse.json({
        state: 'running', broker: 'ibkr_paper', broker_connected: true, next_run: null,
        active_strategies: ['momentum'], approval_mode: 'full_auto', uptime_seconds: 10, last_cycle: null,
      })),
    )
    renderWithProviders(<LiveDashboard />)
    await waitFor(() => expect(screen.getByText('Running')).toBeInTheDocument())
    expect(await screen.findByText('AAPL')).toBeInTheDocument() // position
    expect(await screen.findByText(/operational alert/)).toBeInTheDocument()
  })

  it('shows a stuck-cycle warning when a cycle has run far longer than normal', async () => {
    server.use(
      http.get('http://localhost/api/live/status', () => HttpResponse.json({
        state: 'running', broker: 'ibkr_paper', broker_connected: true, next_run: null,
        active_strategies: ['momentum'], approval_mode: 'full_auto', uptime_seconds: 90000,
        last_cycle: null, cycle_running_seconds: 90000,
      })),
    )
    renderWithProviders(<LiveDashboard />)
    expect(await screen.findByText(/this looks stuck/i)).toBeInTheDocument()
  })

  it('does not show the stuck-cycle warning when no cycle is running', async () => {
    server.use(
      http.get('http://localhost/api/live/status', () => HttpResponse.json({
        state: 'running', broker: 'ibkr_paper', broker_connected: true, next_run: null,
        active_strategies: ['momentum'], approval_mode: 'full_auto', uptime_seconds: 10,
        last_cycle: null, cycle_running_seconds: null,
      })),
    )
    renderWithProviders(<LiveDashboard />)
    await waitFor(() => expect(screen.getByText('Running')).toBeInTheDocument())
    expect(screen.queryByText(/this looks stuck/i)).not.toBeInTheDocument()
  })

  it('start form submits the full payload including selected strategies', async () => {
    let capturedBody: Record<string, unknown> = {}
    server.use(
      http.post('http://localhost/api/live/start', async ({ request }) => {
        capturedBody = (await request.json()) as Record<string, unknown>
        return HttpResponse.json({ status: 'started' })
      }),
    )
    const user = userEvent.setup()
    renderWithProviders(<LiveDashboard />)
    await waitFor(() => expect(screen.getByText('Engine is stopped.')).toBeInTheDocument())

    await user.click(screen.getByText('Start Engine'))
    await waitFor(() => expect(screen.getByText('momentum')).toBeInTheDocument())
    await user.click(screen.getByText('momentum'))

    const startButtons = screen.getAllByText('Start')
    await user.click(startButtons[startButtons.length - 1]!)

    await waitFor(() => expect(capturedBody.strategies).toEqual(['momentum']))
    expect(capturedBody.broker).toBe('alpaca_paper')
  })
})
