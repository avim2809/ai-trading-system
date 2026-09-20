import { describe, it, expect, vi } from 'vitest'
import { screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { http, HttpResponse } from 'msw'
import { server } from '../test/server'
import { renderWithProviders } from '../test/utils'
import LiveDashboard from './LiveDashboard'

describe('LiveDashboard', () => {
  const runningStatus = {
    state: 'running', broker: 'ibkr_paper', broker_connected: true, next_run: null,
    active_strategies: ['momentum'], approval_mode: 'full_auto', uptime_seconds: 10, last_cycle: null,
  }

  it('flattens a position after confirmation and shows the result', async () => {
    let flattenCalled = false
    server.use(
      http.get('http://localhost/api/live/status', () => HttpResponse.json(runningStatus)),
      http.post('http://localhost/api/live/positions/:symbol/flatten', ({ params }) => {
        flattenCalled = true
        return HttpResponse.json({
          symbol: params.symbol, flattened: true,
          order_statuses: [{ order_id: 'o1', symbol: 'AAPL', side: 'sell', quantity: 1, filled_quantity: 1, avg_fill_price: 332.5, status: 'filled', strategy: 'manual_flatten', timestamp: null }],
          failed: [],
        })
      }),
    )
    const confirmSpy = vi.spyOn(window, 'confirm').mockReturnValue(true)
    const user = userEvent.setup()
    renderWithProviders(<LiveDashboard />)
    await screen.findByText('AAPL')

    // The Positions table's row renders before the Strategy Sleeves
    // table's row, so the position's own "Flatten" button is the first.
    await user.click(screen.getAllByText('Flatten')[0]!)
    expect(confirmSpy).toHaveBeenCalled()
    await waitFor(() => expect(flattenCalled).toBe(true))
    expect(await screen.findByText(/AAPL: flattened \(1 order\)/)).toBeInTheDocument()
    confirmSpy.mockRestore()
  })

  it('does not call the flatten endpoint when the confirm dialog is declined', async () => {
    let flattenCalled = false
    server.use(
      http.get('http://localhost/api/live/status', () => HttpResponse.json(runningStatus)),
      http.post('http://localhost/api/live/positions/:symbol/flatten', () => {
        flattenCalled = true
        return HttpResponse.json({ symbol: 'AAPL', flattened: true, order_statuses: [], failed: [] })
      }),
    )
    const confirmSpy = vi.spyOn(window, 'confirm').mockReturnValue(false)
    const user = userEvent.setup()
    renderWithProviders(<LiveDashboard />)
    await screen.findByText('AAPL')

    await user.click(screen.getAllByText('Flatten')[0]!)
    expect(confirmSpy).toHaveBeenCalled()
    expect(flattenCalled).toBe(false)
    confirmSpy.mockRestore()
  })

  it('shows a no-op reason when flattening a strategy sleeve with nothing held', async () => {
    server.use(
      http.get('http://localhost/api/live/status', () => HttpResponse.json(runningStatus)),
      http.post('http://localhost/api/live/sleeves/:strategy/flatten', ({ params }) =>
        HttpResponse.json({ strategy: params.strategy, flattened: false, reason: 'no attributed positions held' })),
    )
    const confirmSpy = vi.spyOn(window, 'confirm').mockReturnValue(true)
    const user = userEvent.setup()
    renderWithProviders(<LiveDashboard />)
    await screen.findByText('Strategy Sleeves')

    // Two "Flatten" buttons render: one per open position (AAPL), one per
    // active strategy sleeve (momentum) — the sleeve row is the last one.
    const flattenButtons = screen.getAllByText('Flatten')
    await user.click(flattenButtons[flattenButtons.length - 1]!)
    expect(await screen.findByText(/momentum: not flattened — no attributed positions held/)).toBeInTheDocument()
    confirmSpy.mockRestore()
  })


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

  it('shows a Reset Kill Switch button when halted, and clearing it re-arms trading', async () => {
    let resetCalled = false
    server.use(
      http.get('http://localhost/api/live/status', () => HttpResponse.json({
        state: 'running', broker: 'ibkr_paper', broker_connected: true, next_run: null,
        active_strategies: ['momentum'], approval_mode: 'full_auto', uptime_seconds: 10, last_cycle: null,
      })),
      http.get('http://localhost/api/live/alerts', () => HttpResponse.json({
        halted: true,
        alerts: [{ timestamp: '2026-07-21T20:02:43', kind: 'drawdown_breach', severity: 'critical', message: 'Drawdown 15% breached kill switch 10%.', cycle_id: 3 }],
      })),
      http.post('http://localhost/api/live/kill-switch/reset', () => {
        resetCalled = true
        return HttpResponse.json({ reset: true, halted: false })
      }),
    )
    const user = userEvent.setup()
    renderWithProviders(<LiveDashboard />)
    expect(await screen.findByText(/Engine halted/i)).toBeInTheDocument()

    const resetButton = screen.getByText('Reset Kill Switch')
    await user.click(resetButton)
    await waitFor(() => expect(resetCalled).toBe(true))
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

  it('shows the market as open with a closes-in countdown', async () => {
    server.use(
      http.get('http://localhost/api/live/status', () => HttpResponse.json({
        state: 'running', broker: 'ibkr_paper', broker_connected: true, next_run: null,
        active_strategies: ['momentum'], approval_mode: 'full_auto', uptime_seconds: 10, last_cycle: null,
        market_open: true, next_market_open: null, next_market_close: '2099-01-01T16:00:00-05:00',
      })),
    )
    renderWithProviders(<LiveDashboard />)
    expect(await screen.findByText('Open')).toBeInTheDocument()
    expect(await screen.findByText(/closes in/)).toBeInTheDocument()
  })

  it('shows the market as closed with an opens-in countdown', async () => {
    server.use(
      http.get('http://localhost/api/live/status', () => HttpResponse.json({
        state: 'running', broker: 'ibkr_paper', broker_connected: true, next_run: null,
        active_strategies: ['momentum'], approval_mode: 'full_auto', uptime_seconds: 10, last_cycle: null,
        market_open: false, next_market_open: '2099-01-01T09:30:00-05:00', next_market_close: null,
      })),
    )
    renderWithProviders(<LiveDashboard />)
    expect(await screen.findByText('Closed')).toBeInTheDocument()
    expect(await screen.findByText(/opens in/)).toBeInTheDocument()
  })

  it('shows a dash for market status when unknown', async () => {
    server.use(
      http.get('http://localhost/api/live/status', () => HttpResponse.json({
        state: 'running', broker: 'ibkr_paper', broker_connected: true, next_run: null,
        active_strategies: ['momentum'], approval_mode: 'full_auto', uptime_seconds: 10, last_cycle: null,
        market_open: null, next_market_open: null, next_market_close: null,
      })),
    )
    renderWithProviders(<LiveDashboard />)
    await waitFor(() => expect(screen.getByText('Running')).toBeInTheDocument())
    expect(screen.queryByText('Open')).not.toBeInTheDocument()
    expect(screen.queryByText('Closed')).not.toBeInTheDocument()
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
