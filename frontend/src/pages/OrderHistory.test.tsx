import { describe, it, expect } from 'vitest'
import { screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { http, HttpResponse } from 'msw'
import { server } from '../test/server'
import { renderWithProviders } from '../test/utils'
import OrderHistory from './OrderHistory'

describe('OrderHistory', () => {
  it('renders orders within the default 7-day range', async () => {
    server.use(http.get('http://localhost/api/live/orders', () => HttpResponse.json([
      { order_id: 'o1', symbol: 'AAPL', side: 'buy', quantity: 1, filled_quantity: 1, avg_fill_price: 332.5, status: 'filled', strategy: 'momentum', timestamp: new Date().toISOString() },
    ])))
    renderWithProviders(<OrderHistory />)
    await waitFor(() => expect(screen.getByText('AAPL')).toBeInTheDocument())
  })

  it('filters out orders older than the selected range', async () => {
    const old = new Date(Date.now() - 60 * 86_400_000).toISOString() // 60 days ago
    server.use(http.get('http://localhost/api/live/orders', () => HttpResponse.json([
      { order_id: 'o1', symbol: 'AAPL', side: 'buy', quantity: 1, filled_quantity: 1, avg_fill_price: 332.5, status: 'filled', strategy: 'momentum', timestamp: old },
    ])))
    renderWithProviders(<OrderHistory />)
    await waitFor(() => expect(screen.getByText('No orders in selected range')).toBeInTheDocument())
  })

  it('switches range filters on click', async () => {
    const user = userEvent.setup()
    server.use(http.get('http://localhost/api/live/orders', () => HttpResponse.json([])))
    renderWithProviders(<OrderHistory />)
    await waitFor(() => expect(screen.getByText('No orders in selected range')).toBeInTheDocument())
    await user.click(screen.getByText('Last 24h'))
    // Still empty, but the click shouldn't throw and the button should be selectable.
    expect(screen.getByText('Last 24h')).toBeInTheDocument()
  })

  it('clears order history after confirming', async () => {
    let cleared = false
    server.use(
      http.get('http://localhost/api/live/orders', () => HttpResponse.json(cleared ? [] : [
        { order_id: 'o1', symbol: 'AAPL', side: 'buy', quantity: 1, filled_quantity: 1, avg_fill_price: 332.5, status: 'filled', strategy: 'momentum', timestamp: new Date().toISOString() },
      ])),
      http.delete('http://localhost/api/live/cycles', () => {
        cleared = true
        return HttpResponse.json({ cleared: 1 })
      }),
    )
    const user = userEvent.setup()
    renderWithProviders(<OrderHistory />)
    await waitFor(() => expect(screen.getByText('AAPL')).toBeInTheDocument())

    await user.click(screen.getByText('Clear History'))
    expect(screen.getByText(/Delete all order history\?/)).toBeInTheDocument()

    await user.click(screen.getByText('Yes, Clear All'))
    await waitFor(() => expect(screen.getByText('No orders in selected range')).toBeInTheDocument())
  })

  it('cancels clearing order history without calling the API', async () => {
    server.use(http.get('http://localhost/api/live/orders', () => HttpResponse.json([
      { order_id: 'o1', symbol: 'AAPL', side: 'buy', quantity: 1, filled_quantity: 1, avg_fill_price: 332.5, status: 'filled', strategy: 'momentum', timestamp: new Date().toISOString() },
    ])))
    const user = userEvent.setup()
    renderWithProviders(<OrderHistory />)
    await waitFor(() => expect(screen.getByText('AAPL')).toBeInTheDocument())

    await user.click(screen.getByText('Clear History'))
    await user.click(screen.getByText('Cancel'))
    expect(screen.getByText('Clear History')).toBeInTheDocument()
    expect(screen.getByText('AAPL')).toBeInTheDocument()
  })
})
