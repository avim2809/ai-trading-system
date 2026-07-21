import { describe, it, expect } from 'vitest'
import { screen, waitFor } from '@testing-library/react'
import { http, HttpResponse } from 'msw'
import { server } from '../test/server'
import { renderWithProviders } from '../test/utils'
import Decisions from './Decisions'

describe('Decisions', () => {
  it('renders a reflected decision with weights, return, and reflection text', async () => {
    renderWithProviders(<Decisions />)
    await waitFor(() => expect(screen.getByText('2026-07-21')).toBeInTheDocument())
    expect(screen.getByText(/AAPL \+5.0%/)).toBeInTheDocument()
    expect(screen.getByText(/AAPL outperformed the benchmark/)).toBeInTheDocument()
    expect(screen.getByText(/Return:/)).toBeInTheDocument()
  })

  it('shows an empty state with no decisions recorded', async () => {
    server.use(http.get('http://localhost/api/memory/decisions', () => HttpResponse.json([])))
    renderWithProviders(<Decisions />)
    await waitFor(() => expect(screen.getByText('No decisions recorded yet.')).toBeInTheDocument())
  })

  it('shows an "awaiting outcome" note for a pending (not yet reflected) decision', async () => {
    server.use(http.get('http://localhost/api/memory/decisions', () => HttpResponse.json([
      { date: '2026-07-22', status: 'pending', proposal_weights: { MSFT: 0.03 }, notes: 'cycle=2', nav_at_decision: 1000000, raw_return: null, benchmark_return: null, reflection: null },
    ])))
    renderWithProviders(<Decisions />)
    await waitFor(() => expect(screen.getByText(/Awaiting outcome/)).toBeInTheDocument())
  })
})
