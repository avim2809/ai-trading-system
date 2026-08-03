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

  it('renders the lessons-learned digest panel', async () => {
    renderWithProviders(<Decisions />)
    await waitFor(() => expect(screen.getByText('Lessons Learned')).toBeInTheDocument())
    expect(screen.getByText(/3 reflected decisions/)).toBeInTheDocument()
    expect(screen.getByText('Trust the signal in trending regimes')).toBeInTheDocument()
  })

  it('shows a per-strategy breakdown when present, hides it when absent', async () => {
    server.use(http.get('http://localhost/api/memory/decisions', () => HttpResponse.json([
      {
        date: '2026-07-24', status: 'pending',
        proposal_weights: { AAPL: 0.05, MSFT: -0.02 },
        per_strategy: {
          momentum: { AAPL: 0.03 },
          danelfin_ai_score: { AAPL: 0.02, MSFT: -0.02 },
        },
        notes: 'cycle=3', nav_at_decision: 1000000, raw_return: null, benchmark_return: null, reflection: null,
      },
    ])))
    renderWithProviders(<Decisions />)
    await waitFor(() => expect(screen.getByText('2026-07-24')).toBeInTheDocument())
    expect(screen.getByText(/Per-strategy breakdown \(2 strategies\)/)).toBeInTheDocument()
    expect(screen.getByText('momentum:')).toBeInTheDocument()
    expect(screen.getByText('danelfin_ai_score:')).toBeInTheDocument()
  })

  it('omits the per-strategy section entirely when the field is absent (older entries)', async () => {
    renderWithProviders(<Decisions />)
    await waitFor(() => expect(screen.getByText('2026-07-21')).toBeInTheDocument())
    expect(screen.queryByText(/Per-strategy breakdown/)).not.toBeInTheDocument()
  })

  it('splits a structured reflection into what-worked / what-failed columns', async () => {
    server.use(http.get('http://localhost/api/memory/decisions', () => HttpResponse.json([
      {
        date: '2026-07-23', status: 'reflected', proposal_weights: { AAPL: 0.05 },
        notes: '', nav_at_decision: 1000000, raw_return: 0.01, benchmark_return: 0.0,
        reflection: 'CORRECT. What worked: thesis held Lesson: trust the signal',
        verdict: 'correct', what_worked: 'thesis held', what_failed: '', lesson: 'trust the signal',
      },
    ])))
    renderWithProviders(<Decisions />)
    await waitFor(() => expect(screen.getByText('What worked')).toBeInTheDocument())
    expect(screen.getByText("What didn't")).toBeInTheDocument()
    expect(screen.getByText('thesis held')).toBeInTheDocument()
    expect(screen.getByText('trust the signal')).toBeInTheDocument()
    expect(screen.getByText('correct')).toBeInTheDocument()
  })
})
