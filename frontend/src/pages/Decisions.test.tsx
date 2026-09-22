import { describe, it, expect } from 'vitest'
import { screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
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

  it('renders no recommendations panel when nothing is pending', async () => {
    renderWithProviders(<Decisions />)
    await waitFor(() => expect(screen.getByText('2026-07-21')).toBeInTheDocument())
    expect(screen.queryByText('Pending Recommendations')).not.toBeInTheDocument()
  })

  it('shows a pending recommendation with its rationale, and applies it after confirmation', async () => {
    let applyCalled = false
    server.use(
      http.get('http://localhost/api/memory/recommendations', () => HttpResponse.json([
        {
          date: '2026-09-18', rollup_reflection: 'stat_arb underperformed for 3 consecutive days.',
          action: 'reduce_position_limit', strategy: 'stat_arb', reduce_by_pct: 0.2,
          rationale: 'stat_arb has a negative rolling Sharpe over the last 5 sessions.',
        },
      ])),
      http.post('http://localhost/api/live/recommendations/:date/apply', ({ params }) => {
        applyCalled = true
        return HttpResponse.json({ date: params.date, action: 'reduce_position_limit', applied: true, strategy: 'stat_arb', new_max_position_pct: 0.04 })
      }),
    )
    const user = userEvent.setup()
    renderWithProviders(<Decisions />)
    await waitFor(() => expect(screen.getByText('Pending Recommendations')).toBeInTheDocument())
    expect(screen.getByText(/stat_arb has a negative rolling Sharpe/)).toBeInTheDocument()
    expect(screen.getByText('Reduce position limit')).toBeInTheDocument()

    await user.click(screen.getByText('Apply'))
    expect(screen.getByText('Apply this recommendation?')).toBeInTheDocument()
    await user.click(screen.getByText('Yes, Apply'))

    await waitFor(() => expect(applyCalled).toBe(true))
    expect(await screen.findByText(/Applied "Reduce position limit" for 2026-09-18/)).toBeInTheDocument()
  })

  it('cancels the confirm step without calling apply', async () => {
    let applyCalled = false
    server.use(
      http.get('http://localhost/api/memory/recommendations', () => HttpResponse.json([
        {
          date: '2026-09-18', rollup_reflection: null,
          action: 'flag_strategy_for_review', strategy: 'gann', reduce_by_pct: 0,
          rationale: 'gann has been flat for 10 sessions.',
        },
      ])),
      http.post('http://localhost/api/live/recommendations/:date/apply', () => {
        applyCalled = true
        return HttpResponse.json({ date: '2026-09-18', action: 'flag_strategy_for_review', applied: true, flagged: true })
      }),
    )
    const user = userEvent.setup()
    renderWithProviders(<Decisions />)
    await waitFor(() => expect(screen.getByText('Pending Recommendations')).toBeInTheDocument())

    await user.click(screen.getByText('Apply'))
    await user.click(screen.getByText('Cancel'))
    expect(applyCalled).toBe(false)
    expect(screen.queryByText('Apply this recommendation?')).not.toBeInTheDocument()
  })

  it('shows an actionable recommendation inline on its own decision entry, with a working Apply button', async () => {
    let applyCalled = false
    server.use(
      http.get('http://localhost/api/memory/decisions', () => HttpResponse.json([
        {
          date: '2026-07-21', status: 'reflected',
          proposal_weights: { AAPL: 0.05 }, notes: 'cycle=1',
          nav_at_decision: 1000000, raw_return: 0.012, benchmark_return: 0.008,
          reflection: 'The directional call was correct.',
          recommendation: {
            action: 'reduce_position_limit', strategy: 'stat_arb', reduce_by_pct: 0.2,
            rationale: 'stat_arb has a negative rolling Sharpe over the last 5 sessions.',
          },
        },
      ])),
      http.post('http://localhost/api/live/recommendations/:date/apply', ({ params }) => {
        applyCalled = true
        return HttpResponse.json({ date: params.date, action: 'reduce_position_limit', applied: true, strategy: 'stat_arb', new_max_position_pct: 0.04 })
      }),
    )
    const user = userEvent.setup()
    renderWithProviders(<Decisions />)
    await waitFor(() => expect(screen.getByText('2026-07-21')).toBeInTheDocument())
    expect(screen.getByText(/stat_arb has a negative rolling Sharpe/)).toBeInTheDocument()

    await user.click(screen.getByText('Apply'))
    await user.click(screen.getByText('Yes, Apply'))

    await waitFor(() => expect(applyCalled).toBe(true))
    expect(await screen.findByText(/Applied "Reduce position limit" for 2026-07-21/)).toBeInTheDocument()
  })

  it('shows an "Applied" badge instead of an Apply button once a recommendation is already applied', async () => {
    server.use(http.get('http://localhost/api/memory/decisions', () => HttpResponse.json([
      {
        date: '2026-07-21', status: 'reflected',
        proposal_weights: { AAPL: 0.05 }, notes: 'cycle=1',
        nav_at_decision: 1000000, raw_return: 0.012, benchmark_return: 0.008,
        reflection: 'The directional call was correct.',
        recommendation: {
          action: 'reduce_position_limit', strategy: 'stat_arb', reduce_by_pct: 0.2,
          rationale: 'Already acted on.', applied: true,
        },
      },
    ])))
    renderWithProviders(<Decisions />)
    await waitFor(() => expect(screen.getByText('2026-07-21')).toBeInTheDocument())
    expect(screen.getByText('Applied')).toBeInTheDocument()
    expect(screen.queryByText('Apply')).not.toBeInTheDocument()
  })

  it('does not render a recommendation card for a no_action entry', async () => {
    server.use(http.get('http://localhost/api/memory/decisions', () => HttpResponse.json([
      {
        date: '2026-07-21', status: 'reflected',
        proposal_weights: { AAPL: 0.05 }, notes: 'cycle=1',
        nav_at_decision: 1000000, raw_return: 0.012, benchmark_return: 0.008,
        reflection: 'The directional call was correct.',
        recommendation: { action: 'no_action', strategy: '', reduce_by_pct: 0, rationale: '' },
      },
    ])))
    renderWithProviders(<Decisions />)
    await waitFor(() => expect(screen.getByText('2026-07-21')).toBeInTheDocument())
    expect(screen.queryByText('No action')).not.toBeInTheDocument()
    expect(screen.queryByText('Apply')).not.toBeInTheDocument()
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
