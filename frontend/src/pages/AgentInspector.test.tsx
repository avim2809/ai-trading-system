import { describe, it, expect } from 'vitest'
import { screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { http, HttpResponse } from 'msw'
import { server } from '../test/server'
import { renderWithProviders } from '../test/utils'
import AgentInspector from './AgentInspector'
import * as m from '../test/mockData'

describe('AgentInspector', () => {
  it('disables Run Step until a strategy is selected', async () => {
    renderWithProviders(<AgentInspector />)
    await waitFor(() => expect(screen.getByText('momentum')).toBeInTheDocument())
    expect(screen.getByText('Run Step')).toBeDisabled()
  })

  it('runs the pipeline and renders the resulting blackboard', async () => {
    const user = userEvent.setup()
    renderWithProviders(<AgentInspector />)
    await waitFor(() => expect(screen.getByText('momentum')).toBeInTheDocument())

    await user.click(screen.getByText('momentum'))
    expect(screen.getByText('Run Step')).toBeEnabled()
    await user.click(screen.getByText('Run Step'))

    await waitFor(() => expect(screen.getByText(/Pipeline snapshot as of/)).toBeInTheDocument())
    // From mockBlackboard: a bull thesis on AAPL and an approved risk decision.
    expect(screen.getByText('APPROVED')).toBeInTheDocument()
  })

  it('does not crash when /strategies returns a non-array shape', async () => {
    server.use(
      http.get('http://localhost/api/strategies', () =>
        HttpResponse.json({ strategies: m.mockStrategies })),
    )
    renderWithProviders(<AgentInspector />)
    await waitFor(() => expect(screen.getByText('Agent Inspector')).toBeInTheDocument())
    expect(screen.queryByText('momentum')).not.toBeInTheDocument()
  })

  it('survives null numeric fields in the blackboard response', async () => {
    server.use(
      http.post('http://localhost/api/agents/step', () =>
        HttpResponse.json({
          ...m.mockBlackboard,
          signal_sets: [{
            domain: 'technical',
            asof: '2026-07-21T19:49:04',
            signals: [{
              symbol: 'AAPL', strategy: 'momentum', score: null, confidence: null,
              horizon: '1d', asof: '2026-07-21T19:49:04', meta: { llm_rationale: { bad: true } },
            }],
          }],
          risk_decision: {
            approved: true,
            adjusted_targets: { AAPL: 0.05 },
          },
        })),
    )
    const user = userEvent.setup()
    renderWithProviders(<AgentInspector />)
    await waitFor(() => expect(screen.getByText('momentum')).toBeInTheDocument())
    await user.click(screen.getByText('momentum'))
    await user.click(screen.getByText('Run Step'))
    await waitFor(() => expect(screen.getByText(/Pipeline snapshot as of/)).toBeInTheDocument())
    expect(screen.getByText('APPROVED')).toBeInTheDocument()
  })
})
