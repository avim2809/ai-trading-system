import { describe, it, expect } from 'vitest'
import { screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { renderWithProviders } from '../test/utils'
import AgentInspector from './AgentInspector'

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
})
