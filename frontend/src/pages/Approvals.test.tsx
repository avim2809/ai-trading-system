import { describe, it, expect } from 'vitest'
import { screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { http, HttpResponse } from 'msw'
import { server } from '../test/server'
import { renderWithProviders } from '../test/utils'
import Approvals from './Approvals'

describe('Approvals', () => {
  it('renders a pending approval card', async () => {
    renderWithProviders(<Approvals />)
    await waitFor(() => expect(screen.getByText(/AAPL/)).toBeInTheDocument())
    expect(screen.getByText('pending')).toBeInTheDocument()
  })

  it('shows an empty state with no pending approvals', async () => {
    server.use(http.get('http://localhost/api/live/approvals', () => HttpResponse.json([])))
    renderWithProviders(<Approvals />)
    await waitFor(() => expect(screen.getByText('No pending approvals.')).toBeInTheDocument())
  })

  it('approve flow requires a confirmation click before mutating', async () => {
    let approveCalled = false
    server.use(http.post('http://localhost/api/live/approvals/:id/approve', () => {
      approveCalled = true
      return HttpResponse.json({ status: 'approved', order_statuses: [] })
    }))
    const user = userEvent.setup()
    renderWithProviders(<Approvals />)
    await waitFor(() => expect(screen.getByText(/AAPL/)).toBeInTheDocument())

    await user.click(screen.getByText(/AAPL/)) // expand the card
    await user.click(screen.getByText('Approve'))
    expect(approveCalled).toBe(false) // not yet — needs confirmation
    expect(screen.getByText('Confirm approval?')).toBeInTheDocument()

    await user.click(screen.getByText('Yes, Approve'))
    await waitFor(() => expect(approveCalled).toBe(true))
  })

  it('reject requires a non-empty reason', async () => {
    const user = userEvent.setup()
    renderWithProviders(<Approvals />)
    await waitFor(() => expect(screen.getByText(/AAPL/)).toBeInTheDocument())

    await user.click(screen.getByText(/AAPL/))
    await user.click(screen.getByText('Reject'))
    const rejectButton = screen.getByRole('button', { name: 'Reject' })
    expect(rejectButton).toBeDisabled()

    await user.type(screen.getByPlaceholderText('Reason for rejection...'), 'Too risky')
    expect(rejectButton).toBeEnabled()
  })
})
