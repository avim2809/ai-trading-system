import { describe, it, expect } from 'vitest'
import { screen, waitFor } from '@testing-library/react'
import { http, HttpResponse } from 'msw'
import { server } from '../test/server'
import { renderWithProviders } from '../test/utils'
import RunDetail from './RunDetail'

describe('RunDetail', () => {
  it('renders run metrics once loaded', async () => {
    renderWithProviders(<RunDetail />, { route: '/runs/run-1', path: '/runs/:runId' })
    await waitFor(() => expect(screen.getByText(/completed/i)).toBeInTheDocument())
  })

  it('renders the turnover section from report.json', async () => {
    renderWithProviders(<RunDetail />, { route: '/runs/run-1', path: '/runs/:runId' })
    await waitFor(() => expect(screen.getByText('Turnover')).toBeInTheDocument())
    expect(screen.getByText('Total Turnover')).toBeInTheDocument()
    expect(screen.getByText('24')).toBeInTheDocument()
  })

  it('shows a not-found state when the run fails to load', async () => {
    server.use(http.get('http://localhost/api/runs/:id', () => HttpResponse.text('not found', { status: 404 })))
    renderWithProviders(<RunDetail />, { route: '/runs/missing', path: '/runs/:runId' })
    await waitFor(() => expect(screen.getByText('Failed to load run')).toBeInTheDocument())
  })
})
