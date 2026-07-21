import { describe, it, expect } from 'vitest'
import { screen, waitFor } from '@testing-library/react'
import { http, HttpResponse } from 'msw'
import { server } from '../test/server'
import { renderWithProviders } from '../test/utils'
import Compare from './Compare'

describe('Compare', () => {
  it('prompts to select 2+ runs when fewer are given', () => {
    renderWithProviders(<Compare />, { route: '/compare?ids=run-1', path: '/compare' })
    expect(screen.getByText('Select at least 2 runs to compare.')).toBeInTheDocument()
  })

  it('renders a comparison table for 2+ runs, highlighting the best value', async () => {
    server.use(
      http.post('http://localhost/api/runs/compare', () =>
        HttpResponse.json({ sharpe_ratio: { 'run-1': 1.2, 'run-2': 0.8 } })),
    )
    renderWithProviders(<Compare />, { route: '/compare?ids=run-1,run-2', path: '/compare' })
    await waitFor(() => expect(screen.getByText('sharpe_ratio')).toBeInTheDocument())
    expect(screen.getByText('1.2000')).toBeInTheDocument()
    expect(screen.getByText('0.8000')).toBeInTheDocument()
  })
})
