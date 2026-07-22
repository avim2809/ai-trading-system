import { describe, it, expect } from 'vitest'
import { screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { http, HttpResponse } from 'msw'
import { server } from '../test/server'
import { renderWithProviders } from '../test/utils'
import SystemResources from './SystemResources'

const mockResources = {
  cpu: { percent: 23.5, count: 2 },
  memory: { used: 1_800_000_000, total: 3_300_000_000, percent: 54.5 },
  disk: { used: 40_000_000_000, total: 100_000_000_000, percent: 40.0, path: '/' },
}

describe('SystemResources', () => {
  it('renders CPU, memory and disk utilization', async () => {
    server.use(
      http.get('http://localhost/api/system/resources', () => HttpResponse.json(mockResources)),
    )
    renderWithProviders(<SystemResources />)
    await waitFor(() => expect(screen.getByText('Server Resources')).toBeInTheDocument())

    expect(screen.getByText('23.5%')).toBeInTheDocument()
    expect(screen.getByText('54.5%')).toBeInTheDocument()
    expect(screen.getByText('40.0%')).toBeInTheDocument()
  })

  it('shows an error block when the resources request fails', async () => {
    server.use(
      http.get('http://localhost/api/system/resources', () => HttpResponse.json({ detail: 'boom' }, { status: 500 })),
    )
    renderWithProviders(<SystemResources />)
    await waitFor(() => expect(screen.getByText('Failed to load server resources')).toBeInTheDocument())
  })

  it('requires confirmation before restarting the service', async () => {
    const user = userEvent.setup()
    let restartCalled = false
    server.use(
      http.get('http://localhost/api/system/resources', () => HttpResponse.json(mockResources)),
      http.post('http://localhost/api/system/restart', () => {
        restartCalled = true
        return HttpResponse.json({ status: 'restarting' })
      }),
    )
    renderWithProviders(<SystemResources />)
    await waitFor(() => expect(screen.getByText('Restart Service')).toBeInTheDocument())

    await user.click(screen.getByText('Restart Service'))
    expect(restartCalled).toBe(false)
    expect(screen.getByText('Restart the trading service now?')).toBeInTheDocument()

    await user.click(screen.getByText('Yes, Restart Service'))
    await waitFor(() => expect(restartCalled).toBe(true))
    await waitFor(() => expect(screen.getByText('Restarting...')).toBeInTheDocument())
  })

  it('cancels the restart confirmation without calling the API', async () => {
    const user = userEvent.setup()
    let restartCalled = false
    server.use(
      http.get('http://localhost/api/system/resources', () => HttpResponse.json(mockResources)),
      http.post('http://localhost/api/system/restart', () => {
        restartCalled = true
        return HttpResponse.json({ status: 'restarting' })
      }),
    )
    renderWithProviders(<SystemResources />)
    await waitFor(() => expect(screen.getByText('Restart Service')).toBeInTheDocument())

    await user.click(screen.getByText('Restart Service'))
    await user.click(screen.getByText('Cancel'))
    expect(restartCalled).toBe(false)
    expect(screen.getByText('Restart Service')).toBeInTheDocument()
  })

  it('requires confirmation before killing the service', async () => {
    const user = userEvent.setup()
    let killCalled = false
    server.use(
      http.get('http://localhost/api/system/resources', () => HttpResponse.json(mockResources)),
      http.post('http://localhost/api/system/kill', () => {
        killCalled = true
        return HttpResponse.json({ status: 'killing' })
      }),
    )
    renderWithProviders(<SystemResources />)
    await waitFor(() => expect(screen.getByText('Kill Service')).toBeInTheDocument())

    await user.click(screen.getByText('Kill Service'))
    expect(killCalled).toBe(false)
    expect(screen.getByText('Force-kill the trading service now?')).toBeInTheDocument()

    await user.click(screen.getByText('Yes, Kill Service'))
    await waitFor(() => expect(killCalled).toBe(true))
    await waitFor(() => expect(screen.getByText('Killing...')).toBeInTheDocument())
  })
})
