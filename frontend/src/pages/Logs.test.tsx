import { describe, it, expect } from 'vitest'
import { screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { http, HttpResponse } from 'msw'
import { server } from '../test/server'
import { renderWithProviders } from '../test/utils'
import Logs from './Logs'

describe('Logs', () => {
  it('renders a fetched log line and shows the live indicator', async () => {
    renderWithProviders(<Logs />)
    await waitFor(() => expect(screen.getByText(/Cycle 1: 20 generated/)).toBeInTheDocument())
    expect(screen.getByText('Live')).toBeInTheDocument()
  })

  it('shows a waiting message before any lines have arrived', () => {
    server.use(http.get('http://localhost/api/logs/tail', () => HttpResponse.json({ lines: [], next_offset: 0, reset: false })))
    renderWithProviders(<Logs />)
    expect(screen.getByText('Waiting for log output…')).toBeInTheDocument()
  })

  it('filters lines by level', async () => {
    server.use(http.get('http://localhost/api/logs/tail', () => HttpResponse.json({
      lines: [
        { ts: '2026-07-21T00:00:00Z', level: 'INFO', logger: 'a', msg: 'info line' },
        { ts: '2026-07-21T00:00:01Z', level: 'ERROR', logger: 'b', msg: 'error line' },
      ],
      next_offset: 100, reset: false,
    })))
    const user = userEvent.setup()
    renderWithProviders(<Logs />)
    await waitFor(() => expect(screen.getByText('info line')).toBeInTheDocument())
    expect(screen.getByText('error line')).toBeInTheDocument()

    await user.selectOptions(screen.getByRole('combobox'), 'ERROR')
    expect(screen.queryByText('info line')).not.toBeInTheDocument()
    expect(screen.getByText('error line')).toBeInTheDocument()
  })

  it('filters lines by search text across logger and message', async () => {
    server.use(http.get('http://localhost/api/logs/tail', () => HttpResponse.json({
      lines: [
        { ts: '2026-07-21T00:00:00Z', level: 'INFO', logger: 'firm.live.engine', msg: 'cycle done' },
        { ts: '2026-07-21T00:00:01Z', level: 'INFO', logger: 'firm.rag.store', msg: 'ingested docs' },
      ],
      next_offset: 100, reset: false,
    })))
    const user = userEvent.setup()
    renderWithProviders(<Logs />)
    await waitFor(() => expect(screen.getByText('cycle done')).toBeInTheDocument())

    await user.type(screen.getByPlaceholderText(/Filter by logger, file, function, or message/), 'rag')
    expect(screen.queryByText('cycle done')).not.toBeInTheDocument()
    expect(screen.getByText('ingested docs')).toBeInTheDocument()
  })

  it('shows file:function:line when present, and matches it via search', async () => {
    server.use(http.get('http://localhost/api/logs/tail', () => HttpResponse.json({
      lines: [
        { ts: '2026-07-21T00:00:00Z', level: 'INFO', logger: 'firm.live.engine', msg: 'cycle done', file: 'engine.py', function: 'run_cycle', line: 42 },
        { ts: '2026-07-21T00:00:01Z', level: 'INFO', logger: 'firm.rag.store', msg: 'ingested docs' },
      ],
      next_offset: 100, reset: false,
    })))
    const user = userEvent.setup()
    renderWithProviders(<Logs />)
    await waitFor(() => expect(screen.getByText('engine.py:run_cycle:42')).toBeInTheDocument())
    expect(screen.getByText('ingested docs')).toBeInTheDocument()

    await user.type(screen.getByPlaceholderText(/Filter by logger, file, function, or message/), 'run_cycle')
    expect(screen.getByText('cycle done')).toBeInTheDocument()
    expect(screen.queryByText('ingested docs')).not.toBeInTheDocument()
  })

  it('Clear button empties the buffer', async () => {
    const user = userEvent.setup()
    renderWithProviders(<Logs />)
    await waitFor(() => expect(screen.getByText(/Cycle 1: 20 generated/)).toBeInTheDocument())
    await user.click(screen.getByText('Clear'))
    expect(screen.queryByText(/Cycle 1: 20 generated/)).not.toBeInTheDocument()
  })

  it('Pause button toggles the live indicator', async () => {
    const user = userEvent.setup()
    renderWithProviders(<Logs />)
    await waitFor(() => expect(screen.getByText('Live')).toBeInTheDocument())
    await user.click(screen.getByText('Pause'))
    expect(screen.getByText('Paused')).toBeInTheDocument()
    expect(screen.getByText('Resume')).toBeInTheDocument()
  })
})
