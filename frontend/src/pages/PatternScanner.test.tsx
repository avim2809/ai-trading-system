import { describe, it, expect } from 'vitest'
import { screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { http, HttpResponse } from 'msw'
import { server } from '../test/server'
import { renderWithProviders } from '../test/utils'
import PatternScanner from './PatternScanner'
import * as m from '../test/mockData'

// Note: table cell text like "cup handle" is intentionally not asserted on
// directly — the Pattern filter <select> renders all 17 humanized pattern
// names as <option> text too, so a plain getByText('cup handle') would
// match both and throw "multiple elements found". Symbol text (AAPL/MSFT)
// is unique to the results table and used for presence/absence checks instead.

describe('PatternScanner', () => {
  it('shows the empty state before any scan has been triggered', async () => {
    renderWithProviders(<PatternScanner />)
    await waitFor(() => expect(screen.getByText('Pattern Scanner')).toBeInTheDocument())
    expect(screen.getByText(/No scans run yet/)).toBeInTheDocument()
    expect(screen.getByText('Total Matches')).toBeInTheDocument()
    expect(screen.getByText('0')).toBeInTheDocument()
  })

  it('triggers a scan and renders the resulting matches', async () => {
    const user = userEvent.setup()
    let triggered = false
    server.use(
      http.post('http://localhost/api/patterns/scan/trigger', () => {
        triggered = true
        return HttpResponse.json({
          scanned: 2,
          matches: 2,
          last_scan: m.mockPatternSummary.last_scan,
        })
      }),
      http.get('http://localhost/api/patterns/scan', () =>
        HttpResponse.json(triggered ? m.mockPatternMatches : [])),
      http.get('http://localhost/api/patterns/summary', () =>
        HttpResponse.json(triggered ? m.mockPatternSummary : { total: 0, by_pattern: {}, by_direction: {}, last_scan: null })),
    )

    renderWithProviders(<PatternScanner />)
    await waitFor(() => expect(screen.getByText(/No scans run yet/)).toBeInTheDocument())

    await user.click(screen.getByText('Scan Now'))

    await waitFor(() => expect(screen.getByText('AAPL')).toBeInTheDocument())
    expect(screen.getByText('MSFT')).toBeInTheDocument()
    expect(screen.getByText(/scanned/)).toBeInTheDocument()
    // Summary tiles refresh too (invalidated on trigger success). Scoped to
    // the tile itself since "2" also appears in the "Last scan: ..." line
    // above (once per scanned/matches span) and would otherwise be ambiguous.
    const totalMatchesTile = screen.getByText('Total Matches').closest('div')!
    await waitFor(() => expect(within(totalMatchesTile).getByText('2')).toBeInTheDocument())
  })

  it('filters displayed rows by pattern, min score, and direction', async () => {
    const user = userEvent.setup()
    server.use(
      http.get('http://localhost/api/patterns/scan', ({ request }) => {
        const url = new URL(request.url)
        let rows = m.mockPatternMatches
        const pattern = url.searchParams.get('pattern')
        const minScore = url.searchParams.get('min_score')
        const direction = url.searchParams.get('direction')
        if (pattern) rows = rows.filter((r) => r.pattern === pattern)
        if (minScore !== null) rows = rows.filter((r) => (r.quality_score ?? 0) >= Number(minScore))
        if (direction) rows = rows.filter((r) => r.direction === direction)
        return HttpResponse.json(rows)
      }),
      http.get('http://localhost/api/patterns/summary', () => HttpResponse.json(m.mockPatternSummary)),
    )

    renderWithProviders(<PatternScanner />)
    await waitFor(() => expect(screen.getByText('AAPL')).toBeInTheDocument())
    expect(screen.getByText('MSFT')).toBeInTheDocument()

    const selects = screen.getAllByRole('combobox')
    const directionSelect = selects.find((el) =>
      Array.from(el.querySelectorAll('option')).some((o) => o.textContent === 'Long'))!
    const patternSelect = selects.find((el) =>
      Array.from(el.querySelectorAll('option')).some((o) => o.textContent === 'All Patterns'))!

    // Direction filter: mockPatternMatches has AAPL=long, MSFT=short.
    await user.selectOptions(directionSelect, 'short')
    await waitFor(() => expect(screen.queryByText('AAPL')).not.toBeInTheDocument())
    expect(screen.getByText('MSFT')).toBeInTheDocument()
    await user.selectOptions(directionSelect, '')
    await waitFor(() => expect(screen.getByText('AAPL')).toBeInTheDocument())

    // Pattern filter: AAPL=cup_handle, MSFT=head_shoulders_top.
    await user.selectOptions(patternSelect, 'head_shoulders_top')
    await waitFor(() => expect(screen.queryByText('AAPL')).not.toBeInTheDocument())
    expect(screen.getByText('MSFT')).toBeInTheDocument()
    await user.selectOptions(patternSelect, '')
    await waitFor(() => expect(screen.getByText('AAPL')).toBeInTheDocument())

    // Min score filter: AAPL=82, MSFT=76.
    await user.type(screen.getByPlaceholderText('0'), '80')
    await waitFor(() => expect(screen.queryByText('MSFT')).not.toBeInTheDocument())
    expect(screen.getByText('AAPL')).toBeInTheDocument()
  })
})
