import { useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { api } from '../api/client'
import type { LiveAttributionHistory } from '../api/types'
import Spinner from '../components/Spinner'
import { formatMetric, signColor } from '../lib/metrics'
import {
  bucketPeriodReturns,
  rangeReturn,
  startOfIsoWeek,
  startOfMonth,
  toIsoDate,
  type PeriodGranularity,
} from '../lib/periodReturns'

const GRANULARITIES: { key: PeriodGranularity; label: string }[] = [
  { key: 'day', label: 'Daily' },
  { key: 'week', label: 'Weekly' },
  { key: 'month', label: 'Monthly' },
  { key: 'year', label: 'Yearly' },
]

// Row cap per granularity so the breakdown table stays a "recent history"
// glance rather than an unbounded scroll -- last 30 days / 12 weeks / 12
// months; years are capped generously since there's rarely more than a
// handful.
const MAX_ROWS: Record<PeriodGranularity, number> = { day: 30, week: 12, month: 12, year: 50 }

function bucketLabel(key: string, granularity: PeriodGranularity): string {
  // 'week' keys are the Monday-start ISO date (see lib/periodReturns.ts) --
  // every other granularity's key is already a display-ready ISO
  // date/month/year prefix.
  return granularity === 'week' ? `Week of ${key}` : key
}

function ReturnCell({ value }: { value: number | undefined }) {
  return (
    <td className={`px-4 py-3 text-right font-mono text-xs ${signColor(value)}`}>
      {value != null ? formatMetric('total_return', value) : '—'}
    </td>
  )
}

// --- Sortable column headers -------------------------------------------
// Shared by all three tables on this page. Clicking an unsorted (or
// differently-sorted) column sorts it descending first; clicking the same
// column again toggles ascending/descending. `undefined` values (missing
// data for that row/column) always sort last regardless of direction, so a
// strategy with no WTD/MTD figure yet doesn't jump to the top of an
// ascending sort.
type SortDirection = 'asc' | 'desc'
interface SortState {
  key: string
  direction: SortDirection
}

function toggleSort(current: SortState | null, key: string): SortState {
  if (current?.key === key) {
    return { key, direction: current.direction === 'desc' ? 'asc' : 'desc' }
  }
  return { key, direction: 'desc' }
}

function compareValues(a: number | string | undefined, b: number | string | undefined, direction: SortDirection): number {
  if (a == null && b == null) return 0
  if (a == null) return 1
  if (b == null) return -1
  if (a < b) return direction === 'asc' ? -1 : 1
  if (a > b) return direction === 'asc' ? 1 : -1
  return 0
}

function SortableTh({
  label,
  sortKey,
  sort,
  onSort,
  align = 'right',
}: {
  label: string
  sortKey: string
  sort: SortState | null
  onSort: (key: string) => void
  align?: 'left' | 'right'
}) {
  const active = sort?.key === sortKey
  return (
    <th
      className={`px-4 py-3 text-slate-400 font-medium cursor-pointer select-none hover:text-slate-200 transition-colors ${
        align === 'right' ? 'text-right' : 'text-left'
      }`}
      onClick={() => onSort(sortKey)}
    >
      <span className="inline-flex items-center gap-1">
        {label}
        <span className={`text-[10px] ${active ? 'text-blue-400' : 'text-slate-600'}`}>
          {active ? (sort.direction === 'desc' ? '▼' : '▲') : '▼'}
        </span>
      </span>
    </th>
  )
}

export default function AttributionHistory() {
  const [granularity, setGranularity] = useState<PeriodGranularity>('day')
  const [rangeStart, setRangeStart] = useState('')
  const [rangeEnd, setRangeEnd] = useState('')
  const [appliedRange, setAppliedRange] = useState<{ start: string; end: string } | null>(null)

  // Independent sort state per table -- see SortableTh/toggleSort above.
  // null = natural/default order (unchanged from before sorting existed).
  const [headlineSort, setHeadlineSort] = useState<SortState | null>(null)
  const [breakdownSort, setBreakdownSort] = useState<SortState | null>(null)
  const [customSort, setCustomSort] = useState<SortState | null>(null)

  const { data: history, isLoading, error } = useQuery<LiveAttributionHistory>({
    queryKey: ['live-attribution-history'],
    queryFn: api.getLiveAttributionHistory,
    refetchInterval: 30000,
  })

  const strategies = useMemo(() => Object.keys(history ?? {}).sort(), [history])

  // Per-strategy bucket-key -> compounded return, at the selected
  // granularity. A plain Map keyed by bucketPeriodReturns' `key` (not the
  // display label) so period rows line up across strategies even when one
  // strategy has gaps the others don't.
  const bucketsByStrategy = useMemo(() => {
    const map: Record<string, Map<string, number>> = {}
    for (const s of strategies) {
      const series = history?.[s]
      if (!series) continue
      map[s] = new Map(
        bucketPeriodReturns(series.dates, series.returns, granularity).map((b) => [b.key, b.return]),
      )
    }
    return map
  }, [history, strategies, granularity])

  const rowKeys = useMemo(() => {
    const keys = new Set<string>()
    for (const m of Object.values(bucketsByStrategy)) {
      for (const key of m.keys()) keys.add(key)
    }
    return Array.from(keys)
      .sort((a, b) => (a < b ? 1 : a > b ? -1 : 0)) // most-recent-first
      .slice(0, MAX_ROWS[granularity])
  }, [bucketsByStrategy, granularity])

  // WTD/MTD boundaries anchor on each strategy's own last observed date,
  // not the browser's local "now" -- see lib/periodReturns.ts -- so results
  // match whatever the backend considers "today" for that strategy's data.
  const headline = useMemo(
    () =>
      strategies.map((s) => {
        const series = history?.[s]
        const lastDate = series?.dates[series.dates.length - 1]
        if (!series || !lastDate) {
          return { strategy: s, wtd: undefined, mtd: undefined } as const
        }
        const lastDateObj = new Date(`${lastDate}T00:00:00Z`)
        const weekStart = toIsoDate(startOfIsoWeek(lastDateObj))
        const monthStart = toIsoDate(startOfMonth(lastDateObj))
        return {
          strategy: s,
          wtd: rangeReturn(series.dates, series.returns, weekStart, lastDate),
          mtd: rangeReturn(series.dates, series.returns, monthStart, lastDate),
        }
      }),
    [history, strategies],
  )

  const sortedHeadline = useMemo(() => {
    if (!headlineSort) return headline
    const { key, direction } = headlineSort
    return [...headline].sort((a, b) =>
      compareValues(key === 'strategy' ? a.strategy : key === 'wtd' ? a.wtd : a.mtd,
        key === 'strategy' ? b.strategy : key === 'wtd' ? b.wtd : b.mtd, direction),
    )
  }, [headline, headlineSort])

  const sortedRowKeys = useMemo(() => {
    if (!breakdownSort) return rowKeys
    const { key, direction } = breakdownSort
    return [...rowKeys].sort((a, b) =>
      compareValues(
        key === 'period' ? a : bucketsByStrategy[key]?.get(a),
        key === 'period' ? b : bucketsByStrategy[key]?.get(b),
        direction,
      ),
    )
  }, [rowKeys, breakdownSort, bucketsByStrategy])

  const customResult = useMemo(() => {
    if (!appliedRange) return null
    const { start, end } = appliedRange
    const rows = strategies.map((s) => {
      const series = history?.[s]
      const value = series ? rangeReturn(series.dates, series.returns, start, end) : undefined
      return { strategy: s, value }
    })
    return { start, end, rows }
  }, [appliedRange, history, strategies])

  const sortedCustomRows = useMemo(() => {
    if (!customResult) return []
    if (!customSort) return customResult.rows
    const { key, direction } = customSort
    return [...customResult.rows].sort((a, b) =>
      compareValues(key === 'strategy' ? a.strategy : a.value, key === 'strategy' ? b.strategy : b.value, direction),
    )
  }, [customResult, customSort])

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Spinner className="h-8 w-8" />
      </div>
    )
  }

  if (error) {
    return (
      <div className="bg-red-900/20 border border-red-700 rounded-xl p-6 text-red-400">
        <h3 className="font-semibold mb-1">Failed to load strategy performance history</h3>
        <p className="text-sm">{(error as Error).message}</p>
      </div>
    )
  }

  return (
    <div>
      <div className="flex items-center justify-between mb-6 flex-wrap gap-2">
        <div>
          <h2 className="text-2xl font-bold text-white">Strategy Performance</h2>
          <p className="text-sm text-slate-400 mt-1">
            Per-strategy return history — day/week/month/year breakdowns, week-to-date/month-to-date, and custom
            date ranges.
          </p>
        </div>
        <Link to="/live" className="text-sm text-blue-400 hover:text-blue-300 flex-shrink-0">
          ← Back to Live Dashboard
        </Link>
      </div>

      {strategies.length === 0 ? (
        <div className="bg-slate-800 rounded-xl border border-slate-700 p-12 text-center">
          <p className="text-slate-400">
            No performance history yet — history accumulates once strategies have traded and a daily return is
            recorded.
          </p>
        </div>
      ) : (
        <>
          {/* WTD / MTD headline */}
          <div className="bg-slate-800 rounded-xl border border-slate-700 overflow-hidden mb-6">
            <div className="px-5 py-3 border-b border-slate-700">
              <h3 className="text-sm font-semibold text-slate-300">Week-to-Date / Month-to-Date</h3>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-slate-700 text-left">
                    <SortableTh label="Strategy" sortKey="strategy" sort={headlineSort} onSort={(k) => setHeadlineSort(toggleSort(headlineSort, k))} align="left" />
                    <SortableTh label="WTD" sortKey="wtd" sort={headlineSort} onSort={(k) => setHeadlineSort(toggleSort(headlineSort, k))} />
                    <SortableTh label="MTD" sortKey="mtd" sort={headlineSort} onSort={(k) => setHeadlineSort(toggleSort(headlineSort, k))} />
                  </tr>
                </thead>
                <tbody>
                  {sortedHeadline.map((row) => (
                    <tr
                      key={row.strategy}
                      className="border-b border-slate-700/50 hover:bg-slate-700/30 transition-colors"
                    >
                      <td className="px-4 py-3 font-mono text-xs text-blue-400">{row.strategy}</td>
                      <ReturnCell value={row.wtd} />
                      <ReturnCell value={row.mtd} />
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Period tabs */}
          <div className="flex gap-2 mb-4">
            {GRANULARITIES.map((g) => (
              <button
                key={g.key}
                onClick={() => setGranularity(g.key)}
                className={`px-4 py-2 text-sm font-medium rounded-lg transition-colors ${
                  granularity === g.key
                    ? 'bg-blue-600/20 text-blue-400 border border-blue-500/40'
                    : 'border border-slate-600 text-slate-400 hover:text-slate-200 hover:bg-slate-800'
                }`}
              >
                {g.label}
              </button>
            ))}
          </div>

          {/* Period breakdown table */}
          <div className="bg-slate-800 rounded-xl border border-slate-700 overflow-hidden mb-6">
            <div className="px-5 py-3 border-b border-slate-700">
              <h3 className="text-sm font-semibold text-slate-300">
                {GRANULARITIES.find((g) => g.key === granularity)?.label} Breakdown
              </h3>
            </div>
            {rowKeys.length === 0 ? (
              <div className="p-8 text-center text-sm text-slate-500">
                Not enough history yet for this granularity.
              </div>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-slate-700 text-left">
                      <SortableTh label="Period" sortKey="period" sort={breakdownSort} onSort={(k) => setBreakdownSort(toggleSort(breakdownSort, k))} align="left" />
                      {strategies.map((s) => (
                        <SortableTh key={s} label={s} sortKey={s} sort={breakdownSort} onSort={(k) => setBreakdownSort(toggleSort(breakdownSort, k))} />
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {sortedRowKeys.map((key) => (
                      <tr key={key} className="border-b border-slate-700/50 hover:bg-slate-700/30 transition-colors">
                        <td className="px-4 py-3 font-mono text-xs text-slate-300">
                          {bucketLabel(key, granularity)}
                        </td>
                        {strategies.map((s) => (
                          <ReturnCell key={s} value={bucketsByStrategy[s]?.get(key)} />
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>

          {/* Custom date range */}
          <div className="bg-slate-800 rounded-xl border border-slate-700 p-5">
            <h3 className="text-sm font-semibold text-slate-300 mb-3">Custom Date Range</h3>
            <div className="flex items-end gap-3 flex-wrap mb-4">
              <div>
                <label className="block text-xs text-slate-400 mb-1">Start Date</label>
                <input
                  type="date"
                  value={rangeStart}
                  onChange={(e) => setRangeStart(e.target.value)}
                  className="px-3 py-2 bg-slate-700 border border-slate-600 rounded-lg text-slate-200 text-sm focus:outline-none focus:ring-1 focus:ring-blue-500"
                />
              </div>
              <div>
                <label className="block text-xs text-slate-400 mb-1">End Date</label>
                <input
                  type="date"
                  value={rangeEnd}
                  onChange={(e) => setRangeEnd(e.target.value)}
                  className="px-3 py-2 bg-slate-700 border border-slate-600 rounded-lg text-slate-200 text-sm focus:outline-none focus:ring-1 focus:ring-blue-500"
                />
              </div>
              <button
                onClick={() => rangeStart && rangeEnd && setAppliedRange({ start: rangeStart, end: rangeEnd })}
                disabled={!rangeStart || !rangeEnd}
                className="px-4 py-2 text-sm font-medium rounded-lg bg-blue-600 text-white hover:bg-blue-500 disabled:opacity-40 transition-colors"
              >
                Apply
              </button>
            </div>

            {customResult && (
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-slate-700 text-left">
                      <SortableTh label="Strategy" sortKey="strategy" sort={customSort} onSort={(k) => setCustomSort(toggleSort(customSort, k))} align="left" />
                      <SortableTh
                        label={`Return (${customResult.start} → ${customResult.end})`}
                        sortKey="value"
                        sort={customSort}
                        onSort={(k) => setCustomSort(toggleSort(customSort, k))}
                      />
                    </tr>
                  </thead>
                  <tbody>
                    {sortedCustomRows.map((row) => (
                      <tr
                        key={row.strategy}
                        className="border-b border-slate-700/50 hover:bg-slate-700/30 transition-colors"
                      >
                        <td className="px-4 py-3 font-mono text-xs text-blue-400">{row.strategy}</td>
                        <ReturnCell value={row.value} />
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </>
      )}
    </div>
  )
}
