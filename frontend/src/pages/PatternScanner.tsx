import { useState } from 'react'
import { useMutation, useQuery, useQueryClient, keepPreviousData } from '@tanstack/react-query'
import { api } from '../api/client'
import type {
  PatternMatchRecord,
  PatternScanQuery,
  PatternScanTriggerRequest,
  PatternSummary,
} from '../api/types'
import Spinner from '../components/Spinner'
import { fmtNum } from '../lib/format'

// Chart pattern detection, Phase 5 (docs/pattern_recognition_plan.md §2) —
// on-demand trigger + filterable results view over the Phase 3 REST API
// (src/firm/patterns/, Strategy #13). The 17 names below are the 4 rule
// families' full detector set (rules/{reversal,triangle,continuation,
// cup_handle}.py) — kept as a literal list since there's no
// `/api/strategies`-style generic endpoint to source pattern names from.
const KNOWN_PATTERNS = [
  'head_shoulders_top',
  'inverse_head_shoulders',
  'double_top',
  'double_bottom',
  'triple_top',
  'triple_bottom',
  'ascending_triangle',
  'descending_triangle',
  'symmetrical_triangle',
  'rising_wedge',
  'falling_wedge',
  'rectangle',
  'bull_flag',
  'bear_flag',
  'pennant',
  'cup_handle',
  'rounding_bottom',
] as const

function humanize(pattern: string): string {
  return pattern.replace(/_/g, ' ')
}

function DirectionBadge({ direction }: { direction: string }) {
  const isLong = direction === 'long'
  return (
    <span
      className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium border ${
        isLong
          ? 'bg-emerald-900/50 text-emerald-400 border-emerald-700'
          : 'bg-red-900/50 text-red-400 border-red-700'
      }`}
    >
      {direction}
    </span>
  )
}

/** Quality score is a 0-100 rubric (scorer.py) — color-band it the same way
 * a reader would eyeball a grade, so a scan of the table surfaces the
 * high-conviction rows without reading every number. */
function qualityColor(score: number | null): string {
  if (score == null) return 'text-slate-400'
  if (score >= 70) return 'text-emerald-400'
  if (score >= 50) return 'text-amber-400'
  return 'text-red-400'
}

function StatTile({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-slate-800 rounded-xl border border-slate-700 px-5 py-4">
      <p className="text-xs font-medium text-slate-400 uppercase tracking-wider">{label}</p>
      <p className="mt-1 text-2xl font-semibold text-white">{value}</p>
    </div>
  )
}

/** Highest-count entry in a {pattern: count} map, humanized + count — e.g. "cup handle (3)". */
function topPatternLabel(byPattern: Record<string, number> | undefined): string {
  if (!byPattern || Object.keys(byPattern).length === 0) return '—'
  const [pattern, count] = Object.entries(byPattern).sort((a, b) => b[1] - a[1])[0]!
  return `${humanize(pattern)} (${count})`
}

export default function PatternScanner() {
  const qc = useQueryClient()

  // Trigger form state
  const [symbols, setSymbols] = useState('AAPL,MSFT,GOOG,AMZN,META')
  const [asofDate, setAsofDate] = useState('2024-01-15')
  const [dataSource, setDataSource] = useState<'synthetic' | 'cache'>('synthetic')

  // Results filter state
  const [filterPattern, setFilterPattern] = useState('')
  const [filterMinScore, setFilterMinScore] = useState('')
  const [filterDirection, setFilterDirection] = useState<'' | 'long' | 'short'>('')

  const scan = useMutation({
    mutationFn: (req: PatternScanTriggerRequest) => api.triggerPatternScan(req),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['pattern-scan'] })
      qc.invalidateQueries({ queryKey: ['pattern-summary'] })
    },
  })

  const handleScan = () => {
    const symbolList = symbols.split(',').map((s) => s.trim()).filter(Boolean)
    if (symbolList.length === 0) return
    scan.mutate({ symbols: symbolList, asof: asofDate, data_source: dataSource })
  }

  const query: PatternScanQuery = {}
  if (filterPattern) query.pattern = filterPattern
  const minScoreNum = filterMinScore.trim() === '' ? NaN : Number(filterMinScore)
  if (!Number.isNaN(minScoreNum)) query.min_score = minScoreNum
  if (filterDirection) query.direction = filterDirection

  const { data: summary } = useQuery<PatternSummary>({
    queryKey: ['pattern-summary'],
    queryFn: api.getPatternSummary,
  })

  // GET /patterns/scan is already best-quality-first server-side (scanner.py)
  // — no client-side re-sort here. Each distinct filter combination is its
  // own query key (server-side filtering via querystring, not client-side
  // array filtering) — placeholderData: keepPreviousData keeps the prior
  // result on screen while a new combination loads instead of blanking the
  // whole page back to the top-level isLoading spinner on every keystroke/
  // dropdown change.
  const { data: matchesRaw, isLoading, isFetching, error } = useQuery({
    queryKey: ['pattern-scan', filterPattern, filterMinScore, filterDirection],
    queryFn: () => api.getPatternScan(query),
    placeholderData: keepPreviousData,
  })
  const matches: PatternMatchRecord[] = Array.isArray(matchesRaw) ? matchesRaw : []

  // last_scan is server-side truth (the in-memory scan cache) for whether a
  // scan has ever run in this backend process — distinct from "ran but the
  // current filters/universe happen to match nothing".
  const hasScanned = summary?.last_scan != null

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
        <h3 className="font-semibold mb-1">Failed to load pattern scan results</h3>
        <p className="text-sm">{(error as Error).message}</p>
      </div>
    )
  }

  return (
    <div>
      <h2 className="text-2xl font-bold text-white mb-6">Pattern Scanner</h2>

      {/* Scan Configuration */}
      <div className="bg-slate-800 rounded-xl border border-slate-700 p-5 mb-6">
        <h3 className="text-sm font-semibold text-slate-300 mb-4">Scan Configuration</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          <div>
            <label className="block text-xs text-slate-400 mb-1">Symbols (comma-separated)</label>
            <input
              type="text"
              value={symbols}
              onChange={(e) => setSymbols(e.target.value)}
              className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-lg text-slate-200 text-sm focus:outline-none focus:ring-1 focus:ring-blue-500"
            />
          </div>

          <div>
            <label className="block text-xs text-slate-400 mb-1">As-of Date</label>
            <input
              type="date"
              value={asofDate}
              onChange={(e) => setAsofDate(e.target.value)}
              className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-lg text-slate-200 text-sm focus:outline-none focus:ring-1 focus:ring-blue-500"
            />
          </div>

          <div>
            <label className="block text-xs text-slate-400 mb-1">Data Source</label>
            <select
              value={dataSource}
              onChange={(e) => setDataSource(e.target.value as 'synthetic' | 'cache')}
              className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-lg text-slate-200 text-sm focus:outline-none focus:ring-1 focus:ring-blue-500"
            >
              <option value="synthetic">Synthetic</option>
              <option value="cache">Cache</option>
            </select>
          </div>
        </div>

        <button
          onClick={handleScan}
          disabled={scan.isPending || symbols.trim() === ''}
          className="mt-4 px-5 py-2.5 bg-blue-600 text-white rounded-lg text-sm font-medium hover:bg-blue-500 disabled:opacity-40 disabled:cursor-not-allowed transition-colors flex items-center gap-2"
        >
          {scan.isPending && <Spinner className="h-4 w-4" />}
          Scan Now
        </button>

        {scan.data && (
          <p className="mt-3 text-xs text-slate-400">
            Last scan: <span className="text-slate-200 font-mono">{scan.data.scanned}</span> symbol
            {scan.data.scanned !== 1 ? 's' : ''} scanned,{' '}
            <span className="text-slate-200 font-mono">{scan.data.matches}</span> match
            {scan.data.matches !== 1 ? 'es' : ''} found.
          </p>
        )}

        {scan.error && (
          <div className="mt-3 bg-red-900/20 border border-red-700 rounded-lg p-3 text-red-400 text-sm">
            {(scan.error as Error).message}
          </div>
        )}
      </div>

      {/* Summary stats */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
        <StatTile label="Total Matches" value={String(summary?.total ?? 0)} />
        <StatTile label="Top Pattern" value={topPatternLabel(summary?.by_pattern)} />
        <StatTile
          label="Long / Short"
          value={`${summary?.by_direction?.long ?? 0} / ${summary?.by_direction?.short ?? 0}`}
        />
      </div>

      {/* Filters */}
      <div className="flex flex-wrap items-end gap-3 mb-4">
        <div>
          <label className="block text-xs text-slate-400 mb-1">Pattern</label>
          <select
            value={filterPattern}
            onChange={(e) => setFilterPattern(e.target.value)}
            className="px-3 py-2 bg-slate-800 border border-slate-600 rounded-lg text-slate-200 text-sm focus:outline-none focus:ring-1 focus:ring-blue-500"
          >
            <option value="">All Patterns</option>
            {KNOWN_PATTERNS.map((p) => (
              <option key={p} value={p}>{humanize(p)}</option>
            ))}
          </select>
        </div>

        <div>
          <label className="block text-xs text-slate-400 mb-1">Min Quality Score</label>
          <input
            type="number"
            min={0}
            max={100}
            placeholder="0"
            value={filterMinScore}
            onChange={(e) => setFilterMinScore(e.target.value)}
            className="w-32 px-3 py-2 bg-slate-800 border border-slate-600 rounded-lg text-slate-200 text-sm focus:outline-none focus:ring-1 focus:ring-blue-500"
          />
        </div>

        <div>
          <label className="block text-xs text-slate-400 mb-1">Direction</label>
          <select
            value={filterDirection}
            onChange={(e) => setFilterDirection(e.target.value as '' | 'long' | 'short')}
            className="px-3 py-2 bg-slate-800 border border-slate-600 rounded-lg text-slate-200 text-sm focus:outline-none focus:ring-1 focus:ring-blue-500"
          >
            <option value="">All</option>
            <option value="long">Long</option>
            <option value="short">Short</option>
          </select>
        </div>

        <p className="text-xs text-slate-500 mb-2 flex items-center gap-2">
          {matches.length} match{matches.length !== 1 ? 'es' : ''}
          {isFetching && !isLoading && <Spinner className="h-3 w-3" />}
        </p>
      </div>

      {/* Results table */}
      <div className="bg-slate-800 rounded-xl border border-slate-700 overflow-hidden">
        {matches.length === 0 ? (
          <div className="p-8 text-center text-sm text-slate-500">
            {hasScanned
              ? 'No pattern matches found for the current filters.'
              : 'No scans run yet — configure symbols above and click "Scan Now" to detect chart patterns.'}
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-slate-700 text-left">
                  <th className="px-4 py-3 text-slate-400 font-medium">Symbol</th>
                  <th className="px-4 py-3 text-slate-400 font-medium">Pattern</th>
                  <th className="px-4 py-3 text-slate-400 font-medium">Direction</th>
                  <th className="px-4 py-3 text-slate-400 font-medium">Stop → Entry → Target</th>
                  <th className="px-4 py-3 text-slate-400 font-medium text-right">Quality</th>
                  <th className="px-4 py-3 text-slate-400 font-medium text-right">R:R</th>
                  <th className="px-4 py-3 text-slate-400 font-medium text-right">Vol Ratio</th>
                  <th className="px-4 py-3 text-slate-400 font-medium text-right">Duration</th>
                  <th className="px-4 py-3 text-slate-400 font-medium">Confirmed</th>
                </tr>
              </thead>
              <tbody>
                {matches.map((m, i) => (
                  <tr
                    key={`${m.symbol}-${m.pattern}-${m.asof}-${i}`}
                    className="border-b border-slate-700/50 hover:bg-slate-700/30 transition-colors"
                  >
                    <td className="px-4 py-3 font-mono text-xs text-blue-400">{m.symbol}</td>
                    <td className="px-4 py-3 text-xs text-slate-200">{humanize(m.pattern)}</td>
                    <td className="px-4 py-3">
                      <DirectionBadge direction={m.direction} />
                    </td>
                    <td className="px-4 py-3 font-mono text-xs text-slate-300">
                      {fmtNum(m.stop, 2)} <span className="text-slate-600">→</span> {fmtNum(m.entry, 2)}{' '}
                      <span className="text-slate-600">→</span> {fmtNum(m.target, 2)}
                    </td>
                    <td className={`px-4 py-3 text-right font-mono text-xs ${qualityColor(m.quality_score)}`}>
                      {fmtNum(m.quality_score, 0)}
                    </td>
                    <td className="px-4 py-3 text-right font-mono text-xs text-slate-300">
                      {m.risk_reward != null ? `${fmtNum(m.risk_reward, 1)}x` : '—'}
                    </td>
                    <td className="px-4 py-3 text-right font-mono text-xs text-slate-300">
                      {m.volume_ratio != null ? `${fmtNum(m.volume_ratio, 2)}x` : '—'}
                    </td>
                    <td className="px-4 py-3 text-right font-mono text-xs text-slate-300">{m.duration_bars}</td>
                    <td className="px-4 py-3 text-xs">
                      {m.confirmed ? (
                        <span className="text-emerald-400">✓</span>
                      ) : (
                        <span className="text-slate-500">—</span>
                      )}{' '}
                      <span className="font-mono text-slate-400">@{m.confirm_index}</span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  )
}
