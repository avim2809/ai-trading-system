import { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { api } from '../api/client'
import type { RunSummary } from '../api/types'
import StatusBadge from '../components/StatusBadge'
import Spinner from '../components/Spinner'
import { formatDateTime } from '../lib/time'

const formatTime = formatDateTime

function metricDisplay(val: number | undefined, fmt: 'pct' | 'ratio'): string {
  if (val === undefined || val === null || isNaN(val)) return '—'
  if (fmt === 'pct') return `${(val * 100).toFixed(1)}%`
  return val.toFixed(2)
}

export default function Dashboard() {
  const navigate = useNavigate()
  const qc = useQueryClient()
  const [selected, setSelected] = useState<Set<string>>(new Set())
  const [confirmClear, setConfirmClear] = useState(false)

  const { data: runs, isLoading, error } = useQuery<RunSummary[]>({
    queryKey: ['runs'],
    queryFn: () => api.getRuns(),
    refetchInterval: (query) => {
      const data = query.state.data
      if (!data) return false
      const hasActive = data.some((r) => r.status === 'running' || r.status === 'pending')
      return hasActive ? 5000 : false
    },
  })

  const clearMut = useMutation({
    mutationFn: () => api.clearRuns(),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['runs'] })
      setConfirmClear(false)
    },
  })

  const toggleSelect = (id: string) => {
    setSelected((prev) => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  }

  const handleCompare = () => {
    if (selected.size < 2) return
    navigate(`/compare?ids=${[...selected].join(',')}`)
  }

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
        <h3 className="font-semibold mb-1">Failed to load runs</h3>
        <p className="text-sm">{(error as Error).message}</p>
      </div>
    )
  }

  return (
    <div>
      <div className="flex items-center justify-between flex-wrap gap-3 mb-6">
        <div>
          <h2 className="text-2xl font-bold text-white">Dashboard</h2>
          <p className="text-sm text-slate-400 mt-1">
            {runs?.length ?? 0} backtest run{runs?.length !== 1 ? 's' : ''}
          </p>
        </div>
        <div className="flex items-center gap-3">
          <button
            onClick={handleCompare}
            disabled={selected.size < 2}
            className="px-4 py-2 text-sm font-medium rounded-lg border border-slate-600 text-slate-300 hover:bg-slate-700 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
          >
            Compare Selected ({selected.size})
          </button>
          {runs && runs.length > 0 && (
            confirmClear ? (
              <div className="flex items-center gap-2">
                <span className="text-xs text-amber-400">Delete all {runs.length} runs?</span>
                <button
                  onClick={() => clearMut.mutate()}
                  disabled={clearMut.isPending}
                  className="px-3 py-2 text-sm font-medium rounded-lg bg-red-600 text-white hover:bg-red-500 disabled:opacity-40 transition-colors flex items-center gap-2"
                >
                  {clearMut.isPending && <Spinner className="h-3.5 w-3.5" />}
                  Yes, Clear All
                </button>
                <button
                  onClick={() => setConfirmClear(false)}
                  className="px-3 py-2 text-sm text-slate-400 hover:text-slate-200 transition-colors"
                >
                  Cancel
                </button>
              </div>
            ) : (
              <button
                onClick={() => setConfirmClear(true)}
                className="px-4 py-2 text-sm font-medium rounded-lg border border-red-700 text-red-400 hover:bg-red-900/20 transition-colors"
              >
                Clear All Runs
              </button>
            )
          )}
          <Link
            to="/new"
            className="px-4 py-2 text-sm font-medium rounded-lg bg-blue-600 text-white hover:bg-blue-500 transition-colors"
          >
            New Backtest
          </Link>
        </div>
      </div>
      {clearMut.error && (
        <div className="bg-red-900/20 border border-red-700 rounded-xl p-4 text-red-400 mb-4 text-sm">
          {(clearMut.error as Error).message}
        </div>
      )}

      {!runs || runs.length === 0 ? (
        <div className="bg-slate-800 rounded-xl border border-slate-700 p-12 text-center">
          <p className="text-slate-400 mb-4">No backtest runs yet.</p>
          <Link
            to="/new"
            className="inline-block px-5 py-2.5 bg-blue-600 text-white rounded-lg hover:bg-blue-500 transition-colors text-sm font-medium"
          >
            Launch Your First Backtest
          </Link>
        </div>
      ) : (
        <div className="bg-slate-800 rounded-xl border border-slate-700 overflow-hidden">
          <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-slate-700 text-left">
                <th className="px-4 py-3 w-10">
                  <span className="sr-only">Select</span>
                </th>
                <th className="px-4 py-3 text-slate-400 font-medium">Run ID</th>
                <th className="px-4 py-3 text-slate-400 font-medium">Status</th>
                <th className="px-4 py-3 text-slate-400 font-medium">Started</th>
                <th className="px-4 py-3 text-slate-400 font-medium">Notes</th>
                <th className="px-4 py-3 text-slate-400 font-medium text-right">Sharpe</th>
                <th className="px-4 py-3 text-slate-400 font-medium text-right">CAGR</th>
                <th className="px-4 py-3 text-slate-400 font-medium text-right">Max DD</th>
              </tr>
            </thead>
            <tbody>
              {runs.map((run) => (
                <tr
                  key={run.run_id}
                  className="border-b border-slate-700/50 hover:bg-slate-700/30 transition-colors"
                >
                  <td className="px-4 py-3">
                    <input
                      type="checkbox"
                      checked={selected.has(run.run_id)}
                      onChange={() => toggleSelect(run.run_id)}
                      className="rounded border-slate-600 bg-slate-700 text-blue-500 focus:ring-blue-500 focus:ring-offset-0"
                      aria-label={`Select run ${run.run_id}`}
                    />
                  </td>
                  <td className="px-4 py-3">
                    <Link
                      to={`/runs/${run.run_id}`}
                      className="text-blue-400 hover:text-blue-300 font-mono text-xs"
                    >
                      {run.run_id.slice(0, 8)}
                    </Link>
                  </td>
                  <td className="px-4 py-3">
                    <StatusBadge status={run.status} />
                  </td>
                  <td className="px-4 py-3 text-slate-400 text-xs">
                    {formatTime(run.start_time)}
                  </td>
                  <td className="px-4 py-3 text-slate-400 text-xs max-w-[200px] truncate">
                    {run.notes || '—'}
                  </td>
                  <td className="px-4 py-3 text-right font-mono text-xs">
                    {metricDisplay(run.metrics['sharpe_ratio'], 'ratio')}
                  </td>
                  <td className="px-4 py-3 text-right font-mono text-xs">
                    <span className={run.metrics['cagr'] != null && run.metrics['cagr']! > 0 ? 'text-emerald-400' : 'text-red-400'}>
                      {metricDisplay(run.metrics['cagr'], 'pct')}
                    </span>
                  </td>
                  <td className="px-4 py-3 text-right font-mono text-xs text-red-400">
                    {metricDisplay(run.metrics['max_drawdown'], 'pct')}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          </div>
        </div>
      )}
    </div>
  )
}
