import { useQuery } from '@tanstack/react-query'
import { api } from '../api/client'
import type { DecisionEntry } from '../api/types'
import Spinner from '../components/Spinner'
import StatusBadge from '../components/StatusBadge'

function formatPct(v: number | null): string {
  if (v == null) return '—'
  return `${v >= 0 ? '+' : ''}${(v * 100).toFixed(2)}%`
}

export default function Decisions() {
  const { data: decisions, isLoading, error } = useQuery<DecisionEntry[]>({
    queryKey: ['decisions'],
    queryFn: () => api.getDecisions(100),
    refetchInterval: 15000,
  })

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
        <h3 className="font-semibold mb-1">Failed to load decision log</h3>
        <p className="text-sm">{(error as Error).message}</p>
      </div>
    )
  }

  return (
    <div>
      <div className="mb-6">
        <h2 className="text-2xl font-bold text-white">Decision Log</h2>
        <p className="text-sm text-slate-400 mt-1">
          Every trading decision (live or backtest) and its outcome once known — the same memory
          fed back into future LLM-enhanced agent prompts for self-reflection.
        </p>
      </div>

      {!decisions || decisions.length === 0 ? (
        <div className="bg-slate-800 rounded-xl border border-slate-700 p-12 text-center">
          <p className="text-slate-400">No decisions recorded yet.</p>
        </div>
      ) : (
        <div className="space-y-3">
          {decisions.map((d) => (
            <div key={d.date} className="bg-slate-800 rounded-xl border border-slate-700 p-5">
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-3">
                  <span className="font-mono text-sm text-slate-200">{d.date}</span>
                  <StatusBadge status={d.status === 'reflected' ? 'completed' : 'pending'} />
                </div>
                {d.raw_return != null && (
                  <div className="flex items-center gap-4 text-sm">
                    <span className={d.raw_return >= 0 ? 'text-emerald-400' : 'text-red-400'}>
                      Return: {formatPct(d.raw_return)}
                    </span>
                    {d.benchmark_return != null && (
                      <span className="text-slate-400">
                        Alpha: {formatPct(d.raw_return - d.benchmark_return)}
                      </span>
                    )}
                  </div>
                )}
              </div>

              <div className="flex flex-wrap gap-2 mb-3">
                {Object.entries(d.proposal_weights).map(([sym, w]) => (
                  <span
                    key={sym}
                    className={`px-2 py-0.5 rounded text-xs font-mono border ${
                      w >= 0
                        ? 'bg-emerald-900/20 border-emerald-700/40 text-emerald-300'
                        : 'bg-red-900/20 border-red-700/40 text-red-300'
                    }`}
                  >
                    {sym} {w >= 0 ? '+' : ''}{(w * 100).toFixed(1)}%
                  </span>
                ))}
              </div>

              {d.notes && <p className="text-xs text-slate-500 mb-2">{d.notes}</p>}

              {d.reflection ? (
                <div className="mt-2 p-3 bg-slate-900/50 border border-slate-700/50 rounded-lg">
                  <p className="text-xs font-medium text-slate-400 mb-1">Reflection</p>
                  <p className="text-sm text-slate-300 leading-relaxed">{d.reflection}</p>
                </div>
              ) : (
                <p className="text-xs text-slate-600 italic">Awaiting outcome — not yet reflected on.</p>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
