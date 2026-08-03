import { useQuery } from '@tanstack/react-query'
import { api } from '../api/client'
import type { DecisionEntry, LessonsDigest } from '../api/types'
import Spinner from '../components/Spinner'
import StatusBadge from '../components/StatusBadge'

function formatPct(v: number | null): string {
  if (v == null) return '—'
  return `${v >= 0 ? '+' : ''}${(v * 100).toFixed(2)}%`
}

const VERDICT_STYLE: Record<string, string> = {
  correct: 'bg-emerald-900/20 border-emerald-700/40 text-emerald-300',
  incorrect: 'bg-red-900/20 border-red-700/40 text-red-300',
  partial: 'bg-amber-900/20 border-amber-700/40 text-amber-300',
  unknown: 'bg-slate-700/40 border-slate-600/50 text-slate-400',
}

function LessonsDigestPanel() {
  const { data } = useQuery<LessonsDigest>({
    queryKey: ['lessons-digest'],
    queryFn: () => api.getLessons(10),
    refetchInterval: 30000,
  })
  if (!data || data.total === 0) return null

  const { counts, recent_lessons: recentLessons } = data
  return (
    <div className="bg-slate-800 rounded-xl border border-slate-700 p-5 mb-6">
      <h3 className="text-sm font-semibold text-slate-300 mb-3">Lessons Learned</h3>
      <div className="flex flex-wrap gap-4 mb-4 text-sm">
        <span className="text-slate-400">
          {data.total} reflected decision{data.total !== 1 ? 's' : ''}:
        </span>
        <span className="text-emerald-400">{counts.correct} correct</span>
        <span className="text-red-400">{counts.incorrect} incorrect</span>
        <span className="text-amber-400">{counts.partial} partial</span>
        {counts.unknown > 0 && <span className="text-slate-500">{counts.unknown} unknown</span>}
      </div>
      {recentLessons.length > 0 && (
        <div>
          <p className="text-xs font-medium text-slate-400 mb-2">Recent lessons</p>
          <ul className="space-y-1.5">
            {recentLessons.map((lesson, i) => (
              <li key={i} className="text-sm text-slate-300 flex gap-2">
                <span className="text-slate-600">•</span>
                {lesson}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
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

      <LessonsDigestPanel />

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

              {d.per_strategy && Object.keys(d.per_strategy).length > 0 && (
                <details className="mb-3">
                  <summary className="text-xs text-slate-500 cursor-pointer hover:text-slate-400">
                    Per-strategy breakdown ({Object.keys(d.per_strategy).length} strategies)
                  </summary>
                  <div className="mt-2 space-y-1.5">
                    {Object.entries(d.per_strategy).map(([strategy, weights]) => (
                      <div key={strategy} className="flex flex-wrap items-center gap-2">
                        <span className="text-xs font-mono text-blue-300">{strategy}:</span>
                        {Object.entries(weights).map(([sym, w]) => (
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
                    ))}
                  </div>
                </details>
              )}

              {d.notes && <p className="text-xs text-slate-500 mb-2">{d.notes}</p>}

              {d.verdict && (d.what_worked || d.what_failed || d.lesson) ? (
                <div className="mt-2 p-3 bg-slate-900/50 border border-slate-700/50 rounded-lg space-y-2">
                  <span
                    className={`inline-block px-2 py-0.5 rounded text-xs font-medium border ${
                      VERDICT_STYLE[d.verdict] ?? VERDICT_STYLE.unknown
                    }`}
                  >
                    {d.verdict}
                  </span>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                    <div>
                      <p className="text-xs font-medium text-slate-400 mb-1">What worked</p>
                      <p className="text-sm text-slate-300 leading-relaxed">
                        {d.what_worked || '—'}
                      </p>
                    </div>
                    <div>
                      <p className="text-xs font-medium text-slate-400 mb-1">What didn't</p>
                      <p className="text-sm text-slate-300 leading-relaxed">
                        {d.what_failed || '—'}
                      </p>
                    </div>
                  </div>
                  {d.lesson && (
                    <div>
                      <p className="text-xs font-medium text-slate-400 mb-1">Lesson</p>
                      <p className="text-sm text-slate-300 leading-relaxed">{d.lesson}</p>
                    </div>
                  )}
                </div>
              ) : d.reflection ? (
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
