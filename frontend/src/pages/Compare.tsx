import { useSearchParams, Link } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { api } from '../api/client'
import Spinner from '../components/Spinner'

function formatMetric(key: string, value: number): string {
  if (value == null || isNaN(value)) return '—'
  const pctKeys = ['total_return', 'cagr', 'max_drawdown', 'volatility', 'turnover']
  if (pctKeys.some((k) => key.toLowerCase().includes(k))) {
    return `${(value * 100).toFixed(2)}%`
  }
  return value.toFixed(4)
}

// Metrics where a *smaller* value is the better outcome (all reported as
// positive magnitudes by firm.eval.metrics — see max_drawdown/annualized_
// volatility/conditional_value_at_risk) — everything else defaults to
// higher-is-better (Sharpe, CAGR, alpha, hit rate, ...).
const LOWER_IS_BETTER = ['max_drawdown', 'annualized_volatility', 'volatility', 'cvar', 'turnover']

function isLowerBetter(key: string): boolean {
  const k = key.toLowerCase()
  return LOWER_IS_BETTER.some((m) => k.includes(m))
}

export default function Compare() {
  const [searchParams] = useSearchParams()
  const idsParam = searchParams.get('ids') ?? ''
  const runIds = idsParam.split(',').filter(Boolean)

  const { data, isLoading, error } = useQuery({
    queryKey: ['compare', runIds],
    queryFn: () => api.compareRuns(runIds),
    enabled: runIds.length >= 2,
  })

  if (runIds.length < 2) {
    return (
      <div className="bg-slate-800 rounded-xl border border-slate-700 p-12 text-center">
        <p className="text-slate-400 mb-4">Select at least 2 runs to compare.</p>
        <Link to="/" className="text-blue-400 hover:text-blue-300 text-sm">
          Back to Dashboard
        </Link>
      </div>
    )
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
        <h3 className="font-semibold mb-1">Comparison failed</h3>
        <p className="text-sm">{(error as Error).message}</p>
      </div>
    )
  }

  if (!data) return null

  const metrics = Object.keys(data)

  return (
    <div>
      <div className="flex items-center justify-between flex-wrap gap-3 mb-6">
        <div>
          <h2 className="text-2xl font-bold text-white">Run Comparison</h2>
          <p className="text-sm text-slate-400 mt-1">{runIds.length} runs</p>
        </div>
        <Link
          to="/"
          className="px-4 py-2 text-sm font-medium rounded-lg border border-slate-600 text-slate-300 hover:bg-slate-700 transition-colors"
        >
          Back to Dashboard
        </Link>
      </div>

      <div className="bg-slate-800 rounded-xl border border-slate-700 overflow-hidden">
        <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-slate-700 text-left">
              <th className="px-4 py-3 text-slate-400 font-medium">Metric</th>
              {runIds.map((id) => (
                <th key={id} className="px-4 py-3 text-slate-400 font-medium text-right">
                  <Link to={`/runs/${id}`} className="text-blue-400 hover:text-blue-300 font-mono text-xs">
                    {id.slice(0, 8)}
                  </Link>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {metrics.map((metric) => {
              const values = runIds.map((id) => data[metric]?.[id])
              const valid = values.filter((v): v is number => v != null && !isNaN(v))
              const best = valid.length === 0 ? NaN : isLowerBetter(metric) ? Math.min(...valid) : Math.max(...valid)
              return (
                <tr key={metric} className="border-b border-slate-700/50">
                  <td className="px-4 py-3 text-slate-300 font-medium">{metric}</td>
                  {runIds.map((id) => {
                    const val = data[metric]?.[id]
                    const isBest = val === best && values.filter((v) => v === best).length === 1
                    return (
                      <td
                        key={id}
                        className={`px-4 py-3 text-right font-mono text-xs ${isBest ? 'text-emerald-400 font-semibold' : 'text-slate-300'}`}
                      >
                        {val != null ? formatMetric(metric, val) : '—'}
                      </td>
                    )
                  })}
                </tr>
              )
            })}
          </tbody>
        </table>
        </div>
      </div>
    </div>
  )
}
