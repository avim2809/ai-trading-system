import { useParams, Link } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { api } from '../api/client'
import MetricCard from '../components/MetricCard'
import StatusBadge from '../components/StatusBadge'
import EquityCurveChart from '../components/EquityCurveChart'
import DrawdownChart from '../components/DrawdownChart'
import MonthlyHeatmap from '../components/MonthlyHeatmap'
import AttributionBar from '../components/AttributionBar'
import Spinner from '../components/Spinner'

export default function RunDetail() {
  const { runId } = useParams<{ runId: string }>()

  const {
    data: run,
    isLoading: runLoading,
    error: runError,
  } = useQuery({
    queryKey: ['run', runId],
    queryFn: () => api.getRun(runId!),
    enabled: !!runId,
    refetchInterval: (query) => {
      const status = query.state.data?.status
      return status === 'running' || status === 'pending' ? 3000 : false
    },
  })

  const isComplete = run?.status === 'completed'

  const { data: report } = useQuery({
    queryKey: ['report', runId],
    queryFn: () => api.getReport(runId!),
    enabled: !!runId && isComplete,
  })

  const { data: equity } = useQuery({
    queryKey: ['equity', runId],
    queryFn: () => api.getEquity(runId!),
    enabled: !!runId && isComplete,
  })

  if (runLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Spinner className="h-8 w-8" />
      </div>
    )
  }

  if (runError || !run) {
    return (
      <div className="bg-red-900/20 border border-red-700 rounded-xl p-6 text-red-400">
        <h3 className="font-semibold mb-1">Failed to load run</h3>
        <p className="text-sm">{(runError as Error)?.message ?? 'Run not found'}</p>
        <Link to="/" className="text-blue-400 hover:text-blue-300 text-sm mt-3 inline-block">
          Back to Dashboard
        </Link>
      </div>
    )
  }

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <div>
          <Link to="/" className="text-sm text-slate-400 hover:text-slate-300 mb-2 inline-block">
            &larr; Dashboard
          </Link>
          <h2 className="text-2xl font-bold text-white flex items-center gap-3">
            Run {run.run_id.slice(0, 8)}
            <StatusBadge status={run.status} />
          </h2>
          <p className="text-sm text-slate-400 mt-1">
            Started {new Date(run.start_time).toLocaleString()}
            {run.end_time && ` · Ended ${new Date(run.end_time).toLocaleString()}`}
          </p>
          {run.notes && (
            <p className="text-sm text-slate-400 mt-1 italic">{run.notes}</p>
          )}
        </div>
      </div>

      {/* Config summary */}
      <div className="bg-slate-800 rounded-xl border border-slate-700 p-5 mb-6">
        <h3 className="text-sm font-semibold text-slate-300 mb-3">Configuration</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-xs">
          {Object.entries(run.config).slice(0, 8).map(([k, v]) => (
            <div key={k}>
              <span className="text-slate-400">{k}: </span>
              <span className="text-slate-200 font-mono">
                {typeof v === 'object' ? JSON.stringify(v) : String(v)}
              </span>
            </div>
          ))}
          <div>
            <span className="text-slate-400">seed: </span>
            <span className="text-slate-200 font-mono">{run.seed}</span>
          </div>
          <div>
            <span className="text-slate-400">config_hash: </span>
            <span className="text-slate-200 font-mono">{run.config_hash.slice(0, 12)}</span>
          </div>
        </div>
      </div>

      {/* Running state */}
      {(run.status === 'running' || run.status === 'pending') && (
        <div className="bg-slate-800 rounded-xl border border-amber-700/50 p-8 text-center">
          <Spinner className="h-10 w-10 mx-auto mb-4" />
          <p className="text-slate-300 font-medium">Backtest is {run.status}...</p>
          <p className="text-xs text-slate-500 mt-1">Auto-refreshing every 3 seconds</p>
        </div>
      )}

      {/* Failed state */}
      {run.status === 'failed' && (
        <div className="bg-red-900/20 border border-red-700 rounded-xl p-6 text-red-400">
          <h3 className="font-semibold mb-1">Backtest Failed</h3>
          <p className="text-sm">{run.notes || 'Unknown error'}</p>
        </div>
      )}

      {/* Completed: metrics + charts */}
      {isComplete && report && (
        <>
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4 mb-6">
            <MetricCard label="CAGR" value={report.portfolio['cagr'] ?? 0} format="pct" />
            <MetricCard label="Sharpe Ratio" value={report.portfolio['sharpe_ratio'] ?? 0} format="ratio" />
            <MetricCard label="Sortino Ratio" value={report.portfolio['sortino_ratio'] ?? 0} format="ratio" />
            <MetricCard label="Max Drawdown" value={report.portfolio['max_drawdown'] ?? 0} format="pct" />
            <MetricCard label="Volatility" value={report.portfolio['annualized_volatility'] ?? 0} format="pct" />
            <MetricCard label="Final NAV" value={report.final_nav ?? 0} format="currency" />
          </div>

          {report.benchmark && Object.keys(report.benchmark).length > 0 && (
            <div className="mb-6">
              <h2 className="text-sm font-medium text-slate-400 uppercase tracking-wider mb-3">
                Benchmark-Relative (vs equal-weight buy &amp; hold)
              </h2>
              <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4">
                <MetricCard label="Alpha (ann.)" value={report.benchmark['alpha'] ?? 0} format="pct" />
                <MetricCard label="Beta" value={report.benchmark['beta'] ?? 0} format="ratio" />
                <MetricCard label="Information Ratio" value={report.benchmark['information_ratio'] ?? 0} format="ratio" />
                <MetricCard label="Excess Return" value={report.benchmark['excess_return'] ?? 0} format="pct" />
                <MetricCard label="Benchmark Return" value={report.benchmark['benchmark_total_return'] ?? 0} format="pct" />
              </div>
            </div>
          )}

          {equity && (
            <div className="space-y-6 mb-6">
              <EquityCurveChart dates={equity.dates} values={equity.values} />
              <DrawdownChart dates={equity.dates} drawdown={equity.drawdown} />
              <MonthlyHeatmap dates={equity.dates} values={equity.values} />
            </div>
          )}

          {report.strategies && Object.keys(report.strategies).length > 0 && (
            <AttributionBar strategies={report.strategies} />
          )}
        </>
      )}
    </div>
  )
}
