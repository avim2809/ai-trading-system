import { formatMetric, signColor, MIN_OBSERVATIONS_FOR_RATIOS } from '../lib/metrics'

interface Props {
  strategies: Record<string, Record<string, number>>
}

export default function StrategyAttributionTable({ strategies }: Props) {
  const names = Object.keys(strategies)
  if (names.length === 0) return null

  // Best performer first — this table exists specifically to answer "which
  // strategy is winning," so ranking by total_return does that at a glance
  // without reading every row.
  const rows = [...names].sort(
    (a, b) => (strategies[b]?.total_return ?? -Infinity) - (strategies[a]?.total_return ?? -Infinity),
  )

  return (
    <div className="bg-slate-800 rounded-xl border border-slate-700 overflow-hidden">
      <div className="px-5 py-3 border-b border-slate-700">
        <h3 className="text-sm font-semibold text-slate-300">Strategy Attribution</h3>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-slate-700 text-left">
              <th className="px-4 py-3 text-slate-400 font-medium">Strategy</th>
              <th className="px-4 py-3 text-slate-400 font-medium text-right">Total Return</th>
              <th className="px-4 py-3 text-slate-400 font-medium text-right">Sharpe</th>
              <th className="px-4 py-3 text-slate-400 font-medium text-right">Max Drawdown</th>
              <th className="px-4 py-3 text-slate-400 font-medium text-right">CAGR</th>
              <th className="px-4 py-3 text-slate-400 font-medium text-right">Days</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((name) => {
              const m = strategies[name] ?? {}
              // n_days is only present on sleeved mode's exact per-sleeve
              // metrics (Orchestrator.get_sleeve_metrics) — blended mode's
              // heuristic PerformanceAttribution carries no sample-size
              // field, so there's nothing to gate on and the ratios are
              // shown as-is.
              const hasEnoughHistory = m.n_days == null || m.n_days >= MIN_OBSERVATIONS_FOR_RATIOS
              return (
                <tr key={name} className="border-b border-slate-700/50 hover:bg-slate-700/30 transition-colors">
                  <td className="px-4 py-3 font-mono text-xs text-blue-400">{name}</td>
                  <td className={`px-4 py-3 text-right font-mono text-xs ${signColor(m.total_return)}`}>
                    {m.total_return != null ? formatMetric('total_return', m.total_return) : '—'}
                  </td>
                  <td className={`px-4 py-3 text-right font-mono text-xs ${hasEnoughHistory ? signColor(m.sharpe_ratio) : 'text-slate-500'}`}>
                    {hasEnoughHistory && m.sharpe_ratio != null ? formatMetric('sharpe_ratio', m.sharpe_ratio) : 'n/a'}
                  </td>
                  <td className="px-4 py-3 text-right font-mono text-xs text-red-400">
                    {m.max_drawdown != null ? formatMetric('max_drawdown', m.max_drawdown) : '—'}
                  </td>
                  <td className={`px-4 py-3 text-right font-mono text-xs ${hasEnoughHistory ? signColor(m.cagr) : 'text-slate-500'}`}>
                    {hasEnoughHistory && m.cagr != null ? formatMetric('cagr', m.cagr) : 'n/a'}
                  </td>
                  <td className="px-4 py-3 text-right font-mono text-xs text-slate-400">
                    {m.n_days != null ? m.n_days.toFixed(0) : '—'}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  )
}
