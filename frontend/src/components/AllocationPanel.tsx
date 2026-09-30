import type { AllocationStatus } from '../api/types'
import { formatDateTime } from '../lib/time'

/** Engine timestamps are naive UTC ISO strings; mark them as UTC before parsing. */
function asUtc(iso: string): string {
  return /([zZ]|[+-]\d{2}:?\d{2})$/.test(iso) ? iso : `${iso}Z`
}

function pct(v: number | undefined | null, digits = 1): string {
  if (v === undefined || v === null || Number.isNaN(v)) return '—'
  return `${(v * 100).toFixed(digits)}%`
}

const DAY_STATUS_LABEL: Record<string, string> = {
  complete: 'orders submitted',
  settled: 'done (all filled)',
  retry: 'retry pending',
}

/** Read-only view of the allocation portfolio (strategy_mode: allocation). */
export default function AllocationPanel({ allocation }: { allocation: AllocationStatus }) {
  const plan = allocation.plan
  const targets = plan?.targets ?? {}
  const current = allocation.current_weights ?? {}
  const planned = new Map((plan?.orders ?? []).map((o) => [o.symbol, o]))
  const symbols = Array.from(new Set([...Object.keys(targets), ...Object.keys(current)]))
    .filter((s) => s in targets || (current[s] ?? 0) !== 0)
    .sort((a, b) => (targets[b] ?? 0) - (targets[a] ?? 0) || a.localeCompare(b))
  const managed = symbols.filter((s) => s in targets)
  const unmanaged = symbols.filter((s) => !(s in targets))
  const totalTarget = Object.values(targets).reduce((a, b) => a + b, 0)

  return (
    <div className="bg-slate-800 rounded-xl border border-slate-700 p-5 mb-6" data-testid="allocation-panel">
      <div className="flex flex-wrap items-start justify-between gap-3 mb-4">
        <div className="min-w-0">
          <h3 className="text-sm font-semibold text-slate-300">Allocation Portfolio</h3>
          <p className="text-xs text-slate-400 mt-0.5">
            Strategy pipeline bypassed — broker positions vs. sleeve targets (band ±{pct(allocation.band_abs)})
          </p>
        </div>
        <div className="flex flex-wrap gap-2 flex-shrink-0">
          <span className="px-2 py-0.5 bg-emerald-900/30 border border-emerald-700/40 rounded text-xs text-emerald-300 font-mono">
            mode: allocation
          </span>
          {allocation.day && (
            <span className="px-2 py-0.5 bg-slate-900/50 border border-slate-600 rounded text-xs text-slate-300">
              {allocation.day}: {DAY_STATUS_LABEL[allocation.day_status ?? ''] ?? allocation.day_status ?? '—'}
            </span>
          )}
        </div>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 mb-4">
        {allocation.sleeves.map((s) => (
          <div key={s.name} className="bg-slate-900/40 border border-slate-700 rounded-lg p-3 min-w-0">
            <div className="flex items-center justify-between gap-2">
              <span className="text-sm text-slate-200 font-mono truncate">{s.name}</span>
              <span className="text-xs text-slate-300 flex-shrink-0">{pct(s.weight)} of NAV</span>
            </div>
            <p className="text-xs text-slate-400 mt-1 truncate">{s.symbols.join(', ')}</p>
            <p className="text-xs text-slate-500 mt-1">
              last rebalance: {s.last_rebalance ? formatDateTime(asUtc(s.last_rebalance)) : 'never'}
            </p>
          </div>
        ))}
        <div className="bg-slate-900/40 border border-slate-700 rounded-lg p-3">
          <div className="flex items-center justify-between gap-2">
            <span className="text-sm text-slate-200 font-mono">cash</span>
            <span className="text-xs text-slate-300">{pct(Math.max(0, 1 - totalTarget))} target</span>
          </div>
          <p className="text-xs text-slate-500 mt-1">
            kill switch {pct(allocation.kill_switch_drawdown, 0)} drawdown
          </p>
        </div>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="text-left text-xs text-slate-400 border-b border-slate-700">
              <th className="py-2 pr-4 font-medium">Symbol</th>
              <th className="py-2 pr-4 font-medium text-right">Target</th>
              <th className="py-2 pr-4 font-medium text-right">Actual</th>
              <th className="py-2 pr-4 font-medium text-right">Drift</th>
              <th className="py-2 font-medium">Last plan</th>
            </tr>
          </thead>
          <tbody>
            {managed.map((sym) => {
              const t = targets[sym] ?? 0
              const a = current[sym] ?? 0
              const drift = a - t
              const order = planned.get(sym)
              const outOfBand = Math.abs(drift) > allocation.band_abs
              return (
                <tr key={sym} className="border-b border-slate-700/50">
                  <td className="py-2 pr-4 font-mono text-slate-200">{sym}</td>
                  <td className="py-2 pr-4 text-right text-slate-300">{pct(t)}</td>
                  <td className="py-2 pr-4 text-right text-slate-300">{pct(a)}</td>
                  <td className={`py-2 pr-4 text-right ${outOfBand ? 'text-amber-400' : 'text-slate-400'}`}>
                    {drift >= 0 ? '+' : ''}{pct(drift)}
                  </td>
                  <td className="py-2 text-xs text-slate-400 whitespace-nowrap">
                    {order ? `${order.side} ${order.fractional ? order.quantity.toFixed(6) : order.quantity}` : '—'}
                  </td>
                </tr>
              )
            })}
            {unmanaged.map((sym) => (
              <tr key={sym} className="border-b border-slate-700/50">
                <td className="py-2 pr-4 font-mono text-slate-400">{sym}</td>
                <td className="py-2 pr-4 text-right text-slate-500">unmanaged</td>
                <td className="py-2 pr-4 text-right text-slate-300">{pct(current[sym])}</td>
                <td className="py-2 pr-4 text-right text-slate-500">—</td>
                <td className="py-2 text-xs text-slate-400 whitespace-nowrap">
                  {planned.get(sym) ? `${planned.get(sym)!.side} ${planned.get(sym)!.quantity}` : '—'}
                </td>
              </tr>
            ))}
            {symbols.length === 0 && (
              <tr>
                <td colSpan={5} className="py-3 text-xs text-slate-500">No allocation plan yet — runs on the first regular-hours cycle.</td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      {plan && (
        <p className="text-xs text-slate-500 mt-3">
          Last plan {formatDateTime(asUtc(plan.asof))}: due {plan.due_sleeves.length ? plan.due_sleeves.join(', ') : 'none'},
          {' '}{plan.orders.length} order(s), gross {pct(plan.gross_before)} → {pct(plan.gross_after)}
        </p>
      )}
      {plan && plan.errors.length > 0 && (
        <ul className="mt-2 space-y-1">
          {plan.errors.map((e, i) => (
            <li key={i} className="text-xs text-amber-400 break-words">{e}</li>
          ))}
        </ul>
      )}
    </div>
  )
}
