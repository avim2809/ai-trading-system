// Shared across backtest runs (completed/running/pending/failed), broker
// orders (filled/partial/pending/cancelled/rejected) and approvals
// (approved/pending/rejected/expired) — every status any caller passes needs
// an explicit entry here, or it silently falls back to a neutral gray that
// can't be told apart from other neutral/negative outcomes (e.g. "approved"
// vs "rejected" both reading as the same gray badge in Approvals history).
const statusStyles: Record<string, string> = {
  completed: 'bg-emerald-900/50 text-emerald-400 border-emerald-700',
  filled:    'bg-emerald-900/50 text-emerald-400 border-emerald-700',
  approved:  'bg-emerald-900/50 text-emerald-400 border-emerald-700',
  running:   'bg-amber-900/50 text-amber-400 border-amber-700',
  partial:   'bg-amber-900/50 text-amber-400 border-amber-700',
  pending:   'bg-blue-900/50 text-blue-400 border-blue-700',
  failed:    'bg-red-900/50 text-red-400 border-red-700',
  rejected:  'bg-red-900/50 text-red-400 border-red-700',
  cancelled: 'bg-slate-700/50 text-slate-400 border-slate-600',
  expired:   'bg-slate-700/50 text-slate-400 border-slate-600',
}

export default function StatusBadge({ status }: { status: string }) {
  const style = statusStyles[status] ?? 'bg-slate-700 text-slate-300 border-slate-600'
  return (
    <span
      className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium border ${style}`}
    >
      {status === 'running' && (
        <span className="w-1.5 h-1.5 rounded-full bg-amber-400 mr-1.5 animate-pulse" />
      )}
      {status}
    </span>
  )
}
