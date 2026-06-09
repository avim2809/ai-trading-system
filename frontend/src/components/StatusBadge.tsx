const statusStyles: Record<string, string> = {
  completed: 'bg-emerald-900/50 text-emerald-400 border-emerald-700',
  running:   'bg-amber-900/50 text-amber-400 border-amber-700',
  pending:   'bg-blue-900/50 text-blue-400 border-blue-700',
  failed:    'bg-red-900/50 text-red-400 border-red-700',
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
