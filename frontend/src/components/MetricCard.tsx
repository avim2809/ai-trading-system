interface MetricCardProps {
  label: string
  value: number | string
  format?: 'pct' | 'currency' | 'ratio' | 'number'
}

function formatValue(value: number | string, format?: string): string {
  if (typeof value === 'string') return value
  if (value == null || isNaN(value)) return '—'
  switch (format) {
    case 'pct':
      return `${(value * 100).toFixed(2)}%`
    case 'currency':
      return `$${value.toLocaleString('en-US', { minimumFractionDigits: 0, maximumFractionDigits: 0 })}`
    case 'ratio':
      return value.toFixed(2)
    case 'number':
      return value.toLocaleString('en-US', { maximumFractionDigits: 2 })
    default:
      return value.toFixed(2)
  }
}

function valueColor(value: number | string, format?: string): string {
  if (typeof value !== 'number' || !format) return 'text-white'
  if (format === 'pct' || format === 'ratio' || format === 'currency') {
    if (value > 0) return 'text-emerald-400'
    if (value < 0) return 'text-red-400'
  }
  return 'text-white'
}

export default function MetricCard({ label, value, format }: MetricCardProps) {
  return (
    <div className="bg-slate-800 rounded-xl border border-slate-700 px-5 py-4">
      <p className="text-xs font-medium text-slate-400 uppercase tracking-wider">
        {label}
      </p>
      <p className={`mt-1 text-2xl font-semibold ${valueColor(value, format)}`}>
        {formatValue(value, format)}
      </p>
    </div>
  )
}
