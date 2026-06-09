import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  Legend,
} from 'recharts'

interface Props {
  strategies: Record<string, Record<string, number>>
}

const COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899']

export default function AttributionBar({ strategies }: Props) {
  const stratNames = Object.keys(strategies)
  if (stratNames.length === 0) return null

  const metricNames = new Set<string>()
  for (const metrics of Object.values(strategies)) {
    for (const k of Object.keys(metrics)) metricNames.add(k)
  }

  const data = [...metricNames].map((metric) => {
    const row: Record<string, string | number> = { metric }
    for (const sn of stratNames) {
      row[sn] = strategies[sn]?.[metric] ?? 0
    }
    return row
  })

  return (
    <div className="bg-slate-800 rounded-xl border border-slate-700 p-5">
      <h3 className="text-sm font-semibold text-slate-300 mb-4">
        Strategy Attribution
      </h3>
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={data} margin={{ top: 5, right: 20, bottom: 5, left: 20 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
          <XAxis
            dataKey="metric"
            tick={{ fontSize: 11, fill: '#94a3b8' }}
          />
          <YAxis tick={{ fontSize: 11, fill: '#94a3b8' }} />
          <Tooltip
            contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #475569', borderRadius: 8 }}
            labelStyle={{ color: '#94a3b8' }}
          />
          <Legend wrapperStyle={{ fontSize: 12, color: '#94a3b8' }} />
          {stratNames.map((name, i) => (
            <Bar
              key={name}
              dataKey={name}
              fill={COLORS[i % COLORS.length]}
              radius={[4, 4, 0, 0]}
            />
          ))}
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}
