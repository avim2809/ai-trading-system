interface Props {
  dates: string[]
  values: number[]
}

const MONTHS = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

function computeMonthlyReturns(dates: string[], values: number[]) {
  const monthly: Record<string, Record<number, number>> = {}
  let prevValue = values[0]!
  let prevYear = dates[0]!.slice(0, 4)
  let prevMonth = parseInt(dates[0]!.slice(5, 7), 10)
  let monthStart = values[0]!

  for (let i = 1; i < dates.length; i++) {
    const year = dates[i]!.slice(0, 4)
    const month = parseInt(dates[i]!.slice(5, 7), 10)

    if (year !== prevYear || month !== prevMonth) {
      if (!monthly[prevYear]) monthly[prevYear] = {}
      monthly[prevYear]![prevMonth - 1] = (prevValue - monthStart) / monthStart
      monthStart = prevValue
      prevYear = year
      prevMonth = month
    }
    prevValue = values[i]!
  }

  if (!monthly[prevYear]) monthly[prevYear] = {}
  monthly[prevYear]![prevMonth - 1] = (prevValue - monthStart) / monthStart

  return monthly
}

function cellStyle(ret: number | undefined): React.CSSProperties {
  if (ret === undefined) return { backgroundColor: '#1e293b' }
  const abs = Math.min(Math.abs(ret) * 10, 1)
  const alpha = 0.3 + abs * 0.7
  if (ret > 0) return { backgroundColor: `rgba(16, 185, 129, ${alpha})` }
  if (ret < 0) return { backgroundColor: `rgba(239, 68, 68, ${alpha})` }
  return { backgroundColor: '#334155' }
}

export default function MonthlyHeatmap({ dates, values }: Props) {
  if (dates.length < 2) return null

  const monthly = computeMonthlyReturns(dates, values)
  const years = Object.keys(monthly).sort()

  return (
    <div className="bg-slate-800 rounded-xl border border-slate-700 p-5 overflow-x-auto">
      <h3 className="text-sm font-semibold text-slate-300 mb-4">Monthly Returns</h3>
      <table className="w-full text-xs">
        <thead>
          <tr>
            <th className="px-2 py-1 text-left text-slate-400 font-medium">Year</th>
            {MONTHS.map((m) => (
              <th key={m} className="px-2 py-1 text-center text-slate-400 font-medium">{m}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {years.map((year) => (
            <tr key={year}>
              <td className="px-2 py-1 font-medium text-slate-300">{year}</td>
              {Array.from({ length: 12 }, (_, m) => {
                const ret = monthly[year]?.[m]
                return (
                  <td
                    key={m}
                    className="px-2 py-1.5 text-center rounded"
                    style={cellStyle(ret)}
                  >
                    {ret !== undefined ? `${(ret * 100).toFixed(1)}%` : ''}
                  </td>
                )
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
