import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useQuery, useMutation } from '@tanstack/react-query'
import { api } from '../api/client'
import type { RunRequest } from '../api/types'
import StrategyConfigForm from '../components/StrategyConfigForm'
import Spinner from '../components/Spinner'

export default function NewBacktest() {
  const navigate = useNavigate()

  const { data: strategies, isLoading: stratLoading } = useQuery({
    queryKey: ['strategies'],
    queryFn: api.getStrategies,
  })

  const { data: defaults } = useQuery({
    queryKey: ['defaults'],
    queryFn: api.getDefaults,
  })

  const [selected, setSelected] = useState<string[]>([])
  const [params, setParams] = useState<Record<string, Record<string, unknown>>>({})
  const [symbols, setSymbols] = useState('')
  const [startDate, setStartDate] = useState('2023-01-01')
  const [endDate, setEndDate] = useState('2024-01-01')
  const [capital, setCapital] = useState(100000)
  const [commission, setCommission] = useState(0.001)
  const [slippage, setSlippage] = useState(0.0005)
  const [rebalance, setRebalance] = useState('daily')
  const [dataSource, setDataSource] = useState('synthetic')
  const [seed, setSeed] = useState(42)
  const [notes, setNotes] = useState('')
  const [showRisk, setShowRisk] = useState(false)
  const [riskOverrides, setRiskOverrides] = useState<Record<string, number>>({})

  const launch = useMutation({
    mutationFn: (req: RunRequest) => api.launchRun(req),
    onSuccess: (data) => {
      navigate(`/runs/${data.run_id}`)
    },
  })

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (selected.length === 0) return

    const universeSymbols = symbols.trim()
      ? symbols.split(',').map((s) => s.trim()).filter(Boolean)
      : null

    launch.mutate({
      strategies: selected,
      strategy_params: params as Record<string, Record<string, unknown>>,
      universe_symbols: universeSymbols,
      start_date: startDate,
      end_date: endDate,
      initial_capital: capital,
      commission_pct: commission,
      slippage_pct: slippage,
      rebalance_frequency: rebalance,
      risk_overrides: riskOverrides,
      data_source: dataSource,
      seed,
      notes,
    })
  }

  if (stratLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Spinner className="h-8 w-8" />
      </div>
    )
  }

  return (
    <div className="max-w-3xl">
      <h2 className="text-2xl font-bold text-white mb-6">New Backtest</h2>

      <form onSubmit={handleSubmit} className="space-y-8">
        {/* Strategy Selection */}
        <section>
          {strategies && (
            <StrategyConfigForm
              strategies={strategies}
              selected={selected}
              params={params}
              onChange={(s, p) => { setSelected(s); setParams(p) }}
            />
          )}
        </section>

        {/* Universe */}
        <section>
          <label className="block text-sm font-medium text-slate-300 mb-2">
            Universe (comma-separated symbols, leave blank for default)
          </label>
          <input
            type="text"
            value={symbols}
            onChange={(e) => setSymbols(e.target.value)}
            placeholder={String(defaults?.universe?.['symbols'] ?? 'AAPL, MSFT, GOOGL, ...')}
            className="w-full px-4 py-2.5 bg-slate-800 border border-slate-600 rounded-lg text-slate-200 placeholder-slate-500 focus:outline-none focus:ring-1 focus:ring-blue-500 text-sm"
          />
        </section>

        {/* Date Range */}
        <section>
          <label className="block text-sm font-medium text-slate-300 mb-2">Date Range</label>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-xs text-slate-400 mb-1">Start Date</label>
              <input
                type="date"
                value={startDate}
                onChange={(e) => setStartDate(e.target.value)}
                className="w-full px-4 py-2.5 bg-slate-800 border border-slate-600 rounded-lg text-slate-200 focus:outline-none focus:ring-1 focus:ring-blue-500 text-sm"
              />
            </div>
            <div>
              <label className="block text-xs text-slate-400 mb-1">End Date</label>
              <input
                type="date"
                value={endDate}
                onChange={(e) => setEndDate(e.target.value)}
                className="w-full px-4 py-2.5 bg-slate-800 border border-slate-600 rounded-lg text-slate-200 focus:outline-none focus:ring-1 focus:ring-blue-500 text-sm"
              />
            </div>
          </div>
        </section>

        {/* Capital & Costs */}
        <section>
          <label className="block text-sm font-medium text-slate-300 mb-2">Capital & Costs</label>
          <div className="grid grid-cols-3 gap-4">
            <div>
              <label className="block text-xs text-slate-400 mb-1">Initial Capital ($)</label>
              <input
                type="number"
                value={capital}
                onChange={(e) => setCapital(Number(e.target.value))}
                className="w-full px-4 py-2.5 bg-slate-800 border border-slate-600 rounded-lg text-slate-200 focus:outline-none focus:ring-1 focus:ring-blue-500 text-sm"
              />
            </div>
            <div>
              <label className="block text-xs text-slate-400 mb-1">Commission (%)</label>
              <input
                type="number"
                step="0.0001"
                value={commission}
                onChange={(e) => setCommission(Number(e.target.value))}
                className="w-full px-4 py-2.5 bg-slate-800 border border-slate-600 rounded-lg text-slate-200 focus:outline-none focus:ring-1 focus:ring-blue-500 text-sm"
              />
            </div>
            <div>
              <label className="block text-xs text-slate-400 mb-1">Slippage (%)</label>
              <input
                type="number"
                step="0.0001"
                value={slippage}
                onChange={(e) => setSlippage(Number(e.target.value))}
                className="w-full px-4 py-2.5 bg-slate-800 border border-slate-600 rounded-lg text-slate-200 focus:outline-none focus:ring-1 focus:ring-blue-500 text-sm"
              />
            </div>
          </div>
        </section>

        {/* Rebalance */}
        <section>
          <label className="block text-sm font-medium text-slate-300 mb-2">Rebalance Frequency</label>
          <select
            value={rebalance}
            onChange={(e) => setRebalance(e.target.value)}
            className="w-full px-4 py-2.5 bg-slate-800 border border-slate-600 rounded-lg text-slate-200 focus:outline-none focus:ring-1 focus:ring-blue-500 text-sm"
          >
            <option value="daily">Daily</option>
            <option value="weekly">Weekly</option>
            <option value="monthly">Monthly</option>
          </select>
        </section>

        {/* Risk Overrides */}
        <section>
          <button
            type="button"
            onClick={() => setShowRisk(!showRisk)}
            className="text-sm font-medium text-slate-300 hover:text-white flex items-center gap-2"
          >
            <span className={`transition-transform ${showRisk ? 'rotate-90' : ''}`}>▶</span>
            Risk Overrides (optional)
          </button>
          {showRisk && (
            <div className="mt-3 grid grid-cols-2 gap-3">
              {['max_position_pct', 'max_sector_pct', 'max_drawdown_pct'].map((key) => (
                <div key={key}>
                  <label className="block text-xs text-slate-400 mb-1">{key}</label>
                  <input
                    type="number"
                    step="0.01"
                    value={riskOverrides[key] ?? ''}
                    onChange={(e) =>
                      setRiskOverrides((prev) => ({
                        ...prev,
                        [key]: Number(e.target.value),
                      }))
                    }
                    placeholder="default"
                    className="w-full px-3 py-1.5 bg-slate-700 border border-slate-600 rounded-md text-slate-200 placeholder-slate-500 focus:outline-none focus:ring-1 focus:ring-blue-500 text-sm"
                  />
                </div>
              ))}
            </div>
          )}
        </section>

        {/* Data Source */}
        <section>
          <label className="block text-sm font-medium text-slate-300 mb-2">Data Source</label>
          <div className="flex gap-6">
            {(['synthetic', 'cache'] as const).map((src) => (
              <label key={src} className="flex items-center gap-2 cursor-pointer">
                <input
                  type="radio"
                  name="dataSource"
                  value={src}
                  checked={dataSource === src}
                  onChange={() => setDataSource(src)}
                  className="text-blue-500 focus:ring-blue-500 focus:ring-offset-0"
                />
                <span className="text-sm text-slate-300 capitalize">{src}</span>
              </label>
            ))}
          </div>
        </section>

        {/* Seed */}
        <section>
          <label className="block text-sm font-medium text-slate-300 mb-2">Random Seed</label>
          <input
            type="number"
            value={seed}
            onChange={(e) => setSeed(Number(e.target.value))}
            className="w-40 px-4 py-2.5 bg-slate-800 border border-slate-600 rounded-lg text-slate-200 focus:outline-none focus:ring-1 focus:ring-blue-500 text-sm"
          />
        </section>

        {/* Notes */}
        <section>
          <label className="block text-sm font-medium text-slate-300 mb-2">Notes</label>
          <textarea
            value={notes}
            onChange={(e) => setNotes(e.target.value)}
            rows={3}
            placeholder="Description of this backtest run..."
            className="w-full px-4 py-2.5 bg-slate-800 border border-slate-600 rounded-lg text-slate-200 placeholder-slate-500 focus:outline-none focus:ring-1 focus:ring-blue-500 text-sm resize-none"
          />
        </section>

        {/* Submit */}
        {launch.error && (
          <div className="bg-red-900/20 border border-red-700 rounded-lg p-4 text-red-400 text-sm">
            {(launch.error as Error).message}
          </div>
        )}

        <button
          type="submit"
          disabled={selected.length === 0 || launch.isPending}
          className="px-6 py-3 bg-blue-600 text-white rounded-lg font-medium hover:bg-blue-500 disabled:opacity-40 disabled:cursor-not-allowed transition-colors flex items-center gap-2"
        >
          {launch.isPending && <Spinner className="h-4 w-4" />}
          Launch Backtest
        </button>
      </form>
    </div>
  )
}
