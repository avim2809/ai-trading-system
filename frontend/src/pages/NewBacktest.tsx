import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useQuery, useMutation } from '@tanstack/react-query'
import { api } from '../api/client'
import type { RunRequest, RegimeOverlayConfig } from '../api/types'
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
  const [regime, setRegime] = useState<RegimeOverlayConfig>({
    enabled: false,
    benchmark_symbol: null,
    n_states: 3,
    retrain_frequency: 21,
    exposure_map: { Bull: 1.5, Bear: 0.5, Chop: 0.25 },
  })

  const [nSplits, setNSplits] = useState(4)

  const buildRequest = (): RunRequest => {
    const universeSymbols = symbols.trim()
      ? symbols.split(',').map((s) => s.trim()).filter(Boolean)
      : null

    return {
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
      regime_overlay: regime,
      data_source: dataSource,
      seed,
      notes,
    }
  }

  const launch = useMutation({
    mutationFn: (req: RunRequest) => api.launchRun(req),
    onSuccess: (data) => {
      navigate(`/runs/${data.run_id}`)
    },
  })

  const walkForward = useMutation({
    mutationFn: () => api.launchWalkForward({ ...buildRequest(), n_splits: nSplits }),
  })

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (selected.length === 0) return
    launch.mutate(buildRequest())
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
              {['max_position_pct', 'max_gross_exposure', 'max_net_exposure', 'max_sector_pct', 'vol_target', 'max_drawdown_pct'].map((key) => (
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

        {/* Market Regime Overlay */}
        <section>
          <label className="flex items-center gap-2 text-sm font-medium text-slate-300 cursor-pointer">
            <input
              type="checkbox"
              checked={regime.enabled}
              onChange={(e) => setRegime((r) => ({ ...r, enabled: e.target.checked }))}
              className="text-blue-500 focus:ring-blue-500 focus:ring-offset-0"
            />
            Market Regime Overlay (HMM)
          </label>
          <p className="mt-1 text-xs text-slate-500">
            Scales gross exposure by the detected market regime (Bull → lever up, Bear/Chop →
            de-risk), blended by HMM posterior confidence.
          </p>
          {regime.enabled && (
            <div className="mt-3 space-y-3">
              <div className="grid grid-cols-3 gap-3">
                <div>
                  <label className="block text-xs text-slate-400 mb-1">Benchmark symbol</label>
                  <input
                    type="text"
                    value={regime.benchmark_symbol ?? ''}
                    onChange={(e) =>
                      setRegime((r) => ({ ...r, benchmark_symbol: e.target.value.trim() || null }))
                    }
                    placeholder="e.g. SPY (blank = universe avg)"
                    className="w-full px-3 py-1.5 bg-slate-700 border border-slate-600 rounded-md text-slate-200 placeholder-slate-500 focus:outline-none focus:ring-1 focus:ring-blue-500 text-sm"
                  />
                </div>
                <div>
                  <label className="block text-xs text-slate-400 mb-1">Num states</label>
                  <input
                    type="number"
                    min={2}
                    value={regime.n_states}
                    onChange={(e) => setRegime((r) => ({ ...r, n_states: Number(e.target.value) }))}
                    className="w-full px-3 py-1.5 bg-slate-700 border border-slate-600 rounded-md text-slate-200 focus:outline-none focus:ring-1 focus:ring-blue-500 text-sm"
                  />
                </div>
                <div>
                  <label className="block text-xs text-slate-400 mb-1">Retrain freq (bars)</label>
                  <input
                    type="number"
                    min={1}
                    value={regime.retrain_frequency}
                    onChange={(e) =>
                      setRegime((r) => ({ ...r, retrain_frequency: Number(e.target.value) }))
                    }
                    className="w-full px-3 py-1.5 bg-slate-700 border border-slate-600 rounded-md text-slate-200 focus:outline-none focus:ring-1 focus:ring-blue-500 text-sm"
                  />
                </div>
              </div>
              <div>
                <label className="block text-xs text-slate-400 mb-1">
                  Gross-exposure factor per regime (at full confidence)
                </label>
                <div className="grid grid-cols-3 gap-3">
                  {(['Bull', 'Bear', 'Chop'] as const).map((label) => (
                    <div key={label}>
                      <label className="block text-xs text-slate-500 mb-1">{label}</label>
                      <input
                        type="number"
                        step="0.05"
                        min={0}
                        value={regime.exposure_map[label]}
                        onChange={(e) =>
                          setRegime((r) => ({
                            ...r,
                            exposure_map: { ...r.exposure_map, [label]: Number(e.target.value) },
                          }))
                        }
                        className="w-full px-3 py-1.5 bg-slate-700 border border-slate-600 rounded-md text-slate-200 focus:outline-none focus:ring-1 focus:ring-blue-500 text-sm"
                      />
                    </div>
                  ))}
                </div>
              </div>
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

        <div className="flex items-end gap-4 flex-wrap">
          <button
            type="submit"
            disabled={selected.length === 0 || launch.isPending}
            className="px-6 py-3 bg-blue-600 text-white rounded-lg font-medium hover:bg-blue-500 disabled:opacity-40 disabled:cursor-not-allowed transition-colors flex items-center gap-2"
          >
            {launch.isPending && <Spinner className="h-4 w-4" />}
            Launch Backtest
          </button>

          <div className="flex items-end gap-2">
            <div>
              <label className="block text-xs text-slate-400 mb-1">Folds</label>
              <input
                type="number"
                min={2}
                max={20}
                value={nSplits}
                onChange={(e) => setNSplits(Number(e.target.value))}
                className="w-20 px-3 py-2.5 bg-slate-800 border border-slate-600 rounded-lg text-slate-200 focus:outline-none focus:ring-1 focus:ring-blue-500 text-sm"
              />
            </div>
            <button
              type="button"
              onClick={() => walkForward.mutate()}
              disabled={selected.length === 0 || walkForward.isPending}
              title="Run a walk-forward analysis: each fold trains on an earlier window and is scored out-of-sample"
              className="px-6 py-3 bg-slate-700 text-white rounded-lg font-medium hover:bg-slate-600 disabled:opacity-40 disabled:cursor-not-allowed transition-colors flex items-center gap-2"
            >
              {walkForward.isPending && <Spinner className="h-4 w-4" />}
              Run Walk-Forward
            </button>
          </div>
        </div>

        {walkForward.error && (
          <div className="bg-red-900/20 border border-red-700 rounded-lg p-4 text-red-400 text-sm">
            {(walkForward.error as Error).message}
          </div>
        )}

        {walkForward.data && (
          <section className="bg-slate-800 border border-slate-700 rounded-xl p-5">
            <h3 className="text-sm font-medium text-slate-300 mb-1">
              Walk-Forward — out-of-sample across {walkForward.data.aggregate.n_folds} folds
            </h3>
            <p className="text-xs text-slate-500 mb-4">
              Mean ± std of each fold's OOS metrics. Stable means with small std suggest the
              edge generalizes rather than overfitting a single window. Each fold is saved as a
              run on the Dashboard.
            </p>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
              {['sharpe_ratio', 'total_return', 'max_drawdown', 'alpha', 'information_ratio', 'excess_return']
                .filter((k) => walkForward.data!.aggregate.metrics[k])
                .map((k) => {
                  const m = walkForward.data!.aggregate.metrics[k]
                  if (!m) return null
                  return (
                    <div key={k} className="bg-slate-900/50 rounded-lg px-4 py-3">
                      <p className="text-xs text-slate-400">{k}</p>
                      <p className="text-lg font-semibold text-white">
                        {m.mean.toFixed(3)}
                        <span className="text-sm text-slate-500"> ± {m.std.toFixed(3)}</span>
                      </p>
                    </div>
                  )
                })}
            </div>
            <div className="mt-4 flex flex-wrap gap-2">
              {walkForward.data.fold_ids.map((id, i) => (
                <button
                  key={id}
                  type="button"
                  onClick={() => navigate(`/runs/${id}`)}
                  className="text-xs px-2 py-1 rounded bg-slate-700 text-slate-200 hover:bg-slate-600"
                >
                  Fold {i + 1}
                </button>
              ))}
            </div>
          </section>
        )}
      </form>
    </div>
  )
}
