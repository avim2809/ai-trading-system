import { useState } from 'react'
import { Link } from 'react-router-dom'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { api } from '../api/client'
import type {
  LiveStatus, LiveStartRequest, BrokerPosition, AccountInfo, CycleRecord,
  PendingApproval, StrategyInfo, LiveAlertsResponse,
} from '../api/types'
import MetricCard from '../components/MetricCard'
import StatusBadge from '../components/StatusBadge'
import Spinner from '../components/Spinner'
import { formatDateTime } from '../lib/time'

const SCHEDULES = [
  { value: 'market_open', label: 'Market Open' },
  { value: 'market_close', label: 'Market Close' },
  { value: 'every_15_min', label: 'Every 15 Minutes' },
  { value: 'hourly', label: 'Hourly' },
]

const inputCls =
  'w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-lg text-slate-200 text-sm focus:outline-none focus:ring-1 focus:ring-blue-500'

function formatUptime(seconds: number | null): string {
  if (seconds == null) return '—'
  const h = Math.floor(seconds / 3600)
  const m = Math.floor((seconds % 3600) / 60)
  const s = Math.floor(seconds % 60)
  if (h > 0) return `${h}h ${m}m ${s}s`
  if (m > 0) return `${m}m ${s}s`
  return `${s}s`
}

const formatTime = (iso: string) => formatDateTime(iso, { seconds: true })

function formatCurrency(val: number): string {
  return val.toLocaleString('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 })
}

export default function LiveDashboard() {
  const qc = useQueryClient()
  const [showStartForm, setShowStartForm] = useState(false)
  const [startBroker, setStartBroker] = useState('alpaca_paper')
  const [startSchedule, setStartSchedule] = useState('market_open')
  const [startInitialCapital, setStartInitialCapital] = useState('100000')
  const [startSymbols, setStartSymbols] = useState('')
  const [startEnabledStrategies, setStartEnabledStrategies] = useState<Set<string>>(new Set())
  const [startAutoApprove, setStartAutoApprove] = useState<Set<string>>(new Set())
  const [startKillSwitch, setStartKillSwitch] = useState('0.10')

  const { data: availableStrategies } = useQuery<StrategyInfo[]>({
    queryKey: ['strategies'],
    queryFn: api.getStrategies,
  })

  const { data: alertsData } = useQuery<LiveAlertsResponse>({
    queryKey: ['live-alerts'],
    queryFn: api.getAlerts,
    refetchInterval: 5000,
  })

  const { data: status, isLoading } = useQuery<LiveStatus>({
    queryKey: ['live-status'],
    queryFn: api.getLiveStatus,
    refetchInterval: (query) =>
      query.state.data?.state === 'running' ? 5000 : 15000,
  })

  const { data: positions } = useQuery<BrokerPosition[]>({
    queryKey: ['live-positions'],
    queryFn: api.getPositions,
    enabled: status?.state === 'running',
    refetchInterval: 5000,
  })

  const { data: account } = useQuery<AccountInfo>({
    queryKey: ['live-account'],
    queryFn: api.getAccount,
    enabled: status?.state === 'running',
    refetchInterval: 5000,
  })

  const { data: cycles } = useQuery<CycleRecord[]>({
    queryKey: ['live-cycles'],
    queryFn: () => api.getCycles(10),
    enabled: status?.state === 'running',
    refetchInterval: 5000,
  })

  const { data: approvals } = useQuery<PendingApproval[]>({
    queryKey: ['live-approvals'],
    queryFn: api.getApprovals,
    enabled: status?.state === 'running',
    refetchInterval: 3000,
  })

  const pendingApprovals = approvals?.filter((a) => a.status === 'pending') ?? []

  const startMut = useMutation({
    mutationFn: (req: LiveStartRequest) => api.startLive(req),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['live-status'] })
      setShowStartForm(false)
    },
  })

  const stopMut = useMutation({
    mutationFn: () => api.stopLive(),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['live-status'] }),
  })

  const triggerMut = useMutation({
    mutationFn: () => api.triggerCycle(),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['live-cycles'] }),
  })

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Spinner className="h-8 w-8" />
      </div>
    )
  }

  const isRunning = status?.state === 'running'

  return (
    <div>
      <div className="flex items-center justify-between flex-wrap gap-3 mb-6">
        <div>
          <h2 className="text-2xl font-bold text-white flex items-center gap-3">
            Live Trading
            {isRunning && (
              <span className="flex items-center gap-1.5 text-sm font-normal text-emerald-400">
                <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
                Running
              </span>
            )}
          </h2>
          <p className="text-sm text-slate-400 mt-1">Real-time trading control panel</p>
        </div>
        <div className="flex gap-3">
          {isRunning ? (
            <>
              <button
                onClick={() => triggerMut.mutate()}
                disabled={triggerMut.isPending}
                className="px-4 py-2 text-sm font-medium rounded-lg border border-slate-600 text-slate-300 hover:bg-slate-700 disabled:opacity-40 transition-colors flex items-center gap-2"
              >
                {triggerMut.isPending && <Spinner className="h-3.5 w-3.5" />}
                Trigger Now
              </button>
              <button
                onClick={() => stopMut.mutate()}
                disabled={stopMut.isPending}
                className="px-4 py-2 text-sm font-medium rounded-lg bg-red-600 text-white hover:bg-red-500 disabled:opacity-40 transition-colors flex items-center gap-2"
              >
                {stopMut.isPending && <Spinner className="h-3.5 w-3.5" />}
                Stop Engine
              </button>
            </>
          ) : (
            <button
              onClick={() => setShowStartForm(!showStartForm)}
              className="px-4 py-2 text-sm font-medium rounded-lg bg-emerald-600 text-white hover:bg-emerald-500 transition-colors"
            >
              Start Engine
            </button>
          )}
        </div>
      </div>

      {/* Pending Approvals Banner */}
      {pendingApprovals.length > 0 && (
        <Link
          to="/live/approvals"
          className="block mb-6 bg-amber-900/20 border border-amber-700/50 rounded-xl p-4 hover:bg-amber-900/30 transition-colors"
        >
          <div className="flex items-center gap-3">
            <span className="w-2 h-2 rounded-full bg-amber-400 animate-pulse" />
            <span className="text-amber-300 font-medium text-sm">
              {pendingApprovals.length} pending approval{pendingApprovals.length !== 1 ? 's' : ''} awaiting review
            </span>
            <span className="ml-auto text-amber-400 text-xs">View &rarr;</span>
          </div>
        </Link>
      )}

      {/* Start Form */}
      {showStartForm && !isRunning && (
        <div className="bg-slate-800 rounded-xl border border-slate-700 p-5 mb-6">
          <h3 className="text-sm font-semibold text-slate-300 mb-4">Start Live Engine</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <label className="block text-xs text-slate-400 mb-1">Broker</label>
              <select
                value={startBroker}
                onChange={(e) => setStartBroker(e.target.value)}
                className={inputCls}
              >
                <option value="alpaca_paper">Alpaca Paper</option>
                <option value="alpaca_live">Alpaca Live</option>
                <option value="ibkr_paper">Interactive Brokers (Paper)</option>
                <option value="ibkr_live">Interactive Brokers (Live)</option>
              </select>
            </div>
            <div>
              <label className="block text-xs text-slate-400 mb-1">Schedule</label>
              <select
                value={startSchedule}
                onChange={(e) => setStartSchedule(e.target.value)}
                className={inputCls}
              >
                {SCHEDULES.map((s) => (
                  <option key={s.value} value={s.value}>{s.label}</option>
                ))}
              </select>
            </div>
            <div>
              <label className="block text-xs text-slate-400 mb-1">Initial Capital</label>
              <input
                type="number"
                value={startInitialCapital}
                onChange={(e) => setStartInitialCapital(e.target.value)}
                className={inputCls}
              />
            </div>
            <div>
              <label className="block text-xs text-slate-400 mb-1">Kill Switch Drawdown</label>
              <input
                type="number"
                step="0.01"
                value={startKillSwitch}
                onChange={(e) => setStartKillSwitch(e.target.value)}
                className={inputCls}
              />
            </div>
            <div className="md:col-span-2">
              <label className="block text-xs text-slate-400 mb-1">Symbols (comma-separated, blank = default 5)</label>
              <input
                type="text"
                value={startSymbols}
                onChange={(e) => setStartSymbols(e.target.value)}
                placeholder="AAPL, MSFT, GOOG, AMZN, META"
                className={inputCls}
              />
            </div>
          </div>

          <div className="mt-4">
            <label className="block text-xs text-slate-400 mb-2">
              Strategies <span className="text-slate-500">(none checked = all {availableStrategies?.length ?? 0} enabled)</span>
            </label>
            <div className="flex flex-wrap gap-2">
              {(availableStrategies ?? []).map((strat) => {
                const enabled = startEnabledStrategies.has(strat.name)
                const auto = startAutoApprove.has(strat.name)
                return (
                  <button
                    key={strat.name}
                    type="button"
                    onClick={() => {
                      setStartEnabledStrategies((prev) => {
                        const next = new Set(prev)
                        if (next.has(strat.name)) next.delete(strat.name)
                        else next.add(strat.name)
                        return next
                      })
                    }}
                    className={`px-3 py-1.5 rounded-lg text-xs font-medium border transition-colors flex items-center gap-2 ${
                      enabled ? 'border-blue-500/60 bg-blue-500/10 text-blue-300' : 'border-slate-700 bg-slate-900/30 text-slate-400'
                    }`}
                  >
                    {strat.name}
                    {enabled && (
                      <span
                        role="button"
                        onClick={(e) => {
                          e.stopPropagation()
                          setStartAutoApprove((prev) => {
                            const next = new Set(prev)
                            if (next.has(strat.name)) next.delete(strat.name)
                            else next.add(strat.name)
                            return next
                          })
                        }}
                        className={auto ? 'text-emerald-400' : 'text-slate-500'}
                        title="Toggle auto-approve"
                      >
                        {auto ? 'auto' : 'manual'}
                      </span>
                    )}
                  </button>
                )
              })}
            </div>
          </div>

          <div className="flex gap-3 mt-4">
            <button
              onClick={() => {
                const symbols = startSymbols.split(',').map((s) => s.trim()).filter(Boolean)
                const strategies = Array.from(startEnabledStrategies)
                const req: LiveStartRequest = {
                  broker: startBroker,
                  schedule: startSchedule,
                  approval_mode: startAutoApprove.size > 0 && strategies.length > 0 && startAutoApprove.size === strategies.length
                    ? 'full_auto' : 'semi_auto',
                  auto_approve_strategies: Array.from(startAutoApprove),
                  symbols,
                  initial_capital: parseFloat(startInitialCapital) || 100_000,
                  strategies,
                  kill_switch_drawdown: parseFloat(startKillSwitch) || 0.10,
                }
                startMut.mutate(req)
              }}
              disabled={startMut.isPending}
              className="px-5 py-2.5 bg-emerald-600 text-white rounded-lg text-sm font-medium hover:bg-emerald-500 disabled:opacity-40 transition-colors flex items-center gap-2"
            >
              {startMut.isPending && <Spinner className="h-4 w-4" />}
              Start
            </button>
            <button
              onClick={() => setShowStartForm(false)}
              className="px-5 py-2.5 border border-slate-600 text-slate-300 rounded-lg text-sm font-medium hover:bg-slate-700 transition-colors"
            >
              Cancel
            </button>
          </div>
          {startMut.error && (
            <div className="mt-3 bg-red-900/20 border border-red-700 rounded-lg p-3 text-red-400 text-sm">
              {(startMut.error as Error).message}
            </div>
          )}
        </div>
      )}

      {/* Stuck-cycle warning — independent of the alerts feed, since a real
          incident showed a cycle can hang for 24+ hours with the alert
          mechanism itself never firing. */}
      {status && status.cycle_running_seconds != null && status.cycle_running_seconds > 1800 && (
        <div className="mb-6 bg-red-900/20 border border-red-700/50 rounded-xl p-4">
          <div className="flex items-center gap-3">
            <span className="w-2 h-2 rounded-full bg-red-400 animate-pulse" />
            <span className="text-red-300 font-medium text-sm">
              A cycle has been running for {formatUptime(status.cycle_running_seconds)} — this looks stuck.
              No new cycles can start until it finishes or the engine is restarted.
            </span>
          </div>
        </div>
      )}

      {/* Alerts */}
      {alertsData && (alertsData.halted || alertsData.alerts.length > 0) && (
        <div className={`mb-6 rounded-xl border p-4 ${alertsData.halted ? 'bg-red-900/20 border-red-700/50' : 'bg-amber-900/10 border-amber-700/40'}`}>
          <div className="flex items-center gap-3 mb-2">
            <span className={`w-2 h-2 rounded-full ${alertsData.halted ? 'bg-red-400 animate-pulse' : 'bg-amber-400'}`} />
            <span className={`font-medium text-sm ${alertsData.halted ? 'text-red-300' : 'text-amber-300'}`}>
              {alertsData.halted ? 'Engine halted — drawdown kill switch tripped' : `${alertsData.alerts.length} operational alert${alertsData.alerts.length !== 1 ? 's' : ''}`}
            </span>
          </div>
          <div className="space-y-1.5 max-h-48 overflow-y-auto">
            {alertsData.alerts.slice(0, 10).map((a, i) => (
              <div key={i} className="text-xs flex items-start gap-2">
                <span className="text-slate-500 font-mono flex-shrink-0">{formatTime(a.timestamp)}</span>
                <span className={a.severity === 'critical' ? 'text-red-400' : 'text-amber-400'}>[{a.kind}]</span>
                <span className="text-slate-300">{a.message}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Status Card */}
      {status && (
        <div className="bg-slate-800 rounded-xl border border-slate-700 p-5 mb-6">
          <h3 className="text-sm font-semibold text-slate-300 mb-3">Engine Status</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4 text-sm">
            <div>
              <span className="text-xs text-slate-400">State</span>
              <div className="mt-1">
                <StatusBadge status={status.state} />
              </div>
            </div>
            <div>
              <span className="text-xs text-slate-400">Broker</span>
              <p className="mt-1 text-slate-200 font-mono text-xs">{status.broker || '—'}</p>
            </div>
            <div>
              <span className="text-xs text-slate-400">Connected</span>
              <p className="mt-1">
                <span className={`inline-flex items-center gap-1.5 text-xs font-medium ${status.broker_connected ? 'text-emerald-400' : 'text-red-400'}`}>
                  <span className={`w-1.5 h-1.5 rounded-full ${status.broker_connected ? 'bg-emerald-400' : 'bg-red-400'}`} />
                  {status.broker_connected ? 'Yes' : 'No'}
                </span>
              </p>
            </div>
            <div>
              <span className="text-xs text-slate-400">Next Run</span>
              <p className="mt-1 text-slate-200 text-xs">
                {status.next_run ? formatTime(status.next_run) : '—'}
              </p>
            </div>
            <div>
              <span className="text-xs text-slate-400">Approval Mode</span>
              <p className="mt-1 text-slate-200 font-mono text-xs">{status.approval_mode || '—'}</p>
            </div>
            <div>
              <span className="text-xs text-slate-400">Uptime</span>
              <p className="mt-1 text-slate-200 font-mono text-xs">{formatUptime(status.uptime_seconds)}</p>
            </div>
          </div>
          {status.active_strategies.length > 0 && (
            <div className="mt-4">
              <span className="text-xs text-slate-400">Active Strategies</span>
              <div className="flex flex-wrap gap-2 mt-1">
                {status.active_strategies.map((s) => (
                  <span key={s} className="px-2 py-0.5 bg-blue-900/30 border border-blue-700/40 rounded text-xs text-blue-300 font-mono">
                    {s}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Account Summary */}
      {account && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
          <MetricCard label="Cash" value={formatCurrency(account.cash)} />
          <MetricCard label="Equity" value={formatCurrency(account.equity)} />
          <MetricCard label="Buying Power" value={formatCurrency(account.buying_power)} />
        </div>
      )}

      {/* Positions Table */}
      {isRunning && (
        <div className="bg-slate-800 rounded-xl border border-slate-700 overflow-hidden mb-6">
          <div className="px-5 py-3 border-b border-slate-700">
            <h3 className="text-sm font-semibold text-slate-300">Positions</h3>
          </div>
          {!positions || positions.length === 0 ? (
            <div className="p-8 text-center text-sm text-slate-500">No open positions</div>
          ) : (
            <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-slate-700 text-left">
                  <th className="px-4 py-3 text-slate-400 font-medium">Symbol</th>
                  <th className="px-4 py-3 text-slate-400 font-medium text-right">Qty</th>
                  <th className="px-4 py-3 text-slate-400 font-medium text-right">Avg Cost</th>
                  <th className="px-4 py-3 text-slate-400 font-medium text-right">Market Value</th>
                  <th className="px-4 py-3 text-slate-400 font-medium text-right">Unrealized P&L</th>
                </tr>
              </thead>
              <tbody>
                {positions.map((pos) => (
                  <tr key={pos.symbol} className="border-b border-slate-700/50 hover:bg-slate-700/30 transition-colors">
                    <td className="px-4 py-3 font-mono text-xs text-blue-400">{pos.symbol}</td>
                    <td className="px-4 py-3 text-right font-mono text-xs">{pos.quantity}</td>
                    <td className="px-4 py-3 text-right font-mono text-xs">${pos.avg_cost.toFixed(2)}</td>
                    <td className="px-4 py-3 text-right font-mono text-xs">{formatCurrency(pos.market_value)}</td>
                    <td className={`px-4 py-3 text-right font-mono text-xs ${pos.unrealized_pnl >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                      {pos.unrealized_pnl >= 0 ? '+' : ''}{formatCurrency(pos.unrealized_pnl)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            </div>
          )}
        </div>
      )}

      {/* Recent Cycles */}
      {isRunning && cycles && cycles.length > 0 && (
        <div className="bg-slate-800 rounded-xl border border-slate-700 p-5">
          <h3 className="text-sm font-semibold text-slate-300 mb-3">Recent Cycles</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5 gap-3">
            {cycles.map((c) => (
              <div key={c.cycle_id} className="bg-slate-900/50 border border-slate-700/50 rounded-lg p-3">
                <div className="flex items-center justify-between mb-2">
                  <span className="font-mono text-xs text-slate-400">{c.cycle_id.slice(0, 8)}</span>
                  <StatusBadge status={c.error ? 'failed' : 'completed'} />
                </div>
                <p className="text-xs text-slate-400">{formatTime(c.timestamp)}</p>
                <div className="flex gap-4 mt-2 text-xs">
                  <span className="text-slate-400">
                    Generated: <span className="text-slate-200 font-mono">{c.orders_generated}</span>
                  </span>
                  <span className="text-slate-400">
                    Submitted: <span className="text-slate-200 font-mono">{c.orders_submitted}</span>
                  </span>
                  <span className="text-slate-400">
                    Queued: <span className="text-slate-200 font-mono">{c.orders_queued}</span>
                  </span>
                </div>
                {c.error && <p className="text-xs text-red-400 mt-1">{c.error}</p>}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Stopped state */}
      {!isRunning && !showStartForm && (
        <div className="bg-slate-800 rounded-xl border border-slate-700 p-12 text-center">
          <p className="text-slate-400 mb-2">Engine is stopped.</p>
          <p className="text-xs text-slate-500">Click "Start Engine" to begin live trading.</p>
        </div>
      )}
    </div>
  )
}
