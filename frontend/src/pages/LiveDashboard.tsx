import { useState } from 'react'
import { Link } from 'react-router-dom'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { api } from '../api/client'
import type { LiveStatus, LiveStartRequest, BrokerPosition, AccountInfo, CycleRecord, PendingApproval } from '../api/types'
import MetricCard from '../components/MetricCard'
import StatusBadge from '../components/StatusBadge'
import Spinner from '../components/Spinner'

function formatUptime(seconds: number | null): string {
  if (seconds == null) return '—'
  const h = Math.floor(seconds / 3600)
  const m = Math.floor((seconds % 3600) / 60)
  const s = Math.floor(seconds % 60)
  if (h > 0) return `${h}h ${m}m ${s}s`
  if (m > 0) return `${m}m ${s}s`
  return `${s}s`
}

function formatTime(iso: string): string {
  return new Date(iso).toLocaleString('en-US', {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  })
}

function formatCurrency(val: number): string {
  return val.toLocaleString('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 })
}

export default function LiveDashboard() {
  const qc = useQueryClient()
  const [showStartForm, setShowStartForm] = useState(false)
  const [startBroker, setStartBroker] = useState('alpaca_paper')

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
      <div className="flex items-center justify-between mb-6">
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
                className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-lg text-slate-200 text-sm focus:outline-none focus:ring-1 focus:ring-blue-500"
              >
                <option value="alpaca_paper">Alpaca Paper</option>
                <option value="alpaca_live">Alpaca Live</option>
                <option value="ibkr_paper">Interactive Brokers (Paper)</option>
                <option value="ibkr_live">Interactive Brokers (Live)</option>
              </select>
            </div>
          </div>
          <div className="flex gap-3 mt-4">
            <button
              onClick={() =>
                startMut.mutate({ broker: startBroker })
              }
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
