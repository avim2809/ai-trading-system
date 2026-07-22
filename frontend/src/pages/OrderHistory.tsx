import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { api } from '../api/client'
import type { OrderRecord } from '../api/types'
import StatusBadge from '../components/StatusBadge'
import Spinner from '../components/Spinner'
import { formatDateTime } from '../lib/time'

type RangeKey = '1d' | '7d' | '30d'

const RANGE_LABELS: Record<RangeKey, string> = { '1d': 'Last 24h', '7d': 'Last 7 days', '30d': 'Last 30 days' }

function rangeMs(key: RangeKey): number {
  const day = 86_400_000
  return key === '1d' ? day : key === '7d' ? 7 * day : 30 * day
}

const formatTime = (iso: string) => formatDateTime(iso, { seconds: true })

export default function OrderHistory() {
  const qc = useQueryClient()
  const [range, setRange] = useState<RangeKey>('7d')
  const [confirmClear, setConfirmClear] = useState(false)

  const { data: allOrders, isLoading, error } = useQuery<OrderRecord[]>({
    queryKey: ['live-orders'],
    queryFn: () => api.getOrders(200),
    refetchInterval: 10000,
  })

  const clearMut = useMutation({
    mutationFn: () => api.clearCycles(),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['live-orders'] })
      qc.invalidateQueries({ queryKey: ['live-cycles'] })
      setConfirmClear(false)
    },
  })

  const cutoff = Date.now() - rangeMs(range)
  const orders = allOrders?.filter((o) => o.timestamp && new Date(o.timestamp).getTime() >= cutoff) ?? []

  const todayCutoff = Date.now() - 86_400_000
  const todayOrders = allOrders?.filter((o) => o.timestamp && new Date(o.timestamp).getTime() >= todayCutoff) ?? []
  const totalToday = todayOrders.length
  const filledToday = todayOrders.filter((o) => o.status === 'filled').length
  const fillRate = totalToday > 0 ? ((filledToday / totalToday) * 100).toFixed(0) : '—'
  const totalVolume = todayOrders.reduce(
    (sum, o) => sum + o.filled_quantity * o.avg_fill_price,
    0,
  )

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Spinner className="h-8 w-8" />
      </div>
    )
  }

  if (error) {
    return (
      <div className="bg-red-900/20 border border-red-700 rounded-xl p-6 text-red-400">
        <h3 className="font-semibold mb-1">Failed to load orders</h3>
        <p className="text-sm">{(error as Error).message}</p>
      </div>
    )
  }

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-2xl font-bold text-white">Order History</h2>
          <p className="text-sm text-slate-400 mt-1">
            {orders.length} order{orders.length !== 1 ? 's' : ''} in range
          </p>
        </div>
        {allOrders && allOrders.length > 0 && (
          confirmClear ? (
            <div className="flex items-center gap-2">
              <span className="text-xs text-amber-400">Delete all order history?</span>
              <button
                onClick={() => clearMut.mutate()}
                disabled={clearMut.isPending}
                className="px-3 py-2 text-sm font-medium rounded-lg bg-red-600 text-white hover:bg-red-500 disabled:opacity-40 transition-colors flex items-center gap-2"
              >
                {clearMut.isPending && <Spinner className="h-3.5 w-3.5" />}
                Yes, Clear All
              </button>
              <button
                onClick={() => setConfirmClear(false)}
                className="px-3 py-2 text-sm text-slate-400 hover:text-slate-200 transition-colors"
              >
                Cancel
              </button>
            </div>
          ) : (
            <button
              onClick={() => setConfirmClear(true)}
              className="px-4 py-2 text-sm font-medium rounded-lg border border-red-700 text-red-400 hover:bg-red-900/20 transition-colors"
            >
              Clear History
            </button>
          )
        )}
      </div>
      {clearMut.error && (
        <div className="bg-red-900/20 border border-red-700 rounded-xl p-4 text-red-400 mb-4 text-sm">
          {(clearMut.error as Error).message}
        </div>
      )}

      {/* Summary stats */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
        <div className="bg-slate-800 rounded-xl border border-slate-700 px-5 py-4">
          <p className="text-xs font-medium text-slate-400 uppercase tracking-wider">Orders Today</p>
          <p className="mt-1 text-2xl font-semibold text-white">{totalToday}</p>
        </div>
        <div className="bg-slate-800 rounded-xl border border-slate-700 px-5 py-4">
          <p className="text-xs font-medium text-slate-400 uppercase tracking-wider">Fill Rate</p>
          <p className="mt-1 text-2xl font-semibold text-white">{fillRate}{fillRate !== '—' ? '%' : ''}</p>
        </div>
        <div className="bg-slate-800 rounded-xl border border-slate-700 px-5 py-4">
          <p className="text-xs font-medium text-slate-400 uppercase tracking-wider">Total Volume (today)</p>
          <p className="mt-1 text-2xl font-semibold text-white">
            ${totalVolume.toLocaleString('en-US', { maximumFractionDigits: 0 })}
          </p>
        </div>
      </div>

      {/* Range filter */}
      <div className="flex gap-2 mb-4">
        {(Object.keys(RANGE_LABELS) as RangeKey[]).map((key) => (
          <button
            key={key}
            onClick={() => setRange(key)}
            className={`px-3 py-1.5 text-xs font-medium rounded-lg transition-colors ${
              range === key
                ? 'bg-blue-600/20 text-blue-400 border border-blue-500/40'
                : 'border border-slate-600 text-slate-400 hover:text-slate-200 hover:bg-slate-800'
            }`}
          >
            {RANGE_LABELS[key]}
          </button>
        ))}
      </div>

      {/* Orders table */}
      <div className="bg-slate-800 rounded-xl border border-slate-700 overflow-hidden">
        {orders.length === 0 ? (
          <div className="p-8 text-center text-sm text-slate-500">No orders in selected range</div>
        ) : (
          <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-slate-700 text-left">
                <th className="px-4 py-3 text-slate-400 font-medium">Timestamp</th>
                <th className="px-4 py-3 text-slate-400 font-medium">Symbol</th>
                <th className="px-4 py-3 text-slate-400 font-medium">Side</th>
                <th className="px-4 py-3 text-slate-400 font-medium text-right">Qty</th>
                <th className="px-4 py-3 text-slate-400 font-medium text-right">Fill Price</th>
                <th className="px-4 py-3 text-slate-400 font-medium">Status</th>
                <th className="px-4 py-3 text-slate-400 font-medium">Strategy</th>
              </tr>
            </thead>
            <tbody>
              {orders.map((o) => (
                <tr key={o.order_id} className="border-b border-slate-700/50 hover:bg-slate-700/30 transition-colors">
                  <td className="px-4 py-3 text-slate-400 text-xs">{o.timestamp ? formatTime(o.timestamp) : '—'}</td>
                  <td className="px-4 py-3 font-mono text-xs text-blue-400">{o.symbol}</td>
                  <td className="px-4 py-3">
                    <span className={`text-xs font-medium uppercase ${o.side === 'buy' ? 'text-emerald-400' : 'text-red-400'}`}>
                      {o.side}
                    </span>
                  </td>
                  <td className="px-4 py-3 text-right font-mono text-xs">
                    {o.filled_quantity}/{o.quantity}
                  </td>
                  <td className="px-4 py-3 text-right font-mono text-xs">
                    {o.avg_fill_price > 0 ? `$${o.avg_fill_price.toFixed(2)}` : '—'}
                  </td>
                  <td className="px-4 py-3">
                    <StatusBadge status={o.status} />
                  </td>
                  <td className="px-4 py-3 text-xs text-slate-400">{o.strategy ?? '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
          </div>
        )}
      </div>
    </div>
  )
}
