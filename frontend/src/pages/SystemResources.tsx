import { useState } from 'react'
import { useQuery, useMutation } from '@tanstack/react-query'
import { api } from '../api/client'
import type { SystemResources } from '../api/types'
import Spinner from '../components/Spinner'

function formatBytes(bytes: number): string {
  if (!bytes) return '0 GB'
  const gb = bytes / 1_000_000_000
  return `${gb.toFixed(1)} GB`
}

function barColor(percent: number): string {
  if (percent >= 90) return 'bg-red-500'
  if (percent >= 75) return 'bg-amber-500'
  return 'bg-emerald-500'
}

function UsageBar({
  label,
  percent,
  detail,
}: {
  label: string
  percent: number
  detail: string
}) {
  const clamped = Math.min(100, Math.max(0, percent))
  return (
    <div className="bg-slate-800 rounded-xl border border-slate-700 px-5 py-4">
      <div className="flex items-center justify-between mb-2">
        <p className="text-xs font-medium text-slate-400 uppercase tracking-wider">{label}</p>
        <p className="text-sm font-semibold text-white">{percent.toFixed(1)}%</p>
      </div>
      <div className="h-2.5 rounded-full bg-slate-700 overflow-hidden">
        <div
          className={`h-full rounded-full transition-all ${barColor(clamped)}`}
          style={{ width: `${clamped}%` }}
        />
      </div>
      <p className="mt-2 text-xs text-slate-400">{detail}</p>
    </div>
  )
}

type ActionState = 'idle' | 'confirming' | 'running'

function ServiceActionButton({
  label,
  runningLabel,
  confirmMessage,
  variant,
  onConfirm,
  isPending,
  error,
}: {
  label: string
  runningLabel: string
  confirmMessage: string
  variant: 'default' | 'danger'
  onConfirm: () => void
  isPending: boolean
  error: unknown
}) {
  const [state, setState] = useState<ActionState>('idle')

  const handleConfirm = () => {
    setState('running')
    onConfirm()
  }

  if (state === 'running' || isPending) {
    return (
      <div className="flex items-center gap-2 px-4 py-2 text-sm font-medium text-slate-300">
        <Spinner className="h-3.5 w-3.5" />
        {runningLabel}
      </div>
    )
  }

  if (state === 'confirming') {
    return (
      <div className="flex items-center gap-2">
        <span className="text-xs text-amber-400">{confirmMessage}</span>
        <button
          onClick={handleConfirm}
          className={`px-3 py-2 text-sm font-medium rounded-lg text-white transition-colors ${
            variant === 'danger' ? 'bg-red-600 hover:bg-red-500' : 'bg-amber-600 hover:bg-amber-500'
          }`}
        >
          Yes, {label}
        </button>
        <button
          onClick={() => setState('idle')}
          className="px-3 py-2 text-sm text-slate-400 hover:text-slate-200 transition-colors"
        >
          Cancel
        </button>
      </div>
    )
  }

  return (
    <div>
      <button
        onClick={() => setState('confirming')}
        className={`px-4 py-2 text-sm font-medium rounded-lg border transition-colors ${
          variant === 'danger'
            ? 'border-red-700 text-red-400 hover:bg-red-900/20'
            : 'border-slate-600 text-slate-300 hover:bg-slate-700'
        }`}
      >
        {label}
      </button>
      {error != null && (
        <p className="mt-2 text-xs text-red-400">{(error as Error).message}</p>
      )}
    </div>
  )
}

export default function SystemResourcesPage() {
  const { data, isLoading, error } = useQuery<SystemResources>({
    queryKey: ['system-resources'],
    queryFn: api.getSystemResources,
    refetchInterval: 5000,
  })

  const restartMut = useMutation({
    mutationFn: () => api.restartService(),
  })

  const killMut = useMutation({
    mutationFn: () => api.killService(),
  })

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
        <h3 className="font-semibold mb-1">Failed to load server resources</h3>
        <p className="text-sm">{(error as Error).message}</p>
      </div>
    )
  }

  return (
    <div>
      <div className="flex items-center justify-between flex-wrap gap-3 mb-6">
        <div>
          <h2 className="text-2xl font-bold text-white">Server Resources</h2>
          <p className="text-sm text-slate-400 mt-1">
            Host CPU, memory and disk utilization for the ai-trading service
          </p>
        </div>
      </div>

      {data && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
          <UsageBar
            label="CPU"
            percent={data.cpu.percent}
            detail={`${data.cpu.count} core${data.cpu.count !== 1 ? 's' : ''}`}
          />
          <UsageBar
            label="Memory"
            percent={data.memory.percent}
            detail={`${formatBytes(data.memory.used)} / ${formatBytes(data.memory.total)}`}
          />
          <UsageBar
            label="Disk"
            percent={data.disk.percent}
            detail={`${formatBytes(data.disk.used)} / ${formatBytes(data.disk.total)} (${data.disk.path})`}
          />
        </div>
      )}

      <div className="bg-slate-800 rounded-xl border border-slate-700 p-5">
        <h3 className="text-sm font-semibold text-slate-300 mb-1">Service Control</h3>
        <p className="text-xs text-slate-400 mb-4">
          Restarting or killing ai-trading.service will briefly drop this page's connection
          to the backend. The service is configured to auto-restart on kill.
        </p>
        <div className="flex flex-wrap gap-4">
          <ServiceActionButton
            label="Restart Service"
            runningLabel="Restarting..."
            confirmMessage="Restart the trading service now?"
            variant="default"
            onConfirm={() => restartMut.mutate()}
            isPending={restartMut.isPending}
            error={restartMut.error}
          />
          <ServiceActionButton
            label="Kill Service"
            runningLabel="Killing..."
            confirmMessage="Force-kill the trading service now?"
            variant="danger"
            onConfirm={() => killMut.mutate()}
            isPending={killMut.isPending}
            error={killMut.error}
          />
        </div>
      </div>
    </div>
  )
}
