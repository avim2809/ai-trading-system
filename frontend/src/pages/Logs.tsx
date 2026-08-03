import { useCallback, useEffect, useRef, useState } from 'react'
import { api } from '../api/client'
import type { LogEntry } from '../api/types'
import { formatClockWithMs } from '../lib/time'

const POLL_MS = 2000
const MAX_LINES = 2000

const LEVELS = ['ALL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'RAW'] as const
type LevelFilter = (typeof LEVELS)[number]

const levelStyles: Record<string, string> = {
  ERROR: 'text-red-400',
  CRITICAL: 'text-red-400',
  WARNING: 'text-amber-400',
  INFO: 'text-slate-300',
  DEBUG: 'text-slate-500',
  RAW: 'text-slate-500 italic',
}

function formatTs(ts: string | null): string {
  if (!ts) return '—'
  return formatClockWithMs(ts)
}

export default function Logs() {
  const [lines, setLines] = useState<LogEntry[]>([])
  const [levelFilter, setLevelFilter] = useState<LevelFilter>('ALL')
  const [search, setSearch] = useState('')
  const [paused, setPaused] = useState(false)
  const [autoScroll, setAutoScroll] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [connected, setConnected] = useState(false)

  const offsetRef = useRef(0)
  const pausedRef = useRef(paused)
  const scrollRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    pausedRef.current = paused
  }, [paused])

  const poll = useCallback(async () => {
    if (pausedRef.current) return
    try {
      const res = await api.tailLogs(offsetRef.current)
      offsetRef.current = res.next_offset
      setConnected(true)
      setError(null)
      if (res.lines.length === 0) return
      setLines((prev) => {
        const merged = res.reset ? res.lines : [...prev, ...res.lines]
        return merged.length > MAX_LINES ? merged.slice(merged.length - MAX_LINES) : merged
      })
    } catch (err) {
      setConnected(false)
      setError((err as Error).message)
    }
  }, [])

  useEffect(() => {
    poll()
    const id = setInterval(poll, POLL_MS)
    return () => clearInterval(id)
  }, [poll])

  useEffect(() => {
    if (autoScroll && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [lines, autoScroll])

  function handleScroll() {
    const el = scrollRef.current
    if (!el) return
    const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 40
    setAutoScroll(atBottom)
  }

  const filtered = lines.filter((l) => {
    if (levelFilter !== 'ALL' && l.level !== levelFilter) return false
    if (search && !`${l.logger} ${l.msg} ${l.file ?? ''} ${l.function ?? ''}`.toLowerCase().includes(search.toLowerCase())) return false
    return true
  })

  return (
    <div className="flex flex-col h-[calc(100vh-5.25rem)] md:h-[calc(100vh-3rem)]">
      <div className="flex items-center justify-between flex-wrap gap-3 mb-4 flex-shrink-0">
        <div>
          <h2 className="text-2xl font-bold text-white flex items-center gap-3">
            Live Logs
            <span className={`flex items-center gap-1.5 text-sm font-normal ${connected ? 'text-emerald-400' : 'text-slate-500'}`}>
              <span className={`w-2 h-2 rounded-full ${connected ? 'bg-emerald-400 animate-pulse' : 'bg-slate-600'}`} />
              {connected ? (paused ? 'Paused' : 'Live') : 'Connecting…'}
            </span>
          </h2>
          <p className="text-sm text-slate-400 mt-1">
            Tails the API process log — covers backtests, live cycles, and RAG/LLM activity in real time.
          </p>
        </div>
        <div className="flex gap-3">
          <button
            onClick={() => setPaused((p) => !p)}
            className="px-4 py-2 text-sm font-medium rounded-lg border border-slate-600 text-slate-300 hover:bg-slate-700 transition-colors"
          >
            {paused ? 'Resume' : 'Pause'}
          </button>
          <button
            onClick={() => setLines([])}
            className="px-4 py-2 text-sm font-medium rounded-lg border border-slate-600 text-slate-300 hover:bg-slate-700 transition-colors"
          >
            Clear
          </button>
        </div>
      </div>

      <div className="flex items-center flex-wrap gap-3 mb-3 flex-shrink-0">
        <select
          value={levelFilter}
          onChange={(e) => setLevelFilter(e.target.value as LevelFilter)}
          className="px-3 py-1.5 bg-slate-800 border border-slate-700 rounded-lg text-slate-200 text-xs focus:outline-none focus:ring-1 focus:ring-blue-500"
        >
          {LEVELS.map((lvl) => (
            <option key={lvl} value={lvl}>{lvl}</option>
          ))}
        </select>
        <input
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="Filter by logger, file, function, or message…"
          className="flex-1 min-w-[140px] px-3 py-1.5 bg-slate-800 border border-slate-700 rounded-lg text-slate-200 text-xs placeholder:text-slate-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
        />
        <span className="text-xs text-slate-500 whitespace-nowrap">
          {filtered.length} / {lines.length} lines
        </span>
      </div>

      {error && (
        <div className="mb-3 bg-red-900/20 border border-red-700 rounded-lg p-3 text-red-400 text-xs flex-shrink-0">
          Failed to fetch logs: {error}
        </div>
      )}

      <div className="relative flex-1 min-h-0">
        <div
          ref={scrollRef}
          onScroll={handleScroll}
          className="h-full overflow-auto bg-slate-950 border border-slate-700 rounded-xl p-3 font-mono text-xs"
        >
          {filtered.length === 0 ? (
            <div className="flex items-center justify-center h-full text-slate-500">
              {lines.length === 0 ? 'Waiting for log output…' : 'No lines match the current filter.'}
            </div>
          ) : (
            filtered.map((l, i) => (
              <div key={i} className="flex gap-2 py-0.5 hover:bg-slate-900/60 whitespace-pre-wrap break-words">
                <span className="text-slate-600 flex-shrink-0">{formatTs(l.ts)}</span>
                <span className={`flex-shrink-0 w-16 ${levelStyles[l.level] ?? 'text-slate-400'}`}>{l.level}</span>
                <span className="text-slate-500 flex-shrink-0">{l.logger}</span>
                {l.file && (
                  <span className="text-slate-700 flex-shrink-0">
                    {l.file}:{l.function}:{l.line}
                  </span>
                )}
                <span className={levelStyles[l.level] ?? 'text-slate-300'}>{l.msg}</span>
                {l.exception && (
                  <pre className="w-full text-red-400/80 mt-0.5">{l.exception}</pre>
                )}
              </div>
            ))
          )}
        </div>
        {!autoScroll && (
          <button
            onClick={() => setAutoScroll(true)}
            className="absolute bottom-4 right-6 px-3 py-1.5 text-xs font-medium rounded-full bg-blue-600 text-white hover:bg-blue-500 shadow-lg transition-colors"
          >
            Jump to latest ↓
          </button>
        )}
      </div>
    </div>
  )
}
