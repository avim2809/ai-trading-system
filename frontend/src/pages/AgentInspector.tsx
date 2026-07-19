import { useState } from 'react'
import { useMutation, useQuery } from '@tanstack/react-query'
import { api } from '../api/client'
import type { StepRequest, BlackboardView, Signal } from '../api/types'
import PipelineStage from '../components/PipelineStage'
import Spinner from '../components/Spinner'

function AIBadge() {
  return (
    <span className="inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-semibold bg-purple-900/50 text-purple-300 border border-purple-700/50 ml-2">
      AI
    </span>
  )
}

const REGIME_STYLES: Record<string, string> = {
  Bull: 'bg-emerald-900/50 text-emerald-300 border-emerald-700/50',
  Bear: 'bg-red-900/50 text-red-300 border-red-700/50',
  Chop: 'bg-amber-900/50 text-amber-300 border-amber-700/50',
}

function RegimeBadge({ label, confidence }: { label: string; confidence?: number }) {
  const style = REGIME_STYLES[label] ?? 'bg-slate-700/50 text-slate-300 border-slate-600/50'
  return (
    <span className={`inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-semibold border ml-2 ${style}`}>
      {label}
      {confidence != null && <span className="ml-1 font-mono opacity-80">{confidence.toFixed(2)}</span>}
    </span>
  )
}

/** Most common regime across regime_hmm signals, with average confidence. */
function marketRegime(view: BlackboardView): { label: string; confidence: number; n: number } | null {
  const regimeSigs = view.signal_sets
    .flatMap((ss) => ss.signals)
    .filter((s) => s.strategy === 'regime_hmm' && typeof s.meta?.regime === 'string')
  if (regimeSigs.length === 0) return null
  const counts: Record<string, { n: number; conf: number }> = {}
  for (const s of regimeSigs) {
    const label = s.meta.regime as string
    const bucket = counts[label] ?? (counts[label] = { n: 0, conf: 0 })
    bucket.n += 1
    bucket.conf += s.confidence
  }
  const [label, agg] = Object.entries(counts).sort((a, b) => b[1].n - a[1].n)[0]!
  return { label, confidence: agg.conf / agg.n, n: regimeSigs.length }
}

function AIReasoning({ rationale }: { rationale: string }) {
  const [expanded, setExpanded] = useState(false)
  return (
    <div className="mt-2">
      <button
        type="button"
        onClick={() => setExpanded(!expanded)}
        className="text-[10px] text-purple-400 hover:text-purple-300 flex items-center gap-1 transition-colors"
      >
        <svg className={`w-3 h-3 transition-transform ${expanded ? 'rotate-90' : ''}`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
        </svg>
        AI Reasoning
      </button>
      {expanded && (
        <div className="mt-1 ml-4 p-2 bg-purple-900/10 border border-purple-800/30 rounded text-xs text-slate-300 leading-relaxed">
          {rationale}
        </div>
      )}
    </div>
  )
}

function LLMUsageSummary({ usage }: { usage: { total_tokens?: number; total_cost?: number; total_calls?: number } }) {
  return (
    <div className="bg-purple-900/10 border border-purple-800/30 rounded-xl p-4 mt-4">
      <h4 className="text-xs font-semibold text-purple-300 mb-2 uppercase tracking-wider">AI Usage Summary</h4>
      <div className="flex gap-6 text-xs">
        {usage.total_tokens != null && (
          <span className="text-slate-400">
            Tokens: <span className="text-slate-200 font-mono">{usage.total_tokens.toLocaleString()}</span>
          </span>
        )}
        {usage.total_cost != null && (
          <span className="text-slate-400">
            Cost: <span className="text-emerald-400 font-mono">~${usage.total_cost.toFixed(4)}</span>
          </span>
        )}
        {usage.total_calls != null && (
          <span className="text-slate-400">
            Calls: <span className="text-slate-200 font-mono">{usage.total_calls}</span>
          </span>
        )}
      </div>
    </div>
  )
}

function SignalRow({ sig }: { sig: Signal }) {
  const isLLMEnhanced = sig.meta?.llm_enhanced === true
  const rationale = sig.meta?.llm_rationale as string | undefined
  const regime = typeof sig.meta?.regime === 'string' ? (sig.meta.regime as string) : undefined
  return (
    <>
      <tr className="text-slate-300">
        <td className="pr-4 py-0.5 font-mono">
          {sig.symbol}
          {isLLMEnhanced && <AIBadge />}
          {regime && <RegimeBadge label={regime} />}
        </td>
        <td className="pr-4 py-0.5">{sig.strategy}</td>
        <td className={`pr-4 py-0.5 text-right font-mono ${sig.score > 0 ? 'text-emerald-400' : sig.score < 0 ? 'text-red-400' : ''}`}>
          {sig.score.toFixed(3)}
        </td>
        <td className="pr-4 py-0.5 text-right font-mono">{sig.confidence.toFixed(2)}</td>
      </tr>
      {rationale && (
        <tr>
          <td colSpan={4} className="pb-2">
            <AIReasoning rationale={rationale} />
          </td>
        </tr>
      )}
    </>
  )
}

export default function AgentInspector() {
  const { data: strategies } = useQuery({
    queryKey: ['strategies'],
    queryFn: api.getStrategies,
  })

  const [selected, setSelected] = useState<string[]>([])
  const [symbols, setSymbols] = useState('AAPL,MSFT,GOOGL')
  const [asofDate, setAsofDate] = useState('2024-01-15')
  const [dataSource, setDataSource] = useState('synthetic')
  const [seed, setSeed] = useState(42)

  const step = useMutation({
    mutationFn: (req: StepRequest) => api.agentStep(req),
  })

  const handleRun = () => {
    if (selected.length === 0) return
    step.mutate({
      strategies: selected,
      strategy_params: {},
      symbols: symbols.split(',').map((s) => s.trim()).filter(Boolean),
      asof_date: asofDate,
      data_source: dataSource,
      seed,
    })
  }

  const result: BlackboardView | undefined = step.data

  return (
    <div>
      <h2 className="text-2xl font-bold text-white mb-6">Agent Inspector</h2>

      {/* Config Form */}
      <div className="bg-slate-800 rounded-xl border border-slate-700 p-5 mb-6">
        <h3 className="text-sm font-semibold text-slate-300 mb-4">Pipeline Configuration</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {/* Strategies */}
          <div className="col-span-full">
            <label className="block text-xs text-slate-400 mb-2">Strategies</label>
            <div className="flex flex-wrap gap-2">
              {strategies?.map((s) => (
                <label
                  key={s.name}
                  className={`flex items-center gap-2 px-3 py-1.5 rounded-lg border text-xs cursor-pointer transition-colors ${
                    selected.includes(s.name)
                      ? 'border-blue-500 bg-blue-500/10 text-blue-300'
                      : 'border-slate-600 bg-slate-700/50 text-slate-400 hover:text-slate-300'
                  }`}
                >
                  <input
                    type="checkbox"
                    checked={selected.includes(s.name)}
                    onChange={() =>
                      setSelected((prev) =>
                        prev.includes(s.name)
                          ? prev.filter((n) => n !== s.name)
                          : [...prev, s.name]
                      )
                    }
                    className="hidden"
                  />
                  {s.name}
                </label>
              ))}
            </div>
          </div>

          <div>
            <label className="block text-xs text-slate-400 mb-1">Symbols (comma-separated)</label>
            <input
              type="text"
              value={symbols}
              onChange={(e) => setSymbols(e.target.value)}
              className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-lg text-slate-200 text-sm focus:outline-none focus:ring-1 focus:ring-blue-500"
            />
          </div>

          <div>
            <label className="block text-xs text-slate-400 mb-1">As-of Date</label>
            <input
              type="date"
              value={asofDate}
              onChange={(e) => setAsofDate(e.target.value)}
              className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-lg text-slate-200 text-sm focus:outline-none focus:ring-1 focus:ring-blue-500"
            />
          </div>

          <div>
            <label className="block text-xs text-slate-400 mb-1">Data Source</label>
            <select
              value={dataSource}
              onChange={(e) => setDataSource(e.target.value)}
              className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-lg text-slate-200 text-sm focus:outline-none focus:ring-1 focus:ring-blue-500"
            >
              <option value="synthetic">Synthetic</option>
              <option value="cache">Cache</option>
            </select>
          </div>

          <div>
            <label className="block text-xs text-slate-400 mb-1">Seed</label>
            <input
              type="number"
              value={seed}
              onChange={(e) => setSeed(Number(e.target.value))}
              className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-lg text-slate-200 text-sm focus:outline-none focus:ring-1 focus:ring-blue-500"
            />
          </div>
        </div>

        <button
          onClick={handleRun}
          disabled={selected.length === 0 || step.isPending}
          className="mt-4 px-5 py-2.5 bg-blue-600 text-white rounded-lg text-sm font-medium hover:bg-blue-500 disabled:opacity-40 disabled:cursor-not-allowed transition-colors flex items-center gap-2"
        >
          {step.isPending && <Spinner className="h-4 w-4" />}
          Run Step
        </button>

        {step.error && (
          <div className="mt-3 bg-red-900/20 border border-red-700 rounded-lg p-3 text-red-400 text-sm">
            {(step.error as Error).message}
          </div>
        )}
      </div>

      {/* Pipeline Visualization */}
      {result && (
        <div className="space-y-4">
          <div className="flex items-center justify-between flex-wrap gap-2">
            <p className="text-xs text-slate-400">
              Pipeline snapshot as of <span className="text-slate-200 font-mono">{result.asof}</span>
            </p>
            {(() => {
              const mr = marketRegime(result)
              return mr ? (
                <span className="text-xs text-slate-400 flex items-center">
                  Market regime:
                  <RegimeBadge label={mr.label} confidence={mr.confidence} />
                  <span className="ml-2 text-slate-500">({mr.n} symbols)</span>
                </span>
              ) : null
            })()}
          </div>

          {/* Analysts */}
          <PipelineStage title="Analysts" status="complete">
            {result.signal_sets.length === 0 ? (
              <p className="text-xs text-slate-500">No signals generated</p>
            ) : (
              result.signal_sets.map((ss) => (
                <div key={ss.domain} className="mb-4 last:mb-0">
                  <p className="text-xs font-medium text-slate-400 mb-2 uppercase">{ss.domain}</p>
                  <table className="w-full text-xs">
                    <thead>
                      <tr className="text-left text-slate-500">
                        <th className="pr-4 pb-1">Symbol</th>
                        <th className="pr-4 pb-1">Strategy</th>
                        <th className="pr-4 pb-1 text-right">Score</th>
                        <th className="pr-4 pb-1 text-right">Confidence</th>
                      </tr>
                    </thead>
                    <tbody>
                      {ss.signals.map((sig, i) => (
                        <SignalRow key={i} sig={sig} />
                      ))}
                    </tbody>
                  </table>
                </div>
              ))
            )}
          </PipelineStage>

          {/* Bull Researcher */}
          <PipelineStage title="Bull Researcher" status="complete">
            {result.theses.filter((t) => t.side === 'bull').length === 0 ? (
              <p className="text-xs text-slate-500">No bull theses</p>
            ) : (
              <div className="space-y-2">
                {result.theses.filter((t) => t.side === 'bull').map((t, i) => (
                  <div key={i} className="bg-emerald-900/10 border border-emerald-800/30 rounded-lg p-3">
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-xs font-mono text-emerald-400">{t.symbol}</span>
                      <span className="text-xs text-slate-400">
                        Conviction: <span className="text-emerald-400 font-mono">{t.conviction.toFixed(2)}</span>
                      </span>
                    </div>
                    <p className="text-xs text-slate-300">{t.rationale}</p>
                  </div>
                ))}
              </div>
            )}
          </PipelineStage>

          {/* Bear Researcher */}
          <PipelineStage title="Bear Researcher" status="complete">
            {result.theses.filter((t) => t.side === 'bear').length === 0 ? (
              <p className="text-xs text-slate-500">No bear theses</p>
            ) : (
              <div className="space-y-2">
                {result.theses.filter((t) => t.side === 'bear').map((t, i) => (
                  <div key={i} className="bg-red-900/10 border border-red-800/30 rounded-lg p-3">
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-xs font-mono text-red-400">{t.symbol}</span>
                      <span className="text-xs text-slate-400">
                        Conviction: <span className="text-red-400 font-mono">{t.conviction.toFixed(2)}</span>
                      </span>
                    </div>
                    <p className="text-xs text-slate-300">{t.rationale}</p>
                  </div>
                ))}
              </div>
            )}
          </PipelineStage>

          {/* Debate */}
          <PipelineStage title="Debate" status="complete">
            {result.debate_results.length === 0 ? (
              <p className="text-xs text-slate-500">No debate results</p>
            ) : (
              <table className="w-full text-xs">
                <thead>
                  <tr className="text-left text-slate-500">
                    <th className="pr-4 pb-1">Symbol</th>
                    <th className="pr-4 pb-1 text-right">Net Conviction</th>
                    <th className="pr-4 pb-1 text-right">Bull</th>
                    <th className="pr-4 pb-1 text-right">Bear</th>
                  </tr>
                </thead>
                <tbody>
                  {result.debate_results.map((d, i) => (
                    <tr key={i} className="text-slate-300">
                      <td className="pr-4 py-0.5 font-mono">{d.symbol}</td>
                      <td className={`pr-4 py-0.5 text-right font-mono ${d.net_conviction > 0 ? 'text-emerald-400' : d.net_conviction < 0 ? 'text-red-400' : ''}`}>
                        {d.net_conviction.toFixed(2)}
                      </td>
                      <td className="pr-4 py-0.5 text-right font-mono text-emerald-400">
                        {d.bull_thesis?.conviction.toFixed(2) ?? '—'}
                      </td>
                      <td className="pr-4 py-0.5 text-right font-mono text-red-400">
                        {d.bear_thesis?.conviction.toFixed(2) ?? '—'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </PipelineStage>

          {/* Portfolio Manager */}
          <PipelineStage title="Portfolio Manager" status={result.proposal ? 'complete' : undefined}>
            {result.proposal ? (
              <div>
                <table className="w-full text-xs mb-2">
                  <thead>
                    <tr className="text-left text-slate-500">
                      <th className="pr-4 pb-1">Symbol</th>
                      <th className="pr-4 pb-1 text-right">Target Weight</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(result.proposal.targets).map(([sym, wt]) => (
                      <tr key={sym} className="text-slate-300">
                        <td className="pr-4 py-0.5 font-mono">{sym}</td>
                        <td className="pr-4 py-0.5 text-right font-mono">{(wt * 100).toFixed(1)}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                {result.proposal.notes && (
                  <p className="text-xs text-slate-400 italic">{result.proposal.notes}</p>
                )}
              </div>
            ) : (
              <p className="text-xs text-slate-500">No proposal generated</p>
            )}
          </PipelineStage>

          {/* Risk Manager */}
          <PipelineStage title="Risk Manager" status={result.risk_decision ? 'complete' : undefined}>
            {result.risk_decision ? (
              <div>
                <div className="flex items-center gap-2 mb-3">
                  <span className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-medium ${
                    result.risk_decision.approved
                      ? 'bg-emerald-900/50 text-emerald-400'
                      : 'bg-red-900/50 text-red-400'
                  }`}>
                    {result.risk_decision.approved ? 'APPROVED' : 'REJECTED'}
                  </span>
                </div>
                {result.risk_decision.violations.length > 0 && (
                  <div className="mb-3">
                    <p className="text-xs text-slate-400 mb-1">Violations:</p>
                    <ul className="list-disc list-inside text-xs text-red-400 space-y-0.5">
                      {result.risk_decision.violations.map((v, i) => (
                        <li key={i}>{v}</li>
                      ))}
                    </ul>
                  </div>
                )}
                {result.risk_decision.actions.length > 0 && (
                  <div className="mb-3">
                    <p className="text-xs text-slate-400 mb-1">Actions taken:</p>
                    <ul className="list-disc list-inside text-xs text-amber-400 space-y-0.5">
                      {result.risk_decision.actions.map((a, i) => (
                        <li key={i}>{a}</li>
                      ))}
                    </ul>
                  </div>
                )}
                {Object.keys(result.risk_decision.adjusted_targets).length > 0 && (
                  <table className="w-full text-xs">
                    <thead>
                      <tr className="text-left text-slate-500">
                        <th className="pr-4 pb-1">Symbol</th>
                        <th className="pr-4 pb-1 text-right">Adjusted Weight</th>
                      </tr>
                    </thead>
                    <tbody>
                      {Object.entries(result.risk_decision.adjusted_targets).map(([sym, wt]) => (
                        <tr key={sym} className="text-slate-300">
                          <td className="pr-4 py-0.5 font-mono">{sym}</td>
                          <td className="pr-4 py-0.5 text-right font-mono">{(wt * 100).toFixed(1)}%</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                )}
              </div>
            ) : (
              <p className="text-xs text-slate-500">No risk decision</p>
            )}
          </PipelineStage>

          {/* Execution */}
          <PipelineStage title="Execution" status={result.execution_report ? 'complete' : undefined}>
            {result.execution_report ? (
              <div>
                <div className="flex gap-6 mb-3 text-xs">
                  <span className="text-slate-400">
                    Turnover: <span className="text-slate-200 font-mono">{(result.execution_report.turnover * 100).toFixed(2)}%</span>
                  </span>
                  <span className="text-slate-400">
                    Costs: <span className="text-slate-200 font-mono">${result.execution_report.costs.toFixed(2)}</span>
                  </span>
                </div>
                {result.execution_report.fills.length > 0 ? (
                  <table className="w-full text-xs">
                    <thead>
                      <tr className="text-left text-slate-500">
                        <th className="pr-4 pb-1">Symbol</th>
                        <th className="pr-4 pb-1">Side</th>
                        <th className="pr-4 pb-1 text-right">Quantity</th>
                        <th className="pr-4 pb-1">Strategy</th>
                      </tr>
                    </thead>
                    <tbody>
                      {result.execution_report.fills.map((f, i) => (
                        <tr key={i} className="text-slate-300">
                          <td className="pr-4 py-0.5 font-mono">{f.symbol}</td>
                          <td className={`pr-4 py-0.5 ${f.side === 'buy' ? 'text-emerald-400' : 'text-red-400'}`}>
                            {f.side}
                          </td>
                          <td className="pr-4 py-0.5 text-right font-mono">{f.quantity}</td>
                          <td className="pr-4 py-0.5">{f.strategy}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                ) : (
                  <p className="text-xs text-slate-500">No fills</p>
                )}
              </div>
            ) : (
              <p className="text-xs text-slate-500">No execution report</p>
            )}
          </PipelineStage>

          {/* LLM Usage Summary */}
          {(result as BlackboardView & { llm_usage?: { total_tokens?: number; total_cost?: number; total_calls?: number } }).llm_usage && (
            <LLMUsageSummary usage={(result as BlackboardView & { llm_usage: { total_tokens?: number; total_cost?: number; total_calls?: number } }).llm_usage} />
          )}
        </div>
      )}
    </div>
  )
}
