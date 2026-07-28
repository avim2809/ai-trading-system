import { useState, useEffect } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { api } from '../api/client'
import type { LiveConfig, StrategyInfo, LLMConfig, LLMProvider, LLMCacheStats, RAGStats, EmbeddingModelInfo, StrategyCircuitBreakerConfig, StrategyRegimeWeightsConfig } from '../api/types'
import Spinner from '../components/Spinner'

const AGENT_DISPLAY_NAMES: Record<string, string> = {
  technical_analyst: 'Technical Analyst',
  fundamental_analyst: 'Fundamental Analyst',
  sentiment_analyst: 'Sentiment Analyst',
  bull_researcher: 'Bull Researcher',
  bear_researcher: 'Bear Researcher',
  debate: 'Debate / Synthesis',
  trader: 'Portfolio Manager',
  risk: 'Risk Manager',
}

const AGENT_MODE_OPTIONS: { value: 'quant' | 'llm_enhanced' | 'llm_only'; label: string }[] = [
  { value: 'quant', label: 'Quant' },
  { value: 'llm_enhanced', label: 'AI-Enhanced' },
  { value: 'llm_only', label: 'AI-Only' },
]

const BACKTEST_POLICIES = [
  { value: 'disabled', label: 'Disabled' },
  { value: 'cache_only', label: 'Cache Only' },
  { value: 'live_calls', label: 'Live Calls' },
]

const RAG_DOC_TYPES = [
  { value: 'sec_filings', label: 'SEC Filings' },
  { value: 'earnings_transcripts', label: 'Earnings Transcripts' },
  { value: 'news', label: 'News' },
  { value: 'research_papers', label: 'Research Papers' },
  { value: 'system_docs', label: 'System Docs' },
]

const BROKERS = [
  { value: 'alpaca_paper', label: 'Alpaca Paper' },
  { value: 'alpaca_live', label: 'Alpaca Live' },
  { value: 'ibkr_paper', label: 'Interactive Brokers (Paper)' },
  { value: 'ibkr_live', label: 'Interactive Brokers (Live)' },
]

const SCHEDULES = [
  { value: 'market_open', label: 'Market Open (9:30 ET)' },
  { value: 'market_close', label: 'Market Close (4:00 ET)' },
  { value: 'every_5_min', label: 'Every 5 Minutes' },
  { value: 'every_15_min', label: 'Every 15 Minutes' },
  { value: 'every_30_min', label: 'Every 30 Minutes' },
  { value: 'hourly', label: 'Hourly' },
  { value: 'custom', label: 'Custom Cron' },
]

const inputCls =
  'w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-lg text-slate-200 text-sm focus:outline-none focus:ring-1 focus:ring-blue-500'

export default function LiveConfig() {
  const qc = useQueryClient()

  const { data: config, isLoading, error } = useQuery<LiveConfig>({
    queryKey: ['live-config'],
    queryFn: api.getLiveConfig,
  })

  const { data: availableStrategies } = useQuery<StrategyInfo[]>({
    queryKey: ['strategies'],
    queryFn: api.getStrategies,
  })

  const [infoOpen, setInfoOpen] = useState<string | null>(null)

  const [broker, setBroker] = useState('')
  const [schedule, setSchedule] = useState('')
  const [customCron, setCustomCron] = useState('')
  const [enabledStrategies, setEnabledStrategies] = useState<string[]>([])
  const [autoApprove, setAutoApprove] = useState<Set<string>>(new Set())
  const [killSwitchDrawdown, setKillSwitchDrawdown] = useState('0.10')
  const [maxDailyTrades, setMaxDailyTrades] = useState('50')
  const [maxDailyTurnover, setMaxDailyTurnover] = useState('0.5')
  const [symbols, setSymbols] = useState('')
  const [strategyParams, setStrategyParams] = useState<Record<string, Record<string, unknown>>>({})

  // Behavioural knobs
  const [newsGuardEnabled, setNewsGuardEnabled] = useState(false)
  const [newsGuardBefore, setNewsGuardBefore] = useState('30')
  const [newsGuardAfter, setNewsGuardAfter] = useState('15')
  const [newsGuardOffline, setNewsGuardOffline] = useState(false)
  const [allocationMethod, setAllocationMethod] = useState('conviction_weighted')
  const [kellyFraction, setKellyFraction] = useState('0.5')
  const [signalCombination, setSignalCombination] = useState('confidence')
  const [circuitBreaker, setCircuitBreaker] = useState<StrategyCircuitBreakerConfig>({
    enabled: false,
    lookback_days: 60,
    min_track_record_days: 20,
    trigger_sharpe: -0.5,
    full_cutoff_sharpe: -1.5,
    damping_floor: 0.25,
  })
  const [regimeWeights, setRegimeWeights] = useState<StrategyRegimeWeightsConfig>({
    enabled: false,
    benchmark_symbol: 'SPY',
    lookback_days: 252,
    retrain_frequency: 21,
    weights: { Bull: {}, Bear: {}, Chop: {} },
  })
  const [regimeWeightsJson, setRegimeWeightsJson] = useState('')

  // AI / LLM state
  const { data: llmProviders } = useQuery<LLMProvider[]>({
    queryKey: ['llm-providers'],
    queryFn: api.getLLMProviders,
  })
  const { data: llmConfig } = useQuery<LLMConfig>({
    queryKey: ['llm-config'],
    queryFn: api.getLLMConfig,
  })
  const { data: cacheStats, refetch: refetchCache } = useQuery<LLMCacheStats>({
    queryKey: ['llm-cache-stats'],
    queryFn: api.getLLMCacheStats,
  })
  const { data: ragStats } = useQuery<RAGStats>({
    queryKey: ['rag-stats'],
    queryFn: api.getRAGStats,
  })
  const { data: embeddingModels } = useQuery<EmbeddingModelInfo[]>({
    queryKey: ['embedding-models'],
    queryFn: api.getEmbeddingModels,
  })

  const [activeProvider, setActiveProvider] = useState('')
  const [activeModel, setActiveModel] = useState('')
  const [temperature, setTemperature] = useState(0.7)
  const [agentModes, setAgentModes] = useState<Record<string, 'quant' | 'llm_enhanced' | 'llm_only'>>({})
  const [cacheEnabled, setCacheEnabled] = useState(true)
  const [compressionEnabled, setCompressionEnabled] = useState(false)
  const [compressionRatio, setCompressionRatio] = useState(0.5)
  const [backtestPolicy, setBacktestPolicy] = useState('disabled')
  const [ragIngestTypes, setRagIngestTypes] = useState<string[]>([])
  const [ragSymbols, setRagSymbols] = useState('')
  const [ragExpanded, setRagExpanded] = useState(false)
  const [testResult, setTestResult] = useState<{ status: 'ok' | 'error'; message: string; time?: number } | null>(null)
  const [testLoading, setTestLoading] = useState(false)
  const [ingestionMsg, setIngestionMsg] = useState('')
  const [selectedEmbeddingModel, setSelectedEmbeddingModel] = useState('')
  const [embeddingReindexWarning, setEmbeddingReindexWarning] = useState(false)
  const [embeddingSaving, setEmbeddingSaving] = useState(false)

  useEffect(() => {
    if (!llmConfig) return
    const model = llmConfig.provider.default_model
    const resolved =
      (llmConfig.provider as LLMConfig['provider'] & { resolved_provider?: string })
        .resolved_provider
      ?? llmProviders?.find(
        (p) =>
          p.models.includes(model)
          || p.default_model === model
          || model.startsWith(`${p.name}/`),
      )?.name
      ?? (model.startsWith('gemini/') ? 'gemini' : model.split('/')[0] || '')
    setActiveProvider(resolved)
    setActiveModel(model)
    setTemperature(llmConfig.provider.temperature)
    setAgentModes(llmConfig.agent_modes)
    setCacheEnabled(llmConfig.optimization.cache_enabled)
    setCompressionEnabled(llmConfig.optimization.compression_enabled)
    setCompressionRatio(llmConfig.optimization.compression_ratio)
    setBacktestPolicy(llmConfig.backtest_policy)
    if (llmConfig.rag?.embedding_model) {
      setSelectedEmbeddingModel(llmConfig.rag.embedding_model)
    }
  }, [llmConfig, llmProviders])

  useEffect(() => {
    if (!config) return
    setBroker(config.broker ?? 'alpaca_paper')
    const sched = config.schedule ?? 'market_open'
    if (SCHEDULES.some((s) => s.value === sched)) {
      setSchedule(sched)
    } else {
      setSchedule('custom')
      setCustomCron(sched)
    }
    setEnabledStrategies(config.strategies?.enabled ?? [])
    setAutoApprove(new Set(config.strategies?.auto_approve ?? []))
    setKillSwitchDrawdown(String(config.risk?.kill_switch_drawdown ?? 0.1))
    setMaxDailyTrades(String(config.risk?.max_daily_trades ?? 50))
    setMaxDailyTurnover(String(config.risk?.max_daily_turnover ?? 0.5))
    setSymbols((config.universe?.symbols ?? []).join(', '))
    setStrategyParams(config.strategy_params ?? {})
    setNewsGuardEnabled(config.news_guard?.enabled ?? false)
    setNewsGuardBefore(String(config.news_guard?.before_min ?? 30))
    setNewsGuardAfter(String(config.news_guard?.after_min ?? 15))
    setNewsGuardOffline(config.news_guard?.offline ?? false)
    setAllocationMethod(config.allocation_method ?? 'conviction_weighted')
    setKellyFraction(String(config.kelly_fraction ?? 0.5))
    setSignalCombination(config.signal_combination?.method ?? 'confidence')
    if (config.strategy_circuit_breaker) {
      setCircuitBreaker((prev) => ({ ...prev, ...config.strategy_circuit_breaker }))
    }
    if (config.strategy_regime_weights) {
      setRegimeWeights((prev) => ({ ...prev, ...config.strategy_regime_weights }))
      setRegimeWeightsJson(JSON.stringify(config.strategy_regime_weights.weights ?? {}, null, 2))
    }
  }, [config])

  const saveMut = useMutation({
    mutationFn: (c: LiveConfig) => api.updateLiveConfig(c),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['live-config'] }),
  })

  const saveLLMMut = useMutation({
    mutationFn: (c: Partial<LLMConfig>) => api.updateLLMConfig(c),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['llm-config'] }),
  })

  const clearCacheMut = useMutation({
    mutationFn: () => api.clearLLMCache(),
    onSuccess: () => refetchCache(),
  })

  const deleteCollectionMut = useMutation({
    mutationFn: (collection: string) => api.deleteRAGCollection(collection),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['rag-stats'] }),
  })

  const handleSave = () => {
    const requireApproval = enabledStrategies.filter((s) => !autoApprove.has(s))
    const payload: LiveConfig = {
      broker,
      schedule: schedule === 'custom' ? customCron : schedule,
      approval_mode: autoApprove.size === enabledStrategies.length ? 'full_auto' : 'semi_auto',
      strategies: {
        enabled: enabledStrategies,
        auto_approve: enabledStrategies.filter((s) => autoApprove.has(s)),
        require_approval: requireApproval,
      },
      risk: {
        kill_switch_drawdown: parseFloat(killSwitchDrawdown) || 0.1,
        max_daily_trades: parseInt(maxDailyTrades) || 50,
        max_daily_turnover: parseFloat(maxDailyTurnover) || 0.5,
      },
      universe: {
        symbols: symbols.split(',').map((s) => s.trim()).filter(Boolean),
      },
      strategy_params: strategyParams,
      news_guard: {
        enabled: newsGuardEnabled,
        before_min: parseInt(newsGuardBefore) || 30,
        after_min: parseInt(newsGuardAfter) || 15,
        offline: newsGuardOffline,
      },
      signal_combination: { method: signalCombination },
      strategy_circuit_breaker: circuitBreaker,
      strategy_regime_weights: (() => {
        if (!regimeWeights.enabled) return { ...regimeWeights, enabled: false }
        let weights = regimeWeights.weights
        if (regimeWeightsJson.trim()) {
          weights = JSON.parse(regimeWeightsJson) as Record<string, Record<string, number>>
        }
        return { ...regimeWeights, weights }
      })(),
      allocation_method: allocationMethod,
      kelly_fraction: parseFloat(kellyFraction) || 0.5,
    }
    saveMut.mutate(payload)

    saveLLMMut.mutate({
      provider: { default_model: activeModel, temperature, max_tokens: llmConfig?.provider.max_tokens ?? 4096 },
      agent_modes: agentModes,
      optimization: { compression_enabled: compressionEnabled, compression_ratio: compressionRatio, cache_enabled: cacheEnabled },
      backtest_policy: backtestPolicy,
    })
  }

  const handleTestConnection = async () => {
    setTestLoading(true)
    setTestResult(null)
    try {
      const res = await api.testLLMConnection(activeModel)
      setTestResult({ status: 'ok', message: `${res.model} responded`, time: res.response_time_ms })
    } catch (err) {
      setTestResult({ status: 'error', message: (err as Error).message })
    } finally {
      setTestLoading(false)
    }
  }

  const handleIngest = async () => {
    if (ragIngestTypes.length === 0) return
    setIngestionMsg('')
    try {
      const syms = ragSymbols.split(',').map((s) => s.trim()).filter(Boolean)
      const res = await api.ingestRAGDocs(ragIngestTypes, syms.length > 0 ? syms : undefined)
      setIngestionMsg(res.message || 'Ingestion started')
    } catch (err) {
      setIngestionMsg(`Error: ${(err as Error).message}`)
    }
  }

  const handleEmbeddingModelChange = async (modelId: string) => {
    setSelectedEmbeddingModel(modelId)
    setEmbeddingReindexWarning(false)
    setEmbeddingSaving(true)
    try {
      const res = await api.setEmbeddingModel(modelId)
      if (res.requires_reindex) {
        setEmbeddingReindexWarning(true)
      }
      qc.invalidateQueries({ queryKey: ['llm-config'] })
    } catch {
      setSelectedEmbeddingModel(llmConfig?.rag?.embedding_model ?? 'all-MiniLM-L6-v2')
    } finally {
      setEmbeddingSaving(false)
    }
  }

  const toggleAutoApprove = (strategy: string) => {
    setAutoApprove((prev) => {
      const next = new Set(prev)
      if (next.has(strategy)) next.delete(strategy)
      else next.add(strategy)
      return next
    })
  }

  const addStrategy = (name: string) => {
    if (!name.trim() || enabledStrategies.includes(name.trim())) return
    setEnabledStrategies((prev) => [...prev, name.trim()])
  }

  const removeStrategy = (name: string) => {
    setEnabledStrategies((prev) => prev.filter((s) => s !== name))
    setAutoApprove((prev) => {
      const next = new Set(prev)
      next.delete(name)
      return next
    })
  }

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
        <h3 className="font-semibold mb-1">Failed to load config</h3>
        <p className="text-sm">{(error as Error).message}</p>
      </div>
    )
  }

  return (
    <div>
      <div className="mb-6">
        <h2 className="text-2xl font-bold text-white">Live Configuration</h2>
        <p className="text-sm text-slate-400 mt-1">Configure broker, schedule, strategies, and risk parameters</p>
      </div>

      <div className="space-y-6">
        {/* Broker */}
        <div className="bg-slate-800 rounded-xl border border-slate-700 p-5">
          <h3 className="text-sm font-semibold text-slate-300 mb-4">Broker</h3>
          <select
            value={broker}
            onChange={(e) => setBroker(e.target.value)}
            className={inputCls + ' max-w-xs'}
          >
            {BROKERS.map((b) => (
              <option key={b.value} value={b.value}>{b.label}</option>
            ))}
          </select>
        </div>

        {/* Schedule */}
        <div className="bg-slate-800 rounded-xl border border-slate-700 p-5">
          <h3 className="text-sm font-semibold text-slate-300 mb-4">Schedule</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-xs text-slate-400 mb-1">Frequency</label>
              <select
                value={schedule}
                onChange={(e) => setSchedule(e.target.value)}
                className={inputCls}
              >
                {SCHEDULES.map((s) => (
                  <option key={s.value} value={s.value}>{s.label}</option>
                ))}
              </select>
            </div>
            {schedule === 'custom' && (
              <div>
                <label className="block text-xs text-slate-400 mb-1">Cron Expression</label>
                <input
                  type="text"
                  value={customCron}
                  onChange={(e) => setCustomCron(e.target.value)}
                  placeholder="*/5 * * * *"
                  className={inputCls}
                />
              </div>
            )}
          </div>
        </div>

        {/* Strategies */}
        <div className="bg-slate-800 rounded-xl border border-slate-700 p-5">
          <div className="flex items-center justify-between flex-wrap gap-2 mb-4">
            <h3 className="text-sm font-semibold text-slate-300">
              Strategies & Approval Mode
              <span className="ml-2 text-xs font-normal text-slate-500">
                ({enabledStrategies.length}/{availableStrategies?.length ?? 0} selected)
              </span>
            </h3>
            <label className="flex items-center gap-2 cursor-pointer text-xs text-slate-400 hover:text-slate-200 transition-colors">
              <input
                type="checkbox"
                checked={availableStrategies != null && availableStrategies.length > 0 && enabledStrategies.length === availableStrategies.length}
                onChange={() => {
                  if (!availableStrategies) return
                  if (enabledStrategies.length === availableStrategies.length) {
                    setEnabledStrategies([])
                    setAutoApprove(new Set())
                  } else {
                    setEnabledStrategies(availableStrategies.map((s) => s.name))
                  }
                }}
                className="rounded border-slate-600 bg-slate-700 text-blue-500 focus:ring-blue-500 focus:ring-offset-0"
              />
              Select All
            </label>
          </div>
          <div className="space-y-2">
            {(availableStrategies ?? []).map((strat) => {
              const isEnabled = enabledStrategies.includes(strat.name)
              return (
                <div key={strat.name} className={`rounded-lg border p-4 transition-colors ${isEnabled ? 'border-blue-500/60 bg-blue-500/5' : 'border-slate-700 bg-slate-900/30'}`}>
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3 min-w-0">
                      <input
                        type="checkbox"
                        checked={isEnabled}
                        onChange={() => {
                          if (isEnabled) {
                            removeStrategy(strat.name)
                          } else {
                            addStrategy(strat.name)
                          }
                        }}
                        className="rounded border-slate-600 bg-slate-700 text-blue-500 focus:ring-blue-500 focus:ring-offset-0 flex-shrink-0"
                      />
                      <div className="min-w-0 group relative">
                        <span className="text-sm font-medium text-slate-200">{strat.name}</span>
                        <span className="ml-2 text-xs text-slate-500">{strat.summary}</span>
                        {strat.description && (
                          <div className="invisible group-hover:visible absolute z-20 left-0 top-full mt-1 w-80 p-3 bg-slate-700 border border-slate-600 rounded-lg shadow-xl text-xs text-slate-300 leading-relaxed">
                            {strat.description}
                          </div>
                        )}
                      </div>
                      {strat.description && (
                        <button
                          type="button"
                          onClick={() => setInfoOpen(infoOpen === strat.name ? null : strat.name)}
                          className="flex-shrink-0 w-5 h-5 rounded-full border border-slate-600 text-slate-400 hover:text-blue-400 hover:border-blue-500 text-xs flex items-center justify-center transition-colors"
                          title="Strategy details"
                        >
                          ?
                        </button>
                      )}
                    </div>
                    {isEnabled && (
                      <button
                        type="button"
                        onClick={() => toggleAutoApprove(strat.name)}
                        className="flex items-center gap-2 text-xs flex-shrink-0 ml-4"
                      >
                        <div className={`w-9 h-5 rounded-full relative transition-colors ${autoApprove.has(strat.name) ? 'bg-emerald-600' : 'bg-slate-600'}`}>
                          <div className={`absolute top-0.5 w-4 h-4 rounded-full bg-white transition-transform ${autoApprove.has(strat.name) ? 'translate-x-4' : 'translate-x-0.5'}`} />
                        </div>
                        <span className={autoApprove.has(strat.name) ? 'text-emerald-400' : 'text-slate-400'}>
                          {autoApprove.has(strat.name) ? 'Auto' : 'Manual'}
                        </span>
                      </button>
                    )}
                  </div>
                  {infoOpen === strat.name && strat.description && (
                    <div className="mt-3 ml-8 p-3 bg-slate-700/50 border border-slate-600/50 rounded-lg text-xs text-slate-300 leading-relaxed">
                      {strat.description}
                    </div>
                  )}
                </div>
              )
            })}
          </div>
        </div>

        {/* Risk Overrides */}
        <div className="bg-slate-800 rounded-xl border border-slate-700 p-5">
          <h3 className="text-sm font-semibold text-slate-300 mb-4">Risk Overrides</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <label className="block text-xs text-slate-400 mb-1">Kill Switch Drawdown</label>
              <input
                type="number"
                step="0.01"
                value={killSwitchDrawdown}
                onChange={(e) => setKillSwitchDrawdown(e.target.value)}
                className={inputCls}
              />
              <p className="text-xs text-slate-500 mt-1">Max portfolio drawdown before auto-stop</p>
            </div>
            <div>
              <label className="block text-xs text-slate-400 mb-1">Max Daily Trades</label>
              <input
                type="number"
                step="1"
                value={maxDailyTrades}
                onChange={(e) => setMaxDailyTrades(e.target.value)}
                className={inputCls}
              />
            </div>
            <div>
              <label className="block text-xs text-slate-400 mb-1">Max Daily Turnover</label>
              <input
                type="number"
                step="0.01"
                value={maxDailyTurnover}
                onChange={(e) => setMaxDailyTurnover(e.target.value)}
                className={inputCls}
              />
              <p className="text-xs text-slate-500 mt-1">Fraction of portfolio value</p>
            </div>
          </div>
          {config?.costs && (
            config.costs.commission_pct != null ||
            config.costs.slippage_pct != null ||
            config.costs.spread_pct != null
          ) && (
            <div className="mt-4 pt-4 border-t border-slate-700 flex flex-wrap gap-x-6 gap-y-1 text-xs text-slate-400">
              <span>
                Commission: <span className="text-slate-200">{((config.costs.commission_pct ?? 0) * 100).toFixed(3)}%</span>
              </span>
              <span>
                Slippage: <span className="text-slate-200">{((config.costs.slippage_pct ?? 0) * 100).toFixed(3)}%</span>
              </span>
              {config.costs.spread_pct != null && (
                <span>
                  Spread: <span className="text-slate-200">{(config.costs.spread_pct * 100).toFixed(3)}%</span>
                </span>
              )}
              <span className="text-slate-500">(set via config/live.yaml costs: block, not editable here)</span>
            </div>
          )}
        </div>

        {/* Allocation & Signal Combination */}
        <div className="bg-slate-800 rounded-xl border border-slate-700 p-5">
          <h3 className="text-sm font-semibold text-slate-300 mb-4">Allocation & Signal Combination</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <label className="block text-xs text-slate-400 mb-1">Allocation method</label>
              <select
                value={allocationMethod}
                onChange={(e) => setAllocationMethod(e.target.value)}
                className={inputCls}
              >
                <option value="conviction_weighted">Conviction Weighted</option>
                <option value="equal_weight">Equal Weight</option>
                <option value="risk_parity">Risk Parity</option>
                <option value="kelly">Kelly</option>
              </select>
            </div>
            <div>
              <label className="block text-xs text-slate-400 mb-1">
                Kelly fraction {allocationMethod !== 'kelly' && <span className="text-slate-600">(kelly only)</span>}
              </label>
              <input
                type="number"
                step="0.05"
                min="0"
                max="1"
                value={kellyFraction}
                disabled={allocationMethod !== 'kelly'}
                onChange={(e) => setKellyFraction(e.target.value)}
                className={inputCls + ' disabled:opacity-40'}
              />
            </div>
            <div>
              <label className="block text-xs text-slate-400 mb-1">Signal combination</label>
              <select
                value={signalCombination}
                onChange={(e) => setSignalCombination(e.target.value)}
                className={inputCls}
              >
                <option value="confidence">Confidence-Weighted</option>
                <option value="optimal">Optimal (inverse-variance)</option>
              </select>
            </div>
          </div>
          <p className="text-xs text-slate-500 mt-2">
            Takes effect on the next cycle (rebuilds the research/trader pipeline).
          </p>
        </div>

        {/* Strategy Circuit Breaker */}
        <div className="bg-slate-800 rounded-xl border border-slate-700 p-5">
          <div className="flex items-center gap-3 mb-4">
            <button
              type="button"
              onClick={() => setCircuitBreaker((c) => ({ ...c, enabled: !c.enabled }))}
              className="flex items-center gap-2 text-xs"
            >
              <div className={`w-9 h-5 rounded-full relative transition-colors ${circuitBreaker.enabled ? 'bg-blue-600' : 'bg-slate-600'}`}>
                <div className={`absolute top-0.5 w-4 h-4 rounded-full bg-white transition-transform ${circuitBreaker.enabled ? 'translate-x-4' : 'translate-x-0.5'}`} />
              </div>
            </button>
            <h3 className="text-sm font-semibold text-slate-300">Strategy Circuit Breaker (experimental)</h3>
          </div>
          <p className="text-xs text-slate-500 mb-4">
            Damps a strategy's signal contribution when its trailing realized Sharpe is
            persistently negative. <span className="text-amber-500">Off by default:</span> a 3-window
            A/B with these thresholds net hurt portfolio Sharpe (over-gated volatile-but-legitimate
            strategies) — see docs/portfolio_construction_diagnosis.md. Enable only for further
            research/calibration.
          </p>
          <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
            <div>
              <label className="block text-xs text-slate-400 mb-1">Lookback (days)</label>
              <input
                type="number"
                min="1"
                disabled={!circuitBreaker.enabled}
                value={circuitBreaker.lookback_days}
                onChange={(e) => setCircuitBreaker((c) => ({ ...c, lookback_days: Number(e.target.value) }))}
                className={inputCls + ' disabled:opacity-40'}
              />
            </div>
            <div>
              <label className="block text-xs text-slate-400 mb-1">Min track record</label>
              <input
                type="number"
                min="1"
                disabled={!circuitBreaker.enabled}
                value={circuitBreaker.min_track_record_days}
                onChange={(e) => setCircuitBreaker((c) => ({ ...c, min_track_record_days: Number(e.target.value) }))}
                className={inputCls + ' disabled:opacity-40'}
              />
            </div>
            <div>
              <label className="block text-xs text-slate-400 mb-1">Trigger Sharpe</label>
              <input
                type="number"
                step="0.1"
                disabled={!circuitBreaker.enabled}
                value={circuitBreaker.trigger_sharpe}
                onChange={(e) => setCircuitBreaker((c) => ({ ...c, trigger_sharpe: Number(e.target.value) }))}
                className={inputCls + ' disabled:opacity-40'}
              />
            </div>
            <div>
              <label className="block text-xs text-slate-400 mb-1">Full cutoff Sharpe</label>
              <input
                type="number"
                step="0.1"
                disabled={!circuitBreaker.enabled}
                value={circuitBreaker.full_cutoff_sharpe}
                onChange={(e) => setCircuitBreaker((c) => ({ ...c, full_cutoff_sharpe: Number(e.target.value) }))}
                className={inputCls + ' disabled:opacity-40'}
              />
            </div>
            <div>
              <label className="block text-xs text-slate-400 mb-1">Damping floor</label>
              <input
                type="number"
                step="0.05"
                min="0"
                max="1"
                disabled={!circuitBreaker.enabled}
                value={circuitBreaker.damping_floor}
                onChange={(e) => setCircuitBreaker((c) => ({ ...c, damping_floor: Number(e.target.value) }))}
                className={inputCls + ' disabled:opacity-40'}
              />
            </div>
          </div>
        </div>

        {/* Strategy Regime Weights */}
        <div className="bg-slate-800 rounded-xl border border-slate-700 p-5">
          <div className="flex items-center gap-3 mb-4">
            <button
              type="button"
              onClick={() => setRegimeWeights((c) => ({ ...c, enabled: !c.enabled }))}
              className="flex items-center gap-2 text-xs"
            >
              <div className={`w-9 h-5 rounded-full relative transition-colors ${regimeWeights.enabled ? 'bg-blue-600' : 'bg-slate-600'}`}>
                <div className={`absolute top-0.5 w-4 h-4 rounded-full bg-white transition-transform ${regimeWeights.enabled ? 'translate-x-4' : 'translate-x-0.5'}`} />
              </div>
            </button>
            <h3 className="text-sm font-semibold text-slate-300">Strategy Regime Weights (experimental)</h3>
          </div>
          <p className="text-xs text-slate-500 mb-4">
            Scales each strategy's raw signal by Bull/Bear/Chop regime before bull/bear combine.
            Off by default — calibrate on historical windows before enabling live.
          </p>
          <div className="grid grid-cols-2 md:grid-cols-3 gap-4 mb-4">
            <div>
              <label className="block text-xs text-slate-400 mb-1">Benchmark</label>
              <input
                disabled={!regimeWeights.enabled}
                value={regimeWeights.benchmark_symbol ?? 'SPY'}
                onChange={(e) => setRegimeWeights((c) => ({ ...c, benchmark_symbol: e.target.value }))}
                className={inputCls + ' disabled:opacity-40'}
              />
            </div>
            <div>
              <label className="block text-xs text-slate-400 mb-1">Lookback (days)</label>
              <input
                type="number"
                min="1"
                disabled={!regimeWeights.enabled}
                value={regimeWeights.lookback_days}
                onChange={(e) => setRegimeWeights((c) => ({ ...c, lookback_days: Number(e.target.value) }))}
                className={inputCls + ' disabled:opacity-40'}
              />
            </div>
            <div>
              <label className="block text-xs text-slate-400 mb-1">Retrain frequency (days)</label>
              <input
                type="number"
                min="1"
                disabled={!regimeWeights.enabled}
                value={regimeWeights.retrain_frequency}
                onChange={(e) => setRegimeWeights((c) => ({ ...c, retrain_frequency: Number(e.target.value) }))}
                className={inputCls + ' disabled:opacity-40'}
              />
            </div>
          </div>
          <div>
            <label className="block text-xs text-slate-400 mb-1">Weights JSON</label>
            <textarea
              rows={6}
              disabled={!regimeWeights.enabled}
              value={regimeWeightsJson}
              onChange={(e) => setRegimeWeightsJson(e.target.value)}
              className={inputCls + ' font-mono text-xs disabled:opacity-40'}
            />
          </div>
          <p className="text-xs text-slate-500 mt-2">
            Takes effect on the next cycle (rebuilds the research/trader pipeline).
          </p>
        </div>

        {/* News-Guard Blackout */}
        <div className="bg-slate-800 rounded-xl border border-slate-700 p-5">
          <div className="flex items-center gap-3 mb-4">
            <button
              type="button"
              onClick={() => setNewsGuardEnabled(!newsGuardEnabled)}
              className="flex items-center gap-2 text-xs"
            >
              <div className={`w-9 h-5 rounded-full relative transition-colors ${newsGuardEnabled ? 'bg-blue-600' : 'bg-slate-600'}`}>
                <div className={`absolute top-0.5 w-4 h-4 rounded-full bg-white transition-transform ${newsGuardEnabled ? 'translate-x-4' : 'translate-x-0.5'}`} />
              </div>
            </button>
            <h3 className="text-sm font-semibold text-slate-300">News-Guard Macro-Event Blackout</h3>
          </div>
          <p className="text-xs text-slate-500 mb-4">
            Blocks new orders around high-impact macro events (rate decisions, CPI, NFP). Uses the
            bundled offline event calendar unless a live source is configured.
          </p>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <label className="block text-xs text-slate-400 mb-1">Blackout before (min)</label>
              <input
                type="number"
                step="1"
                min="0"
                value={newsGuardBefore}
                disabled={!newsGuardEnabled}
                onChange={(e) => setNewsGuardBefore(e.target.value)}
                className={inputCls + ' disabled:opacity-40'}
              />
            </div>
            <div>
              <label className="block text-xs text-slate-400 mb-1">Blackout after (min)</label>
              <input
                type="number"
                step="1"
                min="0"
                value={newsGuardAfter}
                disabled={!newsGuardEnabled}
                onChange={(e) => setNewsGuardAfter(e.target.value)}
                className={inputCls + ' disabled:opacity-40'}
              />
            </div>
            <div className="flex items-end pb-1">
              <label className="flex items-center gap-2 text-xs text-slate-300 cursor-pointer">
                <input
                  type="checkbox"
                  checked={newsGuardOffline}
                  disabled={!newsGuardEnabled}
                  onChange={(e) => setNewsGuardOffline(e.target.checked)}
                  className="rounded border-slate-600 bg-slate-700 text-blue-500 focus:ring-blue-500 focus:ring-offset-0 disabled:opacity-40"
                />
                Force offline calendar
              </label>
            </div>
          </div>
        </div>

        {/* ═══════════════ AI / LLM Configuration ═══════════════ */}
        <div id="ai" className="border-t border-slate-700 pt-6 mt-2">
          <h3 className="text-lg font-bold text-white mb-1">AI / LLM Configuration</h3>
          <p className="text-xs text-slate-400 mb-5">Configure AI providers, agent modes, caching, and knowledge base</p>
        </div>

        {/* Provider & Model */}
        <div className="bg-slate-800 rounded-xl border border-slate-700 p-5">
          <h3 className="text-sm font-semibold text-slate-300 mb-4">Provider & Model</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
            <div>
              <label className="block text-xs text-slate-400 mb-1">Provider</label>
              <select
                value={activeProvider}
                onChange={(e) => {
                  setActiveProvider(e.target.value)
                  const prov = llmProviders?.find((p) => p.name === e.target.value)
                  if (prov && prov.models.length > 0) setActiveModel(prov.models[0] ?? '')
                }}
                className={inputCls}
              >
                <option value="">Select provider...</option>
                {(llmProviders ?? [])
                  .filter((p) => p.configured || p.name.toLowerCase() === 'ollama')
                  .map((p) => (
                    <option key={p.name} value={p.name}>{p.label}</option>
                  ))}
              </select>
            </div>
            <div>
              <label className="block text-xs text-slate-400 mb-1">Model</label>
              <select
                value={activeModel}
                onChange={(e) => setActiveModel(e.target.value)}
                className={inputCls}
              >
                <option value="">Select model...</option>
                {(llmProviders?.find((p) => p.name === activeProvider)?.models ?? []).map((m) => (
                  <option key={m} value={m}>{m}</option>
                ))}
              </select>
            </div>
          </div>
          <div className="mb-4">
            <label className="block text-xs text-slate-400 mb-1">
              Temperature: <span className="text-slate-200 font-mono">{temperature.toFixed(1)}</span>
            </label>
            <input
              type="range"
              min="0"
              max="1"
              step="0.1"
              value={temperature}
              onChange={(e) => setTemperature(parseFloat(e.target.value))}
              className="w-full max-w-xs h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-blue-500"
            />
            <div className="flex justify-between max-w-xs text-[10px] text-slate-500 mt-0.5">
              <span>Precise (0.0)</span>
              <span>Creative (1.0)</span>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <button
              onClick={handleTestConnection}
              disabled={testLoading || !activeModel}
              className="px-4 py-2 bg-slate-700 border border-slate-600 text-slate-200 rounded-lg text-sm hover:bg-slate-600 disabled:opacity-40 transition-colors flex items-center gap-2"
            >
              {testLoading && <Spinner className="h-4 w-4" />}
              Test Connection
            </button>
            {testResult && (
              <span className={`text-sm ${testResult.status === 'ok' ? 'text-emerald-400' : 'text-red-400'}`}>
                {testResult.status === 'ok'
                  ? `Connected — ${testResult.message} (${testResult.time}ms)`
                  : testResult.message}
              </span>
            )}
          </div>
        </div>

        {/* Agent Modes */}
        <div className="bg-slate-800 rounded-xl border border-slate-700 p-5">
          <h3 className="text-sm font-semibold text-slate-300 mb-1">Agent Modes</h3>
          <p className="text-xs text-slate-500 mb-4">Control how each agent generates signals</p>
          <div className="space-y-2">
            {Object.entries(AGENT_DISPLAY_NAMES).map(([key, displayName]) => (
              <div key={key} className="flex items-center justify-between py-2 px-3 rounded-lg bg-slate-900/50 border border-slate-700/50">
                <span className="text-sm text-slate-200">{displayName}</span>
                <div className="flex rounded-lg overflow-hidden border border-slate-600">
                  {AGENT_MODE_OPTIONS.map((opt) => (
                    <button
                      key={opt.value}
                      type="button"
                      onClick={() => setAgentModes((prev) => ({ ...prev, [key]: opt.value }))}
                      className={`px-3 py-1.5 text-xs font-medium transition-colors ${
                        agentModes[key] === opt.value
                          ? opt.value === 'quant'
                            ? 'bg-slate-600 text-white'
                            : opt.value === 'llm_enhanced'
                              ? 'bg-blue-600 text-white'
                              : 'bg-purple-600 text-white'
                          : 'bg-slate-800 text-slate-400 hover:text-slate-200 hover:bg-slate-700'
                      }`}
                    >
                      {opt.label}
                    </button>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Optimization */}
        <div className="bg-slate-800 rounded-xl border border-slate-700 p-5">
          <h3 className="text-sm font-semibold text-slate-300 mb-4">Optimization</h3>
          <div className="space-y-5">
            {/* Cache */}
            <div>
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-3">
                  <button
                    type="button"
                    onClick={() => setCacheEnabled(!cacheEnabled)}
                    className="flex items-center gap-2 text-xs"
                  >
                    <div className={`w-9 h-5 rounded-full relative transition-colors ${cacheEnabled ? 'bg-blue-600' : 'bg-slate-600'}`}>
                      <div className={`absolute top-0.5 w-4 h-4 rounded-full bg-white transition-transform ${cacheEnabled ? 'translate-x-4' : 'translate-x-0.5'}`} />
                    </div>
                  </button>
                  <span className="text-sm text-slate-200">Response Cache</span>
                </div>
                <button
                  type="button"
                  onClick={() => clearCacheMut.mutate()}
                  disabled={clearCacheMut.isPending}
                  className="px-3 py-1 text-xs border border-slate-600 rounded-lg text-slate-400 hover:text-red-400 hover:border-red-600 transition-colors"
                >
                  {clearCacheMut.isPending ? 'Clearing…' : 'Clear Cache'}
                </button>
              </div>
              {cacheStats && (
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                  <div className="bg-slate-900/50 rounded-lg p-3 border border-slate-700/50">
                    <p className="text-[10px] text-slate-500 uppercase">Hits</p>
                    <p className="text-lg font-mono text-emerald-400">{cacheStats.hits.toLocaleString()}</p>
                  </div>
                  <div className="bg-slate-900/50 rounded-lg p-3 border border-slate-700/50">
                    <p className="text-[10px] text-slate-500 uppercase">Misses</p>
                    <p className="text-lg font-mono text-slate-300">{cacheStats.misses.toLocaleString()}</p>
                  </div>
                  <div className="bg-slate-900/50 rounded-lg p-3 border border-slate-700/50">
                    <p className="text-[10px] text-slate-500 uppercase">Cost Saved</p>
                    <p className="text-lg font-mono text-emerald-400">${cacheStats.total_cost_saved.toFixed(2)}</p>
                  </div>
                  <div className="bg-slate-900/50 rounded-lg p-3 border border-slate-700/50">
                    <p className="text-[10px] text-slate-500 uppercase">Size</p>
                    <p className="text-lg font-mono text-slate-300">{cacheStats.db_size_mb.toFixed(1)} MB</p>
                  </div>
                </div>
              )}
            </div>

            {/* Compression */}
            <div className="border-t border-slate-700 pt-4">
              <div className="flex items-center gap-3 mb-3">
                <button
                  type="button"
                  onClick={() => setCompressionEnabled(!compressionEnabled)}
                  className="flex items-center gap-2 text-xs"
                >
                  <div className={`w-9 h-5 rounded-full relative transition-colors ${compressionEnabled ? 'bg-blue-600' : 'bg-slate-600'}`}>
                    <div className={`absolute top-0.5 w-4 h-4 rounded-full bg-white transition-transform ${compressionEnabled ? 'translate-x-4' : 'translate-x-0.5'}`} />
                  </div>
                </button>
                <span className="text-sm text-slate-200">Prompt Compression</span>
              </div>
              {compressionEnabled && (
                <div className="ml-12">
                  <label className="block text-xs text-slate-400 mb-1">
                    Compression Ratio: <span className="text-slate-200 font-mono">{compressionRatio.toFixed(2)}</span>
                  </label>
                  <input
                    type="range"
                    min="0.3"
                    max="0.8"
                    step="0.05"
                    value={compressionRatio}
                    onChange={(e) => setCompressionRatio(parseFloat(e.target.value))}
                    className="w-full max-w-xs h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-blue-500"
                  />
                  <div className="flex justify-between max-w-xs text-[10px] text-slate-500 mt-0.5">
                    <span>Aggressive (0.3)</span>
                    <span>Light (0.8)</span>
                  </div>
                </div>
              )}
            </div>

            {/* Backtest Policy */}
            <div className="border-t border-slate-700 pt-4">
              <label className="block text-xs text-slate-400 mb-1">Backtest LLM Policy</label>
              <select
                value={backtestPolicy}
                onChange={(e) => setBacktestPolicy(e.target.value)}
                className={inputCls + ' max-w-xs'}
              >
                {BACKTEST_POLICIES.map((bp) => (
                  <option key={bp.value} value={bp.value}>{bp.label}</option>
                ))}
              </select>
              <p className="text-xs text-slate-500 mt-1">Whether LLM calls are allowed during backtests</p>
            </div>
          </div>
        </div>

        {/* Knowledge Base (RAG) */}
        <div className="bg-slate-800 rounded-xl border border-slate-700 p-5">
          <h3 className="text-sm font-semibold text-slate-300 mb-1">Knowledge Base (RAG)</h3>
          <p className="text-xs text-slate-500 mb-4">Retrieval-Augmented Generation document collections</p>

          {/* Embedding Model Selector */}
          <div className="mb-5 p-4 bg-slate-900/50 rounded-lg border border-slate-700/50">
            <label className="block text-xs font-medium text-slate-400 mb-2">Embedding Model</label>
            <div className="flex items-center gap-3 mb-3">
              <select
                value={selectedEmbeddingModel}
                onChange={(e) => handleEmbeddingModelChange(e.target.value)}
                disabled={embeddingSaving}
                className={inputCls + ' max-w-md'}
              >
                <option value="">Select model...</option>
                {(embeddingModels ?? []).map((m) => (
                  <option key={m.model_id} value={m.model_id}>
                    {m.name} — {m.dimensions}d, {m.size_mb >= 1000 ? `${(m.size_mb / 1000).toFixed(1)}GB` : `${m.size_mb}MB`}
                  </option>
                ))}
              </select>
              {embeddingSaving && <Spinner className="h-4 w-4" />}
            </div>

            {(() => {
              const info = embeddingModels?.find((m) => m.model_id === selectedEmbeddingModel)
              if (!info) return null
              const qualityColor = info.quality === 'excellent' ? 'bg-emerald-500/20 text-emerald-400' : info.quality === 'better' ? 'bg-blue-500/20 text-blue-400' : 'bg-slate-500/20 text-slate-400'
              const speedLabel = info.speed === 'very_fast' ? 'Very Fast' : info.speed === 'fast' ? 'Fast' : 'Medium'
              return (
                <div className="space-y-2">
                  <div className="flex flex-wrap items-center gap-2">
                    <span className={`inline-flex items-center px-2 py-0.5 rounded text-[10px] font-medium uppercase ${qualityColor}`}>
                      {info.quality}
                    </span>
                    <span className="inline-flex items-center px-2 py-0.5 rounded text-[10px] font-medium uppercase bg-slate-600/40 text-slate-300">
                      {speedLabel}
                    </span>
                    <span className="text-[11px] text-slate-500">
                      {info.dimensions} dimensions · {info.size_mb >= 1000 ? `${(info.size_mb / 1000).toFixed(1)} GB` : `${info.size_mb} MB`}
                    </span>
                  </div>
                  <p className="text-xs text-slate-400">{info.description}</p>
                </div>
              )
            })()}

            {embeddingReindexWarning && (
              <div className="mt-3 flex items-start gap-2 p-3 rounded-lg bg-amber-500/10 border border-amber-500/30">
                <svg className="w-4 h-4 text-amber-400 flex-shrink-0 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                <p className="text-xs text-amber-300">
                  Changing the embedding model requires re-indexing all documents. Current vectors will be incompatible.
                </p>
              </div>
            )}
          </div>

          {ragStats?.collections && Object.keys(ragStats.collections).length > 0 && (
            <div className="mb-4 overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className="text-left text-slate-500 border-b border-slate-700">
                    <th className="pb-2 pr-4">Collection</th>
                    <th className="pb-2 pr-4 text-right">Documents</th>
                    <th className="pb-2">Description</th>
                    <th className="pb-2 pl-4"></th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(ragStats.collections).map(([name, col]) => (
                    <tr key={name} className="text-slate-300 border-b border-slate-700/50">
                      <td className="py-2 pr-4 font-mono">{name}</td>
                      <td className="py-2 pr-4 text-right font-mono">{col.count.toLocaleString()}</td>
                      <td className="py-2 text-slate-400">{col.description}</td>
                      <td className="py-2 pl-4 text-right">
                        <button
                          type="button"
                          onClick={() => {
                            if (confirm(`Delete all ${col.count.toLocaleString()} documents in "${name}"? This cannot be undone.`)) {
                              deleteCollectionMut.mutate(name)
                            }
                          }}
                          disabled={deleteCollectionMut.isPending}
                          className="text-slate-500 hover:text-red-400 disabled:opacity-40 transition-colors"
                          title={`Delete ${name}`}
                        >
                          Delete
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          <div className="border border-slate-700 rounded-lg overflow-hidden">
            <button
              type="button"
              onClick={() => setRagExpanded(!ragExpanded)}
              className="w-full flex items-center justify-between px-4 py-3 bg-slate-900/50 hover:bg-slate-900 transition-colors"
            >
              <span className="text-xs font-medium text-slate-300">Ingest Documents</span>
              <svg
                className={`w-4 h-4 text-slate-400 transition-transform ${ragExpanded ? 'rotate-180' : ''}`}
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
              </svg>
            </button>
            {ragExpanded && (
              <div className="p-4 space-y-3 border-t border-slate-700">
                <div className="flex flex-wrap gap-3">
                  {RAG_DOC_TYPES.map((dt) => (
                    <label key={dt.value} className="flex items-center gap-2 text-xs text-slate-300 cursor-pointer">
                      <input
                        type="checkbox"
                        checked={ragIngestTypes.includes(dt.value)}
                        onChange={() =>
                          setRagIngestTypes((prev) =>
                            prev.includes(dt.value) ? prev.filter((t) => t !== dt.value) : [...prev, dt.value]
                          )
                        }
                        className="rounded border-slate-600 bg-slate-700 text-blue-500 focus:ring-blue-500 focus:ring-offset-0"
                      />
                      {dt.label}
                    </label>
                  ))}
                </div>
                <div>
                  <label className="block text-xs text-slate-400 mb-1">Symbols (comma-separated, optional)</label>
                  <input
                    type="text"
                    value={ragSymbols}
                    onChange={(e) => setRagSymbols(e.target.value)}
                    placeholder={symbols || 'AAPL, MSFT, GOOGL'}
                    className={inputCls}
                  />
                </div>
                <div className="flex items-center gap-3">
                  <button
                    type="button"
                    onClick={handleIngest}
                    disabled={ragIngestTypes.length === 0}
                    className="px-4 py-2 bg-blue-600 text-white rounded-lg text-xs font-medium hover:bg-blue-500 disabled:opacity-40 transition-colors"
                  >
                    Start Ingestion
                  </button>
                  {ingestionMsg && (
                    <span className={`text-xs ${ingestionMsg.startsWith('Error') ? 'text-red-400' : 'text-emerald-400'}`}>
                      {ingestionMsg}
                    </span>
                  )}
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Universe */}
        <div className="bg-slate-800 rounded-xl border border-slate-700 p-5">
          <h3 className="text-sm font-semibold text-slate-300 mb-4">Universe</h3>
          <label className="block text-xs text-slate-400 mb-1">Symbols (comma-separated)</label>
          <textarea
            value={symbols}
            onChange={(e) => setSymbols(e.target.value)}
            rows={3}
            className={inputCls + ' resize-y'}
            placeholder="AAPL, MSFT, GOOG, AMZN, META, TSLA, NVDA"
          />
        </div>

        {/* Save */}
        <div className="flex items-center gap-4">
          <button
            onClick={handleSave}
            disabled={saveMut.isPending || saveLLMMut.isPending}
            className="px-5 py-2.5 bg-blue-600 text-white rounded-lg text-sm font-medium hover:bg-blue-500 disabled:opacity-40 transition-colors flex items-center gap-2"
          >
            {(saveMut.isPending || saveLLMMut.isPending) && <Spinner className="h-4 w-4" />}
            Save Configuration
          </button>
          {saveMut.isSuccess && saveLLMMut.isSuccess && (
            <span className="text-sm text-emerald-400">Configuration saved.</span>
          )}
          {(saveMut.error || saveLLMMut.error) && (
            <span className="text-sm text-red-400">
              {((saveMut.error || saveLLMMut.error) as Error).message}
            </span>
          )}
        </div>
      </div>
    </div>
  )
}
