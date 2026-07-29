// Fixture data mirrors real backend response shapes verified via curl
// against the live running API (2026-07-21/22) — not just the TS types,
// which is exactly what let the /llm/rag/stats shape-mismatch crash slip
// through undetected until it hit a real browser.

import type {
  StrategyInfo, ConfigDefaults, RunSummary, LiveStatus, LiveConfig, LiveAlertsResponse,
  BrokerPosition, AccountInfo, CycleRecord, PendingApproval, ApprovalDetail, DecisionEntry,
  LessonsDigest,
  LLMProvider, LLMConfig, LLMCacheStats, RAGStats, EmbeddingModelInfo, LogTailResponse,
  BlackboardView, OrderRecord, SystemResources,
} from '../api/types'

export const mockStrategies: StrategyInfo[] = [
  { name: 'momentum', default_params: {}, summary: 'Cross-sectional momentum', description: 'Momentum strategy description.' },
  { name: 'trend', default_params: {}, summary: 'Time-series trend following', description: 'Trend strategy description.' },
  { name: 'multi_factor', default_params: {}, summary: 'Multi-factor composite equity', description: 'Multi-factor description.' },
]

export const mockConfigDefaults: ConfigDefaults = {
  universe: { index: 'SP500', min_market_cap: 1000000000, min_avg_volume: 500000 },
  backtest: { start_date: '2018-01-01', end_date: '2023-12-31', initial_capital: 10000000, commission_pct: 0.001, slippage_pct: 0.0005, rebalance_frequency: 'weekly' },
  risk: { max_position_pct: 0.05, max_gross_exposure: 1.5, max_net_exposure: 0.5, max_sector_pct: 0.25, vol_target: 0.12, max_drawdown_pct: 0.15 },
}

export const mockRuns: RunSummary[] = [
  { run_id: 'run-1', status: 'completed', start_time: '2026-07-01T00:00:00', end_time: '2026-07-01T00:05:00', notes: 'test run', metrics: { sharpe: 1.2, cagr: 0.15 } },
]

export const mockLiveStatusStopped: LiveStatus = {
  state: 'stopped', broker: '', broker_connected: false, next_run: null,
  active_strategies: [], approval_mode: '', uptime_seconds: null, last_cycle: null,
  cycle_running_seconds: null,
}

export const mockLiveStatusRunning: LiveStatus = {
  state: 'running', broker: 'ibkr_paper', broker_connected: true,
  next_run: '2026-07-22T09:35:00-04:00',
  active_strategies: ['momentum', 'trend', 'multi_factor'],
  approval_mode: 'full_auto', uptime_seconds: 3600,
  last_cycle: { cycle_id: 1, timestamp: '2026-07-21T19:49:04.066971', orders_generated: 20 },
  cycle_running_seconds: null,
}

export const mockLiveConfig: LiveConfig = {
  broker: 'ibkr_paper', schedule: 'market_open', approval_mode: 'full_auto',
  strategies: { enabled: ['momentum', 'trend', 'multi_factor'], auto_approve: ['momentum'], require_approval: ['trend', 'multi_factor'] },
  risk: { kill_switch_drawdown: 0.08, max_daily_trades: 40, max_daily_turnover: 0.25 },
  universe: { symbols: ['AAPL', 'MSFT', 'NVDA'] },
}

export const mockAlerts: LiveAlertsResponse = {
  halted: false,
  alerts: [
    { timestamp: '2026-07-21T20:02:43.631908', kind: 'daily_limit_breach', severity: 'warning', message: 'Daily limit would be breached (trades 20/40, turnover 100.0%/25.0%); routing orders to manual approval.', cycle_id: 1 },
  ],
}

export const mockPositions: BrokerPosition[] = [
  { symbol: 'AAPL', quantity: 1, avg_cost: 332.5, market_value: 332.5, unrealized_pnl: 0 },
]

export const mockAccount: AccountInfo = {
  cash: 999753.58, equity: 1001712.88, buying_power: 6677369.99, currency: 'USD',
}

export const mockCycles: CycleRecord[] = [
  { cycle_id: 1, timestamp: '2026-07-21T19:49:04.066971', orders_generated: 20, orders_submitted: 0, orders_queued: 20, error: null },
]

export const mockOrders: OrderRecord[] = [
  { order_id: 'o1', symbol: 'AAPL', side: 'buy', quantity: 1, filled_quantity: 1, avg_fill_price: 332.5, status: 'filled', strategy: 'momentum', timestamp: '2026-07-18T19:28:37', source: 'cycle' },
]

export const mockApprovals: PendingApproval[] = [
  {
    approval_id: 'cb28dd9bbd6a', created_at: '2026-07-21T20:02:43.632546', expires_at: '2026-07-21T21:02:43.632546',
    status: 'pending', strategy: 'multi_factor',
    orders: [
      { symbol: 'AAPL', side: 'buy', quantity: 151.58, order_type: 'market', limit_price: null, strategy: 'multi_factor' },
    ],
    reject_reason: null,
  },
]

export const mockBlackboard: BlackboardView = {
  asof: '2026-07-21T19:49:04',
  signal_sets: [
    { domain: 'technical', asof: '2026-07-21T19:49:04', signals: [{ symbol: 'AAPL', strategy: 'momentum', score: 0.8, confidence: 0.7, horizon: '1d', asof: '2026-07-21T19:49:04', meta: {} }] },
  ],
  theses: [{ side: 'bull', symbol: 'AAPL', conviction: 0.6, rationale: 'Strong momentum.', supporting: ['momentum'] }],
  debate_results: [{ symbol: 'AAPL', net_conviction: 0.5, bull_thesis: null, bear_thesis: null }],
  proposal: { asof: '2026-07-21T19:49:04', targets: { AAPL: 0.05 }, per_strategy: {}, notes: 'cycle=1' },
  risk_decision: { approved: true, adjusted_targets: { AAPL: 0.05 }, violations: [], actions: ['Sector concentration cap NOT enforced (no sector_map)'] },
  execution_report: { fills: [{ symbol: 'AAPL', side: 'buy', quantity: 1, strategy: 'momentum' }], turnover: 0.1, costs: 5.0 },
}

export const mockApprovalDetail: ApprovalDetail = {
  ...mockApprovals[0]!,
  blackboard_snapshot: mockBlackboard,
}

export const mockLLMProviders: LLMProvider[] = [
  { name: 'groq', label: 'Groq (free tier)', configured: true, models: ['groq/llama-3.3-70b-versatile'], default_model: 'groq/llama-3.3-70b-versatile' },
  { name: 'gemini', label: 'Google Gemini', configured: false, models: ['gemini/gemini-2.5-flash'], default_model: 'gemini/gemini-2.5-flash' },
  { name: 'anthropic', label: 'Anthropic Claude', configured: false, models: ['anthropic/claude-opus-4-8'], default_model: 'anthropic/claude-opus-4-8' },
]

export const mockLLMConfig: LLMConfig = {
  provider: { default_model: 'groq/llama-3.3-70b-versatile', temperature: 0.3, max_tokens: 2000 },
  agent_modes: { technical_analyst: 'quant', sentiment_analyst: 'llm_enhanced' },
  optimization: { compression_enabled: true, compression_ratio: 0.5, cache_enabled: true },
  rag: { persist_dir: 'data/vectordb', embedding_model: 'voyage-finance-2', default_n_results: 5 },
  backtest_policy: 'cache_only',
}

export const mockCacheStats: LLMCacheStats = {
  hits: 12, misses: 4, total_cost_saved: 0.42, entries: 16, db_size_mb: 0.07,
}

// The real shape after the fix: {"collections": {name: {count, description}}} —
// NOT the flat {name: count} dict VectorStore.stats() returns internally.
export const mockRAGStats: RAGStats = {
  collections: {
    sec_filings: { count: 4205, description: 'SEC filings (10-K/10-Q/8-K)' },
    research: { count: 104, description: 'Academic research papers (arXiv)' },
    system_docs: { count: 35, description: 'Strategy/system documentation' },
  },
}

export const mockEmbeddingModels: EmbeddingModelInfo[] = [
  { model_id: 'all-MiniLM-L6-v2', name: 'MiniLM L6 v2', dimensions: 384, size_mb: 80, quality: 'good', speed: 'very_fast', description: 'Lightweight general-purpose model.' },
]

export const mockLogTail: LogTailResponse = {
  lines: [
    { ts: '2026-07-21T19:49:04.000Z', level: 'INFO', logger: 'firm.live.engine', msg: 'Cycle 1: 20 generated, 0 submitted, 20 queued, 0 failed' },
  ],
  next_offset: 512,
  reset: false,
}

export const mockSystemResources: SystemResources = {
  cpu: { percent: 23.5, count: 2 },
  memory: { used: 1_800_000_000, total: 3_300_000_000, percent: 54.5 },
  disk: { used: 40_000_000_000, total: 100_000_000_000, percent: 40.0, path: '/' },
}

export const mockDecisions: DecisionEntry[] = [
  {
    date: '2026-07-21', status: 'reflected',
    proposal_weights: { AAPL: 0.05, MSFT: 0.03 },
    notes: 'cycle=1', nav_at_decision: 1000000, raw_return: 0.012, benchmark_return: 0.008,
    reflection: 'The directional call was correct; AAPL outperformed the benchmark.',
  },
]

export const mockLessons: LessonsDigest = {
  total: 3,
  counts: { correct: 2, incorrect: 1, partial: 0, unknown: 0 },
  recent_lessons: ['Trust the signal in trending regimes', 'Wait for confirmation before sizing up'],
}
