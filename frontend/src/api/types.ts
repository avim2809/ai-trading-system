export interface StrategyInfo {
  name: string
  default_params: Record<string, unknown>
  summary: string
  description: string
}

export interface RunSummary {
  run_id: string
  status: string
  start_time: string
  end_time: string | null
  notes: string
  metrics: Record<string, number>
}

export interface RunDetail extends RunSummary {
  config: Record<string, unknown>
  config_hash: string
  seed: number
  artifacts_dir: string
}

export interface RegimeOverlayConfig {
  enabled: boolean
  benchmark_symbol: string | null
  n_states: number
  retrain_frequency: number
  exposure_map: { Bull: number; Bear: number; Chop: number }
}

export interface RunRequest {
  strategies: string[]
  strategy_params: Record<string, Record<string, unknown>>
  universe_symbols: string[] | null
  start_date: string
  end_date: string
  initial_capital: number
  commission_pct: number
  slippage_pct: number
  rebalance_frequency: string
  risk_overrides: Record<string, number>
  regime_overlay?: RegimeOverlayConfig
  allocation_method?: string
  kelly_fraction?: number
  signal_combination?: SignalCombinationConfig
  data_source: string
  seed: number
  notes: string
}

export interface SignalCombinationConfig {
  method: string
  [key: string]: unknown
}

export interface MonteCarloSummary {
  n_simulations: number
  confidence: number
  drawdowns: Record<string, number>
  probability_of_loss: Record<string, number>
  confidence_interval: Record<string, number>
}

export interface ReportData {
  portfolio: Record<string, number>
  benchmark?: Record<string, number>
  strategies?: Record<string, Record<string, number>>
  period?: { start: string; end: string }
  final_nav?: number
  data_points: number
  trade_metrics?: Record<string, number>
  monte_carlo?: MonteCarloSummary
}

export interface WalkForwardAggMetric {
  mean: number
  std: number
  min: number
  max: number
  values: number[]
}

export interface WalkForwardOverfitting {
  n_folds: number
  probabilistic_sharpe?: number
  deflated_sharpe?: number
  pbo?: number
  verdict?: string
}

export interface WalkForwardResult {
  fold_ids: string[]
  aggregate: {
    n_folds: number
    fold_ids: string[]
    metrics: Record<string, WalkForwardAggMetric>
    overfitting?: WalkForwardOverfitting
  }
}

export interface EquityData {
  dates: string[]
  values: number[]
  drawdown: number[]
}

export interface Signal {
  symbol: string
  strategy: string
  score: number
  confidence: number
  horizon: string
  asof: string
  meta: Record<string, unknown>
}

export interface SignalSet {
  domain: string
  asof: string
  signals: Signal[]
}

export interface Thesis {
  side: string
  symbol: string
  conviction: number
  rationale: string
  supporting: string[]
}

export interface DebateResult {
  symbol: string
  net_conviction: number
  bull_thesis: Thesis | null
  bear_thesis: Thesis | null
}

export interface TradeProposal {
  asof: string
  targets: Record<string, number>
  per_strategy: Record<string, Record<string, number>>
  notes: string
}

export interface RiskDecision {
  approved: boolean
  adjusted_targets: Record<string, number>
  violations: string[]
  actions: string[]
}

export interface ExecutionFill {
  symbol: string
  side: string
  quantity: number
  strategy: string
}

export interface ExecutionReport {
  fills: ExecutionFill[]
  turnover: number
  costs: number
}

export interface BlackboardView {
  asof: string
  signal_sets: SignalSet[]
  theses: Thesis[]
  debate_results: DebateResult[]
  proposal: TradeProposal | null
  risk_decision: RiskDecision | null
  execution_report: ExecutionReport | null
}

export interface StepRequest {
  strategies: string[]
  strategy_params: Record<string, Record<string, unknown>>
  symbols: string[]
  asof_date: string
  data_source: string
  seed: number
}

export interface ConfigDefaults {
  universe: Record<string, unknown>
  backtest: Record<string, unknown>
  risk: Record<string, unknown>
  strategy_params?: Record<string, Record<string, unknown>>
  allocation_method?: string
  kelly_fraction?: number
  signal_combination?: SignalCombinationConfig
}

// ── Live Trading Types ──

export interface LastCycle {
  cycle_id: string
  timestamp: string
  orders_generated: number
}

export interface LiveStatus {
  state: 'running' | 'stopped'
  broker: string
  broker_connected: boolean
  next_run: string | null
  active_strategies: string[]
  approval_mode: string
  uptime_seconds: number | null
  last_cycle: LastCycle | null
  cycle_running_seconds: number | null
}

export interface LiveStartRequest {
  broker: string
  schedule?: string
  approval_mode?: string
  auto_approve_strategies?: string[]
  symbols?: string[]
  initial_capital?: number
  strategies?: string[]
  kill_switch_drawdown?: number
  max_daily_trades?: number
  max_daily_turnover?: number
  news_guard?: LiveConfigNewsGuard
  signal_combination?: SignalCombinationConfig
  allocation_method?: string
  kelly_fraction?: number
}

export interface LiveAlert {
  timestamp: string
  kind: string
  severity: 'warning' | 'critical'
  message: string
  cycle_id: number
  [key: string]: unknown
}

export interface LiveAlertsResponse {
  halted: boolean
  alerts: LiveAlert[]
}

export interface DecisionEntry {
  date: string
  status: 'pending' | 'reflected'
  proposal_weights: Record<string, number>
  notes: string
  nav_at_decision: number | null
  raw_return: number | null
  benchmark_return: number | null
  reflection: string | null
}

export interface BrokerPosition {
  symbol: string
  quantity: number
  avg_cost: number
  market_value: number
  unrealized_pnl: number
}

export interface AccountInfo {
  cash: number
  equity: number
  buying_power: number
  currency: string
}

export interface OrderRecord {
  order_id: string
  symbol: string
  side: string
  quantity: number
  filled_quantity: number
  avg_fill_price: number
  status: string
  strategy?: string
  timestamp: string | null
  /** ``cycle`` = auto-submitted; ``approval`` = manually approved. */
  source?: 'cycle' | 'approval' | string
  cycle_id?: number
  approval_id?: string
}

export interface CycleRecord {
  cycle_id: string
  timestamp: string
  orders_generated: number
  orders_submitted: number
  orders_queued: number
  error: string | null
}

export interface ApprovalOrder {
  symbol: string
  side: string
  quantity: number
  order_type: string
  limit_price: number | null
  strategy: string
}

export interface PendingApproval {
  approval_id: string
  created_at: string
  expires_at: string
  status: string
  strategy: string
  orders: ApprovalOrder[]
  reject_reason: string | null
}

export interface ApprovalDetail extends PendingApproval {
  blackboard_snapshot: BlackboardView
}

export interface LiveConfigStrategies {
  enabled: string[]
  auto_approve: string[]
  require_approval: string[]
}

export interface LiveConfigRisk {
  kill_switch_drawdown: number
  max_daily_trades: number
  max_daily_turnover: number
}

export interface LiveConfigUniverse {
  symbols: string[]
}

export interface LiveConfigNewsGuard {
  enabled: boolean
  before_min: number
  after_min: number
  offline: boolean
}

export interface LiveConfig {
  broker: string
  schedule: string
  approval_mode: string
  strategies: LiveConfigStrategies
  strategy_params?: Record<string, Record<string, unknown>>
  risk: LiveConfigRisk
  universe: LiveConfigUniverse
  news_guard?: LiveConfigNewsGuard
  signal_combination?: SignalCombinationConfig
  allocation_method?: string
  kelly_fraction?: number
}

// ── LLM / AI Types ──

export interface LLMProvider {
  name: string
  label: string
  models: string[]
  configured: boolean
  default_model: string
}

export interface LLMConfig {
  provider: {
    default_model: string
    temperature: number
    max_tokens: number
  }
  agent_modes: Record<string, 'quant' | 'llm_enhanced' | 'llm_only'>
  optimization: {
    compression_enabled: boolean
    compression_ratio: number
    cache_enabled: boolean
  }
  rag: {
    persist_dir: string
    embedding_model: string
    default_n_results: number
  }
  backtest_policy: string
}

export interface LLMCacheStats {
  hits: number
  misses: number
  total_cost_saved: number
  entries: number
  db_size_mb: number
}

export interface RAGStats {
  collections: Record<string, { count: number; description: string }>
}

export interface LLMTestResult {
  status: string
  model: string
  response_time_ms: number
}

export interface EmbeddingModelInfo {
  model_id: string
  name: string
  dimensions: number
  size_mb: number
  quality: 'good' | 'better' | 'excellent'
  speed: 'very_fast' | 'fast' | 'medium'
  description: string
}

export interface LogEntry {
  ts: string | null
  level: string
  logger: string
  msg: string
  exception?: string
}

export interface LogTailResponse {
  lines: LogEntry[]
  next_offset: number
  reset: boolean
}

// ── System Resources ──

export interface SystemResources {
  cpu: {
    percent: number
    count: number
  }
  memory: {
    used: number
    total: number
    percent: number
  }
  disk: {
    used: number
    total: number
    percent: number
    path: string
  }
}
