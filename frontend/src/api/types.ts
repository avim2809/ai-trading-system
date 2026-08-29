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
  spread_pct?: number
  short_borrow_annual_pct?: number
  market_impact_coefficient?: number
  market_impact_crossover_participation?: number | null
  rebalance_frequency: string
  risk_overrides: Record<string, number>
  regime_overlay?: RegimeOverlayConfig
  allocation_method?: string
  kelly_fraction?: number
  signal_combination?: SignalCombinationConfig
  strategy_circuit_breaker?: StrategyCircuitBreakerConfig
  strategy_regime_weights?: StrategyRegimeWeightsConfig
  data_source: string
  seed: number
  notes: string
}

export interface SignalCombinationConfig {
  method: string
  [key: string]: unknown
}

// Generic per-strategy rolling-Sharpe circuit breaker (see
// firm.agents.research._circuit_breaker). Disabled by default — an A/B with
// these exact default thresholds was found to net *hurt* portfolio Sharpe
// in every tested window (see docs/portfolio_construction_diagnosis.md), so
// treat this as an experimental/research knob, not a recommended default.
export interface StrategyCircuitBreakerConfig {
  enabled: boolean
  lookback_days?: number
  min_track_record_days?: number
  trigger_sharpe?: number
  full_cutoff_sharpe?: number
  damping_floor?: number
}

// Regime-conditional per-strategy score multipliers (see
// firm.agents.research._regime_weights). Disabled by default — calibrate
// before enabling live.
export interface StrategyRegimeWeightsConfig {
  enabled: boolean
  benchmark_symbol?: string
  lookback_days?: number
  retrain_frequency?: number
  weights?: Record<string, Record<string, number>>
  min_multiplier?: number
  max_multiplier?: number
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
  // avg_turnover / total_turnover (fraction of NAV) / rebalance_count, from
  // firm.backtest.analyzers.TurnoverAnalyzer. Omitted when the report has no
  // rebalance activity to measure.
  turnover?: Record<string, number>
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
  // Only present when at least one fold ran a genuine multi-candidate
  // param_grid selection on its train window — PBO needs real competing
  // trials to mean anything, so it's omitted (not estimated) otherwise.
  pbo?: number
  pbo_n_folds?: number
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

export interface HealthResponse {
  status: string
  broker: {
    type: string | null
    connected: boolean | null
    live_engine_running: boolean
  }
}

export interface ConfigDefaults {
  universe: Record<string, unknown>
  backtest: Record<string, unknown>
  risk: Record<string, unknown>
  strategy_params?: Record<string, Record<string, unknown>>
  allocation_method?: string
  kelly_fraction?: number
  signal_combination?: SignalCombinationConfig
  strategy_circuit_breaker?: StrategyCircuitBreakerConfig
  strategy_regime_weights?: StrategyRegimeWeightsConfig
}

// ── Live Trading Types ──

export interface LastCycle {
  cycle_id: number
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
  /** null = unknown (no broker instance to ask yet), not "closed". */
  market_open: boolean | null
  next_market_open: string | null
  next_market_close: string | null
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
  strategy_circuit_breaker?: StrategyCircuitBreakerConfig
  strategy_regime_weights?: StrategyRegimeWeightsConfig
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
  /** {strategy: {symbol: weight}} attribution of proposal_weights — absent
   * (undefined, not just empty) on entries recorded before this field
   * existed. */
  per_strategy?: Record<string, Record<string, number>>
  notes: string
  nav_at_decision: number | null
  raw_return: number | null
  benchmark_return: number | null
  reflection: string | null
  /** Structured reflection fields (added alongside `reflection`, which stays
   * as a rendered fallback for older entries / older prompt-injection
   * consumers). `verdict` is null until reflected; "unknown" means the LLM
   * reflection call failed and only the unstructured fallback exists. */
  verdict?: 'correct' | 'incorrect' | 'partial' | 'unknown' | null
  what_worked?: string | null
  what_failed?: string | null
  lesson?: string | null
}

export interface LessonsDigest {
  total: number
  counts: { correct: number; incorrect: number; partial: number; unknown: number }
  recent_lessons: string[]
}

export interface BrokerPosition {
  symbol: string
  quantity: number
  avg_cost: number
  market_value: number
  unrealized_pnl: number
  side: 'long' | 'short'
}

export interface PositionsSummary {
  long_value: number
  short_value: number
  n_long: number
  n_short: number
}

export interface LivePortfolioHistory {
  dates: string[]
  values: number[]
  drawdown: number[]
  metrics: Record<string, number>
  n_observations: number
}

export type LiveAttribution = Record<string, Record<string, number>>

export interface CapitalGateCriterion {
  label: string
  threshold?: string
  value: number | null
  passing: boolean | null
  /** Present only on `realized_sharpe`: the non-bootstrapped point estimate. */
  point_estimate?: number
  /** Present only on `realized_sharpe`: daily-return observation count. */
  n_observations?: number
  /** Present only on `kill_switch_trips`: not persisted across restarts. */
  durable?: boolean
  currently_halted?: boolean
  /** Present only on `llm_ab`: always false — a manual runbook item. */
  applicable?: boolean
  /** Human-readable caveat, shown when `passing` is null. */
  caveat?: string
}

export interface CapitalGateStatus {
  engine_running: boolean
  broker: string | null
  overall_passing: boolean
  n_passing: number
  blocking: string[]
  criteria: {
    duration?: CapitalGateCriterion
    trade_count?: CapitalGateCriterion
    realized_sharpe?: CapitalGateCriterion
    max_drawdown?: CapitalGateCriterion
    kill_switch_trips?: CapitalGateCriterion
    llm_ab?: CapitalGateCriterion
  }
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
  cycle_id: number
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

export interface LiveConfigCosts {
  commission_pct?: number
  slippage_pct?: number
  spread_pct?: number
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
  strategy_circuit_breaker?: StrategyCircuitBreakerConfig
  strategy_regime_weights?: StrategyRegimeWeightsConfig
  allocation_method?: string
  kelly_fraction?: number
  costs?: LiveConfigCosts
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
  /** Absent on "RAW" lines (unparseable/non-JSON) and any entry logged
   * before this field existed. */
  file?: string
  function?: string
  line?: number
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
