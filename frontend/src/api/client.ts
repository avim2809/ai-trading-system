import type {
  StrategyInfo,
  ConfigDefaults,
  RunSummary,
  RunDetail,
  ReportData,
  EquityData,
  RunRequest,
  StepRequest,
  BlackboardView,
  LiveStatus,
  LiveStartRequest,
  BrokerPosition,
  AccountInfo,
  OrderRecord,
  CycleRecord,
  PendingApproval,
  ApprovalDetail,
  LiveConfig,
  LLMProvider,
  LLMConfig,
  LLMCacheStats,
  RAGStats,
  LLMTestResult,
  EmbeddingModelInfo,
} from './types'

const BASE = '/api'

async function fetchJson<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${url}`, {
    headers: { 'Content-Type': 'application/json' },
    ...init,
  })
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText)
    throw new Error(`${res.status}: ${text}`)
  }
  return res.json() as Promise<T>
}

export const api = {
  health: () => fetchJson<{ status: string }>('/health'),

  getStrategies: () => fetchJson<StrategyInfo[]>('/strategies'),

  getDefaults: () => fetchJson<ConfigDefaults>('/config/defaults'),

  getRuns: (status?: string) =>
    fetchJson<RunSummary[]>(status ? `/runs?status=${status}` : '/runs'),

  getRun: (id: string) => fetchJson<RunDetail>(`/runs/${id}`),

  getReport: (id: string) => fetchJson<ReportData>(`/runs/${id}/report`),

  getEquity: (id: string) => fetchJson<EquityData>(`/runs/${id}/equity`),

  launchRun: (req: RunRequest) =>
    fetchJson<{ run_id: string }>('/runs', {
      method: 'POST',
      body: JSON.stringify(req),
    }),

  compareRuns: (ids: string[]) =>
    fetchJson<Record<string, Record<string, number>>>('/runs/compare', {
      method: 'POST',
      body: JSON.stringify({ run_ids: ids }),
    }),

  agentStep: (req: StepRequest) =>
    fetchJson<BlackboardView>('/agents/step', {
      method: 'POST',
      body: JSON.stringify(req),
    }),

  // ── Live Trading ──

  getLiveStatus: () => fetchJson<LiveStatus>('/live/status'),

  startLive: (req: LiveStartRequest) =>
    fetchJson<{ status: string }>('/live/start', {
      method: 'POST',
      body: JSON.stringify(req),
    }),

  stopLive: () =>
    fetchJson<{ status: string }>('/live/stop', { method: 'POST' }),

  triggerCycle: () =>
    fetchJson<{ cycle_id: string }>('/live/trigger', { method: 'POST' }),

  getPositions: () => fetchJson<BrokerPosition[]>('/live/positions'),

  getAccount: () => fetchJson<AccountInfo>('/live/account'),

  getOrders: (_limit = 50) =>
    fetchJson<OrderRecord[]>('/live/orders'),

  getCycles: (_limit = 20) =>
    fetchJson<CycleRecord[]>('/live/cycles'),

  getApprovals: () => fetchJson<PendingApproval[]>('/live/approvals'),

  getApprovalDetail: (id: string) =>
    fetchJson<ApprovalDetail>(`/live/approvals/${id}`),

  approveOrder: (id: string) =>
    fetchJson<{ status: string; order_statuses: unknown[] }>(
      `/live/approvals/${id}/approve`,
      { method: 'POST' },
    ),

  rejectOrder: (id: string, reason: string) =>
    fetchJson<{ status: string }>(`/live/approvals/${id}/reject`, {
      method: 'POST',
      body: JSON.stringify({ reason }),
    }),

  getLiveConfig: () => fetchJson<LiveConfig>('/live/config'),

  updateLiveConfig: (config: LiveConfig) =>
    fetchJson<{ status: string }>('/live/config', {
      method: 'PUT',
      body: JSON.stringify(config),
    }),

  // ── LLM / AI ──

  getLLMProviders: () => fetchJson<LLMProvider[]>('/llm/providers'),

  getLLMConfig: () => fetchJson<LLMConfig>('/llm/config'),

  updateLLMConfig: (config: Partial<LLMConfig>) =>
    fetchJson<{ status: string }>('/llm/config', {
      method: 'PUT',
      body: JSON.stringify(config),
    }),

  getLLMCacheStats: () => fetchJson<LLMCacheStats>('/llm/cache/stats'),

  clearLLMCache: () =>
    fetchJson<{ status: string }>('/llm/cache', { method: 'DELETE' }),

  getRAGStats: () => fetchJson<RAGStats>('/llm/rag/stats'),

  ingestRAGDocs: (types: string[], symbols?: string[]) =>
    fetchJson<{ status: string; message: string }>('/llm/rag/ingest', {
      method: 'POST',
      body: JSON.stringify({ types, symbols }),
    }),

  testLLMConnection: () =>
    fetchJson<LLMTestResult>('/llm/test', { method: 'POST' }),

  getEmbeddingModels: () =>
    fetchJson<EmbeddingModelInfo[]>('/llm/embedding-models'),

  setEmbeddingModel: (model_id: string) =>
    fetchJson<{ status: string; requires_reindex: boolean; model: EmbeddingModelInfo }>(
      '/llm/rag/embedding-model',
      { method: 'PUT', body: JSON.stringify({ model_id }) },
    ),
}
