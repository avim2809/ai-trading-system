import { http, HttpResponse } from 'msw'
import * as m from './mockData'

const API = 'http://localhost/api'

// Default handlers — every endpoint the app calls gets a realistic
// response so tests fail loudly (via setup.ts's onUnhandledRequest:
// 'error') if a component starts calling something new and unmocked.
export const handlers = [
  http.get(`${API}/health`, () => HttpResponse.json({
    status: 'ok',
    broker: { type: null, connected: null, live_engine_running: false },
  })),
  http.get(`${API}/strategies`, () => HttpResponse.json(m.mockStrategies)),
  http.get(`${API}/config/defaults`, () => HttpResponse.json(m.mockConfigDefaults)),

  http.get(`${API}/runs`, () => HttpResponse.json(m.mockRuns)),
  http.get(`${API}/runs/:id`, () => HttpResponse.json({ ...m.mockRuns[0], config: {}, config_hash: 'abc', seed: 42, artifacts_dir: 'runs/run-1' })),
  http.get(`${API}/runs/:id/report`, () => HttpResponse.json({
    portfolio: { sharpe: 1.2 },
    data_points: 100,
    turnover: { avg_turnover: 0.05, total_turnover: 1.2, rebalance_count: 24 },
  })),
  http.get(`${API}/runs/:id/equity`, () => HttpResponse.json({ dates: ['2026-01-01'], values: [100000], drawdown: [0] })),
  http.post(`${API}/runs`, () => HttpResponse.json({ run_id: 'new-run-1' })),
  http.delete(`${API}/runs`, () => HttpResponse.json({ cleared: m.mockRuns.length })),
  http.post(`${API}/runs/compare`, () => HttpResponse.json({ 'run-1': { sharpe: 1.2 } })),
  http.post(`${API}/runs/walk_forward`, () => HttpResponse.json({ fold_ids: ['fold-1'], aggregate: { n_folds: 1, fold_ids: ['fold-1'], metrics: {} } })),

  http.post(`${API}/agents/step`, () => HttpResponse.json(m.mockBlackboard)),

  http.get(`${API}/live/status`, () => HttpResponse.json(m.mockLiveStatusStopped)),
  http.post(`${API}/live/start`, () => HttpResponse.json({ status: 'started' })),
  http.post(`${API}/live/stop`, () => HttpResponse.json({ status: 'stopped' })),
  http.post(`${API}/live/trigger`, () => HttpResponse.json({ cycle_id: 1, timestamp: '2026-07-21T19:49:04', orders_generated: 0, orders_submitted: 0, orders_queued: 0, orders_failed: 0, skipped: false, error: null })),
  http.get(`${API}/live/positions`, () => HttpResponse.json(m.mockPositions)),
  http.get(`${API}/live/account`, () => HttpResponse.json(m.mockAccount)),
  http.get(`${API}/live/orders`, () => HttpResponse.json(m.mockOrders)),
  http.get(`${API}/live/cycles`, () => HttpResponse.json(m.mockCycles)),
  http.delete(`${API}/live/cycles`, () => HttpResponse.json({ cleared: m.mockCycles.length })),
  http.get(`${API}/live/alerts`, () => HttpResponse.json(m.mockAlerts)),
  http.post(`${API}/live/kill-switch/reset`, () => HttpResponse.json({ reset: true, halted: false })),
  http.get(`${API}/live/approvals`, () => HttpResponse.json(m.mockApprovals)),
  http.delete(`${API}/live/approvals`, () => HttpResponse.json({ cleared: m.mockApprovals.length })),
  http.get(`${API}/live/approvals/:id`, () => HttpResponse.json(m.mockApprovalDetail)),
  http.post(`${API}/live/approvals/:id/approve`, () => HttpResponse.json({ status: 'approved', order_statuses: [] })),
  http.post(`${API}/live/approvals/:id/reject`, () => HttpResponse.json({ status: 'rejected' })),
  http.get(`${API}/live/config`, () => HttpResponse.json(m.mockLiveConfig)),
  http.put(`${API}/live/config`, () => HttpResponse.json({ status: 'updated' })),

  http.get(`${API}/llm/providers`, () => HttpResponse.json({ providers: m.mockLLMProviders })),
  http.get(`${API}/llm/config`, () => HttpResponse.json(m.mockLLMConfig)),
  http.put(`${API}/llm/config`, () => HttpResponse.json({ status: 'updated' })),
  http.get(`${API}/llm/cache/stats`, () => HttpResponse.json(m.mockCacheStats)),
  http.delete(`${API}/llm/cache`, () => HttpResponse.json({ status: 'cleared' })),
  http.get(`${API}/llm/rag/stats`, () => HttpResponse.json(m.mockRAGStats)),
  http.post(`${API}/llm/rag/ingest`, () => HttpResponse.json({ status: 'ingestion_started', message: 'Ingestion started' })),
  http.delete(`${API}/llm/rag/:collection`, () => HttpResponse.json({ status: 'deleted' })),
  // Mimics the real backend's validation: every TestRequest field is
  // optional, but FastAPI still 422s "Field required" on a truly empty
  // body — catches any regression back to the bug where the client sent
  // no body at all and the button silently "did nothing".
  http.post(`${API}/llm/test`, async ({ request }) => {
    const raw = await request.text()
    if (!raw) return HttpResponse.json({ detail: [{ msg: 'Field required' }] }, { status: 422 })
    const body = JSON.parse(raw) as { model?: string }
    return HttpResponse.json({ status: 'ok', response: 'Hello!', model: body.model ?? 'groq/llama-3.3-70b-versatile', response_time_ms: 250 })
  }),
  http.get(`${API}/llm/embedding-models`, () => HttpResponse.json(m.mockEmbeddingModels)),
  http.put(`${API}/llm/rag/embedding-model`, () => HttpResponse.json({ status: 'updated', requires_reindex: false, model: m.mockEmbeddingModels[0] })),

  http.get(`${API}/logs/tail`, () => HttpResponse.json(m.mockLogTail)),

  http.get(`${API}/memory/decisions`, () => HttpResponse.json(m.mockDecisions)),
  http.get(`${API}/memory/lessons`, () => HttpResponse.json(m.mockLessons)),

  http.get(`${API}/system/resources`, () => HttpResponse.json(m.mockSystemResources)),
  http.post(`${API}/system/restart`, () => HttpResponse.json({ status: 'restarting' })),
  http.post(`${API}/system/kill`, () => HttpResponse.json({ status: 'killing' })),
]
