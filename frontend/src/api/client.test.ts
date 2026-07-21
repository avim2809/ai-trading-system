import { describe, it, expect } from 'vitest'
import { http, HttpResponse } from 'msw'
import { server } from '../test/server'
import { api } from './client'

describe('api client', () => {
  it('health() returns the parsed response', async () => {
    const res = await api.health()
    expect(res.status).toBe('ok')
  })

  it('getLLMProviders() unwraps the {providers: [...]} envelope into a bare array', async () => {
    // Regression: the backend returns {"providers": [...]}, not a bare
    // array — a mismatch here would throw at runtime (.filter/.find on an
    // object), not just fail a type check.
    const providers = await api.getLLMProviders()
    expect(Array.isArray(providers)).toBe(true)
    expect(providers[0]?.name).toBe('groq')
  })

  it('throws with status + body text on a non-ok response', async () => {
    server.use(
      http.get('http://localhost/api/health', () =>
        HttpResponse.text('Internal Server Error', { status: 500 })),
    )
    await expect(api.health()).rejects.toThrow('500')
  })

  it('getRuns() appends a status query param only when provided', async () => {
    let capturedUrl = ''
    server.use(
      http.get('http://localhost/api/runs', ({ request }) => {
        capturedUrl = request.url
        return HttpResponse.json([])
      }),
    )
    await api.getRuns()
    expect(capturedUrl).not.toContain('status=')

    await api.getRuns('completed')
    expect(capturedUrl).toContain('status=completed')
  })

  it('tailLogs() passes the offset through as a query param', async () => {
    let capturedUrl = ''
    server.use(
      http.get('http://localhost/api/logs/tail', ({ request }) => {
        capturedUrl = request.url
        return HttpResponse.json({ lines: [], next_offset: 42, reset: false })
      }),
    )
    const res = await api.tailLogs(42)
    expect(capturedUrl).toContain('offset=42')
    expect(res.next_offset).toBe(42)
  })

  it('startLive() posts the request body as JSON', async () => {
    let body: unknown
    server.use(
      http.post('http://localhost/api/live/start', async ({ request }) => {
        body = await request.json()
        return HttpResponse.json({ status: 'started' })
      }),
    )
    await api.startLive({ broker: 'ibkr_paper', strategies: ['momentum'] })
    expect(body).toMatchObject({ broker: 'ibkr_paper', strategies: ['momentum'] })
  })

  it('testLLMConnection() always sends a JSON body, never an empty one', async () => {
    // Regression: the real backend's TestRequest fields are all optional,
    // but FastAPI still 422s "Field required" on a truly empty POST body —
    // the previous implementation sent no body at all, so clicking "Test
    // Connection" in the UI always failed before the handler even ran.
    // MSW doesn't replicate FastAPI's validation on its own, so this
    // handler does it explicitly to make sure a regression here is caught.
    let rawBody = ''
    server.use(
      http.post('http://localhost/api/llm/test', async ({ request }) => {
        rawBody = await request.text()
        if (!rawBody) return HttpResponse.json({ detail: 'Field required' }, { status: 422 })
        return HttpResponse.json({ status: 'ok', response: 'hi', model: 'groq/llama-3.3-70b-versatile', response_time_ms: 10 })
      }),
    )
    const res = await api.testLLMConnection()
    expect(rawBody).not.toBe('')
    expect(res.status).toBe('ok')
  })

  it('testLLMConnection(model) includes the model override in the request body', async () => {
    let body: unknown
    server.use(
      http.post('http://localhost/api/llm/test', async ({ request }) => {
        body = await request.json()
        return HttpResponse.json({ status: 'ok', response: 'hi', model: 'anthropic/claude-opus-4-8', response_time_ms: 10 })
      }),
    )
    await api.testLLMConnection('anthropic/claude-opus-4-8')
    expect(body).toMatchObject({ model: 'anthropic/claude-opus-4-8' })
  })

  it('getDecisions() defaults to limit=50', async () => {
    let capturedUrl = ''
    server.use(
      http.get('http://localhost/api/memory/decisions', ({ request }) => {
        capturedUrl = request.url
        return HttpResponse.json([])
      }),
    )
    await api.getDecisions()
    expect(capturedUrl).toContain('limit=50')
  })
})
