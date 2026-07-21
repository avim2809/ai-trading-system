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
