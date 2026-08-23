import '@testing-library/jest-dom/vitest'
import { afterAll, afterEach, beforeAll } from 'vitest'
import { server } from './server'

// jsdom has no ResizeObserver implementation; recharts' ResponsiveContainer
// (used by EquityCurveChart/DrawdownChart/MonthlyHeatmap) calls it on mount
// and throws otherwise. Previously an unhandled exception that could crash
// an in-flight render depending on async timing (report vs. equity fetch
// resolution order) without failing the test that happened to be running —
// silent test-suite flakiness, not a real assertion failure.
class ResizeObserverStub {
  observe() {}
  unobserve() {}
  disconnect() {}
}
global.ResizeObserver = ResizeObserverStub

beforeAll(() => server.listen({ onUnhandledRequest: 'error' }))
afterEach(() => server.resetHandlers())
afterAll(() => server.close())
