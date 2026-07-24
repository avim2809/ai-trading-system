/** Safe numeric display — backend may emit null for NaN/Inf via serialize_signal. */
export function fmtNum(value: number | null | undefined, digits = 2, fallback = '—'): string {
  if (value == null || Number.isNaN(value)) return fallback
  return value.toFixed(digits)
}

export function fmtPct(value: number | null | undefined, digits = 1, fallback = '—'): string {
  if (value == null || Number.isNaN(value)) return fallback
  return `${(value * 100).toFixed(digits)}%`
}

export function asString(value: unknown, fallback = ''): string {
  if (value == null) return fallback
  if (typeof value === 'string') return value
  if (typeof value === 'number' || typeof value === 'boolean') return String(value)
  try {
    return JSON.stringify(value)
  } catch {
    return fallback
  }
}
