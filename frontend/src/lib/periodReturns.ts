/**
 * Pure, unit-testable helpers for turning a strategy's raw daily return
 * series (`StrategyReturnSeries` from `GET /live/attribution/history`, see
 * `api/types.ts`) into day/week/month/year period breakdowns, WTD/MTD
 * headline figures, and custom date-range totals.
 *
 * All date boundaries are computed in UTC. Callers should derive WTD/MTD
 * boundaries from the *series' own last date* (its most recent observation)
 * rather than the browser's local "now" -- see `AttributionHistory.tsx` --
 * so results match whatever the backend considers "today", regardless of
 * the viewer's clock or timezone.
 */

export type PeriodGranularity = 'day' | 'week' | 'month' | 'year'

export interface PeriodBucket {
  /** Grouping key. Sorts chronologically as a plain string (the ISO date of
   * a day/week-start, or a 'YYYY-MM'/'YYYY' prefix) -- not meant for
   * display, use `start`/`end` for that. */
  key: string
  /** First contributing date (ISO `YYYY-MM-DD`) actually observed in this bucket. */
  start: string
  /** Last contributing date (ISO `YYYY-MM-DD`) actually observed in this bucket. */
  end: string
  /** Compounded return across every daily return that fell in this bucket. */
  return: number
}

/** Compounds a list of fractional per-period returns into one total return:
 * `(1+r1)*(1+r2)*...*(1+rn) - 1`. Returns 0 (no change) for an empty list. */
export function compoundReturns(returns: number[]): number {
  return returns.reduce((acc, r) => acc * (1 + r), 1) - 1
}

/** Parses an ISO `YYYY-MM-DD` string as a UTC-midnight Date, avoiding
 * local-timezone drift from date-only arithmetic. */
function parseIsoDate(iso: string): Date {
  return new Date(`${iso}T00:00:00Z`)
}

/** Formats a Date as an ISO `YYYY-MM-DD` string (UTC) — the inverse of
 * `parseIsoDate`, exported for callers converting `startOfIsoWeek`/
 * `startOfMonth`'s Date result back into a string for `rangeReturn`. */
export function toIsoDate(d: Date): string {
  return d.toISOString().slice(0, 10)
}

/** Start (Monday, UTC midnight) of the ISO week containing `date`. */
export function startOfIsoWeek(date: Date): Date {
  const d = new Date(Date.UTC(date.getUTCFullYear(), date.getUTCMonth(), date.getUTCDate()))
  const daysSinceMonday = (d.getUTCDay() + 6) % 7 // Mon=0 .. Sun=6
  d.setUTCDate(d.getUTCDate() - daysSinceMonday)
  return d
}

/** Start (1st of the month, UTC midnight) of the calendar month containing `date`. */
export function startOfMonth(date: Date): Date {
  return new Date(Date.UTC(date.getUTCFullYear(), date.getUTCMonth(), 1))
}

function bucketKey(iso: string, granularity: PeriodGranularity): string {
  switch (granularity) {
    case 'day':
      return iso
    case 'week':
      // The Monday-start date of the week, not an ISO week *number* --
      // simpler, sorts correctly as a plain string, and doubles as a
      // ready-to-display label ("week of 2024-01-15").
      return toIsoDate(startOfIsoWeek(parseIsoDate(iso)))
    case 'month':
      return iso.slice(0, 7)
    case 'year':
      return iso.slice(0, 4)
    default:
      return iso
  }
}

/** Groups a strategy's `(dates, returns)` into calendar buckets at the given
 * granularity, compounding every daily return that falls in each bucket,
 * and returns the buckets most-recent-first. `dates`/`returns` need not be
 * pre-sorted or aligned to calendar-bucket boundaries (e.g. weekend/holiday
 * gaps are fine -- `start`/`end` reflect the actual observations, not the
 * calendar bucket's own boundaries). */
export function bucketPeriodReturns(
  dates: string[],
  returns: number[],
  granularity: PeriodGranularity,
): PeriodBucket[] {
  const buckets = new Map<string, { start: string; end: string; returns: number[] }>()
  for (let i = 0; i < dates.length; i++) {
    const iso = dates[i]
    const r = returns[i]
    if (iso === undefined || r === undefined) continue
    const key = bucketKey(iso, granularity)
    const existing = buckets.get(key)
    if (existing) {
      if (iso < existing.start) existing.start = iso
      if (iso > existing.end) existing.end = iso
      existing.returns.push(r)
    } else {
      buckets.set(key, { start: iso, end: iso, returns: [r] })
    }
  }

  return Array.from(buckets.entries())
    .map(([key, b]) => ({ key, start: b.start, end: b.end, return: compoundReturns(b.returns) }))
    .sort((a, b) => (a.key < b.key ? 1 : a.key > b.key ? -1 : 0))
}

/** Compounds only the returns whose date falls within the inclusive range
 * `[startIso, endIso]` (both ISO `YYYY-MM-DD`). Returns 0 when nothing in
 * `dates` falls in the range. */
export function rangeReturn(
  dates: string[],
  returns: number[],
  startIso: string,
  endIso: string,
): number {
  const inRange: number[] = []
  for (let i = 0; i < dates.length; i++) {
    const iso = dates[i]
    const r = returns[i]
    if (iso !== undefined && r !== undefined && iso >= startIso && iso <= endIso) {
      inRange.push(r)
    }
  }
  return compoundReturns(inRange)
}
