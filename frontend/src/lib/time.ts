/** Shared timestamp formatting — always renders in Israel time (Asia/Jerusalem),
 * regardless of the viewing device's own timezone, so times read consistently
 * whether checked from a desktop in the office or a phone anywhere else.
 * Uses the IANA zone name (not a fixed UTC+3 offset) so it stays correct
 * across DST transitions (IDT/UTC+3 in summer, IST/UTC+2 in winter).
 */

const TIME_ZONE = 'Asia/Jerusalem'

/** Compact date+time, e.g. "Jul 22, 06:13 AM" — optionally with seconds. */
export function formatDateTime(iso: string, opts: { seconds?: boolean } = {}): string {
  return new Date(iso).toLocaleString('en-US', {
    timeZone: TIME_ZONE,
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
    ...(opts.seconds ? { second: '2-digit' as const } : {}),
  })
}

/** Full date+time using default locale formatting (no explicit field list). */
export function formatFullDateTime(iso: string): string {
  return new Date(iso).toLocaleString('en-US', { timeZone: TIME_ZONE })
}

/** HH:MM:SS.mmm — for the live log tail, where sub-second precision matters. */
export function formatClockWithMs(ts: string): string {
  const d = new Date(ts)
  if (Number.isNaN(d.getTime())) return ts
  return (
    d.toLocaleTimeString('en-US', {
      timeZone: TIME_ZONE,
      hour12: false,
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
    }) +
    '.' +
    String(d.getMilliseconds()).padStart(3, '0')
  )
}
