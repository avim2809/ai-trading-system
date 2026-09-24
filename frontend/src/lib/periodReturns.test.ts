import { describe, it, expect } from 'vitest'
import {
  compoundReturns,
  bucketPeriodReturns,
  rangeReturn,
  startOfIsoWeek,
  startOfMonth,
  toIsoDate,
} from './periodReturns'

describe('compoundReturns', () => {
  it('returns 0 for an empty list', () => {
    expect(compoundReturns([])).toBe(0)
  })

  it('returns the single return unchanged for a one-element list', () => {
    expect(compoundReturns([0.1])).toBeCloseTo(0.1)
  })

  it('compounds multiple positive returns multiplicatively, not additively', () => {
    // (1.1 * 1.1) - 1 = 0.21, not 0.1 + 0.1 = 0.2 -- the whole point of
    // compounding over simple summation.
    expect(compoundReturns([0.1, 0.1])).toBeCloseTo(0.21)
  })

  it('handles a mix of gains and losses', () => {
    expect(compoundReturns([0.1, -0.1])).toBeCloseTo(-0.01)
  })
})

describe('startOfIsoWeek', () => {
  it('returns the same date when it is already a Monday', () => {
    // 2024-01-15 is a Monday.
    const result = startOfIsoWeek(new Date('2024-01-15T00:00:00Z'))
    expect(toIsoDate(result)).toBe('2024-01-15')
  })

  it('backs up to the preceding Monday for a mid-week date', () => {
    // 2024-01-17 is a Wednesday in the same week as 2024-01-15.
    const result = startOfIsoWeek(new Date('2024-01-17T00:00:00Z'))
    expect(toIsoDate(result)).toBe('2024-01-15')
  })

  it('backs up to the preceding Monday for a Sunday (end of week)', () => {
    // 2024-01-14 is a Sunday, part of the *prior* week (Monday 2024-01-08).
    const result = startOfIsoWeek(new Date('2024-01-14T00:00:00Z'))
    expect(toIsoDate(result)).toBe('2024-01-08')
  })

  it('handles a year boundary correctly', () => {
    // 2024-01-01 is itself a Monday, so the week containing it starts there
    // even though the *prior* calendar day (2023-12-31, a Sunday) is in the
    // previous ISO week.
    expect(toIsoDate(startOfIsoWeek(new Date('2024-01-01T00:00:00Z')))).toBe('2024-01-01')
    expect(toIsoDate(startOfIsoWeek(new Date('2023-12-31T00:00:00Z')))).toBe('2023-12-25')
  })
})

describe('startOfMonth', () => {
  it('returns the 1st of the month for a mid-month date', () => {
    expect(toIsoDate(startOfMonth(new Date('2024-02-15T00:00:00Z')))).toBe('2024-02-01')
  })

  it('returns the same date when already the 1st', () => {
    expect(toIsoDate(startOfMonth(new Date('2024-02-01T00:00:00Z')))).toBe('2024-02-01')
  })
})

describe('bucketPeriodReturns', () => {
  it('groups by calendar day 1:1 and sorts most-recent-first', () => {
    const buckets = bucketPeriodReturns(['2024-01-01', '2024-01-02'], [0.01, 0.02], 'day')
    expect(buckets).toEqual([
      { key: '2024-01-02', start: '2024-01-02', end: '2024-01-02', return: expect.closeTo(0.02) },
      { key: '2024-01-01', start: '2024-01-01', end: '2024-01-01', return: expect.closeTo(0.01) },
    ])
  })

  it('compounds every daily return that falls within the same ISO week', () => {
    // 2024-01-15/16 fall in the week starting Monday 2024-01-15;
    // 2024-01-22 starts the next week.
    const buckets = bucketPeriodReturns(
      ['2024-01-15', '2024-01-16', '2024-01-22'],
      [0.01, 0.02, 0.03],
      'week',
    )
    expect(buckets).toHaveLength(2)
    expect(buckets[0]).toMatchObject({ key: '2024-01-22', start: '2024-01-22', end: '2024-01-22' })
    expect(buckets[0]!.return).toBeCloseTo(0.03)
    expect(buckets[1]).toMatchObject({ key: '2024-01-15', start: '2024-01-15', end: '2024-01-16' })
    expect(buckets[1]!.return).toBeCloseTo(1.01 * 1.02 - 1)
  })

  it('splits across a month boundary even for adjacent calendar days', () => {
    const buckets = bucketPeriodReturns(['2024-01-31', '2024-02-01'], [0.01, 0.02], 'month')
    expect(buckets.map((b) => b.key)).toEqual(['2024-02', '2024-01'])
  })

  it('splits across a year boundary', () => {
    const buckets = bucketPeriodReturns(['2023-12-31', '2024-01-01'], [0.01, 0.02], 'year')
    expect(buckets.map((b) => b.key)).toEqual(['2024', '2023'])
  })

  it('start/end reflect the actual observed dates, not the calendar bucket boundary', () => {
    // Tuesday + Thursday of the week starting Monday 2024-01-15 -- sparse
    // (e.g. weekend/holiday gaps), but start/end must be the actual first
    // and last dates seen, not the week's own Monday/Sunday boundary.
    const buckets = bucketPeriodReturns(['2024-01-16', '2024-01-18'], [0.01, 0.02], 'week')
    expect(buckets[0]).toMatchObject({ key: '2024-01-15', start: '2024-01-16', end: '2024-01-18' })
  })

  it('does not require pre-sorted input', () => {
    const sorted = bucketPeriodReturns(['2024-01-01', '2024-01-02'], [0.01, 0.02], 'day')
    const unsorted = bucketPeriodReturns(['2024-01-02', '2024-01-01'], [0.02, 0.01], 'day')
    expect(unsorted).toEqual(sorted)
  })

  it('returns an empty array for an empty series', () => {
    expect(bucketPeriodReturns([], [], 'day')).toEqual([])
  })
})

describe('rangeReturn', () => {
  const dates = ['2024-01-01', '2024-01-02', '2024-01-03']
  const returns = [0.01, 0.02, 0.03]

  it('compounds only the returns within an inclusive range', () => {
    expect(rangeReturn(dates, returns, '2024-01-01', '2024-01-02')).toBeCloseTo(1.01 * 1.02 - 1)
  })

  it('includes a range boundary that matches a single date exactly', () => {
    expect(rangeReturn(dates, returns, '2024-01-02', '2024-01-02')).toBeCloseTo(0.02)
  })

  it('returns 0 when no date falls within the range', () => {
    expect(rangeReturn(dates, returns, '2024-02-01', '2024-02-28')).toBe(0)
  })

  it('returns 0 for an empty series', () => {
    expect(rangeReturn([], [], '2024-01-01', '2024-01-31')).toBe(0)
  })
})
