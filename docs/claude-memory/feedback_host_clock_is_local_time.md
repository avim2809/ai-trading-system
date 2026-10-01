---
name: feedback-host-clock-is-local-time
description: "Host clock and Python log timestamps are Asia/Jerusalem (UTC+3), not UTC; take UTC from `date -u` or git %ad"
metadata:
  node_type: memory
  type: feedback
  originSessionId: c4c796e1-061f-42c7-9fa9-255931a43502
  modified: 2026-09-30T19:20:53.743Z
---

The VPS runs in the **Asia/Jerusalem** time zone (IDT, UTC+3 in summer), and Python `logging` timestamps are local time. On 2026-09-30 I copied log times into pre-registration fields as if they were UTC:
- The insider prereg `PREREGISTERED_AT` label read 17:15Z. The real freeze commit was 16:57:30Z, and the run started at 16:57:44Z.
- The first cleaner `frozen_at` labels had to be corrected before the shortlist freeze.

**Why:** a freeze timestamp that is later than the run makes a pre-registration look back-dated, even when git proves the order was fine.
**How to apply:** for any audit timestamp, take UTC from `date -u +%Y-%m-%dT%H:%M:%SZ` or from the commit time (`git log --date=iso-strict`), never from a log line. The freeze commit's own time is the authoritative record. Related: [[project-eodhd-data-and-insider-verdict-sep30]].
