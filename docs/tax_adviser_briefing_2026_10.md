# Briefing for the Israeli tax adviser (OD-12): US-listed ETFs held by an Israeli resident

Date: 2026-10-13. Written for: the owner, to hand to an Israeli tax professional. Not tax advice. Source quality is stated for every item; "established" means an official primary source was read, "likely" a reputable secondary source, "open" means the research could not establish it and the adviser must answer.

The position being reviewed: an Israeli tax-resident individual who may hold US-listed ETFs (SPY, IEF, TLT, EFA, EEM, GLD, DBC and similar) in a US broker account (Alpaca or Interactive Brokers). The accounts are paper-trading today; nothing is live. The core book is a 60/40 SPY/IEF allocation plus a small BTC satellite.

## What the first research pass established (2026-10-13, 99 agent calls, 25 claims verified)

| # | Item | Status | Detail |
|---|---|---|---|
| 1 | US estate tax on a non-US person | established (IRS) | US-situs assets above USD 60,000 are exposed (rates up to 40%); the executor files Form 706-NA (Aug 2025 revision). The 2025 US law change raised the citizen/resident exemption only; the non-resident USD 60,000 is unchanged. |
| 2 | Estate treaty relief | established | The 1975 US-Israel convention is income-tax only. Israel is not on the IRS estate-treaty list. No relief. |
| 3 | US-listed ETFs are US-situs | established for fund-form ETFs | The look-through rule lapsed after 2011, so US-listed regulated funds are US-situs regardless of what they hold. GLD (grantor trust) and DBC (commodity pool) are structured differently: situs needs a separate answer. |
| 4 | US dividend withholding | likely | 30% statutory; 25% under the treaty for an individual. Interest-related and short-term-gain dividends are generally exempt, which matters for IEF and TLT (recalled, not read from a primary source). |
| 5 | Israeli tax on dividends | likely (secondary) | 25% (30% for a 10% shareholder). Surtax 3%; an extra 2% on capital income above about ILS 721,560 from 2025, reported to run 2025-2027. The threshold appears identical for 2025 and 2026: confirm per year. |
| 6 | Irish UCITS ETFs | likely (one blog) | Not US-situs, so no US estate exposure; about 15% US withholding inside the fund (not reclaimable). Luxembourg funds typically bear 30%. No prospectus was checked. |

## Open questions to put to the adviser (the research could not establish these)

1. Israeli reporting duties for foreign accounts and assets, 2025-2026: thresholds, annual return forms, any separate foreign-asset disclosure form. (A 2011 article gave NIS 1,768,000; that figure is stale and the claim built on it was refuted.)
2. The Israeli capital gains rate on foreign ETF sales (25% is widely quoted but its verification failed 1-2), and whether the inflationary (CPI) adjustment applies to foreign ETFs.
3. How Israel classifies foreign-domiciled ETFs and UCITS, any special regime for foreign funds, treatment of accumulating funds.
4. How the US withholding is credited against Israeli tax (foreign tax credit mechanics and limits).
5. W-8BEN requirements and renewal for an Israeli individual.
6. Whether Interactive Brokers (and Alpaca) offer Irish or Luxembourg UCITS ETFs to Israeli-resident clients, given EU PRIIPs/KID rules.
7. Estate-tax situs of GLD and DBC for a non-US person.
8. Practical estate mitigations beyond UCITS (not researched).

A second, narrower research pass on questions 1-7 is running; its result will be appended below.

## Consequences for the repo's after-tax model

`config/tax_il.yaml` (frozen, hash-pinned by the core_v1 pre-registration; do not edit) carries a US withholding placeholder of 0%. The research suggests roughly 25% for the equity ETFs and largely nothing for the bond ETFs. Any future after-tax comparison should use a new, versioned config (`config/tax_il_proposed_2026_10.yaml`, unverified, awaiting the adviser's answers) rather than editing the frozen file.
