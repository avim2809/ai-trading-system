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

The second research pass on questions 1-7 is appended below.

## Consequences for the repo's after-tax model

`config/tax_il.yaml` (frozen, hash-pinned by the core_v1 pre-registration; do not edit) carries a US withholding placeholder of 0%. The research suggests roughly 25% for the equity ETFs and largely nothing for the bond ETFs. Any future after-tax comparison should use a new, versioned config (`config/tax_il_proposed_2026_10.yaml`, unverified, awaiting the adviser's answers) rather than editing the frozen file.

## Second research pass (2026-10-13): what it found and what it could not

Limitation first: the agent could not read any Israeli primary source (gov.il and the IBKR knowledge base returned HTTP 403; the Israel Tax Authority site and Nevo were not reached). Israeli answers are therefore "likely" at best, from secondary sources and search snippets. Treat every Israeli figure below as a lead for the adviser, not a fact.

| # | Question | Status | What was found | Source (date) |
|---|---|---|---|---|
| 1 | Israeli reporting duties for foreign accounts and assets | likely structure, unknown current thresholds and forms | Section 131 of the Income Tax Ordinance: an annual report for a resident with a foreign account above a balance threshold and income above an income threshold (a snippet gave about NIS 1.87m balance and NIS 337,000 income; about NIS 2.09m for foreign-held assets), indexed annually. Securities sales above about NIS 2.81m a year (2023 figure) trigger an annual return. Form 5328 (declaration of income sources in Israel and abroad) exists; a snippet also named Form 150 for holdings in foreign-incorporated entities. Form numbers 1301, 1322, 1325, 1399 and 1324 could not be confirmed for a foreign ETF holder. | Globes 2011-01-23 (stale, historical only); bshcpa.co.il article 2024; gov.il service page (not readable) |
| 2 | Israeli capital gains rate; CPI adjustment | 25% likely; foreign-ETF CPI treatment unknown | Real gain on securities bought from 2003-01-01: 25% (30% for a 10% shareholder). The inflationary amount is exempt to the extent it accrued after 1994. For a security denominated in foreign currency, a snippet says the exchange-rate move is deemed the index rate, so for a USD ETF the adjustment may be the USD/NIS move rather than Israeli CPI; currency losses are reportedly not deductible. No primary source confirms this for foreign-listed ETFs. | PwC Tax Summaries, last reviewed 2026-06-29 |
| 3 | Classification of foreign ETFs and UCITS; foreign tax credit | classification unknown; credit likely | Nothing found on whether Israel treats foreign-domiciled ETFs or UCITS as securities or as fund units taxed as ordinary income, or on accumulating funds. This is the highest-priority adviser question. The foreign tax credit is unilateral, capped at Israeli tax on the same income, with baskets by income type and a carry-forward of up to 5 years. With the Israeli dividend rate also 25%, a 25% US withholding would probably absorb most of the credit (the agent's inference, not sourced). Whether the surtaxes can be offset is unknown. | PwC foreign tax relief page, 2026-06-29 |
| 4 | W-8BEN | established | A signed form stays in effect to the end of the third succeeding calendar year (signed in 2026: valid to 2029-12-31) unless circumstances change; notify the withholding agent within 30 days of a change. A foreign TIN is required for financial-account holders; treaty claims go on line 10. | IRS Form W-8BEN instructions (page shows the October 2021 revision) |
| 5 | UCITS ETFs at IBKR and Alpaca for Israeli residents | IBKR likely; Alpaca unknown | IBKR blocks products without a PRIIPs KID for EEA and UK retail clients; Israel is outside both, and secondary sources describe Israelis buying Irish UCITS at IBKR, but no IBKR page explicitly lists Israel. Constraints may include market-data subscriptions, listing currency and the client entity. Nothing found on Alpaca for Israeli residents or UCITS; its non-US live trading was described as an invite-only beta with a USD 30,000 minimum. Ask Alpaca directly. | IBKR PRIIPs page (snippets only); myetfjourney.com 2026-06; brokerchooser.com |
| 6 | GLD and DBC: situs and withholding | GLD estate established; rest likely or unknown | GLD's own 10-K says its shares may be considered US-situs and would then be in a non-resident alien's US gross estate: treat GLD as exposed above USD 60,000. As a grantor trust it distributes only rarely, so dividend withholding is normally not an issue, and a non-US holder generally owes no US income tax on sale gains (full withholding paragraph not read). DBC is a commodity pool taxed as a partnership and issues a Schedule K-1; the Invesco "Qualified Notices" under Treas. Reg. 1.1446-4 could not be parsed, so its withholding rate, income character and US-situs status are unknown (the usual view is that a partnership interest in a US-organised entity is US-situs; not verified). A K-1 fund also raises Israeli reporting and credit questions that were not sourced. | SPDR Gold Trust FY2025 10-K (SEC); Invesco notice 2025-12-09 (unreadable) |

## The adviser's priority list

1. Are accumulating UCITS taxed under the ordinary 25% capital regime in Israel, or differently?
2. The current section 131 thresholds and the forms that apply to a foreign ETF holder.
3. For USD-denominated foreign ETFs, is the CPI adjustment replaced by the currency move?
4. DBC: US-situs for estate purposes, US withholding, Israeli treatment of a K-1 commodity pool. (Consider whether to hold it at all.)
5. Alpaca's offering to Israeli residents; confirm IBKR UCITS availability for an Israeli client entity.
6. Does the foreign tax credit absorb the 25% treaty withholding, and can the 2% and 3% surtaxes be offset?

Nothing here changes the repo's after-tax model: the proposed withholding values stay unwritten until the adviser answers. The frozen `config/tax_il.yaml` is untouched.
