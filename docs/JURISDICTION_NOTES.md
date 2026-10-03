# Jurisdiction notes (Israel-resident owner)

> **NOT ADVICE. Information for discussion with a professional only. Verify with an Israeli tax professional before relying on anything here.** This memo states no tax outcome for any person and recommends no tax position. It was written offline (no browsing, no network) from the repo's own documents plus general background knowledge; every general-knowledge claim is tagged `[verify:Qn]` and points at a numbered adviser question in section 5.

**Ticket:** P2-06. **Decision it feeds:** OD-12 (signed direction: run P2-05 now on US-listed BM2, UCITS variant is a placeholder, information not advice). **Feeds:** P2-05 (`config/tax_il.yaml`), P7-01 (capital path).

## 1. Purpose and how to read this
Tax drag decides whether a modest pre-tax edge survives (G-RESEARCH 7 is after-tax), and fund domicile decides the vehicle choice (P2-01). Writing rules: nothing is stated as settled unless it is a direct quote or paraphrase of a repo document (marked **[repo]**); background knowledge is marked **[general]** and always carries a `[verify:Qn]` tag; no thresholds or rates are given that the repo does not already state. Where a number is needed and the repo has none, the memo says "open".

## 2. What the repo already says
All from `docs/research_findings_beyond_equities_2026_09_30.md` (the "brief") unless noted:
- **[repo]** The brief's own caveat: its figures are "not independently checked" (L53, listing the Israeli tax item among unchecked claims). The brief is framed for a "Small, Israeli-Resident Systematic Trading System" (L63, title).
- **[repo]** Israeli residents pay "the standard 25% domestic capital-gains rate on TASE securities, not a tax-favored rate"; the exemption applies only to non-residents (L185, repeated L211 and L242).
- **[repo]** No US-style Section 1256 mark-to-market exists in Israel for futures (L256).
- **[repo]** CFDs (any underlying) are on the brief's "don't bother" list; "trade the listed underlying via IBKR instead" (L232).
- **[repo]** Four unresolved questions (L256): (1) futures P&L characterisation; (2) FX currency-linkage loss-offset asymmetry (Section 9(13)/29); (3) whether perpetual-futures funding payments are capital gains or ordinary income; (4) whether higher-turnover strategies risk Israel Tax Authority (ITA) "trader status" reclassification from capital-gains to labour-income rates.
- **[repo]** `docs/claude-memory/user_profile.md` L17 (updated 2026-09-30): Israel resident, a Colmex CFD account (MT4 only), open to any regulated broker; a Plus500 CFD account was ruled out (no API).
- **[repo]** The source plan's figures as carried in `plan/tickets/P2-05.md` (L6, L51): 25% on realised real gains, plus a 3% surtax and an additional 2% on capital income above ILS 721,560 "per source plan; verify". These are repeated here only as "per source plan, verify current-year thresholds".
- **Line-reference discrepancy.** The P2-06 ticket cites L173 of the brief for "IBKR accepts Israeli residents". L173 is actually text on the CBOE BuyWrite Index. The nearest support is L185: IBKR is "the only verified broker connecting retail clients to TASE" (that statement is about TASE connectivity, not about account eligibility), and L53 lists IBKR TASE-in-paper-account support as not independently checked. IBKR acceptance of Israeli residents is therefore treated as unconfirmed here (Q14).

## 3. Topics
Each topic: generally true / uncertain / adviser question / parameter affected.

### 3.1 Self-reporting of foreign broker accounts and income
- **[general]** Israeli tax residents are generally taxed on worldwide income and are generally expected to report foreign-account income and gains. `[verify:Q1]`
- **[general]** There may be separate reporting duties for foreign assets/accounts above some thresholds. Thresholds not known to this memo; open. `[verify:Q2]`
- **Uncertain:** who withholds and who reports for a foreign broker (IBKR, Alpaca); whether any form-level duty applies at the owner's scale.
- **Adviser question:** Q1, Q2. **Affects:** none directly in the model; affects P7-01 operating process (record-keeping needs).

### 3.2 Capital gains versus business income
- **[repo]** Trader-status reclassification is an open question (L256, item 4).
- **[general]** Classification as capital vs business/labour income can depend on frequency, systematic nature and intent of trading; the ITA may reclassify. `[verify:Q3]`
- **Uncertain:** where the system's expected turnover sits relative to any line the ITA applies (the brief at L185 notes 20-50 bets/yr for one candidate; trend systems differ by path). The relevant turnover for each candidate path is a P2-05/P5 output, not known today.
- **Adviser question:** Q3, Q4. **Affects:** `rate_real_gain` (a reclassification changes the whole rate stack), so P2-05 should run a sensitivity with the reclassified rate once the adviser supplies it.

### 3.3 Rate stack: real gains, surtax, inflation adjustment, currency
- **[repo]** 25% on real gains (L185). **Per source plan, verify:** 3% surtax and an additional 2% on capital income above ILS 721,560.
- **[general]** "Real" gains may mean an inflation-adjusted (CPI-indexed) basis, and the gain may be computed in ILS so that currency moves enter the gain. `[verify:Q5]` `[verify:Q6]`
- **Uncertain:** the current-year surtax threshold and whether the 2% layer applies to capital income as modelled; whether the inflation adjustment applies to each instrument type (listed foreign ETF, futures); the translation date convention.
- **Questions:** Q5-Q7. **Affects:** `rate_real_gain`, `surtax_rate_1`, `surtax_rate_2`, `surtax_threshold_ils`, `inflation_adjust`.

### 3.4 US-situs estate-tax exposure for non-US persons; UCITS alternative
- **[general]** US estate tax may apply to non-US persons on US-situs assets, which can include shares of US-listed ETFs, with a nonresident exemption that is much lower than the US citizen exemption; futures margin positions at US brokers may also be raised by advisers. Whether and how any Israel-US estate treaty modifies this is not known. `[verify:Q8]`
- **[general]** An Ireland-domiciled UCITS fund is generally not a US-situs asset, which is why UCITS is often discussed as an alternative; the repo has data and broker-access gaps for UCITS (OD-12 trade-offs). `[verify:Q9]`
- **Uncertain:** the exemption amount, rates, and treaty effect (all open; no figures are given here), and whether the exposure is material at the owner's real capital (OD-01 not yet answered).
- **Questions:** Q8, Q9. **Affects:** OD-12 (vehicle choice), P2-01; the P2-05 `fund_domicile` parameter (`US`, `IE`). Not a return-model parameter unless the owner chooses to price it.

### 3.4a Mixed-vehicle note for OD-12
OD-12's direction is US-listed BM2 now with a UCITS placeholder. This memo does not decide between them. The adviser's answers to Q8-Q11 are the inputs to that decision before P7.

### 3.5 Dividend withholding by fund domicile; treaty effects
- **[general]** US-listed funds typically suffer US withholding on dividends for non-US persons, reduced under treaty where eligible; an Ireland-domiciled UCITS has its own withholding at the underlying-holdings level and a treaty/domicile effect that differs. Rates are open and are NOT stated here. `[verify:Q10]`
- **[general]** Whether Israel allows a foreign-tax credit against Israeli tax on the same dividends, and on what basis. `[verify:Q11]`
- **Questions:** Q10, Q11. **Affects:** `withholding` (by domicile `US`, `IE`), foreign-tax-credit switch in P2-05.

### 3.6 Loss offsets and ILS translation asymmetries
- **[repo]** The FX currency-linkage loss-offset asymmetry (Section 9(13)/29) is an open question (L256, item 2).
- **[general]** Capital losses may offset same-year gains and may be carried forward, possibly with restrictions on which income they can offset. `[verify:Q12]`
- **Uncertain:** whether a loss in USD terms can become a gain in ILS terms (and the reverse) and how each is treated.
- **Questions:** Q12. **Affects:** `loss_carryforward`, the in-year offset logic, the FX translation assumption.

### 3.7 Futures and perpetual-funding characterisation
- **[repo]** No Section 1256-style mark-to-market exists (L256); funding payment character is open (L256, item 3); futures P&L characterisation is open (L256, item 1).
- **[general]** Futures gains/losses and funding payments may be capital or ordinary in character, and may be subject to different offset rules. `[verify:Q13]`
- **Question:** Q13. **Affects:** treatment of futures P&L in P2-05 (placeholder only; ETF path is the baseline) and the P7-01 path choice if futures are pursued.

### 3.8 Broker and jurisdiction access
- **[repo]** IBKR is "the only verified broker connecting retail clients to TASE" (L185); TASE-in-paper-account support is unverified (L53). CFDs are excluded (L232). Colmex CFD account exists, Plus500 ruled out (`user_profile.md` L17).
- **Open:** IBKR acceptance of Israeli residents is not confirmed by any repo document (see the line-ref discrepancy in section 2); Alpaca live availability for Israeli residents is unconfirmed (OD-20). `[verify:Q14]`
- **Question:** Q14. **Affects:** P7-01, P7-02 (OD-20).

## 4. Parameter table for `config/tax_il.yaml`
`Source status`: `source-plan` (stated in the source plan as carried in `plan/tickets/P2-05.md`), `repo-brief` (stated in the brief), `agent-knowledge` (background; value open). No working value is invented: a blank means unset until the adviser answers; P2-05 must run with the parameter explicitly marked unset or as a labelled sensitivity.

| Parameter | Working value | Source status | Verify-by (adviser question) |
|---|---|---|---|
| `rate_real_gain` | 0.25 | repo-brief (L185) and source-plan | Q3, Q5 |
| `surtax_rate_1` | 0.03 | source-plan | Q6 |
| `surtax_rate_2` | 0.02 | source-plan | Q6 |
| `surtax_threshold_ils` | 721560 (per source plan; current-year value open) | source-plan | Q6 |
| `inflation_adjust` | unset (modelled as a switchable assumption) | agent-knowledge | Q5 |
| `loss_carryforward` | unset (switchable) | agent-knowledge | Q12 |
| `withholding.US` | unset (open) | agent-knowledge | Q10 |
| `withholding.IE` | unset (open) | agent-knowledge | Q10 |
| foreign-tax-credit switch | unset | agent-knowledge | Q11 |
| FX translation convention | unset (trade-date translation is the P2-05 default assumption) | agent-knowledge | Q7 |
| trader-status reclassification rate | unset (sensitivity only) | repo-brief (open question, L256) | Q3, Q4 |
| futures P&L / funding character | unset | repo-brief (open question, L256) | Q13 |
| US estate-tax exposure (US-situs assets) | not a return parameter; vehicle-choice input | agent-knowledge | Q8, Q9 |

## 5. Questions for the tax adviser
Answerable yes/no or by number where possible. Please also give the legal source for each answer.
1. Must income and gains from a foreign broker account (for example IBKR) be self-reported, and on which form and deadline?
2. Are there separate reporting duties for foreign accounts or assets, and at what thresholds?
3. At what trading frequency or pattern (number of trades per year, holding period, systematic or automated trading, use of leverage) does the ITA tend to classify trading as business or labour income? Does a daily-signal ETF rotation or monthly-rebalanced trend system fall on either side?
4. If reclassified, what rate stack applies and from which date is it applied (prospectively or retroactively)?
5. How is the "real" gain computed: is the cost basis CPI-indexed, and does the adjustment apply to foreign-listed ETFs and futures?
6. What are the current-year surtax rates and the income threshold, and does the additional layer apply to capital income as the source plan states (3% plus 2% above ILS 721,560)?
7. Is the gain computed in ILS, and at which exchange-rate dates (trade date, year end)?
8. Does US estate tax apply to the owner as a non-US person holding US-listed ETFs or futures margin at a US broker; what are the exemption and rate; does any treaty modify this?
9. Is an Ireland-domiciled UCITS ETF outside US-situs estate exposure, and does the UCITS route change the Israeli tax treatment (for example how gains and distributions are taxed)?
10. What are the effective dividend withholding rates for the owner on (a) US-listed ETFs and (b) Ireland-domiciled UCITS ETFs?
11. Is a foreign-tax credit available against Israeli tax on foreign dividends, and with what limits?
12. Can capital losses offset gains in the same year and be carried forward, against which income types, and how does the FX asymmetry under Sections 9(13)/29 affect this?
13. How are futures P&L and perpetual-futures funding payments characterised (capital or ordinary), and may losses be offset?
14. Is there any reason, tax or regulatory, why an Israeli resident could not hold a live account at IBKR or Alpaca (OD-20)?

> **NOT ADVICE. Verify with an Israeli tax professional.** Every `[verify:Qn]` tag above has a matching question in section 5.
