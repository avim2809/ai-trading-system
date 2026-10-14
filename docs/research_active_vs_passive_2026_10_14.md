# Active or passive? Evidence review for a small Israeli-resident systematic investor

Date: 2026-10-14. Deep-research workflow run `wf_f6ed816d-42d` (108 agents, 26 sources fetched, 128 claims extracted, 25 verified, 6 refuted, 9 findings after synthesis). Written for the owner as background for a decision. **Not investment or tax advice.** Raw machine-readable result: `docs/research_active_vs_passive_2026_10_14.json`.

How to read it: each finding carries a confidence and the adversarial verification vote (three verifiers per claim; two refutes kill a claim). "ESTABLISHED" passages quote primary sources; passages labelled as project records or synthesis arithmetic are the project's own material and are not independent evidence. The final section is the operating session's own assessment, kept separate from the research output.

## Coverage warning

Verified outside evidence exists only for: individual-trader performance, the multiple-testing debate, post-publication decay of equity anomalies, and unlevered trend following and risk parity. **No outside claim survived** on currency hedging for a shekel investor, Israeli bonds or TA-125 trackers, UCITS versus US-listed ETFs, the stock-bond correlation since 2022, crypto trend, time-series momentum after 2010, volatility-managed portfolios, carry, low-volatility or quality, or ML and LLM return prediction. Findings on those topics rest on project records and are low confidence.

## Summary (research output)

This is background for the owner's decision, not investment or tax advice. The verified evidence supports staying passive and putting effort into structure rather than into another strategy search. Large-sample studies find that trading lowers individual investors' net returns, mostly through costs, and that lasting skill is rare (Barber and Odean 2000; under 1% of Taiwanese day traders in Barber, Lee, Liu and Odean 2014). Backtest Sharpe ratios of algorithms built by retail users barely predicted their later results (Quantopian, 2016), and most published equity anomalies are statistically real but shrink after publication and earn little after costs (McLean and Pontiff 2016; Jensen, Kelly and Pedersen 2023; Chen 2025).

Our testing standard sits at the strict end of a field that disagrees with itself, so it is not the field norm. It is also miscalibrated in both directions:
- The deflated-Sharpe rule is far stricter than a 5% familywise test.
- It measures Sharpe against zero and gives no interest on idle cash. That flatters any portfolio heavy in bonds.

Neither problem decided the trend core. It also failed on design feasibility and on a statistically significant after-tax shortfall against 60/40 in shekels. That outcome matches the literature on unlevered long-only trend and risk parity: volatility and drawdowns fall, but equity-like returns need about 2x gross exposure. Two design errors found by our own audit made it worse: a 9% volatility target that could not be reached under a 1.0 gross cap, and sizing capital fixed at USD 100,000 while equity grew. For a shekel investor, the USD/ILS exchange rate swamps any low-volatility USD book.

The biggest open risks are structural: currency, tax and broker access for the passive core, and the unproven BTC satellite. This run verified no outside evidence on those angles.

Ranked next steps:
1. Keep the passive book.
2. Settle the currency structure of the bond sleeve before any real money goes in.
3. Settle tax and account structure with the adviser, and run the core with low realised turnover.
4. Keep BTC small and treat it as speculation, not a proven edge.
5. Recalibrate the testing standard, but only for future strategy families.
6. Test any new active idea forward-only.

## Findings

### 1. [high] vote: 3-0 (claims 0, 1, 4, 5)

**Claim.** Large-sample studies find that trading lowers individual investors' net returns compared with passive holding, mainly through costs. Lasting skill is rare: under 1% of Taiwanese day traders show it. A single winning year is mostly luck. These studies cover stock pickers and day traders in older, high-cost markets, not systematic ETF investors. They carry over to our case as evidence about costs and persistence, not as a direct test of our setup.

**Evidence.**

ESTABLISHED. Barber and Odean (2000), 'Trading Is Hazardous to Your Wealth', Journal of Finance 55(2), April 2000. Peer-reviewed. Sample: 66,465 households at a US discount broker, February 1991 to January 1997, common stocks only, about 4 stocks per household.
- The highest-turnover fifth of households (turnover above 250% a year) earned 11.4% a year net. The value-weighted market earned 17.9%, the lowest-turnover fifth 18.5%, and the Vanguard Index 500 fund 17.8%.
- The high-minus-low gap in Fama-French alpha was -0.80% a month (t = -4.59). It was still -0.468% a month with a momentum factor added.
- The average household roughly matched the market before costs (18.7% vs 17.9%) but earned 16.4% after costs. Its Fama-French alpha was -0.311% a month (3.7% a year). With momentum added this shrinks to -0.041% a month and is not significant.
- 43.4% of households beat the index after costs, which is significantly below 50%.
- The penalty came mostly from 1990s costs: about 3% round-trip commission plus about 1% bid-ask spread. Gross returns barely differed by turnover.

Barber, Lee, Liu and Odean (2014), 'The cross-section of speculator skill: Evidence from day trading', Journal of Financial Markets 18. Accepted manuscript dated April 2013. Data: complete Taiwan Stock Exchange records, 1992-2006, about 450,000 day traders in an average year.
- Fewer than 1% (1,000 to 4,000, depending on the commission assumed) predictably earn positive alpha after fees the following year. Alpha here is measured against a market index.
- About 20% of those trading more than NT$600,000 a year are profitable in a typical year.
- Repeat winners: 6.6% observed against 3.9% expected by chance (6.2% vs 3.1% among the larger traders).
- The authors write that 'most of successful day traders are merely lucky'. But they also find real skill in a small group: the top 500, ranked on the prior year's Sharpe ratio, earned 37.9 basis points a day net the next year.
- Fees modelled are commissions plus a 0.3% tax on sales. No income or capital-gains tax is modelled.

QUALIFYING EVIDENCE, named by the verifiers but not separately verified here:
- Welch (Journal of Finance 2022): the combined Robinhood portfolio had positive alpha from mid-2018 to mid-2020.
- Even-Tov et al. (2023, eToro data): zero commissions raised trading and improved net performance.
- Coval, Hirshleifer and Shumway (Review of Asset Pricing Studies 2021): skill persists among the top 10% of traders.
- Barber et al. (Review of Financial Studies 2009) replicate the aggregate losses: individuals in Taiwan lost 3.8 percentage points a year.

INTERPRETATION. For a rules-based ETF investor with near-zero commissions, the behavioural part of these results carries over only weakly. What does carry over: gross edges are small, and costs plus an Israeli 25% tax on realised gains can turn a small gross edge into a net loss. One good period, live or backtested, is weak evidence unless it persists out of sample.

**Sources:** https://faculty.haas.berkeley.edu/odean/papers%20current%20versions/individual_investor_performance_final.pdf; https://ideas.repec.org/a/eee/finmar/v18y2014icp1-24.html; https://escholarship.org/uc/item/7k75v0qx

### 2. [medium] vote: 3-0 (claims 2, 3; claim 2 rated medium by its verifier)

**Claim.** For strategies built by retail users on a quant platform, backtest Sharpe ratios barely predicted results after deployment. Authors who ran more backtesting saw a larger drop from backtest to live. This supports penalising Sharpe ratios for the number of trials, but says nothing about how large the penalty should be.

**Evidence.**

ESTABLISHED. Wiecki, Campbell, Lent and Stauth, Quantopian staff. This is practitioner research: an SSRN working paper from March 2016, published in The Journal of Investing 25(3), Fall 2016, a peer-reviewed practitioner journal. Sample: 888 US-equity algorithms, filtered from 7,152, written January to June 2015 and backtested from 2010 with modelled costs and slippage.
- Backtest Sharpe explained 2% of the variance in out-of-sample Sharpe (R² = 0.02).
- Backtest annual return was slightly negatively related to out-of-sample return (R² = 0.015).
- For comparison, the within-backtest baseline was R² = 0.21.
- The log of total backtest days correlated with the Sharpe shortfall (Spearman R² = 0.017, p < 0.0001). The authors read this as support for trial-penalised measures such as the deflated Sharpe ratio, and leave the size of the penalty to future work.

LIMITS
- The out-of-sample period was only 6 to 12 months, simulated rather than traded with real money.
- The backtest period was a calm bull market; the out-of-sample period was flat to bearish.
- The SSRN draft is internally inconsistent: it gives both 'R² < 0.025' and 'R² < 0.25', and one figure caption calls the results insignificant.
- Backtests still carried a little information in aggregate. The top 10 by backtest Sharpe reached an out-of-sample Sharpe of 0.7 (not significant), and a machine-learning model on backtest features reached R² of 0.17.
- A verifier also cites Suhonen, Lennkh and Perez (Journal of Portfolio Management, Winter 2017): a median 73% Sharpe drop from backtest to live across about 215 bank alternative-beta indices. Not separately verified here.

INTERPRETATION. Expect most of any backtested Sharpe to disappear live. That supports some trial-count penalty. It does not show whether deflating at the raw count of 843 trials, rather than the family count of 31, is the right size.

**Sources:** https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2745220

### 3. [high] vote: 3-0 (claims 7, 8, 9, 11, 14, 15); 2-1 (claim 10)

**Claim.** Our pass/fail standard sits at the strict end of a field that disagrees with itself; it is not a field norm. The strict school (López de Prado and Lewis) deliberately controls the chance of any false positive (the familywise error rate) and puts false positives ahead of false negatives. The most recent peer-reviewed work (Harvey and Liu 2020; Jensen, Kelly and Pedersen 2023; Chen 2025) argues that no single hurdle fits every dataset, that strict corrections cause costly false negatives, and that most published equity anomalies are statistically real. The two camps agree on two points: hurdles must rise with the number of trials mined from the data, and 'statistically real' is not the same as 'profitable after costs'.

**Evidence.**

ESTABLISHED

1. López de Prado and Lewis, 'Detection of False Investment Strategies Using Unsupervised Learning Methods'. SSRN 3167017, first version April 2018, revised November 2018.
- They control the chance of any false positive (FWER) rather than the false discovery rate (FDR), and put false positives first, because false positives are 'actual economic losses' while missed discoveries are 'opportunity losses'.
- They concede that FWER 'is considered overly punitive' in most fields.
- Their justification: fund managers hold a free option on false positives, and investors are only shown the best strategy.
- They count effectively independent trials, which is fewer than the raw number of trials.

2. Harvey and Liu, 'False (and Missed) Discoveries in Financial Economics'. Journal of Finance 75(5), October 2020. 'p0' below is the assumed share of strategies that are truly profitable.
- For 484 backtested CAPIQ strategies (1985-2014), neither the usual t = 2.0 nor the stricter 3.0 hurdle is optimal.
- With p0 = 10%, a 5% false-discovery target gives t = 2.4 (18% of strategies survive). Treating a false discovery as five times as costly as a missed one gives t = 2.6 (15% survive).
- With p0 = 0, the 5% cutoff is 3.2 for the 484 strategies but 4.9 for 18,113 data-mined anomalies.
- Footnote 24: the 3.0 hurdle was 'never the intention' as a universal rule.
- They call the Benjamini-Yekutieli correction 'far too conservative', though it was the only method that held its false-discovery target.
- They still find only about 0.1% (about 18) of the 18,113 anomalies true at a 10% false discovery rate.

3. Jensen, Kelly and Pedersen, 'Is There a Replication Crisis in Finance?'. Journal of Finance 78(5), October 2023. Kelly and Pedersen are AQR principals, which the paper discloses; Harvey was the discussant.
- Factors that their empirical-Bayes method accepted but the Benjamini-Yekutieli correction rejected later earned an out-of-sample CAPM alpha of 0.35% a month (t = 5.1). Information ratio over 1990-2020: 0.93 in the US and 1.10 outside it. These figures are before costs.
- For a factor with an annual Sharpe of 0.5, raising the cutoff from 1.96 to 2.78 delays investing from about 15 years to about 31.

4. Chen, 'Most claimed statistical findings in cross-sectional return predictability are likely true'. arXiv 2206.15365, version 10, November 2025. Federal Reserve Board working paper; the author lists it as forthcoming in Journal of Finance: Insights and Perspectives.
- False discovery rate is at most 25% in 8 of 9 prior studies; the exception (Jacobs-Müller) gives 32%.
- At most 8.5% (±0.9%) across 29,000 equal-weighted data-mined signals.
- Using Harvey, Liu and Zhu's own estimates, the false discovery rate at t = 2.0 is about 9%. The 3.0 hurdle throws out 44% of factors above 2.0, including factors earning about 100 basis points a month.

5. McLean and Pontiff, Journal of Finance, February 2016. Returns of published predictors shrink after publication but do not disappear. That suggests few were pure false positives.

CONFLICTS
- On how many findings are false: Chen against Harvey, Liu and Zhu (2016, 'most claimed research findings... are likely false'), Harvey and Liu (2020, about 0.1% true) and Chordia, Goyal and Saretto (Review of Financial Studies 2020, about 45% false discovery rate). Chen attributes the gap to interpretation and calibration.
- On missed discoveries: López de Prado and Lewis treat them as opportunity losses only; Harvey and Liu say both types of error matter economically.

POINTS IN THE SAME SOURCES THAT FAVOUR STRICTNESS FOR US
- Chen's own bound for value-weighted, factor-adjusted alphas (large, liquid stocks) is a 41.7% false discovery rate, with a t of about 3.3 needed for a 10% rate.
- Harvey and Liu's hurdles for data-mined sets reach 4.9.
- Jensen, Kelly and Pedersen's argument covers published factors grouped by theme, not a private search over hundreds of rule variants.

INTERPRETATION. A strict bar is defensible for a private, data-mined search with a passive fallback, because false positives cost real money, tax and time. But the free-option argument does not apply to a self-directed investor, and the field offers no consensus number. The bar should be stated as a choice, with its false-negative cost shown.

**Sources:** https://ssrn.com/abstract=3167017; https://doi.org/10.1111/jofi.12951; https://doi.org/10.1111/jofi.13249; https://arxiv.org/abs/2206.15365; https://doi.org/10.1111/jofi.12365

### 4. [medium] vote: 2-1 (claim 6); project-record and arithmetic checks were not voted

**Claim.** Our statistical gate is miscalibrated in both directions, but the trend-core verdict does not depend on it. The deflated-Sharpe rule (DSR > 0.95) is far stricter than a 5% familywise test. The trend core failed it because of the cross-trial variance term, not the trial count. But the gate measured Sharpe against zero with idle cash earning nothing, a test that a passive bond holding would also pass. On the null that matters economically, the trend core's case is weak. It also failed three point gates that do not use the DSR, including a statistically significant after-tax shortfall against 60/40.

**Evidence.**

ESTABLISHED: THE FORMULA. Bailey and López de Prado, 'The Deflated Sharpe Ratio', Journal of Portfolio Management, 2014 (SSRN 2460551).
- The benchmark Sharpe (SR*) is the expected maximum Sharpe ratio of K zero-skill trials: sqrt(V) x [(1-γ)Φ⁻¹(1-1/K) + γΦ⁻¹(1-1/(K·e))]. V is the variance of Sharpe ratios across trials; γ is Euler's constant.
- The multiplier is 1.19 at K = 5, 1.57 at K = 10, 2.09 at K = 31 and 3.21 at K = 843. It grows roughly like sqrt(ln K).
- Passing requires DSR > 0.95. The null hypothesis is a Sharpe of zero, not 'no better than 60/40 after tax'.
- Per the verification notes, the paper's Appendix A.3 says K must count independent trials, and raw counts of correlated trials overstate the expected maximum.

OUR RECORDS (REPORT.md, run 2026-10-13)
- Daily Sharpe 0.0610, which is 0.968 a year, measured against zero over the full window 1994-02-02 to 2026-09-29 (8,218 days).
- DSR 0.628 at the family count of 31 trials and 0.0075 at the raw count of 843.
- The 12 grid variants correlate at a median of 0.983, so the grid is effectively about 1.1 independent trials.

VERIFICATION ARITHMETIC (reproduced in this synthesis)
- The project's V rule in config/gates.yaml gives 6.07 times the candidate's sampling variance under zero skill. That builds in about 0.40 of annual spread in true Sharpe across the family.
- With V equal to the zero-skill sampling variance, the DSR would be 0.9996 at N = 31 and 0.988 at N = 843. Both pass.
- Under the project's V, passing at N = 31 needs an annual Sharpe of about 1.20 against zero; with zero-skill V it needs about 0.66. Under the project's V, passing also needs the effective trial count to be 9 or fewer.
- The DSR > 0.95 rule amounts to the expected maximum plus 1.645 standard errors: 3.73 SE at K = 31, against 2.94 for an exact 5% familywise (Šidák) test; 4.85 against 3.84 at K = 843.
- So when trials are independent and have zero skill, the rule's real false-positive rate is about 0.05% to 0.3%, not 5%.

THE OTHER DIRECTION (my check of the engine code)
- In src/firm/backtest/vector_engine.py, with rf = None the Sharpe is measured against zero and idle cash earns nothing.
- A long-only portfolio of bond ETFs, which earn positive carry, therefore scores a high Sharpe with no timing skill. A buy-and-hold short-Treasury ETF would also score near or above 1. In the project's ETF data, SHY returned about 1.9% a year over its history; its volatility of roughly 1-2% is from memory.
- My rough estimate, not computed from data: crediting T-bill interest on the invested part (T-bills averaged about 1.8% a year over 2005-2026 and about 2.5% over 1994-2026, from memory) gives an excess-over-cash Sharpe of roughly 0.4 to 0.6. Over 32.6 years that is a t-statistic of about 2.3 to 3.4: at best borderline against the 5% familywise cutoff at K = 31, and failing at K = 843.
- No comparison was run against the same assets held static at the same weights, so timing skill was never separated from simply holding the assets.

THE AFTER-TAX BENCHMARK GATE (G7)
- A strategy correlated with 60/40 passes only if its point estimate beats 60/40 by the minimum detectable gap: 0.55 Sharpe at 80% power over about 22 years, with a standard error of about 0.20.
- By my normal approximation, a genuine after-tax edge of +0.3 Sharpe would pass only about 10% of the time.

FAILURES THAT DO NOT USE THE DSR (FINISHER.md)
- G7: Sharpe gap of -0.56 against 60/40 after tax in shekels (95% CI -0.95 to -0.18).
- G6: the gross cap was binding on 42% of days.
- G5: stress-test breach in the February 2018 'Volmageddon' episode.
- Diversification measure (ENB) of 2.448 against a minimum of 2.5.
Each of these alone forces the lowest verdict (Tier D).

INTERPRETATION. The gate tested a null that is too easy, using mechanics that are too strict. The two errors partly offset. The verdict was decided anyway by the economics and the design.

**Sources:** https://ssrn.com/abstract=3167017; https://ssrn.com/abstract=2460551; `research/reports/core_v1/REPORT.md`; `research/reports/core_v1/FINISHER.md`; `config/gates.yaml`; `src/firm/backtest/vector_engine.py`

### 5. [medium] vote: 3-0 (claims 17, 18)

**Claim.** The trend-core outcome is what the literature on unlevered long-only trend and risk parity predicts: volatility and drawdowns fall, but returns stay low without about 2x leverage. Design errors in our own specification made it worse: an unreachable volatility target and sizing capital fixed at USD 100,000. For a shekel investor, USD/ILS exposure then swamps a low-volatility USD portfolio. The design errors shrink exposure; they do not flip the verdict.

**Evidence.**

ESTABLISHED (peer-reviewed but in-sample). Clare, Seaton, Smith and Thomas, 'The trend is our friend', Journal of Behavioral and Experimental Finance 9, 2016; accepted manuscript July 2015.
- Data: five USD index series, 1994-2015: MSCI World, MSCI EM, Citigroup World Government Bond Index, DJ-UBS commodities, and FTSE/EPRA REITs.
- A long-only, unlevered 10-month moving-average rule (assets below their average move to T-bills) raised the Sharpe ratio from 0.33 to 0.79. Volatility fell from 12.09% to 6.80%, maximum drawdown from 46.60% to 11.58%, and annual return rose from 6.61% to 8.02%. The 6-, 8- and 12-month signals gave Sharpe ratios of 0.72-0.80.
- Limits: no costs or taxes, index data rather than ETFs, many variants and no multiple-testing correction, no holdout. The data end in March 2015, so they miss 2020 and 2022.
- Unlevered inverse-volatility risk parity returned 6.59% at 5.91% volatility (Sharpe 0.67): the same return as equal weight at half the risk. The authors credit this mainly to bonds' run over 1994-2015; the bond index alone returned 5.05% at 2.99% volatility. Within each asset class, risk parity added almost nothing.
- The paper notes that proponents rely on leverage. It cites Inker (GMO white paper, 2010, practitioner research) arguing that recent decades flattered bond-heavy risk parity.
- The bond index's 2.99% volatility suggests a currency-hedged or local-currency series. That inflates the bond weight and does not carry over to an unhedged USD book held by a shekel investor.

OUR RECORDS (FINISHER.md, vol_shortfall_decomposition.out and the 2026-10-13 proposal)
- Realised volatility was 2.35% against a 9% target, 26% of target.
- Reaching the target needed 1.99x gross exposure, against a 1.0 cap.
- Sources of the volatility shortfall: the gross cap removed 52% of volatility; sizing capital fixed at USD 100,000 while equity grew removed 38%; the long/flat floor removed 22%; the per-instrument cap removed 10% and could not be met on 2,355 days.
- Some ETF closing prices were not split-adjusted, costing about 0.1-0.4 percentage points a year in the shekel results.

IN SHEKELS, 2005-2026 (REPORT.md)
- USD/ILS fell about 1.6% a year.
- The trend core's 2.25% a year in USD became 0.47% a year in shekels before tax, at 8.2% volatility with a 23% maximum drawdown. After tax: -0.24% a year, 8.4% volatility, 26% maximum drawdown.
- The 60/40 benchmark (BM2) returned 6.42% a year before tax and 5.41% after.
- The gap is in return, not just Sharpe: about 2.25% against about 8% a year in USD (60/40's USD figure is implied by its shekel return plus the FX drift). Currency movement can neither create nor close that gap.

INTERPRETATION
- The outcome has the shape the literature predicts: lower volatility and drawdown, modest return.
- The unreachable target and the fixed sizing capital are real design errors that an up-front feasibility check should have caught.
- Fixing them would scale exposure up. It would not create the roughly 0.55 after-tax Sharpe advantage that G7 required.
- What long-only trend has been shown to deliver is lower volatility and drawdowns. An unlevered investor cannot turn that into the return a core holding needs.

**Sources:** https://openaccess.city.ac.uk/id/eprint/17841/; `research/reports/core_v1/FINISHER.md`; `research/reports/core_v1/REPORT.md`; `research/reports/core_v1/diagnostics/vol_shortfall_decomposition.out`; `plan/proposals/next_research_step_2026_10_13.md`

### 6. [high] vote: 3-0 (claims 12, 13, 16)

**Claim.** Verified evidence on strategy families covers only published cross-sectional equity anomalies and factors. Most are statistically real, but they shrink after publication and earn little after trading costs, especially in large, liquid stocks and after 2005. How often a factor 'replicates' depends heavily on how it is constructed. No verified evidence in this run covers time-series momentum or trend after 2010, carry, low-volatility or quality, volatility-managed portfolios, or machine-learning and LLM return prediction.

**Evidence.**

ESTABLISHED

1. McLean and Pontiff, Journal of Finance 71(1), February 2016. 97 predictors from peer-reviewed studies, CRSP data to 2013.
- The average monthly long-short return falls from 0.582% in-sample to 0.402% out-of-sample before publication, and to 0.264% after publication.
- Both 'disappears entirely' and 'no change' are statistically rejected.
- The post-publication drop is 53% with time fixed effects.
- Returns are before costs and mostly equal-weighted. Per the verification notes, the 95% confidence interval on the share due to statistical bias runs from about 0% to 52%.

2. Jensen, Kelly and Pedersen, Journal of Finance, 2023.
- 82.4% of the originally significant US factors (98 of 119) have significant CAPM alpha; 75.6% after a Benjamini-Yekutieli correction. Estimated Bayesian false discovery rate: 0.1%.
- CONFLICT: Hou, Xue and Zhang (2020) report 35% replication, falling to 18% after a similar correction.
- Most of the gap comes from construction choices: CAPM alpha instead of raw return (on raw returns, 61.3% of the same 119 replicate), capped value weights, and one-month holding.
- The authors find post-publication decay somewhat stronger than their model predicts.

3. Chen (2025) on what 'true' means: a non-zero alpha before trading costs, information costs and arbitrage. Citing Chen and Velikov (Journal of Financial and Quantitative Analysis 58(3), 2023), he says most findings 'produce measly profits' after those costs.

VERIFICATION NOTES (not separately voted). From the Chen and Velikov abstract and full texts:
- The average anomaly's expected return after spreads, post-publication decay and the post-2005 era is about 4 basis points a month.
- Gross decay rises to about 72% when post-publication data are restricted to after 2005.

CLAIMS THAT FAILED VERIFICATION, so their figures are not relied on:
- Three standalone claims about Chen and Velikov (1-2).
- A claim built on Jensen, Kelly and Pedersen's post-sample decay estimates (0-3).
- A McLean and Pontiff '26% / 58% lower' claim read as 'about half survives live' (0-3).

INTERPRETATION. A long-only ETF account under USD 25,000 cannot harvest single-stock anomalies at low cost. Factor ETFs carry the decayed premia plus fees. With a 25% tax on realised gains, the expected net edge over a cap-weighted core is small and uncertain.

**Sources:** https://doi.org/10.1111/jofi.12365; https://doi.org/10.1111/jofi.13249; https://arxiv.org/abs/2206.15365; https://www.cambridge.org/core/journals/journal-of-financial-and-quantitative-analysis/article/zeroing-in-on-the-expected-returns-of-anomalies/945133D5A3ECEEAF466AEE91551FD225

### 7. [low] vote: none (no claim survived for this angle)

**Claim.** For a shekel-based investor, currency and tax structure, not strategy choice, are the largest unresolved design questions for the passive core. This run produced no externally verified evidence on hedged versus unhedged holdings, home bias, Israeli bonds or TA-125 trackers, Irish UCITS versus US-listed ETFs, or the stock-bond correlation since 2022. This finding therefore rests on our own records and the project's tax briefing.

**Evidence.**

OUR RECORDS
- USD/ILS fell about 1.6% a year over 2005-2026. Its volatility dominated a 2.35%-volatility USD portfolio in shekel terms: 8.2% volatility and a 23% maximum drawdown.
- The same exposure applies to the 40% IEF bond sleeve of the 60/40 core. My interpretation, still to be measured: a large share of an unhedged Treasury sleeve's risk in shekels is USD/ILS.
- The 60/40 benchmark returned 6.42% a year before tax and 5.41% after tax in shekels over 2005-2026, at 10.0-10.8% volatility with a 33% maximum drawdown (REPORT.md).
- The 92/8 replay (USD, before tax, February 2015 to September 2026) lost 15.9% in calendar 2022, when stocks and bonds fell together (allocation_forward_test_plan.md).
- The equity sleeve is US-only (SPY). This run did not test it against a global equity index.

PROJECT TAX BRIEFING (tax_adviser_briefing_2026_10.md, dated 2026-10-13). Israeli items are rated 'likely' at best and need the adviser.
- US estate tax applies to US-situs assets above USD 60,000, and there is no US-Israel estate treaty. Established from IRS sources.
- US-listed fund ETFs count as US-situs.
- Irish UCITS ETFs do not, and pay about 15% US withholding inside the fund (source: one blog).
- Treaty withholding on US dividends is 25%. Interest-related dividends, which matter for IEF and TLT, are generally exempt (recalled, not from a primary source).
- For securities in foreign currency, the inflation adjustment may follow the USD/ILS rate, and currency losses are reportedly not deductible (secondary source).

THE AFTER-TAX MODEL (REPORT.md). The core_v1 model used 0% US withholding as a placeholder, no inflation adjustment, no loss carry-forward and trade-date FX translation. After-tax comparisons are therefore provisional.

PRACTICAL ACCESS (project notes, unverified)
- Alpaca's live trading for non-US residents was described as an invite-only beta with a USD 30,000 minimum, above this account's size.
- IBKR's spot crypto via Paxos was described as US-residents-only.
- Whether TASE instruments can be traded in an IBKR paper account is unverified.

INTERPRETATION. None of this is alpha, but each item moves the investor's realised shekel results by more than any edge our research found.

**Sources:** `research/reports/core_v1/REPORT.md`; `research/reports/core_v1/FINISHER.md`; `docs/tax_adviser_briefing_2026_10.md`; `docs/allocation_forward_test_plan.md`; `docs/research_findings_beyond_equities_2026_09_30.md`

### 8. [low] vote: none (no claim survived for this angle)

**Claim.** Nothing verified in this run supports the BTC 4-week trend satellite as a proven edge. Our own analysis shows its timing gain over simply holding BTC cannot be detected statistically over any realistic horizon. Most of the 92/8 book's backtest advantage came from Bitcoin's 2015-2026 rise, and Bitcoin is a survivor picked with hindsight.

**Evidence.**

No external claim on crypto time-series momentum survived verification in this run.

OUR RECORDS
- BTC 4-week trend against BTC held at the same average exposure: Sharpe 1.35 vs 1.07 (+0.28) since February 2015. It landed in the inconclusive tier (Tier C) (edge_search_verdict_2026_09.md).
- Telling a +0.28 Sharpe gap from noise would take about 43 years at the two strategies' 0.73 correlation, or about 158 years assuming no correlation (allocation_forward_test_plan.md).
- The 92/8 replay (USD, before tax, February 2015 to September 2026) earned 12.0% a year at 10.1% volatility, with a Sharpe above cash of 0.96 and a 19.2% maximum drawdown. 60/40 alone earned 8.7% a year.
- The plan itself says most of that gap is Bitcoin's run and is not expected to repeat. It also rules out scaling the satellite up on forward-test results.
- S5 crypto cross-sectional momentum was inconclusive. Its data check confirmed that delisted coins are present in the data.

INTERPRETATION
- The satellite is mostly an 8% BTC position with a drawdown filter.
- In an Israeli taxable account, each trend exit realises a gain taxed at about 25%. That erodes the pre-tax timing advantage; the effect is not quantified.
- Where an Israeli resident could hold it is unresolved (unverified project notes: Alpaca's non-US live minimum may be about USD 30,000; IBKR spot crypto is reportedly US-only).
- A US-listed spot bitcoin ETF would add US-situs estate exposure.

**Sources:** `docs/allocation_forward_test_plan.md`; `docs/edge_search_verdict_2026_09.md`; `docs/eodhd_s5_crypto_gate_2026_10.md`

### 9. [medium] vote: n/a (synthesis and interpretation)

**Claim.** Ranked next steps for the owner's decision (my interpretation, not investment or tax advice; consistent with the owner's 2026-10-13 decision to stop active research and work on the live prerequisites):
1. Keep the passive book and put no real money into active strategies from this research program.
2. Before any real money, decide the currency structure of the defensive (bond) sleeve, measured in shekels.
3. Settle tax and account structure with the adviser, and run the core with low realised turnover.
4. Treat BTC as a small speculative allocation, not a validated edge, and do not scale it up on forward-test results.
5. Recalibrate the testing standard, only for a new strategy family and never retroactively.
6. Revisit active ideas only through forward-only tests, with feasibility proven in advance and leverage decided explicitly.

**Evidence.**

1. KEEP PASSIVE. Highest confidence.
- Our record: 0 of 11 legacy strategies had an edge, every later idea was inconclusive, and the trend core got the lowest verdict (Tier D). It had three point failures that do not involve the DSR and a significant after-tax shortfall against 60/40 (findings 4 and 5).
- Outside evidence: individual traders lose to passive after costs (finding 1), backtests barely predict live results (finding 2), and anomalies decay and earn little net (finding 6).
- Stakes: on USD 25,000, a 1 percentage point a year after-tax edge is USD 250 a year, small next to the cost of data feeds, a server and LLM APIs.
- The IBKR blended control has little information value; the project's own plan says a months-long paper race means nothing statistically. Consider retiring its LLM arm, or running it on the cheapest model that does the job.

2. CURRENCY OF THE BOND SLEEVE
- Our shekel figures: USD/ILS fell 1.6% a year, and a 2.35%-volatility USD book had 8.2% volatility in shekels.
- On pre-seal 2005-2026 data, compare three options: unhedged IEF; Israeli government bonds, nominal or CPI-linked, through TASE trackers if IBKR access is confirmed; and ILS-hedged global bonds.
- Keep equities unhedged unless that analysis says otherwise.
- The outside literature (for example Campbell, Serfaty-de Medeiros and Viceira, Journal of Finance 2010) is still to be verified.

3. TAX AND ACCOUNT STRUCTURE
- Estate tax starts above USD 60,000 of US-situs assets. That does not bind under USD 25,000, but plan the Irish UCITS route before approaching it, if IBKR offers UCITS to Israeli residents.
- Once the adviser answers, replace the 0% withholding placeholder in a new versioned tax config. The frozen file stays unchanged.
- Deferral arithmetic (mine, nominal, ignoring the inflation adjustment and surtax): at a 7% pre-tax return over 20 years, realising all gains every year at 25% compounds at about 5.25% a year, against about 5.91% if tax is deferred to the end. That is about 0.65 percentage points a year.
- In our own model the trend core paid about 0.6 percentage points a year of tax on about 0.4% a year of pre-tax shekel return, with no loss offsets modelled. Buy-and-hold 60/40 paid about 1.0 point a year on 6.4%.
- The 92/8 replay turns over about 1x NAV a year, which is high for a passive book in a taxable account. A real-money version should rebalance mostly through cash flows and with wider bands. The running forward test stays frozen.
- Confirm a live broker first: Alpaca's non-US minimum may exceed the account.

4. BTC: see finding 8.

5. TESTING STANDARD (findings 3 and 4)
- Use returns in excess of cash, with interest credited on idle cash.
- Add a comparison against the same assets held static at the same weights, to isolate timing skill.
- Compute the DSR at the effective number of independent trials, with a cross-trial variance consistent with the null. Report familywise results (Holm, Šidák) and false-discovery results (Benjamini-Hochberg, Storey) side by side.
- Replace G7's 'point estimate must exceed the minimum detectable gap' rule, which gives a genuine +0.3 Sharpe edge only about a 10% chance of passing, with an explicit decision rule.
- Or state plainly that the standard means 'passive unless the evidence is overwhelming'.
- config/gates.yaml is a protected file. Any change needs the owner, applies to a new family, and must never be used to rescue core_v1.

6. FUTURE ACTIVE IDEAS
- History to 2026-09-30 has already been examined, so retesting on it can only reject an idea, never confirm it.
- Risk-parity-style trend needs about 2x gross exposure. My arithmetic: retail margin at about the benchmark rate plus 1.5% (IBKR, small balances, from memory) would more than halve an excess-over-cash Sharpe of about 0.5 at 2x leverage.
- The owner has already ruled that any successor stays ETF long-only with gross exposure of 1.0 or less.

**Sources:** https://faculty.haas.berkeley.edu/odean/papers%20current%20versions/individual_investor_performance_final.pdf; https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2745220; https://doi.org/10.1111/jofi.12951; https://doi.org/10.1111/jofi.13249; https://openaccess.city.ac.uk/id/eprint/17841/; `research/reports/core_v1/REPORT.md`; `plan/proposals/next_research_step_2026_10_13.md`; `docs/tax_adviser_briefing_2026_10.md`; `docs/allocation_forward_test_plan.md`

## Caveats (research output)

COVERAGE. Verified outside claims exist only for angles 1-3 and the testing-standard question. No claim survived for:
- angle 4: currency hedging, home bias, Israeli bonds or TA-125 trackers, UCITS versus US-listed ETFs, and the stock-bond correlation since 2022;
- angle 5: crypto time-series momentum;
- much of angle 2: time-series momentum after 2010, long-only ETF versus long-short futures trend, carry, low-volatility and quality, volatility-managed portfolios and their critiques, and ML or LLM return prediction including LLM look-ahead bias.
Findings on those topics rest on project records and are low confidence.

AGE AND RELEVANCE OF THE RETAIL STUDIES. Barber and Odean (data 1991-96, about 3% round-trip commissions) and the Taiwan study (1992-2006) predate zero-commission retail trading. Newer evidence cited by the verifiers is more favourable to retail traders (Welch 2022; Even-Tov et al. 2023) and was not separately verified.

LIMITS OF OTHER SOURCES
- Quantopian's out-of-sample period was only 6-12 months, simulated, and in a different market regime from its backtests.
- Clare et al. is in-sample, before costs, based on indices, ends in 2015, and appears to use a currency-hedged bond index.

OUR OWN ARITHMETIC. These figures are mine, not published results: the DSR multipliers and false-positive rates, the excess-over-cash Sharpe of about 0.4-0.6, the roughly 10% G7 pass chance, the 0.65 percentage-point tax-deferral figure, and the leverage cost. The T-bill averages, SHY's volatility and IBKR margin spreads are from memory. They should be recomputed read-only from the stored return series and the pre-seal FRED data, using firm.research.data_access in a research session.

REFUTED CLAIMS. These did not survive, and no figure in this report depends on them:
- a detailed claim about the ONC clustering method and its effective trial count (0-3);
- a Jensen, Kelly and Pedersen post-sample decay claim (0-3);
- a McLean and Pontiff '26% / 58% lower' claim read as 'about half survives live' (0-3);
- three Chen and Velikov claims (1-2).
The Chen and Velikov headline figures appear only as verifier notes. The effective-trial-count point rests on the verifiers' reading of Bailey and López de Prado (2014), Appendix A.3, not on the refuted ONC claim.

CONFLICTS BETWEEN SOURCES
- Jensen, Kelly and Pedersen against Hou, Xue and Zhang on replication rates.
- Chen against Harvey, Liu and Zhu (2016), Harvey and Liu (2020) and Chordia, Goyal and Saretto (2020) on false discovery rates.
- López de Prado and Lewis against Harvey and Liu on the cost of false negatives.
- Barber and Odean against Welch on how modern retail traders fare.

INTERESTS. Kelly and Pedersen are AQR principals. Chen cites his own work with Velikov. The Quantopian authors were platform staff.

PROJECT-SPECIFIC FACTS. These come from our own reports (REPORT.md, FINISHER.md, the vol-shortfall diagnostic and the 2026-10-13 proposal). They are not independent evidence. The membership of the 31-trial family is provisional and the owner has not confirmed it.

TAX AND BROKER ACCESS. No Israeli tax or broker-access item was verified in this run; the project's own briefing rates them 'likely' at best. This report is background for the owner, not investment or tax advice.

TIME SENSITIVITY. All project results use data to 2026-09-30, which has already been examined in-sample. No data after that date was looked at. The 92/8 forward test is two weeks old and cannot yet say anything about edge.

## Open questions (research output)

- For a shekel investor, after tax, which bond sleeve gave the best risk and return over 2005-2026 (pre-seal data)? The candidates are unhedged US Treasuries (IEF), Israeli government bonds (nominal or CPI-linked, through TASE trackers if IBKR access is confirmed) and ILS-hedged global bonds. Given how stocks, bonds and the currency have moved together since 2022, does leaving equities unhedged still help? Sources to verify first: Campbell, Serfaty-de Medeiros and Viceira (Journal of Finance 2010) on global currency hedging, and practitioner work on the positive stock-bond correlation since 2022 (for example Brixton et al., AQR, Journal of Portfolio Management 2023).
- Once the adviser answers, does the after-tax ranking of US-listed versus Irish UCITS holdings change? The answers needed are: Israeli treatment of accumulating UCITS; whether the inflation adjustment follows USD/ILS and whether currency losses are deductible; capital-loss offsets and carry-forward; the foreign tax credit for 25% US withholding; and the interest-related-dividend exemption for bond ETFs. At what account size does the USD 60,000 estate threshold make UCITS the better choice? And which broker can actually serve an Israeli resident with under USD 25,000, given Alpaca's reported USD 30,000 non-US minimum and the open questions about UCITS and crypto availability at IBKR?
- What does post-publication, after-cost evidence say for the strategy families this run could not verify? These are leads from memory, unverified here: time-series momentum after 2010, long-only ETF versus long-short futures (Moskowitz, Ooi and Pedersen, Journal of Financial Economics 2012; Hurst, Ooi and Pedersen, AQR practitioner, 2017); its critiques (Kim, Tse and Wald, Journal of Financial Markets 2016; Huang, Li, Wang and Zhou, Journal of Financial Economics 2020); volatility-managed portfolios (Moreira and Muir, Journal of Finance 2017, against Cederburg et al., Journal of Financial Economics 2020); crypto momentum (Liu and Tsyvinski, Review of Financial Studies 2021; Liu, Tsyvinski and Wu, Journal of Finance 2022); and LLM look-ahead bias (for example Glasserman and Lin, 2023 working paper).
- On the economically correct comparison, how much of core_v1's 0.97 Sharpe against zero survives? Two comparisons are needed: returns in excess of T-bills with idle cash credited, and returns against the same 14 ETFs held static at the same weights. Both can be computed read-only from the stored return series, with no new trial. Separately, how often would the full gate set (DSR at N = 31 plus G7's rule that the point estimate must exceed the minimum detectable gap) reject a strategy with a genuine after-tax edge of +0.2 to +0.3 Sharpe? That tells the owner whether 'passive unless the evidence is overwhelming' is the intended standard.

## Refuted claims (did not survive verification; nothing above depends on them)

- [0-3] Raw trial counts overstate the multiple-testing penalty in this method's own terms. Trials are usually correlated, so the number of effectively independent trials K is at most the raw count N. The paper prescribes clustering trials (its ONC algorithm) into K << N cluster-level strategies and using that E[K], plus the variance of Sharpe ratios across those clusters, as the DSR inputs. It also warns that assuming a constant correlation across trials biases both E[K] and the false-positive probability. INTERPRETATION: a DSR computed at a raw count of 843 mostly-correlated variants is stricter than the authors prescribe; a clustered count (the 31 families, or fewer if the families co-move) is the closer analogue. CAVEAT: ONC was validated only by Monte Carlo on synthetic block-correlation matrices with N = 20, 40, 80 and 160, not on real backtest libraries or N in the hundreds. Source: https://ssrn.com/abstract=3167017
- [0-3] Factor alphas decay materially after the original study's sample period. The average in-sample CAPM alpha of 0.49% a month falls to 0.26% a month afterwards, a 47% decline. A GLS regression of later alpha on in-sample alpha across factors gives a slope of 0.26, against about 0.9 expected if there were no alpha-hacking or arbitrage. The authors concede that the strongest in-sample alphas may have been data-mined or arbitraged away, although 83.3% of the originally significant US factors still had positive returns after the original sample. This is a three-way CONFLICT with McLean and Pontiff (2016), who find a 58% post-publication decline, and Chen and Zimmermann (2020b), who estimate a bias-corrected drop of only 12%. A haircut of roughly half on published backtest alphas fits this paper's numbers. Source: https://doi.org/10.1111/jofi.13249
- [1-2] Published JFQA version, covering 204 US stock-market anomalies built as long-short portfolios. After accounting for effective bid-ask spreads, post-publication decay and the trading era that began in the early 2000s, the average anomaly's expected net return is about 4 bps a month (roughly 0.5% a year). The authors call this negligible even though the cost-mitigation techniques produce impressive net returns in-sample. The estimate leaves out price impact, so for larger traders it is if anything too high. SCOPE: these are cross-sectional single-stock long-short anomalies (peer-reviewed, Federal Reserve Board and Penn State authors). They are not ETF-level time-series trend, carry or allocation strategies. The paper therefore speaks to angle 2 (factor decay and costs) and does not test long-only ETF trend directly. The authors say spreads are a lower-bound cost because they match the smallest orders, so their cost model fits a sub-USD 25k account reasonably well. Source: https://www.cambridge.org/core/journals/journal-of-financial-and-quantitative-analysis/article/zeroing-in-on-the-expected-returns-of-anomalies/945133D5A3ECEEAF466AEE91551FD225
- [1-2] Picking the best anomalies or combining many signals does not rescue net returns. After data-mining controls, the strongest single anomalies net at most about 10 bps a month (roughly 1.2% a year). Several combination methods net about 20 bps a month (roughly 2.4% a year), and that is still before price impact and before any investor-level tax. VERSION CONFLICT: the May 2020 working paper says it did not study combinations ('A limitation of our study is that we do not allow for combining multiple anomalies'). The 20 bps combination figure therefore exists only in the published version, whose full text (methods, sample, turnover) was paywalled and could not be checked here. INTERPRETATION (ours, not the authors'): a high-turnover multi-signal book earning about 2.4% a year before tax, taxed at 25% on realised gains, leaves little edge over a passive core. That bears on the blended 11-strategy control. Source: https://www.cambridge.org/core/journals/journal-of-financial-and-quantitative-analysis/article/zeroing-in-on-the-expected-returns-of-anomalies/945133D5A3ECEEAF466AEE91551FD225
- [1-2] On the testing standard. The paper's data-mining adjustments are not a fixed hurdle like Bonferroni or a deflated Sharpe ratio at raw trial counts. It uses two methods: (a) a clean out-of-sample sort on in-sample statistics, and (b) empirical-Bayes shrinkage that estimates how much of the spread across anomalies is luck. Both reach the same near-zero verdict. In post-publication, post-2005 net data, only 9% of anomaly t-statistics exceed 2 in absolute value, against 5% expected under the null. Even the 90th-percentile anomaly is worth only 20 bps a month equal-weighted, or 6 bps value-weighted. INTERPRETATION: a gentler shrinkage method also finds almost nothing in recent net-of-cost data. That suggests a null result for stock anomalies is not just a by-product of an overly strict bar. Our deflated-Sharpe gate at raw trial counts is stricter than this paper's method. CROSS-CHECK FLAG: the authors' framing is 'real in-sample but decayed and eaten by costs' (they cite Chen and Zimmermann 2019 for empirical Bayes). That differs from the 'most findings are false' view behind high t-statistic hurdles, so the two should not be treated as the same argument. Figures are from the May 2020 working paper. Source: https://www.cambridge.org/core/journals/journal-of-financial-and-quantitative-analysis/article/zeroing-in-on-the-expected-returns-of-anomalies/945133D5A3ECEEAF466AEE91551FD225
- [0-3] Across 97 cross-sectional stock-return predictors from 79 peer-reviewed studies (US stocks, CRSP, sample ending 2013), average long-short extreme-quintile portfolio returns are 26% lower in the window after the original sample but before publication, and 58% lower after publication. The authors treat 26% as an upper bound on data-mining or statistical bias and attribute about 32 percentage points of the decline to publication-informed trading. Interpretation: an investor should expect roughly half of a published in-sample anomaly return to survive live. Source: https://doi.org/10.1111/jofi.12365

## Sources fetched

| Quality | Angle | URL |
|---|---|---|
| primary | Retail and individual systematic traders vs passive, after costs and Israeli tax | https://faculty.haas.berkeley.edu/odean/papers%20current%20versions/individual_investor_performance_final.pdf |
| primary | Retail and individual systematic traders vs passive, after costs and Israeli tax | https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2745220 |
| primary | Retail and individual systematic traders vs passive, after costs and Israeli tax | https://ideas.repec.org/a/eee/finmar/v18y2014icp1-24.html |
| secondary | Retail and individual systematic traders vs passive, after costs and Israeli tax | https://taxsummaries.pwc.com/israel/individual/income-determination |
| primary | What survives publication and multiple testing, and is our bar too strict? | https://ssrn.com/abstract=3167017 |
| primary | What survives publication and multiple testing, and is our bar too strict? | https://doi.org/10.1111/jofi.12951 |
| primary | What survives publication and multiple testing, and is our bar too strict? | https://doi.org/10.1111/jofi.13249 |
| primary | What survives publication and multiple testing, and is our bar too strict? | https://www.cambridge.org/core/journals/journal-of-financial-and-quantitative-analysis/article/zeroing-in-on-the-expected-returns-of-anomalies/945133D5A3ECEEAF466AEE91551FD225 |
| primary | What survives publication and multiple testing, and is our bar too strict? | https://doi.org/10.1111/jofi.12365 |
| primary | What survives publication and multiple testing, and is our bar too strict? | https://arxiv.org/abs/2206.15365 |
| primary | Trend following reality check and the core_v1 design | https://openaccess.city.ac.uk/id/eprint/17841/ |
| primary | Trend following reality check and the core_v1 design | https://ideas.repec.org/a/taf/ufajxx/v68y2012i6p75-93.html |
| primary | Trend following reality check and the core_v1 design | https://ideas.repec.org/a/eee/finmar/v30y2016icp103-124.html |
| primary | Trend following reality check and the core_v1 design | https://www.aqr.com/Insights/Research/Journal-Article/A-Century-of-Evidence-on-Trend-Following-Investing |
| primary | Trend following reality check and the core_v1 design | https://ideas.repec.org/a/eee/jfinec/v135y2020i3p774-794.html |
| primary | Passive core for a shekel-based investor | https://danielrnathan.com/assets/pdf/BZN_RAPS.pdf |
| primary | Passive core for a shekel-based investor | https://www.nber.org/papers/w13088 |
| secondary | Passive core for a shekel-based investor | https://en.globes.co.il/en/article-1001548827 |
| primary | Passive core for a shekel-based investor | https://www.pictet.com/content/dam/www/documents/publications/pictet-research-institute/Currency-hedging-publication-final-may-2026.pdf.coredownload.pdf |
| secondary | Passive core for a shekel-based investor | https://en.globes.co.il/en/article-israelis-overseas-exposure-keeps-rising-and-with-it-the-risks-1001479722 |
| primary | Passive core for a shekel-based investor | https://ideas.repec.org/a/eee/jbfina/v47y2014icp29-40.html |
| primary | BTC trend satellite evidence | https://link.springer.com/article/10.1007/s10479-019-03357-1 |
| primary | BTC trend satellite evidence | https://ideas.repec.org/a/taf/apeclt/v30y2023i12p1685-1691.html |
| primary | BTC trend satellite evidence | https://ideas.repec.org/a/eee/reveco/v93y2024ipbp858-874.html |
| primary | BTC trend satellite evidence | https://acfr.aut.ac.nz/__data/assets/pdf_file/0009/918729/Time_Series_and_Cross_Sectional_Momentum_in_the_Cryptocurrency_Market_with_IA.pdf |
| primary | BTC trend satellite evidence | https://ideas.repec.org/a/bla/finmgt/v50y2021i1p107-137.html |

## Assessment by the operating session (2026-10-14, separate from the research output)

1. **The decision stands.** The verified evidence supports the owner's 2026-10-13 decision (A+D): stay passive and work on the structure of the passive book. It adds nothing that would revive core_v1.
2. **Our testing standard was miscalibrated both ways, but the verdict does not depend on it.** The DSR > 0.95 rule is far stricter than a 5% familywise test (a real false-positive rate of roughly 0.05-0.3% under independent zero-skill trials). It also measured Sharpe against zero with idle cash earning nothing, which flatters bond-heavy books. core_v1 also failed the economically relevant gates (a significant after-tax shortfall against 60/40 in shekels, stress, robustness, Kelly). Any recalibration applies only to a future strategy family, never retroactively.
3. **The most decision-relevant questions came back unverified.** For a shekel investor the 40% IEF sleeve is mostly a USD/ILS position (about 8% volatility over 2005-2026), so the currency structure of the bond sleeve is the most important open item before real money. It needs its own check (unhedged US Treasuries versus Israeli government bonds versus ILS-hedged bonds, on pre-seal data).
4. **Broker access is a hidden constraint.** An undated snippet from the earlier tax research says Alpaca's non-US live trading is an invite-only beta with a USD 30,000 minimum. If current, an account under USD 25,000 would make IBKR the live broker. Unverified; ask Alpaca.
5. **Weak spots of this run.** Much of the verified evidence concerns stock-picking anomalies, only loosely relevant to an ETF allocator; the retail-trader studies use 1990s-2006 data from before zero commissions; several figures (an excess-over-cash Sharpe of about 0.4-0.6 for core_v1, a 0.65-point tax-deferral benefit) are the synthesis's own arithmetic and are flagged as such.
6. **Cheap follow-ups, no new trial and no ledger rows:** (a) core_v1's Sharpe in excess of T-bills and against the same 14 ETFs held static, read-only from the stored return series; (b) the bond-sleeve currency comparison above; (c) the owner asks Alpaca about live access for Israeli residents. The adviser questions in `docs/tax_adviser_briefing_2026_10.md` are unchanged.
7. **Cost note.** This run used about 10 million subagent tokens over almost three hours. For the open questions, a single targeted agent or a small read-only script is far cheaper.

Run statistics: {"angles": 5, "sourcesFetched": 26, "claimsExtracted": 128, "claimsVerified": 25, "confirmed": 19, "killed": 6, "unverified": 0, "afterSynthesis": 9, "urlDupes": 0, "budgetDropped": 4, "agentCalls": 108}
