import pandas as pd

cache = "/tmp/claude-0/-local-store-git-ai-trading-system/c4c796e1-061f-42c7-9fa9-255931a43502/scratchpad/insider_data"
events = pd.read_parquet(f"{cache}/cluster_events.parquet")
universe = {"AAPL", "MSFT", "NVDA", "GOOG", "AMZN", "META", "TSLA", "AVGO", "AMD", "CRM", "NFLX", "ADBE",
            "JPM", "GS", "BAC", "V", "MA", "JNJ", "UNH", "LLY", "XOM", "CVX", "SPY", "QQQ", "IWM"}
print("events with ticker in current 25-name universe:", events["ticker"].isin(universe).sum(), "/", len(events))
print(events[events["ticker"].isin(universe)][["ticker", "cluster_trans_date", "known_date", "n_distinct_insiders"]].to_string())

purchases = pd.read_parquet(f"{cache}/point_in_time_purchases.parquet")
print()
print("distinct tickers total:", purchases["ticker"].nunique(dropna=False))
print("unresolved ticker rows:", purchases["ticker"].isna().sum())

print()
print("tickers per issuer_cik, top 5 by count:")
print(purchases.groupby("issuer_cik")["ticker"].nunique().sort_values(ascending=False).head(5))

print()
print("events by n_distinct_insiders:")
print(events["n_distinct_insiders"].value_counts().sort_index())
