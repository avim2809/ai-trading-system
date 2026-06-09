"""CLI for RAG document ingestion.

Usage:
    python scripts/ingest_docs.py --all
    python scripts/ingest_docs.py --sec --earnings --symbols AAPL,MSFT,GOOGL
    python scripts/ingest_docs.py --news --symbols AAPL --days 60
    python scripts/ingest_docs.py --research --topics "portfolio optimization,risk management"
    python scripts/ingest_docs.py --system
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Ensure project root is on path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))


def _load_default_symbols() -> list[str]:
    """Load default symbols from settings.yaml universe or use fallback."""
    try:
        import yaml
        config_path = _ROOT / "config" / "settings.yaml"
        if config_path.exists():
            with open(config_path) as f:
                data = yaml.safe_load(f) or {}
            # Use universe symbols if defined, otherwise common large-caps
            return data.get("universe", {}).get("symbols", [])
    except Exception:
        pass
    return ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM", "V", "JNJ"]


def _load_rag_config() -> dict:
    """Load RAG config from config/llm.yaml."""
    try:
        import yaml
        config_path = _ROOT / "config" / "llm.yaml"
        if config_path.exists():
            with open(config_path) as f:
                data = yaml.safe_load(f) or {}
            return data.get("rag", {})
    except Exception:
        pass
    return {}


def main():
    parser = argparse.ArgumentParser(description="Ingest documents into RAG vector store")
    parser.add_argument("--all", action="store_true", help="Ingest all document types")
    parser.add_argument("--sec", action="store_true", help="Ingest SEC EDGAR filings")
    parser.add_argument("--earnings", action="store_true", help="Ingest earnings transcripts")
    parser.add_argument("--news", action="store_true", help="Ingest financial news")
    parser.add_argument("--research", action="store_true", help="Ingest arXiv research papers")
    parser.add_argument("--system", action="store_true", help="Ingest system documentation")
    parser.add_argument("--symbols", type=str, default=None,
                        help="Comma-separated symbols (default: from config)")
    parser.add_argument("--years", type=str, default="2020-2024",
                        help="Year range for SEC/earnings (e.g. 2020-2024)")
    parser.add_argument("--days", type=int, default=30,
                        help="Lookback days for news (default: 30)")
    parser.add_argument("--topics", type=str, default=None,
                        help="Comma-separated research topics")
    parser.add_argument("--max-results", type=int, default=100,
                        help="Max results per source (default: 100)")

    args = parser.parse_args()

    # Determine what to ingest
    ingest_all = args.all
    if not any([args.all, args.sec, args.earnings, args.news, args.research, args.system]):
        parser.print_help()
        sys.exit(1)

    # Load config
    rag_config = _load_rag_config()
    persist_dir = rag_config.get("persist_dir", "data/vectordb")
    chunk_size = rag_config.get("chunk_size", 500)
    chunk_overlap = rag_config.get("chunk_overlap", 50)

    # Parse arguments
    symbols = args.symbols.split(",") if args.symbols else _load_default_symbols()
    year_parts = args.years.split("-")
    start_year = int(year_parts[0])
    end_year = int(year_parts[1]) if len(year_parts) > 1 else start_year
    topics = args.topics.split(",") if args.topics else None

    # Initialize store and chunker
    from firm.rag.store import VectorStore
    from firm.rag.chunker import DocumentChunker

    store = VectorStore(persist_dir=persist_dir)
    chunker = DocumentChunker(chunk_size=chunk_size, overlap=chunk_overlap)

    print(f"RAG Ingestion")
    print(f"  Vector DB: {persist_dir}")
    print(f"  Symbols: {', '.join(symbols[:10])}{'...' if len(symbols) > 10 else ''}")
    print(f"  Years: {start_year}-{end_year}")
    print()

    total_docs = 0
    start_time = time.time()

    if ingest_all or args.sec:
        print("[SEC] Ingesting EDGAR filings...")
        from firm.rag.ingestors.sec_ingestor import SECIngestor
        ingestor = SECIngestor(store, chunker)
        count = ingestor.ingest(symbols=symbols, start_year=start_year, end_year=end_year)
        print(f"  -> {count} document chunks ingested")
        total_docs += count

    if ingest_all or args.earnings:
        print("[Earnings] Ingesting call transcripts...")
        from firm.rag.ingestors.earnings_ingestor import EarningsIngestor
        ingestor = EarningsIngestor(store, chunker)
        count = ingestor.ingest(symbols=symbols, start_year=start_year, end_year=end_year)
        print(f"  -> {count} document chunks ingested")
        total_docs += count

    if ingest_all or args.news:
        print("[News] Ingesting financial news...")
        from firm.rag.ingestors.news_ingestor import NewsIngestor
        ingestor = NewsIngestor(store, chunker)
        count = ingestor.ingest(symbols=symbols, days=args.days)
        print(f"  -> {count} document chunks ingested")
        total_docs += count

    if ingest_all or args.research:
        print("[Research] Ingesting arXiv papers...")
        from firm.rag.ingestors.research_ingestor import ResearchIngestor
        ingestor = ResearchIngestor(store, chunker)
        count = ingestor.ingest(topics=topics, max_results=args.max_results)
        print(f"  -> {count} document chunks ingested")
        total_docs += count

    if ingest_all or args.system:
        print("[System] Ingesting system documentation...")
        from firm.rag.ingestors.system_ingestor import SystemIngestor
        ingestor = SystemIngestor(store, chunker)
        count = ingestor.ingest()
        print(f"  -> {count} document chunks ingested")
        total_docs += count

    elapsed = time.time() - start_time
    print()
    print(f"Done! {total_docs} total chunks ingested in {elapsed:.1f}s")
    print(f"Vector store stats: {store.stats()}")


if __name__ == "__main__":
    main()
