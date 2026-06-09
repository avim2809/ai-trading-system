# AI Multi-Agent Stock Investment Firm

A modular, reproducible **multi-agent "AI investment firm"** in Python. Ten quant
strategies feed an analyst-to-execution agent pipeline, backtested with
[Backtrader](https://www.backtrader.com/) under strict **point-in-time /
no-look-ahead** guarantees, using paid market-data APIs.

> **Status: Phase 1 (foundation).** This repository currently contains the
> scaffold, typed data contracts, the point-in-time data layer (providers, cache,
> PIT store, survivorship-aware universe), and the **shared interfaces** that the
> strategy, agent, engine, and evaluation layers will be built against in later
> phases.

## Architecture

Three layers with strict boundaries:

| Layer | Package | Responsibility |
| --- | --- | --- |
| **Quant** | `firm.strategies` | 10 pluggable alpha engines emitting standardized `Signal`s from point-in-time data. |
| **Agent / governance** | `firm.agents` | Analysts aggregate signals by domain -> bull/bear researchers debate -> PM allocates -> risk vets (veto/scale) -> execution places tagged orders, coordinated by an `Orchestrator` over a per-timestep `Blackboard`. |
| **Engine** | `firm.backtest` | A single Backtrader `Strategy` bridge invokes the orchestrator each rebalance bar; Cerebro handles fills, costs, accounting, analyzers. |

Supporting packages: `firm.data` (point-in-time data), `firm.contracts` (typed
inter-agent messages), `firm.portfolio` (ledger + attribution), `firm.eval`,
`firm.experiments`.

```
Cerebro (per bar) -> FirmStrategy.next() --rebalance--> Orchestrator -> Blackboard
   PIT DataStore (date <= now) -> Analysts -> Bull/Bear -> Debate -> PM
   -> Risk (veto/scale) -> Execution -> orders (strategy-tagged) -> Cerebro -> Ledger
```

**Look-ahead safety.** All data flows through `firm.data.pit_store.PointInTimeDataStore`,
which only ever returns rows with `time <= asof`. Strategies receive a read-only
`PitView` bound to a single decision timestamp and physically cannot see the future.

## Layout

```
src/firm/
  config.py            # pydantic-settings (.env) + YAML loader
  logging_setup.py     # structured JSON logging
  contracts/models.py  # frozen dataclasses (Signal, SignalSet, ... PortfolioState)
  data/
    providers/         # DataProvider ABC + polygon/tiingo/alphavantage/fmp adapters
    cache.py           # Parquet read/write cache
    pit_store.py       # PointInTimeDataStore + PitDataView (no look-ahead)
    universe.py        # survivorship-aware universe resolution
    schemas.py         # canonical column schemas
  strategies/          # BaseStrategy + PitView protocol + registry  (engines: later phase)
  agents/              # Agent base + analysts/research pkgs          (logic: later phase)
  portfolio/state.py   # PortfolioLedger (+ Position) attribution ledger
  eval/ experiments/   # (later phases)
config/                # settings.yaml, strategies/*.yaml, experiments/*.yaml
scripts/fetch_data.py  # CLI: build a point-in-time (date, symbol) panel into the cache
tests/                 # PIT no-look-ahead + cache round-trip unit tests
```

## Quick Install

Requires Python >= 3.10. Node.js >= 18 for the web UI (optional).

**Windows (PowerShell):**
```powershell
.\setup.ps1                      # core + web UI
.\setup.ps1 -Components all      # everything (API, live trading, LLM/RAG)
.\setup.ps1 -Components api,live # pick specific extras
.\setup.ps1 -SkipFrontend        # skip Node.js build
```

**macOS / Linux:**
```bash
chmod +x setup.sh
./setup.sh                        # core + web UI
./setup.sh --components all       # everything
./setup.sh --components api,live  # pick specific extras
./setup.sh --skip-frontend        # skip Node.js build
```

The setup script will:
1. Create a virtual environment (`.venv`)
2. Install Python dependencies (core + selected extras)
3. Create `.env` from `.env.example`
4. Create data directories (`data/cache`, `data/vectordb`, `runs`)
5. Build the frontend production bundle
6. Verify the installation

After setup, edit `.env` with your API keys.

### Manual Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev,api,live,llm]"
copy .env.example .env    # edit with your API keys
cd frontend && npm install && npm run build && cd ..
```

### Available Extras

| Extra | What it adds |
|-------|-------------|
| `api` | FastAPI web server + REST API |
| `live` | Alpaca + Interactive Brokers live/paper trading |
| `llm` | LLM agents (OpenAI, Anthropic, Ollama), RAG pipeline, ChromaDB |
| `dev` | pytest, ruff, httpx |

### API Keys (`.env`)

| Key | Purpose | Required? |
|-----|---------|-----------|
| `POLYGON_API_KEY` | Market data (prices) | For real data |
| `TIINGO_API_KEY` | Prices + news sentiment | For real data |
| `ALPHAVANTAGE_API_KEY` | Prices + news | For real data |
| `FMP_API_KEY` | Fundamentals + earnings | For real data |
| `OPENAI_API_KEY` | GPT models for AI agents | For LLM mode |
| `ANTHROPIC_API_KEY` | Claude models | For LLM mode |
| `GROQ_API_KEY` | Groq inference | For LLM mode |
| `ALPACA_API_KEY` / `SECRET` | Alpaca paper/live trading | For live trading |
| `IBKR_HOST` / `PORT` | Interactive Brokers | For live trading |

None are required for backtesting with synthetic data.

## How to run

**Fetch a point-in-time data panel into the Parquet cache:**

```powershell
python scripts/fetch_data.py --symbols AAPL,MSFT,NVDA --start 2020-01-01 --end 2021-12-31
# or, after install, the console script:
firm-fetch-data --symbols AAPL,MSFT --start 2020-01-01 --end 2021-12-31
```

**Run the test suite:**

```powershell
pytest
```

**Sanity-check imports without installing** (PowerShell):

```powershell
$env:PYTHONPATH = "src"; python -c "import firm; print(firm.__version__)"
```

## Web Interface

The system includes a full web interface for managing backtests, viewing results, and inspecting the agent pipeline.

### Quick Start

1. Install API dependencies:
   ```bash
   pip install -e ".[api]"
   ```

2. Start the server:
   ```bash
   firm-api
   ```
   This serves both the API (at `/api`) and the frontend (at `/`) on http://localhost:8000.

### Development Mode

For frontend development with hot reload:

1. Start the API server:
   ```bash
   firm-api
   ```

2. In a separate terminal, start the Vite dev server:
   ```bash
   cd frontend
   npm install
   npm run dev
   ```
   Open http://localhost:5173 (proxies API calls to :8000).

### Frontend Pages

- **Dashboard** (`/`) — View all backtest runs, status, key metrics; compare runs side-by-side
- **New Backtest** (`/new`) — Configure strategies, universe, dates, capital, risk params; launch backtests
- **Run Detail** (`/runs/:id`) — Equity curve, drawdown chart, monthly heatmap, per-strategy attribution
- **Agent Inspector** (`/inspector`) — Step through the agent pipeline: analysts → bull/bear debate → PM → risk → execution

### API Endpoints

All endpoints are under `/api`:
- `GET /api/health` — Health check
- `GET /api/strategies` — List available strategies with default params
- `GET /api/config/defaults` — Default configuration
- `GET /api/runs` — List all backtest runs
- `GET /api/runs/{id}` — Run details
- `GET /api/runs/{id}/report` — Full performance report
- `GET /api/runs/{id}/equity` — Equity curve data
- `POST /api/runs` — Launch a new backtest
- `POST /api/runs/compare` — Compare multiple runs
- `POST /api/agents/step` — Run one agent pipeline step

## Configuration

* Secrets come from `.env` via `firm.config.Settings` (pydantic-settings).
* Reproducible parameters live in `config/*.yaml`, loaded via
  `firm.config.load_settings_yaml()`, `load_strategy_config(name)`,
  `load_experiment_config(name)`. Runs are seeded for determinism.

## Engineering standards

Typed contracts, dependency-injected providers, structured JSON logging of every
signal/decision/trade, strict no-look-ahead in the PIT store, and deterministic,
config-driven runs. The Backtrader bridge keeps strategy/agent logic
engine-agnostic so the engine could be swapped later.
