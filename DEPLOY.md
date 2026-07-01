# Deploying the AI Trading System

One Docker image serves the whole app: FastAPI mounts the built React frontend, so
you only run **one container** on **one port (8000)**. No GPU is needed.

> **AI models are external (hosted).** The LLM uses Groq, and RAG embeddings +
> reranking use the **Voyage API** — no local torch, no model downloads. That keeps
> the image small and RAM low: **~1 GB is enough** (a $6/mo / 1 GB Droplet works).
> To run RAG models locally instead, see [§6 Local AI fallback](#6-running-ai-models-locally-fallback).

---

## 1. Secrets (environment variables)

Set only the keys for the features you use. The default LLM is Groq's free tier, so
`GROQ_API_KEY` is the one you most likely need.

| Variable | Needed for |
|---|---|
| `GROQ_API_KEY` | Default LLM (free tier) |
| `VOYAGE_API_KEY` | RAG embeddings + reranking (hosted Voyage; has a free tier) |
| `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` | Alternative LLM providers (optional) |
| `GEMINI_API_KEY` / `MISTRAL_API_KEY` / `DEEPSEEK_API_KEY` | Further optional LLM providers (see the provider registry) |
| `POLYGON_API_KEY` | Price data (default provider) |
| `FMP_API_KEY` | Fundamentals |
| `TIINGO_API_KEY` | Sentiment/news |
| `ALPHAVANTAGE_API_KEY` | Alt news source (optional) |
| `ALPACA_API_KEY` / `ALPACA_SECRET_KEY` | Live trading via Alpaca (only if you go live) |
| `IBKR_USERID` / `IBKR_PASSWORD` | IB Gateway login (see [§5 IBKR](#5-ibkr-gateway-connectivity)) |
| `IBKR_HOST` / `IBKR_PORT` / `IBKR_PAPER_PORT` / `IBKR_CLIENT_ID` | App → Gateway connection (set automatically by Compose) |

Create a `.env` file (never commit it — it's already in `.dockerignore`/`.gitignore`):

```env
GROQ_API_KEY=...
VOYAGE_API_KEY=...
POLYGON_API_KEY=...
FMP_API_KEY=...
TIINGO_API_KEY=...
```

---

## 2. Test locally first

```bash
docker build -t firm .
docker run --rm -p 8000:8000 --env-file .env \
  -v firm_data:/app/data -v firm_runs:/app/runs \
  firm
```

Open http://localhost:8000 — frontend + API are both there (`/api/...`, `/metrics`).
The named volumes (`firm_data`, `firm_runs`) persist Chroma/DuckDB/caches/runs across
restarts.

---

## 3. DigitalOcean Droplet (recommended — $200 / 60-day trial)

A single small Droplet runs everything, keeps state on its disk, and stays on 24/7 for
live trading later.

1. **Create the Droplet**: Ubuntu 24.04, **Basic / Regular, 1 GB RAM** (~$6/mo; the
   $200 credit covers ~33 months). AI models are hosted, so 1 GB is plenty. Add your
   SSH key. (Pick 2 GB only if you switch to the local-models fallback in §6.)
2. **Install Docker**:
   ```bash
   ssh root@YOUR_DROPLET_IP
   curl -fsSL https://get.docker.com | sh
   ```
3. **Get the code & secrets** onto the box:
   ```bash
   git clone https://github.com/avim2809/ai-trading-system.git
   cd ai-trading-system
   nano .env          # paste the keys from step 1
   ```
4. **Build & run** (auto-restart on reboot/crash):
   ```bash
   docker build -t firm .
   docker run -d --name firm --restart unless-stopped \
     -p 80:8000 --env-file .env \
     -v /srv/firm/data:/app/data -v /srv/firm/runs:/app/runs \
     firm
   ```
   State now lives in `/srv/firm/` on the Droplet — survives container rebuilds.
5. **Lock it down** (single private user — don't expose this to the world):
   ```bash
   ufw allow OpenSSH && ufw allow 80 && ufw enable
   ```
   App is now at `http://YOUR_DROPLET_IP`.

**To update later:** `git pull && docker build -t firm . && docker rm -f firm && <run cmd again>`.

### Optional: HTTPS + a domain
Point a domain's A record at the Droplet, then put Caddy in front (auto-TLS):
```bash
docker run -d --name caddy --restart unless-stopped \
  -p 443:443 -p 80:80 \
  caddy caddy reverse-proxy --from your-domain.com --to localhost:8000
```
(Run the app container on `-p 8000:8000` instead of `-p 80:8000` when fronting it with Caddy.)

---

## 4. Railway (alternative — push-to-deploy, no Linux box)

Railway auto-detects the `Dockerfile`. ~$5/mo Hobby + usage.

1. New Project → **Deploy from GitHub repo** → pick this repo.
2. **Variables**: add the keys from step 1. (`PORT` is injected automatically — the
   image already honors it.)
3. **Add a Volume** mounted at **`/app/data`** so Chroma/DuckDB/caches persist.
   (Railway gives one volume per service; put run artifacts under `data/` too, or set
   `rag.runs_dir` in `config/llm.yaml` to a path under `/app/data` if you need them
   persisted as well.)
4. Deploy. Railway gives you a public URL. Watch the metered billing once an
   always-on instance + volume are running.

---

## 5. IBKR Gateway connectivity

The app connects to IBKR's API as a **client** — it does not host the Gateway. IB
Gateway is a separate process, and two things break a naive Docker setup:

- **`127.0.0.1` inside the app container is the app container**, not the host or the
  Gateway. You must point the app at wherever the Gateway actually listens.
- **IB Gateway only accepts API connections from localhost by default**, needs a GUI to
  log in, and forces a daily re-auth.

The robust answer for a server is the included **[docker-compose.yml](docker-compose.yml)**,
which runs the app alongside the `gnzsnz/ib-gateway` image. That image runs Gateway
headless (IBC auto-login), and proxies its API to `0.0.0.0` so the app container can
reach it over the private network.

```bash
# Add to .env (the Gateway login — NOT an API key):
#   IBKR_USERID=your_ib_username
#   IBKR_PASSWORD=your_ib_password
#   IBKR_TRADING_MODE=paper        # paper | live
#   IBKR_VNC_PASSWORD=somepass     # optional, to watch the GUI over VNC

docker compose up -d --build
```

How the wiring works (already set in the compose file):

- App reaches Gateway at **`IBKR_HOST=ib-gateway`** (the service name) — never `127.0.0.1`.
- **`IBKR_PORT=4001` (live) / `IBKR_PAPER_PORT=4002` (paper)** — IB Gateway's ports.
  The app's code defaults to TWS ports (`7496/7497`), so these overrides are required.
- The API ports are **not published to the host** — only the app, on the shared
  `firmnet` network, can reach them. Keep it that way; an open IBKR API port is dangerous.
- Start in **`paper`** mode and confirm an order round-trips before switching to `live`.

**Debugging the Gateway:** if auto-login fails, tunnel VNC and watch the GUI:
`ssh -L 5900:localhost:5900 root@YOUR_DROPLET_IP`, then connect a VNC viewer to
`localhost:5900`. Check the Gateway's API settings ("Enable ActiveX and Socket Clients"
on, "Read-Only API" off) and that the trusted-IP / connection prompts are handled by IBC.

> **Alternative — Gateway runs on the host (not a container):** start the app with
> `--add-host=host.docker.internal:host-gateway` and set `IBKR_HOST=host.docker.internal`.
> You must also open the Gateway's API to non-localhost: uncheck *"Allow connections from
> localhost only"* and add the Docker bridge subnet (e.g. `172.17.0.0/16`) to Trusted IPs.
> This means running the Gateway GUI on the box — fine on your laptop, awkward on a Droplet.

> The `gnzsnz/ib-gateway` env var names can change between releases — confirm against that
> image's README if a variable doesn't take effect.

---

## Notes

- **Why a persistent volume matters:** Chroma vectordb, DuckDB/SQLite caches, the
  Parquet price cache, and run artifacts are all on local disk. Without a volume they
  vanish on every restart and you re-fetch/re-embed everything.
- **Going live (24/7 trading):** the always-on Droplet/Railway instance is what the
  APScheduler loop needs. Keep the instance private and use **paper** broker keys until
  you've validated behavior.
- **Re-indexing the vector store:** the RAG corpus must be embedded with whatever
  provider/model is configured. If you bring an old `data/vectordb` built with the local
  `all-MiniLM` model (384-dim), it is incompatible with Voyage (1024-dim) — delete it and
  re-ingest (`python scripts/ingest_docs.py`) so vectors match the active embedder.

---

## 6. Running AI models locally (fallback)

The default is hosted (Voyage) so you don't pay the RAM/size cost of local models. If
you ever want to run RAG fully local (offline, no per-call API):

1. Install the optional extra: `pip install ".[local]"` (pulls torch — large, needs ≥2 GB RAM).
   For Docker, add `local` to the extras in the `Dockerfile`'s `pip install` line.
2. In `config/llm.yaml` set:
   ```yaml
   rag:
     embedding_provider: local
     embedding_model: all-MiniLM-L6-v2
     reranker_provider: local
     reranker_model: cross-encoder/ms-marco-MiniLM-L-6-v2
   ```
3. Re-ingest so the vectors match the local model's dimensions (see the re-index note above).

The LLM itself is independent of this — it's hosted (Groq/OpenAI/Anthropic) regardless.
