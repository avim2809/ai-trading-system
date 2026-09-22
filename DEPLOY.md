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

## 6. Bare-metal VPS (no Docker)

Use this path if you want lower overhead, easier debugging, or direct access to
`python` / `pytest` without a container shell. The `setup.sh` script installs
every system dependency (Python 3.14, Java 17, Node.js, IB Gateway) in one shot.

### 6a. Install everything

```bash
ssh user@YOUR_VPS_IP
git clone https://github.com/avim2809/ai-trading-system.git
cd ai-trading-system
chmod +x setup.sh
./setup.sh --components all        # installs Python 3.14, IB Gateway, Node.js, all extras
cp .env.example .env
nano .env                          # fill in API keys (see §1)
```

### 6b. Headless IB Gateway with IBC

`./setup.sh` installs IB Gateway at `/opt/ibgateway`. To make it run headlessly
(no GUI, auto-login, survives daily re-auth), install **IBC** on top:

```bash
# Download the latest IBC release
IBC_VERSION=$(curl -s https://api.github.com/repos/IbcAlpha/IBC/releases/latest \
  | grep tag_name | cut -d'"' -f4)
wget -q "https://github.com/IbcAlpha/IBC/releases/download/${IBC_VERSION}/IBCLinux-${IBC_VERSION}.zip" \
  -O /tmp/ibc.zip
sudo mkdir -p /opt/ibc
sudo unzip -q /tmp/ibc.zip -d /opt/ibc
sudo chmod +x /opt/ibc/*.sh /opt/ibc/scripts/*.sh

# Create the IBC config with your credentials
mkdir -p ~/.ibc
cat > ~/.ibc/config.ini << 'EOF'
FIX=no
IbLoginId=YOUR_IBKR_USERNAME
IbPassword=YOUR_IBKR_PASSWORD
TradingMode=paper                  # paper | live
ReadOnlyLogin=no
AcceptIncomingConnectionAction=accept
HandshakeTimeout=10
EOF
chmod 600 ~/.ibc/config.ini        # credentials — keep private
```

Test it once interactively (you should see "API connection established"):

```bash
/opt/ibc/scripts/ibgateway.sh \
  /opt/ibgateway \
  ~/.ibc/config.ini \
  /opt/ibc
```

Press Ctrl-C when satisfied; systemd will manage it going forward.

### 6c. systemd services

Production runs **two independent live instances** side by side — IBKR on
`:8000` and Alpaca on `:8001` (see the pipeline overview in `CLAUDE.md`/
`AGENTS.md` for why: blended vs. sleeved capital, broker-comparison control).
Each is its own `firm-api` process, same checkout and venv, distinguished
only by `FIRM_API_PORT`/`FIRM_DATA_DIR`/`FIRM_LIVE_CONFIG`. All unit files
below live in `deploy/` — copy them rather than retyping, so a fresh box
can't drift from what's actually deployed the way this doc once did.

**IB Gateway** — name matters: it must match `ai-trading.service`'s
`After=`/`Wants=ibgateway.service` (no hyphen), or systemd silently won't
order/wait for it before starting the API:

```bash
sudo cp deploy/ibgateway.service /etc/systemd/system/ibgateway.service
# Edit paths/user if this checkout isn't at /local/store/git/ai-trading-system
# run as root — see the WorkingDirectory/ExecStart/User lines.
```

**IBKR instance** (`deploy/ai-trading.service`, port 8000):

```bash
sudo cp deploy/ai-trading.service /etc/systemd/system/ai-trading.service
# Edit paths/user if needed, then:
sudo systemctl daemon-reload
sudo systemctl enable ai-trading
```

Key settings in that unit:

- `ExecStart=.../.venv/bin/firm-api` — single process for API + web UI + live engine
- `EnvironmentFile=.../.env` — API keys and broker credentials
- `Environment=FIRM_AUTO_START_LIVE=1` — boots live from `config/live.yaml` on startup
- `After=ibgateway.service` — waits for the IB Gateway *process* to fork
- `ExecStartPre=scripts/wait_for_ibgateway.sh` — waits (up to 90s, configurable via
  `IBGATEWAY_WAIT_TIMEOUT`) for IBC's headless login to actually open the API port,
  since `After=`/`Wants=` alone don't — closes a real boot race that has silently
  stopped the live engine before (see `docs/PROJECT_CONTEXT.md` "Broker & host
  failover"). Always exits 0; never blocks `firm-api` from starting indefinitely.
- `RestartSteps=4`/`RestartMaxDelaySec=160` — geometric restart backoff (10s →
  160s) so a persistent failure doesn't hammer whatever it's failing against
  forever, while a single transient crash still recovers in 10s.

`POST /api/live/start` merges missing fields (universe, strategies, risk, `strategy_params`) from `config/live.yaml`. See [docs/PROJECT_CONTEXT.md](docs/PROJECT_CONTEXT.md).

**Alpaca instance** (`deploy/ai-trading-alpaca.service`, port 8001) — a second,
independent paper-trading instance; only set it up if you actually want both
running:

```bash
sudo cp deploy/ai-trading-alpaca.service /etc/systemd/system/ai-trading-alpaca.service
sudo systemctl daemon-reload
sudo systemctl enable ai-trading-alpaca
```

It shares the checkout/venv/`.env` with the IBKR instance but sets its own
`FIRM_API_PORT=8001`, `FIRM_DATA_DIR=data_alpaca`, and
`FIRM_LIVE_CONFIG=config/live_alpaca.yaml`, and has no IB Gateway dependency
(Alpaca's REST API needs no local gateway process). Add `ALPACA_API_KEY`/
`ALPACA_SECRET_KEY` to `.env` before starting it.

Add to `.env` (applies to both instances):

```env
FIRM_AUTO_START_LIVE=1    # set to 0 to start live manually from the dashboard
```

Xvfb (virtual framebuffer) is needed by IB Gateway's Java UI even in headless mode.
Install it if not present: `sudo apt-get install -y xvfb`.

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable ibgateway ai-trading ai-trading-alpaca
sudo systemctl start ibgateway

# Wait ~60s for Gateway to complete login, then:
sudo systemctl start ai-trading
sudo systemctl start ai-trading-alpaca   # only if running both
sudo systemctl status ai-trading      # should show "active (running)"
```

To start live manually instead of on boot, comment out the
`Environment=FIRM_AUTO_START_LIVE=1` line in your copy of the unit and use
the dashboard or `POST /api/live/start`.

### 6d. Quick firewall (direct access, no nginx yet)

```bash
sudo ufw limit 22/tcp               # SSH, rate-limited against brute-force
sudo ufw allow 8000/tcp             # IBKR instance, direct
sudo ufw allow 8001/tcp             # Alpaca instance, direct (skip if not running it)
sudo ufw enable
```

The frontend is now at **`http://YOUR_VPS_IP:8000`** (and `:8001` for Alpaca).
This is fine for initial testing; before leaving either instance running
unattended on the internet, go to §6e and put nginx + TLS + basic auth in
front, then close these two ports.

### 6e. nginx + TLS + basic auth (production — do this before going live unattended)

`firm-api` binds `127.0.0.1:<port>` only (see `run()` in `src/firm/api/app.py`)
— it controls live trading (start/stop, order approval, account data) and
must only ever be reached through a reverse proxy that adds TLS + auth, not
directly. **Basic auth is not optional for a bare-metal box exposed to the
internet** — this endpoint was briefly open with no auth in an earlier
deployment; don't repeat that.

Production fronts both instances with one nginx config: `:443` serves the
IBKR instance's frontend and proxies `/api/alpaca/` to the Alpaca instance
(so one page reaches both — see `frontend/src/api/client.ts`), and a second
server block on `:8443` serves the Alpaca instance directly. The template is
[`deploy/nginx-ai-trading.conf`](deploy/nginx-ai-trading.conf); if you're
only running the IBKR instance, delete the `location /api/alpaca/` block and
the `:8443` server block.

```bash
sudo apt-get install -y nginx apache2-utils

# HTTP basic auth — nginx workers run as www-data, so the file must stay
# group-readable by that user even after tightening permissions below.
sudo htpasswd -c /etc/nginx/.htpasswd youroperatorname
sudo chgrp www-data /etc/nginx/.htpasswd && sudo chmod 640 /etc/nginx/.htpasswd

# Rate-limit zone and site config — copied from deploy/, not retyped (see
# §6c rationale). TLS cert/key and the htpasswd file above are generated
# per-deployment, never committed — deploy/nginx-ai-trading.conf only
# references their paths.
sudo cp deploy/nginx-rate-limit.conf /etc/nginx/conf.d/rate-limit.conf
sudo cp deploy/nginx-ai-trading.conf /etc/nginx/sites-available/ai-trading
# If using a real domain instead of IP-only access, replace `server_name _;`
# in both server blocks with your domain first.

sudo ln -sf /etc/nginx/sites-available/ai-trading /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx

# Only nginx should be internet-reachable now — close the direct ports
# opened in §6d and open the two nginx ports instead.
sudo ufw allow 443/tcp && sudo ufw allow 8443/tcp
sudo ufw delete allow 8000/tcp
sudo ufw delete allow 8001/tcp   # skip if you never opened it
```

**TLS cert — pick one:**

```bash
# Option A: real domain pointed at this host (free, auto-renews)
sudo apt-get install -y certbot python3-certbot-nginx
sudo certbot --nginx -d trading.yourdomain.com

# Option B: no public domain / IP-only access (self-signed — browsers will
# warn on first visit; that's expected, not a misconfiguration)
sudo mkdir -p /etc/nginx/ssl
sudo openssl req -x509 -nodes -days 3650 -newkey rsa:2048 \
    -keyout /etc/nginx/ssl/ai-trading.key -out /etc/nginx/ssl/ai-trading.crt \
    -subj "/CN=ai-trading-system"
```

**fail2ban** — install with defaults; no app-specific jail config is needed
beyond what ships with the package:

```bash
sudo apt-get install -y fail2ban
sudo systemctl enable --now fail2ban
```

The bundled `sshd`, `nginx-http-auth`, and `nginx-limit-req` jails already
cover SSH brute-forcing and repeated failed logins/rate-limit hits against
the nginx site set up above, once nginx's logs exist for them to read.

### 6f. Backup timer

Daily backup of live-trading state (kill-switch, `live_state.db`, decision
memory, pending approvals, execution audit trail) to a second on-disk
location — protects against an accidental deletion or bad edit inside
`data/`/`data_alpaca/`, not against losing the disk itself (see
`scripts/backup_live_state.sh` for what it excludes and why).

```bash
sudo cp deploy/backup-live-state.service deploy/backup-live-state.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now backup-live-state.timer
```

Backs up to `/local/store/backups/ai-trading-live-state` by default,
retaining 14 days; override with `LIVE_STATE_BACKUP_DIR`/
`LIVE_STATE_BACKUP_RETAIN` env vars on the service if needed.

### 6g. Useful maintenance commands

```bash
# Logs
sudo journalctl -u ai-trading -f
sudo journalctl -u ai-trading-alpaca -f
sudo journalctl -u ibgateway -f

# Restart after config change
sudo systemctl restart ai-trading

# Pull latest + restart
git pull
sudo systemctl restart ai-trading

# Check Gateway connectivity
nc -zv 127.0.0.1 4002 && echo "Gateway reachable" || echo "Gateway not up"

# Start trading manually (if FIRM_AUTO_START_LIVE=0)
curl -X POST http://localhost:8000/api/live/start \
  -H "Content-Type: application/json" \
  -d '{"broker":"ibkr_paper","approval_mode":"full_auto"}'
```

---

## 7. Running AI models locally (fallback)

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
