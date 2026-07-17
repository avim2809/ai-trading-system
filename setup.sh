#!/usr/bin/env bash
# AI Trading System - One-command setup script (Ubuntu 26 / Linux / macOS)
#
# Usage:
#   ./setup.sh                        # install core + api
#   ./setup.sh --components all       # install everything (live + IB Gateway + IBC)
#   ./setup.sh --components live      # live trading stack + IB Gateway + IBC
#   ./setup.sh --install-services     # also write + enable systemd services (bare-metal VPS)
#   ./setup.sh --skip-ibkr            # skip IB Gateway installation
#   ./setup.sh --skip-ibc             # skip IBC download (manual gateway login instead)
#   ./setup.sh --skip-frontend        # skip Node.js build
#   ./setup.sh --skip-venv            # skip venv creation
#   ./setup.sh --skip-system          # skip apt system packages
#   ./setup.sh --uninstall            # remove everything this script installed
#
# Environment overrides:
#   IBKR_INSTALL_DIR=/custom/path     IB Gateway install location (default: /opt/ibgateway)
#   IBC_INSTALL_DIR=/custom/path      IBC install location        (default: /opt/ibc)
set -e

COMPONENTS="api"
SKIP_FRONTEND=false
SKIP_VENV=false
SKIP_SYSTEM=false
SKIP_IBKR=false
SKIP_IBC=false
INSTALL_SERVICES=false
UNINSTALL=false
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
PYTHON_MIN=(3 14)
IBKR_INSTALL_DIR="${IBKR_INSTALL_DIR:-/opt/ibgateway}"
IBC_INSTALL_DIR="${IBC_INSTALL_DIR:-/opt/ibc}"
IBKR_INSTALLER_URL="https://download2.interactivebrokers.com/installers/ibgateway/stable-standalone/ibgateway-stable-standalone-linux-x64.sh"

usage() {
    echo "Usage: ./setup.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --components <list>   Comma-separated extras to install (default: api)"
    echo "                        Values: api, live, llm, dev, all"
    echo "  --install-services    Write + enable systemd service files (bare-metal VPS)"
    echo "  --skip-ibkr           Skip IB Gateway installation"
    echo "  --skip-ibc            Skip IBC download (manual gateway login)"
    echo "  --skip-frontend       Skip Node.js / frontend build"
    echo "  --skip-venv           Skip virtual environment creation"
    echo "  --skip-system         Skip apt system package installation"
    echo "  --uninstall           Remove everything this script installed"
    echo "  --help                Show this help message"
    echo ""
    echo "Environment:"
    echo "  IBKR_INSTALL_DIR      IB Gateway install path (default: /opt/ibgateway)"
    echo "  IBC_INSTALL_DIR       IBC install path        (default: /opt/ibc)"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --components)      COMPONENTS="$2"; shift 2 ;;
        --skip-frontend)   SKIP_FRONTEND=true; shift ;;
        --skip-venv)       SKIP_VENV=true; shift ;;
        --skip-system)     SKIP_SYSTEM=true; shift ;;
        --skip-ibkr)       SKIP_IBKR=true; shift ;;
        --skip-ibc)        SKIP_IBC=true; shift ;;
        --install-services) INSTALL_SERVICES=true; shift ;;
        --uninstall)       UNINSTALL=true; shift ;;
        --help|-h)         usage; exit 0 ;;
        *)                 echo "Unknown option: $1"; echo ""; usage; exit 1 ;;
    esac
done

step()  { echo -e "\n\033[36m==> $1\033[0m"; }
ok()    { echo -e "    \033[32m[OK]\033[0m $1"; }
warn()  { echo -e "    \033[33m[!]\033[0m $1"; }
fail()  { echo -e "    \033[31m[X]\033[0m $1"; }

echo ""
echo -e "\033[34m============================================\033[0m"
echo -e "\033[34m  AI Multi-Agent Trading System\033[0m"
echo -e "\033[34m============================================\033[0m"
echo ""

# ===============================================================
# UNINSTALL
# ===============================================================
if [ "$UNINSTALL" = true ]; then
    step "Uninstalling AI Trading System"

    # Stop and disable systemd services
    for svc in firm-api ib-gateway; do
        if systemctl is-active --quiet "$svc" 2>/dev/null; then
            sudo systemctl stop "$svc"
            ok "Stopped $svc"
        fi
        if systemctl is-enabled --quiet "$svc" 2>/dev/null; then
            sudo systemctl disable "$svc"
            ok "Disabled $svc"
        fi
        if [ -f "/etc/systemd/system/${svc}.service" ]; then
            sudo rm -f "/etc/systemd/system/${svc}.service"
            ok "Removed /etc/systemd/system/${svc}.service"
        fi
    done
    command -v systemctl &>/dev/null && sudo systemctl daemon-reload

    # IBC
    if [ -d "$IBC_INSTALL_DIR" ]; then
        sudo rm -rf "$IBC_INSTALL_DIR"
        ok "Removed IBC at $IBC_INSTALL_DIR"
    fi
    if [ -f "$HOME/.ibc/config.ini" ]; then
        rm -f "$HOME/.ibc/config.ini"
        ok "Removed ~/.ibc/config.ini"
    fi

    # IB Gateway
    if [ -d "$IBKR_INSTALL_DIR" ]; then
        sudo rm -rf "$IBKR_INSTALL_DIR"
        ok "Removed IB Gateway at $IBKR_INSTALL_DIR"
    fi
    [ -f "$PROJECT_ROOT/scripts/start_ibgateway.sh" ] && rm -f "$PROJECT_ROOT/scripts/start_ibgateway.sh" && ok "Removed scripts/start_ibgateway.sh"

    # Python venv
    if [ -d "$PROJECT_ROOT/.venv" ]; then
        rm -rf "$PROJECT_ROOT/.venv"
        ok "Removed .venv"
    fi

    # Data directories — ask first
    if [ -d "$PROJECT_ROOT/data" ] || [ -d "$PROJECT_ROOT/runs" ]; then
        read -r -p "    Remove data/ and runs/ directories? [y/N] " confirm
        if [[ "$confirm" =~ ^[Yy]$ ]]; then
            rm -rf "$PROJECT_ROOT/data" "$PROJECT_ROOT/runs"
            ok "Removed data/ and runs/"
        else
            warn "Kept data/ and runs/ (manual cleanup if needed)"
        fi
    fi

    echo ""
    echo -e "\033[32m  Uninstall complete.\033[0m"
    echo "  System packages (Python, Java, Node.js, xvfb) were NOT removed."
    echo "  Remove them manually with: sudo apt-get remove python3.14 openjdk-17-jre-headless nodejs xvfb"
    echo ""
    exit 0
fi

# ===============================================================
# INSTALL
# ===============================================================

# ---------------------------------------------------------------
# 1. System packages (Ubuntu / Debian only)
# ---------------------------------------------------------------
if [ "$SKIP_SYSTEM" = false ] && command -v apt-get &>/dev/null; then
    step "Installing system packages (apt)"

    UBUNTU_CODENAME=$(. /etc/os-release 2>/dev/null && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}" || echo "")
    UBUNTU_VER=$(. /etc/os-release 2>/dev/null && echo "${VERSION_ID:-0}" || echo "0")

    sudo apt-get update -qq

    # Core build tools and libraries needed to compile Python C extensions
    sudo apt-get install -y --no-install-recommends \
        build-essential \
        gfortran \
        git \
        curl \
        wget \
        ca-certificates \
        gnupg \
        libssl-dev \
        libffi-dev \
        libopenblas-dev \
        liblapack-dev \
        libbz2-dev \
        liblzma-dev \
        libreadline-dev \
        libsqlite3-dev \
        libncurses-dev \
        libgdbm-dev \
        libdb-dev \
        uuid-dev \
        zlib1g-dev \
        tk-dev \
        liblzma-dev \
        pkg-config \
        xvfb                  # virtual display — required for headless IB Gateway
    ok "Build dependencies installed"

    # Java 17 — required by IB Gateway
    if ! command -v java &>/dev/null; then
        sudo apt-get install -y --no-install-recommends openjdk-17-jre-headless
        ok "Java installed ($(java -version 2>&1 | head -1))"
    else
        ok "Java already present ($(java -version 2>&1 | head -1))"
    fi

    # Python 3.14 — available natively on Ubuntu 26+; fall back to deadsnakes PPA
    if ! dpkg -l python3.14 &>/dev/null 2>&1 && ! command -v python3.14 &>/dev/null; then
        warn "python3.14 not found via apt — attempting deadsnakes PPA"
        sudo apt-get install -y --no-install-recommends software-properties-common
        sudo add-apt-repository -y ppa:deadsnakes/ppa
        sudo apt-get update -qq
    fi

    sudo apt-get install -y --no-install-recommends \
        python3.14 \
        python3.14-venv \
        python3.14-dev \
        python3.14-distutils 2>/dev/null || \
    sudo apt-get install -y --no-install-recommends \
        python3.14 \
        python3.14-venv \
        python3.14-dev
    ok "Python 3.14 installed"

    # Node.js — required for frontend build
    if ! command -v node &>/dev/null; then
        curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash - 2>/dev/null
        sudo apt-get install -y --no-install-recommends nodejs
        ok "Node.js installed ($(node --version))"
    else
        ok "Node.js already present ($(node --version))"
    fi
else
    [ "$SKIP_SYSTEM" = true ] && step "Skipping system packages (--skip-system)" \
                               || step "Non-apt system — skipping apt block"
fi

# ---------------------------------------------------------------
# 2. IB Gateway
# ---------------------------------------------------------------
if [[ "$COMPONENTS" == *"live"* || "$COMPONENTS" == "all" ]] && [ "$SKIP_IBKR" = false ]; then
    step "Installing IB Gateway (Interactive Brokers)"

    if [ -f "$IBKR_INSTALL_DIR/ibgateway" ] || [ -f "$IBKR_INSTALL_DIR/Jts/ibgateway" ]; then
        ok "IB Gateway already installed at $IBKR_INSTALL_DIR"
    else
        if ! command -v java &>/dev/null; then
            fail "Java not found — install Java 17+ before IB Gateway"
            fail "  Ubuntu: sudo apt-get install openjdk-17-jre-headless"
            fail "  macOS:  brew install openjdk@17"
            exit 1
        fi

        IBKR_INSTALLER="/tmp/ibgateway-installer.sh"
        echo "    Downloading IB Gateway stable standalone..."
        if curl -fsSL "$IBKR_INSTALLER_URL" -o "$IBKR_INSTALLER"; then
            ok "Installer downloaded"
        else
            fail "Download failed — check the URL or your internet connection:"
            fail "  $IBKR_INSTALLER_URL"
            fail "  Manual download: https://www.interactivebrokers.com/en/trading/ibgateway-unattended.php"
            exit 1
        fi

        chmod +x "$IBKR_INSTALLER"
        echo "    Running installer (this may take a minute)..."
        sudo mkdir -p "$IBKR_INSTALL_DIR"
        sudo sh "$IBKR_INSTALLER" -q -dir "$IBKR_INSTALL_DIR" 2>/dev/null || {
            fail "IB Gateway installer failed"
            fail "  Try running manually: sh $IBKR_INSTALLER"
            exit 1
        }
        rm -f "$IBKR_INSTALLER"
        ok "IB Gateway installed at $IBKR_INSTALL_DIR"
    fi

    # Gateway launch wrapper (used directly or by IBC)
    IBKR_WRAPPER="$PROJECT_ROOT/scripts/start_ibgateway.sh"
    if [ ! -f "$IBKR_WRAPPER" ]; then
        mkdir -p "$PROJECT_ROOT/scripts"
        cat > "$IBKR_WRAPPER" <<'WRAPPER'
#!/usr/bin/env bash
# Launch IB Gateway in paper or live mode.
# Usage: ./scripts/start_ibgateway.sh [paper|live]
MODE="${1:-paper}"
INSTALL_DIR="${IBKR_INSTALL_DIR:-/opt/ibgateway}"

if [ ! -f "$INSTALL_DIR/ibgateway" ] && [ ! -f "$INSTALL_DIR/Jts/ibgateway" ]; then
    echo "IB Gateway not found at $INSTALL_DIR"
    echo "Run: ./setup.sh --components live"
    exit 1
fi

if [ -f "$INSTALL_DIR/ibgateway" ]; then
    exec "$INSTALL_DIR/ibgateway" "$MODE"
else
    exec "$INSTALL_DIR/Jts/ibgateway" "$MODE"
fi
WRAPPER
        chmod +x "$IBKR_WRAPPER"
        ok "Launch wrapper created at scripts/start_ibgateway.sh"
    else
        ok "Launch wrapper already exists"
    fi
else
    [ "$SKIP_IBKR" = true ] && step "Skipping IB Gateway (--skip-ibkr)"
fi

# ---------------------------------------------------------------
# 3. IBC — headless auto-login for IB Gateway
# ---------------------------------------------------------------
if [[ "$COMPONENTS" == *"live"* || "$COMPONENTS" == "all" ]] && \
   [ "$SKIP_IBKR" = false ] && [ "$SKIP_IBC" = false ]; then
    step "Installing IBC (headless IB Gateway auto-login)"

    if [ -f "$IBC_INSTALL_DIR/scripts/ibgateway.sh" ]; then
        ok "IBC already installed at $IBC_INSTALL_DIR"
    else
        IBC_VERSION=$(curl -fsSL https://api.github.com/repos/IbcAlpha/IBC/releases/latest \
            2>/dev/null | grep '"tag_name"' | cut -d'"' -f4 || echo "")

        if [ -z "$IBC_VERSION" ]; then
            warn "Could not fetch latest IBC version — skipping IBC install"
            warn "  Install manually: https://github.com/IbcAlpha/IBC/releases"
        else
            IBC_ZIP="/tmp/ibc-${IBC_VERSION}.zip"
            echo "    Downloading IBC ${IBC_VERSION}..."
            if curl -fsSL \
                "https://github.com/IbcAlpha/IBC/releases/download/${IBC_VERSION}/IBCLinux-${IBC_VERSION}.zip" \
                -o "$IBC_ZIP"; then
                sudo mkdir -p "$IBC_INSTALL_DIR"
                sudo unzip -q "$IBC_ZIP" -d "$IBC_INSTALL_DIR"
                sudo chmod +x "$IBC_INSTALL_DIR"/*.sh \
                               "$IBC_INSTALL_DIR"/scripts/*.sh 2>/dev/null || true
                rm -f "$IBC_ZIP"
                ok "IBC ${IBC_VERSION} installed at $IBC_INSTALL_DIR"
            else
                warn "IBC download failed — install manually from https://github.com/IbcAlpha/IBC/releases"
            fi
        fi
    fi

    # Create IBC config template if it doesn't exist
    mkdir -p "$HOME/.ibc"
    if [ ! -f "$HOME/.ibc/config.ini" ]; then
        cat > "$HOME/.ibc/config.ini" <<'IBC_CONFIG'
# IBC configuration — fill in your IBKR credentials before starting
# Full reference: https://github.com/IbcAlpha/IBC/blob/master/userguide.md

[IBController]
FIX=no
IbLoginId=YOUR_IBKR_USERNAME
IbPassword=YOUR_IBKR_PASSWORD
TradingMode=paper                   # paper | live
ReadOnlyLogin=no
AcceptIncomingConnectionAction=accept
HandshakeTimeout=10
IbcAlpha=no
ForceTwsApiPort=0
IBC_CONFIG
        chmod 600 "$HOME/.ibc/config.ini"
        ok "IBC config template created at ~/.ibc/config.ini"
        warn "Edit ~/.ibc/config.ini and set IbLoginId / IbPassword before starting"
    else
        ok "IBC config already exists at ~/.ibc/config.ini"
    fi
else
    if [[ "$COMPONENTS" == *"live"* || "$COMPONENTS" == "all" ]] && [ "$SKIP_IBC" = true ]; then
        step "Skipping IBC (--skip-ibc) — gateway will require manual login"
    fi
fi

# ---------------------------------------------------------------
# 4. Check Python 3.14
# ---------------------------------------------------------------
step "Checking Python installation"

if command -v python3.14 &>/dev/null; then
    PYTHON=python3.14
elif command -v python3 &>/dev/null; then
    PYTHON=python3
else
    fail "Python not found. Install Python >= 3.14"
    exit 1
fi

PY_VERSION=$($PYTHON --version)
ok "Found $PY_VERSION (via $PYTHON)"

$PYTHON -c "import sys; exit(0 if sys.version_info >= (${PYTHON_MIN[0]}, ${PYTHON_MIN[1]}) else 1)" || {
    fail "Python >= ${PYTHON_MIN[0]}.${PYTHON_MIN[1]} required. Found: $PY_VERSION"
    exit 1
}

# ---------------------------------------------------------------
# 5. Virtual environment
# ---------------------------------------------------------------
if [ "$SKIP_VENV" = false ]; then
    step "Setting up virtual environment"
    if [ ! -d "$PROJECT_ROOT/.venv" ]; then
        $PYTHON -m venv "$PROJECT_ROOT/.venv"
        ok "Created .venv (Python $($PYTHON --version))"
    else
        ok ".venv already exists"
    fi
    source "$PROJECT_ROOT/.venv/bin/activate"
    ok "Activated .venv"

    python -m pip install --quiet --upgrade pip setuptools wheel
    ok "pip/setuptools/wheel up to date"
fi

# ---------------------------------------------------------------
# 6. Parse components
# ---------------------------------------------------------------
step "Parsing components: $COMPONENTS"
if [ "$COMPONENTS" = "all" ]; then
    EXTRAS="dev,api,live,llm"
else
    EXTRAS="$COMPONENTS"
fi
ok "Will install: core + [$EXTRAS]"

# ---------------------------------------------------------------
# 7. Install Python package
# ---------------------------------------------------------------
step "Installing Python dependencies"
pip install -e ".[$EXTRAS]" --quiet
ok "Python packages installed"

# ---------------------------------------------------------------
# 8. Create .env
# ---------------------------------------------------------------
step "Configuring environment"
if [ ! -f "$PROJECT_ROOT/.env" ]; then
    if [ -f "$PROJECT_ROOT/.env.example" ]; then
        cp "$PROJECT_ROOT/.env.example" "$PROJECT_ROOT/.env"
        ok "Created .env from .env.example"
        warn "Edit .env to add your API keys"
    fi
else
    ok ".env already exists"
fi

# ---------------------------------------------------------------
# 9. Create data directories
# ---------------------------------------------------------------
step "Creating data directories"
for d in data/cache data/vectordb runs; do
    mkdir -p "$PROJECT_ROOT/$d"
    ok "$d/"
done

# ---------------------------------------------------------------
# 10. Frontend build
# ---------------------------------------------------------------
if [ "$SKIP_FRONTEND" = false ]; then
    step "Building frontend"
    if command -v npm &>/dev/null; then
        cd "$PROJECT_ROOT/frontend"
        echo "    Installing npm packages..."
        npm install --silent 2>/dev/null || npm install
        ok "npm packages installed"
        echo "    Building production bundle..."
        if npm run build 2>/dev/null; then
            ok "Frontend built -> frontend/dist/"
        else
            warn "Frontend build failed (non-critical)"
        fi
        cd "$PROJECT_ROOT"
    else
        warn "Node.js/npm not found - skipping frontend build"
        warn "Install Node.js from https://nodejs.org for the web UI"
    fi
else
    step "Skipping frontend (--skip-frontend)"
fi

# ---------------------------------------------------------------
# 11. systemd services (bare-metal VPS — only with --install-services)
# ---------------------------------------------------------------
if [ "$INSTALL_SERVICES" = true ] && command -v systemctl &>/dev/null; then
    step "Installing systemd services"

    VENV_UVICORN="$PROJECT_ROOT/.venv/bin/uvicorn"

    # ib-gateway.service
    sudo tee /etc/systemd/system/ib-gateway.service > /dev/null <<SERVICE
[Unit]
Description=IB Gateway (headless via IBC)
After=network.target
StartLimitIntervalSec=300
StartLimitBurst=5

[Service]
Type=simple
User=$USER
Environment=DISPLAY=:99
ExecStartPre=/usr/bin/Xvfb :99 -screen 0 1024x768x24 -ac
ExecStart=$IBC_INSTALL_DIR/scripts/ibgateway.sh $IBKR_INSTALL_DIR $HOME/.ibc/config.ini $IBC_INSTALL_DIR
Restart=always
RestartSec=30
TimeoutStartSec=90

[Install]
WantedBy=multi-user.target
SERVICE
    ok "Created /etc/systemd/system/ib-gateway.service"

    # firm-api.service
    sudo tee /etc/systemd/system/firm-api.service > /dev/null <<SERVICE
[Unit]
Description=AI Trading System API
After=network.target ib-gateway.service
Wants=ib-gateway.service

[Service]
Type=simple
User=$USER
WorkingDirectory=$PROJECT_ROOT
EnvironmentFile=$PROJECT_ROOT/.env
ExecStart=$VENV_UVICORN firm.api.app:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
SERVICE
    ok "Created /etc/systemd/system/firm-api.service"

    sudo systemctl daemon-reload
    sudo systemctl enable ib-gateway firm-api
    ok "Services enabled (will start on next boot)"

    warn "Before starting, complete these steps:"
    warn "  1. Edit ~/.ibc/config.ini — set IbLoginId and IbPassword"
    warn "  2. sudo systemctl start ib-gateway"
    warn "  3. Wait ~60s for Gateway login, then: sudo systemctl start firm-api"
    warn "  Monitor: sudo journalctl -u firm-api -f"
elif [ "$INSTALL_SERVICES" = true ]; then
    warn "systemd not found — skipping service installation"
fi

# ---------------------------------------------------------------
# 12. Verify
# ---------------------------------------------------------------
step "Verifying installation"
ALL_PASSED=true

verify() {
    local name="$1" cmd="$2"
    result=$(eval "$cmd" 2>&1) && ok "$name: $result" || { fail "$name: $result"; ALL_PASSED=false; }
}

verify "Core package"    'python -c "import firm; print(\"OK\")"'
verify "Contracts"       'python -c "from firm.contracts.models import Signal; print(\"OK\")"'
verify "Strategies"      'python -c "from firm.strategies import list_strategies; print(len(list_strategies()),\"strategies\")"'
verify "Data providers"  'python -c "from firm.data.providers.base import DataProvider; print(\"OK\")"'

[[ "$EXTRAS" == *"api"* ]]  && verify "API (FastAPI)"  'python -c "from firm.api.app import create_app; print(\"OK\")"'
[[ "$EXTRAS" == *"live"* ]] && verify "Live trading"   'python -c "from firm.brokers.base import Broker; print(\"OK\")"'
[[ "$EXTRAS" == *"llm"* ]]  && verify "LLM service"    'python -c "from firm.llm.provider import LLMService; print(\"OK\")"'

if [[ "$COMPONENTS" == *"live"* || "$COMPONENTS" == "all" ]] && [ "$SKIP_IBKR" = false ]; then
    if [ -f "$IBKR_INSTALL_DIR/ibgateway" ] || [ -f "$IBKR_INSTALL_DIR/Jts/ibgateway" ]; then
        ok "IB Gateway: installed at $IBKR_INSTALL_DIR"
    else
        fail "IB Gateway: not found at $IBKR_INSTALL_DIR"
        ALL_PASSED=false
    fi
fi

if [[ "$COMPONENTS" == *"live"* || "$COMPONENTS" == "all" ]] && \
   [ "$SKIP_IBKR" = false ] && [ "$SKIP_IBC" = false ]; then
    if [ -f "$IBC_INSTALL_DIR/scripts/ibgateway.sh" ]; then
        ok "IBC: installed at $IBC_INSTALL_DIR"
    else
        warn "IBC: not installed — gateway will require manual login"
    fi
fi

# ---------------------------------------------------------------
# 13. Summary
# ---------------------------------------------------------------
echo ""
echo -e "\033[34m============================================\033[0m"
if [ "$ALL_PASSED" = true ]; then
    echo -e "\033[32m  Setup complete!\033[0m"
else
    echo -e "\033[33m  Setup complete (with warnings)\033[0m"
fi
echo -e "\033[34m============================================\033[0m"
echo ""
echo "  Next steps:"
echo "    1. Edit .env with your API keys"

if [[ "$COMPONENTS" == *"live"* || "$COMPONENTS" == "all" ]] && \
   [ "$SKIP_IBKR" = false ] && [ "$SKIP_IBC" = false ]; then
    echo "    2. Edit ~/.ibc/config.ini with your IBKR credentials"
    if [ "$INSTALL_SERVICES" = true ]; then
        echo "    3. sudo systemctl start ib-gateway   # starts headless Gateway"
        echo "    4. sudo systemctl start firm-api     # starts API + frontend"
        echo "    5. Open http://localhost:8000"
    else
        echo "    3. Start IB Gateway:  sudo systemctl start ib-gateway  (if --install-services was used)"
        echo "       or manually:       ./scripts/start_ibgateway.sh paper"
        echo "    4. Start the server:  firm-api"
        echo "    5. Open http://localhost:8000"
    fi
else
    echo "    2. Start the server:  firm-api"
    echo "    3. Open http://localhost:8000"
fi
echo ""
echo "  To uninstall: ./setup.sh --uninstall"
echo ""
echo "  Optional:"
echo "    - Run tests:       pytest"
echo "    - Fetch data:      python scripts/fetch_data.py --symbols AAPL,MSFT --start 2023-01-01 --end 2024-01-01"
echo "    - Dev frontend:    cd frontend && npm run dev"
echo ""
