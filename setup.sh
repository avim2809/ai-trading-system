#!/usr/bin/env bash
# AI Trading System - One-command setup script (macOS/Linux)
#
# Usage:
#   ./setup.sh                  # install core + api
#   ./setup.sh --components all # install everything
#   ./setup.sh --skip-frontend  # skip Node.js build
#   ./setup.sh --skip-venv      # skip venv creation
set -e

COMPONENTS="api"
SKIP_FRONTEND=false
SKIP_VENV=false
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --components)  COMPONENTS="$2"; shift 2 ;;
        --skip-frontend) SKIP_FRONTEND=true; shift ;;
        --skip-venv)   SKIP_VENV=true; shift ;;
        *)             echo "Unknown option: $1"; exit 1 ;;
    esac
done

step()  { echo -e "\n\033[36m==> $1\033[0m"; }
ok()    { echo -e "    \033[32m[OK]\033[0m $1"; }
warn()  { echo -e "    \033[33m[!]\033[0m $1"; }
fail()  { echo -e "    \033[31m[X]\033[0m $1"; }

echo ""
echo -e "\033[34m============================================\033[0m"
echo -e "\033[34m  AI Multi-Agent Trading System - Setup\033[0m"
echo -e "\033[34m============================================\033[0m"
echo ""

# ---------------------------------------------------------------
# 1. Check Python
# ---------------------------------------------------------------
step "Checking Python installation"
if ! command -v python3 &>/dev/null; then
    fail "Python3 not found. Install Python >= 3.10"
    exit 1
fi
PY_VERSION=$(python3 --version)
ok "Found $PY_VERSION"

python3 -c "import sys; exit(0 if sys.version_info >= (3,10) else 1)" || {
    fail "Python >= 3.10 required. Found: $PY_VERSION"
    exit 1
}

# ---------------------------------------------------------------
# 2. Virtual environment
# ---------------------------------------------------------------
if [ "$SKIP_VENV" = false ]; then
    step "Setting up virtual environment"
    if [ ! -d "$PROJECT_ROOT/.venv" ]; then
        python3 -m venv "$PROJECT_ROOT/.venv"
        ok "Created .venv"
    else
        ok ".venv already exists"
    fi
    source "$PROJECT_ROOT/.venv/bin/activate"
    ok "Activated .venv"
fi

# ---------------------------------------------------------------
# 3. Parse components
# ---------------------------------------------------------------
step "Parsing components: $COMPONENTS"
if [ "$COMPONENTS" = "all" ]; then
    EXTRAS="dev,api,live,llm"
else
    EXTRAS="$COMPONENTS"
fi
ok "Will install: core + [$EXTRAS]"

# ---------------------------------------------------------------
# 4. Install Python package
# ---------------------------------------------------------------
step "Installing Python dependencies"
pip install -e ".[$EXTRAS]" --quiet
ok "Python packages installed"

# ---------------------------------------------------------------
# 5. Create .env
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
# 6. Create data directories
# ---------------------------------------------------------------
step "Creating data directories"
for d in data/cache data/vectordb runs; do
    mkdir -p "$PROJECT_ROOT/$d"
    ok "$d/"
done

# ---------------------------------------------------------------
# 7. Frontend build
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
# 8. Verify
# ---------------------------------------------------------------
step "Verifying installation"
ALL_PASSED=true

verify() {
    local name="$1" cmd="$2"
    result=$(eval "$cmd" 2>&1) && ok "$name: $result" || { fail "$name: $result"; ALL_PASSED=false; }
}

verify "Core package"    'python3 -c "import firm; print(\"OK\")"'
verify "Contracts"       'python3 -c "from firm.contracts.models import Signal; print(\"OK\")"'
verify "Strategies"      'python3 -c "from firm.strategies import list_strategies; print(len(list_strategies()),\"strategies\")"'
verify "Data providers"  'python3 -c "from firm.data.providers.base import DataProvider; print(\"OK\")"'

[[ "$EXTRAS" == *"api"* ]] && verify "API (FastAPI)" 'python3 -c "from firm.api.app import create_app; print(\"OK\")"'
[[ "$EXTRAS" == *"live"* ]] && verify "Live trading" 'python3 -c "from firm.brokers.base import Broker; print(\"OK\")"'
[[ "$EXTRAS" == *"llm"* ]] && verify "LLM service"  'python3 -c "from firm.llm.provider import LLMService; print(\"OK\")"'

# ---------------------------------------------------------------
# 9. Summary
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
echo "    2. Start the server:  firm-api"
echo "    3. Open http://localhost:8000"
echo ""
echo "  Optional:"
echo "    - Run tests:       pytest"
echo "    - Fetch data:      python scripts/fetch_data.py --symbols AAPL,MSFT --start 2023-01-01 --end 2024-01-01"
echo "    - Dev frontend:    cd frontend && npm run dev"
echo ""
