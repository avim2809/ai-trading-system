#Requires -Version 5.1
<#
.SYNOPSIS
    AI Trading System - One-command setup script (Windows PowerShell)
.DESCRIPTION
    Installs Python dependencies, builds the frontend, creates .env from
    template, initializes data directories, and verifies the installation.
.PARAMETER Components
    Comma-separated list of optional components to install.
    Options: api, live, llm, dev, all (default: api)
.PARAMETER SkipFrontend
    Skip Node.js/frontend build (useful if Node.js is not installed)
.PARAMETER SkipVenv
    Skip virtual environment creation (use if already in a venv)
.EXAMPLE
    .\setup.ps1
    .\setup.ps1 -Components all
    .\setup.ps1 -Components api,live -SkipFrontend
#>

param(
    [string]$Components = "api",
    [switch]$SkipFrontend,
    [switch]$SkipVenv
)

$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot

function Write-Step($msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }
function Write-Ok($msg)   { Write-Host "    [OK] $msg" -ForegroundColor Green }
function Write-Warn($msg) { Write-Host "    [!] $msg" -ForegroundColor Yellow }
function Write-Fail($msg) { Write-Host "    [X] $msg" -ForegroundColor Red }

Write-Host ""
Write-Host "============================================" -ForegroundColor Blue
Write-Host "  AI Multi-Agent Trading System - Setup"     -ForegroundColor Blue
Write-Host "============================================" -ForegroundColor Blue
Write-Host ""

# ---------------------------------------------------------------
# 1. Check Python
# ---------------------------------------------------------------
Write-Step "Checking Python installation"
$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) {
    Write-Fail "Python not found. Install Python >= 3.10 from https://python.org"
    exit 1
}
$pyVersion = python --version 2>&1
Write-Ok "Found $pyVersion"

$versionCheck = python -c "import sys; exit(0 if sys.version_info >= (3,10) else 1)" 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Fail "Python >= 3.10 required. Found: $pyVersion"
    exit 1
}

# ---------------------------------------------------------------
# 2. Virtual environment
# ---------------------------------------------------------------
if (-not $SkipVenv) {
    Write-Step "Setting up virtual environment"
    if (-not (Test-Path "$ProjectRoot\.venv")) {
        python -m venv "$ProjectRoot\.venv"
        Write-Ok "Created .venv"
    } else {
        Write-Ok ".venv already exists"
    }

    $activateScript = "$ProjectRoot\.venv\Scripts\Activate.ps1"
    if (Test-Path $activateScript) {
        . $activateScript
        Write-Ok "Activated .venv"
    }
}

# ---------------------------------------------------------------
# 3. Parse components
# ---------------------------------------------------------------
Write-Step "Parsing components: $Components"
$extras = @()
if ($Components -eq "all") {
    $extras = @("dev", "api", "live", "llm")
} else {
    $extras = $Components.Split(",") | ForEach-Object { $_.Trim() }
}

$extrasStr = ($extras | ForEach-Object { $_ }) -join ","
Write-Ok "Will install: core + [$extrasStr]"

# ---------------------------------------------------------------
# 4. Install Python package
# ---------------------------------------------------------------
Write-Step "Installing Python dependencies"
$pipTarget = ".[" + $extrasStr + "]"
pip install -e $pipTarget 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) {
    pip install -e $pipTarget
    if ($LASTEXITCODE -ne 0) {
        Write-Fail "pip install failed"
        exit 1
    }
}
Write-Ok "Python packages installed"

# ---------------------------------------------------------------
# 5. Create .env from template
# ---------------------------------------------------------------
Write-Step "Configuring environment"
$envFile = "$ProjectRoot\.env"
$envExample = "$ProjectRoot\.env.example"
if (-not (Test-Path $envFile)) {
    if (Test-Path $envExample) {
        Copy-Item $envExample $envFile
        Write-Ok "Created .env from .env.example"
        Write-Warn "Edit .env to add your API keys"
    } else {
        Write-Warn ".env.example not found, skipping"
    }
} else {
    Write-Ok ".env already exists"
}

# ---------------------------------------------------------------
# 6. Create data directories
# ---------------------------------------------------------------
Write-Step "Creating data directories"
$dirs = @("data/cache", "data/vectordb", "runs")
foreach ($d in $dirs) {
    $path = Join-Path $ProjectRoot $d
    if (-not (Test-Path $path)) {
        New-Item -ItemType Directory -Path $path -Force | Out-Null
        Write-Ok "Created $d/"
    } else {
        Write-Ok "$d/ exists"
    }
}

# ---------------------------------------------------------------
# 7. Frontend build
# ---------------------------------------------------------------
if (-not $SkipFrontend) {
    Write-Step "Building frontend"

    $npmPaths = @(
        "C:\Program Files\nodejs\npm.cmd",
        (Get-Command npm -ErrorAction SilentlyContinue).Source
    ) | Where-Object { $_ -and (Test-Path $_) } | Select-Object -First 1

    if ($npmPaths) {
        $npmCmd = $npmPaths
        Push-Location "$ProjectRoot\frontend"
        try {
            Write-Host "    Installing npm packages..." -ForegroundColor Gray
            & $npmCmd install 2>&1 | Out-Null
            Write-Ok "npm packages installed"

            Write-Host "    Building production bundle..." -ForegroundColor Gray
            & $npmCmd run build 2>&1 | Out-Null
            if ($LASTEXITCODE -eq 0) {
                Write-Ok "Frontend built -> frontend/dist/"
            } else {
                Write-Warn "Frontend build failed (non-critical, dev server still works)"
            }
        } finally {
            Pop-Location
        }
    } else {
        Write-Warn "Node.js/npm not found - skipping frontend build"
        Write-Warn "Install Node.js from https://nodejs.org for the web UI"
    }
} else {
    Write-Step "Skipping frontend (--SkipFrontend)"
}

# ---------------------------------------------------------------
# 8. Verify installation
# ---------------------------------------------------------------
Write-Step "Verifying installation"

$checks = @(
    @{ Name = "Core package";    Cmd = 'python -c "import firm; print(\"OK\")"' },
    @{ Name = "Contracts";       Cmd = 'python -c "from firm.contracts.models import Signal; print(\"OK\")"' },
    @{ Name = "Strategies";      Cmd = 'python -c "from firm.strategies import list_strategies; print(len(list_strategies()), \"strategies\")"' },
    @{ Name = "Data providers";  Cmd = 'python -c "from firm.data.providers.base import DataProvider; print(\"OK\")"' }
)

if ($extras -contains "api") {
    $checks += @{ Name = "API (FastAPI)"; Cmd = 'python -c "from firm.api.app import create_app; print(\"OK\")"' }
}
if ($extras -contains "live") {
    $checks += @{ Name = "Live trading"; Cmd = 'python -c "from firm.brokers.base import Broker; print(\"OK\")"' }
}
if ($extras -contains "llm") {
    $checks += @{ Name = "LLM service"; Cmd = 'python -c "from firm.llm.provider import LLMService; print(\"OK\")"' }
}

$allPassed = $true
foreach ($check in $checks) {
    try {
        $result = Invoke-Expression $check.Cmd 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Ok "$($check.Name): $result"
        } else {
            Write-Fail "$($check.Name): $result"
            $allPassed = $false
        }
    } catch {
        Write-Fail "$($check.Name): $($_.Exception.Message)"
        $allPassed = $false
    }
}

# ---------------------------------------------------------------
# 9. Summary
# ---------------------------------------------------------------
Write-Host ""
Write-Host "============================================" -ForegroundColor Blue
if ($allPassed) {
    Write-Host "  Setup complete!" -ForegroundColor Green
} else {
    Write-Host "  Setup complete (with warnings)" -ForegroundColor Yellow
}
Write-Host "============================================" -ForegroundColor Blue
Write-Host ""
Write-Host "  Next steps:" -ForegroundColor White
Write-Host "    1. Edit .env with your API keys" -ForegroundColor Gray
Write-Host "    2. Start the server:  firm-api" -ForegroundColor Gray
Write-Host "    3. Open http://localhost:8000" -ForegroundColor Gray
Write-Host ""
Write-Host "  Optional:" -ForegroundColor White
Write-Host "    - Run tests:       pytest" -ForegroundColor Gray
Write-Host "    - Fetch data:      python scripts/fetch_data.py --symbols AAPL,MSFT --start 2023-01-01 --end 2024-01-01" -ForegroundColor Gray
Write-Host "    - Dev frontend:    cd frontend && npm run dev" -ForegroundColor Gray
Write-Host ""
