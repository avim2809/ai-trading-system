#!/usr/bin/env python
"""Live-startup smoke test for LIVE-IMPORT-PATH changes (PLAN.md section 8, P0-06).

For each live config (``config/live.yaml`` and ``config/live_alpaca.yaml``) it
starts a fresh interpreter that

1. imports ``firm.runtime``, ``firm.live.engine`` and ``firm.api.app``,
2. builds the engine config exactly as ``firm.api.routers.live._start_live_engine``
   does (``resolve_live_startup`` plus ``agent_modes``/``llm_config`` from the LLM
   config), and
3. runs ``firm.runtime.build_orchestrator`` on it,

without connecting to any broker, without network access and without reading
``.env`` or the live ``data/`` / ``data_alpaca/`` directories:

* the child gets a minimal allow-listed environment plus the ``Environment=``
  lines of the matching systemd unit (``FIRM_AUTO_START_LIVE``,
  ``FIRM_ALLOW_TRADING``, ``FIRM_DATA_DIR`` and ``FIRM_API_PORT`` are dropped);
* cwd and ``FIRM_DATA_DIR`` are a temp dir, and the config is a COPY whose
  state/cache paths point into that temp dir (``FIRM_DATA_DIR`` alone does not
  redirect paths named inside the YAML), so only the STATIC universe is asserted;
* proxies point at a dead port and ``socket.create_connection`` /
  ``socket.socket.connect`` raise for IP sockets.

Exit code 0 = pass, 1 = fail. One line per check on stdout.
"""

from __future__ import annotations

import argparse
import logging
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml

log = logging.getLogger("live_import_smoke")

REPO_ROOT = Path(__file__).resolve().parents[1]

# config -> systemd unit whose Environment= lines production runs it with
DEFAULT_CONFIGS: dict[str, str] = {
    "config/live.yaml": "deploy/ai-trading.service",
    "config/live_alpaca.yaml": "deploy/ai-trading-alpaca.service",
}

# Production-only variables the smoke test must never inherit from a unit file.
_DROPPED_UNIT_VARS = frozenset(
    {"FIRM_AUTO_START_LIVE", "FIRM_ALLOW_TRADING", "FIRM_DATA_DIR", "FIRM_API_PORT"}
)
# Unit variables whose value is a path relative to the repo root (the unit's WorkingDirectory).
_PATH_UNIT_VARS = frozenset({"FIRM_LLM_CONFIG", "FIRM_LIVE_CONFIG"})
_PASSTHROUGH_ENV = ("PATH", "LANG", "LC_ALL", "PYTHONPATH", "VIRTUAL_ENV", "TZ")
_DEAD_PROXY = "http://127.0.0.1:9"


def out(line: str) -> None:
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


# --------------------------------------------------------------------------- #
# helpers shared by parent and child
# --------------------------------------------------------------------------- #
def parse_unit_environment(unit_path: Path) -> dict[str, str]:
    """Return the ``Environment=`` assignments of a systemd unit file."""
    env: dict[str, str] = {}
    for raw in unit_path.read_text().splitlines():
        line = raw.strip()
        if not line.startswith("Environment="):
            continue
        for token in shlex.split(line[len("Environment="):]):
            key, sep, value = token.partition("=")
            if sep:
                env[key] = value
    return env


def _rewrite_state_paths(node: Any, state_dir: Path) -> Any:
    """Point every relative ``*_path`` / ``*_db`` / ``*_file`` string into ``state_dir``."""
    if isinstance(node, dict):
        rewritten = {}
        for key, value in node.items():
            if (
                isinstance(value, str)
                and isinstance(key, str)
                and key.endswith(("_path", "_db", "_file"))
                and not os.path.isabs(value)
            ):
                rewritten[key] = str(state_dir / Path(value).name)
            else:
                rewritten[key] = _rewrite_state_paths(value, state_dir)
        return rewritten
    if isinstance(node, list):
        return [_rewrite_state_paths(v, state_dir) for v in node]
    return node


def sanitize_config(config_path: Path, work_dir: Path) -> Path:
    """Write a copy of ``config_path`` whose state/cache paths live under ``work_dir``."""
    cfg = yaml.safe_load(config_path.read_text()) or {}
    state_dir = work_dir / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    copy = work_dir / config_path.name
    copy.write_text(yaml.safe_dump(_rewrite_state_paths(cfg, state_dir), sort_keys=False))
    return copy


def build_child_env(unit_env: dict[str, str], work_dir: Path, live_config: Path) -> dict[str, str]:
    env = {k: os.environ[k] for k in _PASSTHROUGH_ENV if k in os.environ}
    for key, value in unit_env.items():
        if key in _DROPPED_UNIT_VARS:
            continue
        if key in _PATH_UNIT_VARS and not os.path.isabs(value):
            value = str(REPO_ROOT / value)
        env[key] = value
    env.update(
        {
            "FIRM_DATA_DIR": str(work_dir / "data"),
            "FIRM_LIVE_CONFIG": str(live_config),
            "HTTP_PROXY": _DEAD_PROXY,
            "HTTPS_PROXY": _DEAD_PROXY,
            "ALL_PROXY": _DEAD_PROXY,
            "NO_PROXY": "",
            "MPLCONFIGDIR": str(work_dir / "mpl"),
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    return env


# --------------------------------------------------------------------------- #
# child-side checks (run in the fresh interpreter)
# --------------------------------------------------------------------------- #
def _block_network() -> None:
    """Make IP socket connects raise (HTTPS SDKs and raw TCP clients ignore HTTP_PROXY)."""
    import socket

    def _refuse(*_a: Any, **_k: Any) -> None:
        raise OSError("network disabled by scripts/live_import_smoke.py")

    orig_connect = socket.socket.connect
    orig_connect_ex = socket.socket.connect_ex

    def _connect(self: socket.socket, address: Any) -> Any:
        if self.family in (socket.AF_INET, socket.AF_INET6):
            _refuse()
        return orig_connect(self, address)

    def _connect_ex(self: socket.socket, address: Any) -> Any:
        if self.family in (socket.AF_INET, socket.AF_INET6):
            _refuse()
        return orig_connect_ex(self, address)

    socket.create_connection = _refuse  # type: ignore[assignment]
    socket.socket.connect = _connect  # type: ignore[method-assign]
    socket.socket.connect_ex = _connect_ex  # type: ignore[method-assign]


def check_imports() -> None:
    """Import the modules both running services load at startup."""
    import importlib

    for name in ("firm.runtime", "firm.live.engine", "firm.api.app"):
        importlib.import_module(name)
        log.info("imported %s", name)
    import firm

    # Guards the "main .venv tests a worktree" mistake: the code under test must be this checkout's.
    if not Path(firm.__file__).resolve().is_relative_to(REPO_ROOT / "src"):
        raise AssertionError(f"firm imported from {firm.__file__}, expected under {REPO_ROOT / 'src'}")
    out("PASS imports: firm.runtime firm.live.engine firm.api.app")


def check_build(config_path: str) -> None:
    """Build the engine config as production does, then ``build_orchestrator`` (no connect)."""
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"live config not found: {config_path}")
    os.environ["FIRM_LIVE_CONFIG"] = str(path)
    yaml_cfg = yaml.safe_load(path.read_text()) or {}

    from firm.live.provider_utils import resolve_live_startup
    from firm.llm.config import llm_service_config, load_llm_config, provider_config
    from firm.runtime import build_orchestrator

    # --- LLM arm: must be the unit's FIRM_LLM_CONFIG, not the default config/llm.yaml
    llm_env = os.environ.get("FIRM_LLM_CONFIG", "")
    if not llm_env:
        raise AssertionError("FIRM_LLM_CONFIG not set by the unit environment")
    llm_loaded = load_llm_config()
    llm_expected = yaml.safe_load(Path(llm_env).read_text()) or {}
    if llm_loaded != llm_expected:
        raise AssertionError(f"load_llm_config() did not resolve {llm_env}")
    if Path(llm_env).resolve() == (REPO_ROOT / "config" / "llm.yaml").resolve():
        raise AssertionError("LLM config resolved to the default config/llm.yaml")
    if llm_service_config() != llm_service_config(llm_expected):
        raise AssertionError("llm_service_config() does not reflect FIRM_LLM_CONFIG")
    out(f"PASS llm config: {Path(llm_env).name} (agent_modes={len(llm_loaded.get('agent_modes', {}))})")

    # --- startup resolution (static universe only: dynamic-universe state is in the temp dir)
    resolved = resolve_live_startup()
    static_symbols = list((yaml_cfg.get("universe") or {}).get("symbols") or [])
    if resolved["symbols"] != static_symbols:
        raise AssertionError(
            f"symbols differ from the static universe ({len(resolved['symbols'])} vs {len(static_symbols)})"
        )
    expected_strategies = list((yaml_cfg.get("strategies") or {}).get("enabled") or []) or None
    if resolved["strategies"] != expected_strategies:
        raise AssertionError(f"strategies differ: {resolved['strategies']} vs {expected_strategies}")
    engine_config = resolved["engine_config"]
    if not engine_config.get("sector_map"):
        raise AssertionError("no risk.sector_map: _start_live_engine would refuse to start")
    expected_mode = str(yaml_cfg.get("strategy_mode") or "pipeline")
    mode = str(engine_config.get("strategy_mode") or "pipeline")
    if mode != expected_mode:
        raise AssertionError(f"strategy_mode {mode!r} != {expected_mode!r}")
    if path.name == "live_alpaca.yaml" and mode != "allocation":
        raise AssertionError(f"Alpaca config must be strategy_mode=allocation, got {mode!r}")
    out(
        f"PASS resolve {path.name}: broker={resolved['broker']} symbols={len(static_symbols)} "
        f"strategies={len(resolved['strategies'] or [])} strategy_mode={mode}"
    )

    # --- config exactly as api/routers/live.py:_start_live_engine builds it
    config = {
        **engine_config,
        "symbols": resolved["symbols"],
        "strategies": resolved["strategies"],
        "strategy_params": resolved["strategy_params"],
        "agent_modes": load_llm_config().get("agent_modes", {}),
        "llm_config": provider_config(),
    }
    config.setdefault("respect_market_hours", True)
    orchestrator = build_orchestrator(config)
    if orchestrator is None:
        raise AssertionError("build_orchestrator returned None")
    out(f"PASS build_orchestrator {path.name}: {type(orchestrator).__name__}")


def _child_main(config_path: str) -> int:
    _block_network()
    try:
        check_imports()
        check_build(config_path)
    except Exception:
        log.exception("smoke check failed for %s", config_path)
        out(f"FAIL {Path(config_path).name}")
        return 1
    return 0


# --------------------------------------------------------------------------- #
# parent
# --------------------------------------------------------------------------- #
def _run_one(config: str, unit: str | None) -> bool:
    cfg_path = Path(config)
    if not cfg_path.is_absolute():
        cfg_path = REPO_ROOT / cfg_path
    if not cfg_path.is_file():
        log.error("live config does not exist: %s", cfg_path)
        out(f"FAIL {config}: config file not found")
        return False
    unit_env: dict[str, str] = {}
    if unit:
        unit_path = REPO_ROOT / unit
        if not unit_path.is_file():
            log.error("systemd unit does not exist: %s", unit_path)
            out(f"FAIL {config}: unit {unit} not found")
            return False
        unit_env = parse_unit_environment(unit_path)
    with tempfile.TemporaryDirectory(prefix="live_import_smoke_") as tmp:
        work = Path(tmp)
        copy = sanitize_config(cfg_path, work)
        env = build_child_env(unit_env, work, copy)
        cmd = [sys.executable, str(Path(__file__).resolve()), "--child", str(copy)]
        log.info("smoke child for %s (cwd=%s)", config, work)
        proc = subprocess.run(cmd, cwd=work, env=env, capture_output=True, text=True, timeout=600)
    for line in proc.stdout.splitlines():
        out(f"[{Path(config).name}] {line}")
    if proc.returncode != 0:
        for line in proc.stderr.splitlines()[-25:]:
            log.error("[%s] %s", Path(config).name, line)
    return proc.returncode == 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--config",
        action="append",
        help="live config to check (repeatable; default: config/live.yaml and config/live_alpaca.yaml)",
    )
    parser.add_argument("--child", metavar="SANITIZED_CONFIG", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s", stream=sys.stderr)

    if args.child:
        return _child_main(args.child)

    configs = args.config or list(DEFAULT_CONFIGS)
    ok = True
    for config in configs:
        unit = DEFAULT_CONFIGS.get(config)
        if unit is None:
            log.warning("no systemd unit known for %s: running without unit environment", config)
        ok = _run_one(config, unit) and ok
    out("SMOKE PASS" if ok else "SMOKE FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
