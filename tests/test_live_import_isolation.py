"""The live import graph must not contain any credibility-plan research module.

``firm.runtime``, ``firm.live.engine`` and ``firm.api.app`` are loaded by both
running firm-api services (IBKR :8000, Alpaca :8001), so anything they import
is deployed code. New research modules (PLAN.md section 3) are additive and
must stay out of that graph until the owner-approved integration ticket
(P6-01). CODEOWNERS-protected (P0-04): change the forbidden list only through a
diff the owner reviews.

The graph is inspected in a fresh interpreter with a temp ``FIRM_DATA_DIR`` and
a temp cwd, because importing ``firm.api.app`` runs ``create_app()`` and
``setup_logging`` at import time and must never touch the live ``data/`` dirs.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"

FORBIDDEN_PREFIXES = (
    "firm.validation",
    "firm.research",
    "firm.signals",
    "firm.costs",
    "firm.risk",
    "firm.monitoring",
    "firm.lifecycle",
    "firm.reporting",
    "firm.data.cleaning",
    "firm.data.etf_loader",
    "firm.data.manifest",
    "firm.data.qa",
    "firm.backtest.vector_engine",
    "firm.portfolio.forecast_combine",
    "firm.portfolio.sizing",
    "firm.portfolio.weights",
    "firm.allocation.carver_sleeve",
)

_CHILD_CODE = """\
PREFIXES = {prefixes!r}
import sys, importlib
for m in ('firm.runtime', 'firm.live.engine', 'firm.api.app'):
    importlib.import_module(m)
# Lazily imported by request handlers; load them too, then re-check.
from firm.data.pit_store import PointInTimeDataStore  # noqa: F401
from firm.eval.robustness import MonteCarloAnalyzer  # noqa: F401
import firm
assert firm.__file__.startswith({src!r}), firm.__file__
bad = [k for k in sys.modules if any(k == p or k.startswith(p + '.') for p in PREFIXES)]
print('BAD:' + ','.join(sorted(bad)))
"""


def is_forbidden(module_name: str) -> bool:
    """Exact-name or dotted-prefix match (``firm.risk`` but not ``firm.riskfoo``)."""
    return any(module_name == p or module_name.startswith(p + ".") for p in FORBIDDEN_PREFIXES)


def _bad_modules(src_root: Path, tmp_path: Path) -> subprocess.CompletedProcess[str]:
    code = _CHILD_CODE.format(prefixes=FORBIDDEN_PREFIXES, src=str(src_root))
    env = {**os.environ, "FIRM_DATA_DIR": str(tmp_path), "PYTHONPATH": str(src_root)}
    for var in ("FIRM_AUTO_START_LIVE", "FIRM_LIVE_CONFIG", "FIRM_ALLOW_TRADING"):
        env.pop(var, None)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=300,
    )


def test_live_graph_excludes_new_modules(tmp_path):
    out = _bad_modules(SRC, tmp_path)
    assert out.returncode == 0, out.stderr
    # Exact line `BAD:` (empty list): stray logging cannot mask a failure, and a
    # non-empty list prints `BAD:<names>`, which does not match.
    assert "BAD:" in out.stdout.splitlines(), out.stdout + out.stderr


def test_prefix_matching_is_exact():
    assert not is_forbidden("firm.agents.risk")
    assert not is_forbidden("firm.portfolio.optimizer")
    assert not is_forbidden("firm.riskfoo")
    assert not is_forbidden("firm.data.pit_store")
    assert is_forbidden("firm.risk")
    assert is_forbidden("firm.risk.limits")
    assert is_forbidden("firm.research.seal")
    assert is_forbidden("firm.portfolio.sizing")


def test_probe_in_live_graph_is_detected(tmp_path):
    """Mutation test: a research module imported from firm.runtime must be caught."""
    overlay = tmp_path / "overlay"
    shutil.copytree(
        SRC / "firm", overlay / "firm", ignore=shutil.ignore_patterns("__pycache__")
    )
    research = overlay / "firm" / "research"
    research.mkdir(exist_ok=True)
    if not (research / "__init__.py").exists():
        (research / "__init__.py").write_text('"""probe overlay"""\n')
    (research / "_probe.py").write_text("PROBE = True\n")
    runtime = overlay / "firm" / "runtime.py"
    runtime.write_text(runtime.read_text() + "\nimport firm.research._probe  # noqa: E402,F401\n")

    work = tmp_path / "work"
    work.mkdir()
    out = _bad_modules(overlay, work)
    assert out.returncode == 0, out.stderr
    lines = out.stdout.splitlines()
    assert "BAD:" not in lines
    assert any(
        ln.startswith("BAD:") and "firm.research._probe" in ln for ln in lines
    ), out.stdout + out.stderr


@pytest.mark.parametrize("name", ["firm.runtime", "firm.live.engine", "firm.api.app"])
def test_root_modules_are_not_themselves_forbidden(name):
    assert not is_forbidden(name)
