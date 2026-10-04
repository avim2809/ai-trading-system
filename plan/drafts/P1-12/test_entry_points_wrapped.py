"""Every backtest entry point is wrapped by ``capture_run`` (P1-12).

DRAFT for the owner-protected integrity test directory (owner commits it unchanged).
"""

from __future__ import annotations

import ast
from pathlib import Path

_HERE = Path(__file__).resolve()
REPO = next(p for p in _HERE.parents if (p / "src" / "firm").is_dir())

WRAPPED = {
    "src/firm/backtest/run.py": "execute_backtest",
    "src/firm/runtime.py": "run_backtest_from_config",
    "src/firm/experiments/runner.py": "run",
    "src/firm/scripts/run_backtest.py": "main",
}
# Frozen scripts that predate the ledger: allowlisted BY NAME, never edited.
FROZEN_ALLOWLIST = {
    "run_alt_premia_evaluation.py", "run_backtest.py", "run_best_stocks_arm.py", "run_combination_evaluation.py",
    "run_eodhd_s1_evaluation.py", "run_eodhd_s2_evaluation.py", "run_eodhd_s3_evaluation.py",
    "run_eodhd_s4_evaluation.py", "run_eodhd_s5_evaluation.py", "run_insider_cluster_evaluation.py",
    "run_live_trading.py", "run_pbo_trial_audit.py", "run_standalone_strategy_evaluation.py",
    "run_until_profitable.py", "run_walk_forward_pbo_audit.py",
}


def _functions(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node


def _calls(node: ast.AST, name: str) -> bool:
    return any(
        isinstance(c, ast.Call) and (
            (isinstance(c.func, ast.Name) and c.func.id == name)
            or (isinstance(c.func, ast.Attribute) and c.func.attr == name)
        )
        for c in ast.walk(node)
    )


def test_entry_points_call_capture_run():
    for rel, fn in WRAPPED.items():
        tree = ast.parse((REPO / rel).read_text())
        found = [f for f in _functions(tree) if f.name == fn and _calls(f, "capture_run")]
        assert found, f"{rel}: {fn} must call capture_run"


def test_every_engine_construction_is_inside_a_wrapped_function():
    allowed = set(WRAPPED.values()) | {f"_{v}_impl" for v in WRAPPED.values()}
    bad: list[str] = []
    for root in ("src", "scripts"):
        for p in (REPO / root).rglob("*.py"):
            rel = p.relative_to(REPO).as_posix()
            if rel == "src/firm/backtest/engine.py":
                continue
            tree = ast.parse(p.read_text())
            for fn in _functions(tree):
                for c in ast.walk(fn):
                    if isinstance(c, ast.Call) and isinstance(c.func, ast.Name) and c.func.id == "BacktestEngine":
                        if fn.name not in allowed:
                            bad.append(f"{rel}:{c.lineno} in {fn.name}")
            for stmt in tree.body:  # module-level construction
                if not isinstance(stmt, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)) and _calls(
                    stmt, "BacktestEngine"
                ):
                    bad.append(f"{rel}: module-level BacktestEngine(")
    assert not bad, f"unwrapped direct BacktestEngine construction: {sorted(set(bad))}"


def test_new_run_scripts_import_firm_research():
    missing = [
        p.name for p in sorted((REPO / "scripts").glob("run_*.py"))
        if p.name not in FROZEN_ALLOWLIST and "firm.research" not in p.read_text()
    ]
    assert not missing, f"scripts/run_*.py must go through firm.research (ledger): {missing}"


def test_allowlist_has_no_stale_entries():
    gone = [n for n in FROZEN_ALLOWLIST if not (REPO / "scripts" / n).exists()]
    assert not gone, f"allowlisted scripts no longer exist (keep the list honest): {gone}"
