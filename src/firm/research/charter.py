"""Charter front-matter parser, validator and charter-before-ledger ordering check (ticket P5-02).

A charter is a human-written, owner-committed markdown file with machine-readable YAML front matter. It pre-commits the
*procedures* (tau, max-DD bootstrap, decommission rules) before any result exists. This module only READS charters; it never
writes one. The integrity test that applies it to the committed charters is owner-authored (a draft lives under plan/drafts/P5-02/).

Timestamps come from git (``git log``), never from file mtimes or log lines. Importing this module must never happen from a live
module (``firm.research`` is on the forbidden list of ``tests/test_live_import_isolation.py``).
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
import subprocess
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

log = logging.getLogger(__name__)

__all__ = [
    "REQUIRED_KEYS",
    "SURVIVAL_REFERENCE",
    "CharterError",
    "charter_added_time",
    "check_charter_precedes_ledger",
    "gates_sha256",
    "iter_charters",
    "parse_charter",
    "validate_body",
    "validate_charter_file",
    "validate_front_matter",
]

REPO_DIR = Path(__file__).resolve().parents[3]
CHARTER_DIR = Path("research") / "charters"
GATES_PATH = Path("config") / "gates.yaml"
TEMPLATE_NAME = "TEMPLATE.md"

# Every key the ticket's front matter lists, in ticket order.
REQUIRED_KEYS: tuple[str, ...] = (
    "family", "charter_version", "evidence_base", "rebalance_frequency", "expected_turnover", "expected_annual_cost_bps",
    "expected_worst_year", "expected_longest_flat_months", "correlation_expectations", "falsification", "gates_yaml_sha256",
    "mechanism_committed_at_utc", "tau", "tau_derivation", "gross_cap", "gross_cap_bound_ceiling", "max_dd_procedure",
    "max_dd_analytic_2p5_tau", "survival_reference", "stress_reference", "expected_sharpe_haircut_min",
    "combined_forecast_avg_abs_expected", "speed_weights_rule", "decommission", "approved_by", "approved_commit",
)
# Filled only at approval; a placeholder is tolerated unless ``strict`` (so a draft can be validated before sign-off).
APPROVAL_KEYS: frozenset[str] = frozenset({"mechanism_committed_at_utc", "approved_by", "approved_commit"})
PROCEDURE_KEYS: tuple[str, ...] = (
    "method", "block_length", "draws", "quantile", "seed", "path_length_days", "path_length_rationale", "applied_to",
)
DECOMMISSION_TRIGGERS: frozenset[str] = frozenset({"dd", "cusum", "fidelity_2_months", "mechanism"})

# The ticket's integrity test fixes this exact string (its front-matter example differs: see the w10a report).
SURVIVAL_REFERENCE = "max(bootstrap_p95,2.5*tau)"
GROSS_CAP_BOUND_CEILING_MAX = 0.20
HAIRCUT_MIN = 0.5
MIN_DRAWS = 10_000

_FRONT = re.compile(r"\A---[ \t]*\r?\n(.*?)\r?\n---[ \t]*(?:\r?\n|\Z)", re.DOTALL)
_HEADING = re.compile(r"^#{1,6}[ \t]+(.*?)[ \t]*#*[ \t]*$", re.MULTILINE)
# A hand-typed max-DD *result*: a drawdown term, a connective, then a percentage. Multiples ("1.5 x survival_dd") and bare
# literature figures ("max drawdown 20.61%") do not match. Heuristic by design; the real guard is review of the owner's text.
_DD_LITERAL = re.compile(
    r"(?:max(?:imum)?[ \-_]*(?:dd|draw[ \-]?down)|survival[_ ]dd|stress[_ ]dd|hard[ _]decommission[ _]dd)"
    r"\s*(?:=|:|<=?|>=?|\bof\b|\bis\b|\bat\b)\s*[-−+~]?\s*\d+(?:\.\d+)?\s*%",
    re.IGNORECASE,
)


class CharterError(ValueError):
    """The file is not a parseable charter."""


def _is_placeholder(v: Any) -> bool:
    return v is None or (isinstance(v, str) and (not v.strip() or v.strip().startswith("<")))


def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", s)


# ------------------------------------------------------------------ parsing

def parse_charter(text: str) -> tuple[dict, str]:
    """Split ``text`` into (front-matter mapping, body). Raises :class:`CharterError` when absent or malformed."""
    m = _FRONT.match(text)
    if not m:
        raise CharterError("no YAML front matter (file must start with a '---' block)")
    try:
        fm = yaml.safe_load(m.group(1))
    except yaml.YAMLError as exc:
        raise CharterError(f"front matter is not valid YAML: {exc}") from exc
    if not isinstance(fm, dict):
        raise CharterError("front matter must be a YAML mapping")
    return fm, text[m.end():]


# ------------------------------------------------------------------ validation

def validate_front_matter(
    fm: dict, *, gates_sha256: str | None = None, gates_path_length_days: int | None = None, strict: bool = False
) -> list[str]:
    """Return a list of problems (empty = valid). ``gates_sha256`` / ``gates_path_length_days`` come from config/gates.yaml."""
    p: list[str] = []
    for k in REQUIRED_KEYS:
        if k not in fm:
            p.append(f"missing key: {k}")
        elif _is_placeholder(fm[k]) and (strict or k not in APPROVAL_KEYS):
            p.append(f"{k} is empty or an unfilled placeholder")

    def has(k: str) -> bool:
        return k in fm and not _is_placeholder(fm[k])

    if has("gates_yaml_sha256"):
        h = str(fm["gates_yaml_sha256"])
        if not re.fullmatch(r"[0-9a-f]{64}", h):
            p.append("gates_yaml_sha256 must be 64 lowercase hex characters")
        elif gates_sha256 is not None and h != gates_sha256:
            p.append("gates_yaml_sha256 does not match the current config/gates.yaml")
    if "tau" in fm and not _is_placeholder(fm["tau"]):
        t = fm["tau"]
        if not _is_num(t) or not 0 < t < 1:
            p.append("tau must be an annualised vol as a fraction in (0, 1), e.g. 0.09")
    if has("gross_cap") and not (_is_num(fm["gross_cap"]) and 0 < fm["gross_cap"] <= 1.0):
        p.append("gross_cap must be in (0, 1.0] (ETF path, no leverage)")
    if has("gross_cap_bound_ceiling") and not (
        _is_num(fm["gross_cap_bound_ceiling"]) and 0 < fm["gross_cap_bound_ceiling"] <= GROSS_CAP_BOUND_CEILING_MAX
    ):
        p.append(f"gross_cap_bound_ceiling must be in (0, {GROSS_CAP_BOUND_CEILING_MAX}]")
    if has("expected_sharpe_haircut_min") and not (_is_num(fm["expected_sharpe_haircut_min"]) and fm["expected_sharpe_haircut_min"] >= HAIRCUT_MIN):
        p.append(f"expected_sharpe_haircut_min must be >= {HAIRCUT_MIN}")
    if has("max_dd_analytic_2p5_tau") and fm["max_dd_analytic_2p5_tau"] is not True:
        p.append("max_dd_analytic_2p5_tau must be true (the 2.5 tau companion is always reported)")
    if has("survival_reference") and _norm(str(fm["survival_reference"])) != SURVIVAL_REFERENCE:
        p.append("survival_reference must be exactly max(bootstrap_p95, 2.5*tau)")
    if has("stress_reference") and "bootstrap_p95" not in str(fm["stress_reference"]):
        p.append("stress_reference must be a bootstrap p95 reference")
    if has("combined_forecast_avg_abs_expected") and not (_is_num(fm["combined_forecast_avg_abs_expected"]) and fm["combined_forecast_avg_abs_expected"] > 0):
        p.append("combined_forecast_avg_abs_expected must be a positive number")

    proc = fm.get("max_dd_procedure")
    if "max_dd_procedure" in fm:
        if not isinstance(proc, dict):
            p.append("max_dd_procedure must be a mapping")
        else:
            for k in PROCEDURE_KEYS:
                if k not in proc:
                    p.append(f"max_dd_procedure.{k} is missing")
                elif _is_placeholder(proc[k]):
                    p.append(f"max_dd_procedure.{k} is empty or an unfilled placeholder")
            if "method" in proc and proc["method"] != "stationary_bootstrap":
                p.append("max_dd_procedure.method must be stationary_bootstrap")
            if "block_length" in proc and proc["block_length"] != "politis_white":
                p.append("max_dd_procedure.block_length must be politis_white")
            d = proc.get("draws")
            if "draws" in proc and not (isinstance(d, int) and not isinstance(d, bool) and d >= MIN_DRAWS):
                p.append(f"max_dd_procedure.draws must be an integer >= {MIN_DRAWS}")
            if "quantile" in proc and not (_is_num(proc["quantile"]) and proc["quantile"] == 0.95):
                p.append("max_dd_procedure.quantile must be 0.95")
            if "seed" in proc and not (isinstance(proc["seed"], int) and not isinstance(proc["seed"], bool)):
                p.append("max_dd_procedure.seed must be an integer")
            pl = proc.get("path_length_days")
            if "path_length_days" in proc:
                if not (isinstance(pl, int) and not isinstance(pl, bool) and pl > 0):
                    p.append("max_dd_procedure.path_length_days must be a positive integer")
                elif gates_path_length_days is not None and pl != gates_path_length_days:
                    p.append(f"max_dd_procedure.path_length_days must equal config/gates.yaml ({gates_path_length_days})")
            if "applied_to" in proc and proc["applied_to"] != "selected_config_1x_cost":
                p.append("max_dd_procedure.applied_to must be selected_config_1x_cost")

    dec = fm.get("decommission")
    if "decommission" in fm:
        if not isinstance(dec, dict):
            p.append("decommission must be a mapping")
        else:
            if dec.get("soft_dd_mult") != 1.0 or dec.get("hard_dd_mult") != 1.5:
                p.append("decommission soft_dd_mult / hard_dd_mult must be 1.0 / 1.5 (G-DECOMMISSION)")
            trig = dec.get("triggers")
            if not isinstance(trig, list) or set(trig) != DECOMMISSION_TRIGGERS:
                p.append(f"decommission.triggers must be exactly {sorted(DECOMMISSION_TRIGGERS)}")
    return p


def _headings(body: str) -> list[str]:
    return [m.group(1).lower() for m in _HEADING.finditer(body)]


def validate_body(body: str) -> list[str]:
    """Body must contain Mechanism and Falsification sections and no hand-typed numeric max-DD result."""
    p: list[str] = []
    hs = _headings(body)
    for name in ("Mechanism", "Falsification"):
        if not any(name.lower() in h for h in hs):
            p.append(f"body has no {name} section heading")
    for m in _DD_LITERAL.finditer(body):
        p.append(f"numeric max-drawdown literal in body ({m.group(0)!r}); the result is a derived ledger artefact")
    return p


def gates_sha256(repo_dir: Path | None = None) -> str:
    return hashlib.sha256((Path(repo_dir or REPO_DIR) / GATES_PATH).read_bytes()).hexdigest()


def _gates_path_length(repo: Path) -> int | None:
    try:
        g = yaml.safe_load((repo / GATES_PATH).read_text())
        v = g["charter"]["max_dd_procedure"]["path_length_trading_days"]
    except (OSError, KeyError, TypeError, yaml.YAMLError):
        return None
    return v if isinstance(v, int) else None


def iter_charters(repo_dir: Path | None = None) -> Iterator[Path]:
    """Committed charter files (sorted), TEMPLATE excluded."""
    d = Path(repo_dir or REPO_DIR) / CHARTER_DIR
    if d.is_dir():
        yield from (f for f in sorted(d.glob("*.md")) if f.name != TEMPLATE_NAME)


# ------------------------------------------------------------------ ordering

def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", "safe.directory=*", *args], cwd=repo, capture_output=True, text=True, check=True, timeout=60
    ).stdout


def charter_added_time(repo_dir: Path, rel: str) -> datetime | None:
    """UTC committer time of the commit that first ADDED ``rel`` (later edits do not move it); None if never committed."""
    try:
        out = _git(Path(repo_dir), "log", "--diff-filter=A", "--format=%cI", "--", rel).split()
    except (OSError, subprocess.SubprocessError):
        return None
    if not out:
        return None
    return min(datetime.fromisoformat(s).astimezone(UTC) for s in out)


def _parse_utc(s: str) -> datetime:
    dt = datetime.fromisoformat(s)
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt.astimezone(UTC)


def check_charter_precedes_ledger(
    repo_dir: Path, rel: str, family: str, *, ledger: pd.DataFrame | None = None
) -> list[str]:
    """G-RESEARCH 8 ordering: the charter's add-commit is STRICTLY earlier than the family's first ledger row.

    ``ledger`` is a frame with ``family`` and ``created_at`` (the shape of ``firm.research.ledger.trials()``); default reads the
    real host ledger. A family with no ledger rows passes; a family with rows and no committed charter fails.
    """
    if ledger is None:
        # lazy: the host ledger is only read on the real path
        from firm.research import ledger as L

        ledger = L.trials(family=family)
    rows = ledger[ledger["family"] == family]
    stamps = [_parse_utc(str(s)) for s in rows["created_at"] if isinstance(s, str) and s]
    if len(rows) == 0:
        return []
    if not stamps:
        return [f"{family}: ledger rows have no created_at timestamp; ordering cannot be verified"]
    first = min(stamps)
    added = charter_added_time(repo_dir, rel)
    if added is None:
        return [f"{rel}: not committed in git but family {family} already has ledger rows"]
    if not added < first:
        return [f"{rel}: charter commit {added.isoformat()} is not strictly earlier than first ledger row {first.isoformat()}"]
    return []


# ------------------------------------------------------------------ one file

def validate_charter_file(
    path: Path, *, repo_dir: Path | None = None, ledger: pd.DataFrame | None = None, strict: bool = False,
    check_ordering: bool = True,
) -> list[str]:
    """All checks for one committed charter. Reads committed files only; no market data."""
    repo = Path(repo_dir or REPO_DIR)
    path = Path(path)
    try:
        fm, body = parse_charter(path.read_text())
    except CharterError as exc:
        return [f"{path.name}: {exc}"]
    problems = validate_front_matter(
        fm, gates_sha256=gates_sha256(repo), gates_path_length_days=_gates_path_length(repo), strict=strict
    )
    problems += validate_body(body)
    if check_ordering and isinstance(fm.get("family"), str):
        rel = path.resolve().relative_to(repo.resolve()).as_posix()
        problems += check_charter_precedes_ledger(repo, rel, fm["family"], ledger=ledger)
    return [f"{path.name}: {x}" if not x.startswith(path.name) else x for x in problems]
