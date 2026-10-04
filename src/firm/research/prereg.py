"""Pre-registration index and approval check (ticket P1-09, decisions OD-06 / OD-13).

Two kinds of pre-registration exist:

* ``frozen_module`` - the committed ``scripts/*_preregistered*.py`` modules whose ``bars_fingerprint()``
  is the immutable record. They are INDEXED here, never replaced or edited.
* ``yaml`` - new families: a :class:`PreregSpec` YAML under ``research/preregistration/`` committed by the
  owner (CODEOWNERS-protected; this code never writes there). The index entry's ``fingerprint`` is then the
  :func:`spec_hash` recorded at approval.

:func:`is_approved` is what ``firm.research.ledger.run_trial`` calls for ``mode="registered"``. Any failure
returns ``False``; the ledger then raises ``UnapprovedPreregError`` (``mode="exploratory"`` stays possible:
logged, counted, never promotable). Timestamps come from git (committer time converted to UTC), never from
the module ``PREREGISTERED_AT`` constants (the host clock is Asia/Jerusalem).

Importing this module must never happen from a live module (``tests/test_live_import_isolation.py``).
"""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import yaml

__all__ = [
    "IndexEntry",
    "PreregSpec",
    "covers",
    "find_freeze_commit",
    "freeze_time_utc",
    "gates_hash",
    "grid_size",
    "is_approved",
    "load_index",
    "load_spec",
    "recompute_fingerprint",
    "spec_from_dict",
    "spec_hash",
    "validate_spec",
    "verify_index",
]

log = logging.getLogger(__name__)

REPO_DIR = Path(__file__).resolve().parents[3]
DEFAULT_INDEX = REPO_DIR / "research" / "preregistration" / "INDEX.yaml"
GATES_PATH = "config/gates.yaml"
POST_SEAL_DECLARATION = "No data after 2026-09-30 was examined."
_UNHASHED = ("approver", "approved_at_utc")
_STATUSES = ("APPROVED", "DRAFT", "SUPERSEDED", "RUN", "REJECTED")
_KINDS = ("frozen_module", "yaml")
_FP_SOURCES = ("trial_history", "freeze_commit", "none")
# config keys that are bookkeeping, not strategy parameters
_NON_PARAM_KEYS = frozenset({"seed", "function", "_provenance"})


@dataclass(frozen=True)
class IndexEntry:
    family: str
    kind: Literal["frozen_module", "yaml"]
    module_path: str | None
    yaml_path: str | None
    fingerprint: str  # frozen_module: bars_fingerprint(); yaml: spec_hash recorded at approval
    freeze_commit: str
    freeze_time_utc: str
    ledger_path: str | None
    fingerprint_source: Literal["trial_history", "freeze_commit", "none"]
    status: Literal["APPROVED", "DRAFT", "SUPERSEDED", "RUN", "REJECTED"]
    notes: str = ""


@dataclass(frozen=True)
class PreregSpec:
    family: str
    hypothesis: str
    mechanism_ref: str  # path to the approved charter (P5-02)
    universe: list[str]
    date_range: tuple[str, str]  # pre-seal only
    param_grid: dict[str, list]
    max_grid_size: int  # <= 12 for core_v1
    n_planned_trials: int
    metrics: list[str]
    gates_hash: str  # sha256 of config/gates.yaml
    charter_ref: str
    falsification: str
    post_seal_data_declaration: str  # POST_SEAL_DECLARATION
    approver: str
    approved_at_utc: str


# ------------------------------------------------------------------ helpers

def _canon(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _git(repo: Path, *args: str, timeout: int = 60) -> str:
    return subprocess.run(
        ["git", "-c", "safe.directory=*", *args], cwd=repo, capture_output=True, text=True, check=True,
        timeout=timeout,
    ).stdout


def gates_hash(repo_dir: Path | None = None) -> str:
    return hashlib.sha256((Path(repo_dir or REPO_DIR) / GATES_PATH).read_bytes()).hexdigest()


def freeze_time_utc(commit: str, repo_dir: Path | None = None) -> str:
    """Committer time of ``commit`` as an ISO-8601 UTC string (from git, never from module constants)."""
    iso = _git(Path(repo_dir or REPO_DIR), "log", "-1", "--format=%cI", commit).strip()
    return datetime.fromisoformat(iso).astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_utc(s: str) -> datetime:
    dt = datetime.fromisoformat(s)  # py>=3.11 parses a trailing Z
    if dt.tzinfo is None:
        raise ValueError(f"timestamp {s!r} has no timezone")
    return dt.astimezone(UTC)


# ------------------------------------------------------------------ spec

def spec_from_dict(d: Mapping[str, Any]) -> PreregSpec:
    names = {f.name for f in fields(PreregSpec)}
    unknown, missing = set(d) - names, names - set(d)
    if unknown or missing:
        raise ValueError(f"spec fields: unknown={sorted(unknown)} missing={sorted(missing)}")
    dd = dict(d)
    dr = dd["date_range"]
    if not isinstance(dr, (list, tuple)) or len(dr) != 2:
        raise TypeError("date_range must be [start, end]")
    dd["date_range"] = (str(dr[0]), str(dr[1]))
    return PreregSpec(**dd)


def load_spec(path: Path) -> PreregSpec:
    d = yaml.safe_load(Path(path).read_text())
    if not isinstance(d, dict):
        raise TypeError(f"{path}: not a mapping")
    return spec_from_dict(d)


def spec_hash(spec: PreregSpec) -> str:
    """sha256 of the canonical JSON of every field except approver / approved_at_utc."""
    d = {k: v for k, v in asdict(spec).items() if k not in _UNHASHED}
    d["date_range"] = list(d["date_range"])
    return hashlib.sha256(_canon(d).encode()).hexdigest()


def grid_size(param_grid: Mapping[str, list]) -> int:
    n = 1
    for v in param_grid.values():
        n *= len(v)
    return n


def validate_spec(spec: PreregSpec) -> list[str]:
    """Human-readable schema problems (empty list = valid)."""
    p: list[str] = []
    for name in ("family", "hypothesis", "mechanism_ref", "charter_ref", "falsification", "gates_hash"):
        if not isinstance(getattr(spec, name), str) or not getattr(spec, name).strip():
            p.append(f"{name} must be a non-empty string")
    if not spec.universe:
        p.append("universe is empty")
    if not spec.metrics:
        p.append("metrics is empty")
    if spec.post_seal_data_declaration != POST_SEAL_DECLARATION:
        p.append(f"post_seal_data_declaration must be exactly {POST_SEAL_DECLARATION!r}")
    try:
        s, e = (datetime.fromisoformat(x) for x in spec.date_range)
        if s >= e:
            p.append("date_range start must precede end")
        if e > datetime(2026, 9, 30):  # noqa: DTZ001 - naive on purpose, compared with naive ISO dates
            p.append("date_range end is after the 2026-09-30 burn date (post-seal data)")
    except ValueError:
        p.append("date_range entries must be ISO dates")
    if not spec.param_grid or any(not isinstance(v, list) or not v for v in spec.param_grid.values()):
        p.append("param_grid must map each parameter to a non-empty list")
    else:
        n = grid_size(spec.param_grid)
        if n > spec.max_grid_size:
            p.append(f"param_grid expands to {n} configs > max_grid_size {spec.max_grid_size}")
    if not (isinstance(spec.max_grid_size, int) and spec.max_grid_size >= 1):
        p.append("max_grid_size must be a positive int")
    elif not (isinstance(spec.n_planned_trials, int) and 1 <= spec.n_planned_trials <= spec.max_grid_size):
        p.append("n_planned_trials must be an int in [1, max_grid_size]")
    if len(spec.gates_hash) != 64:
        p.append("gates_hash must be a 64-hex sha256")
    if not spec.approver.strip():
        p.append("approver is empty")
    try:
        _parse_utc(spec.approved_at_utc)
    except ValueError:
        p.append("approved_at_utc must be a timezone-aware ISO timestamp")
    return p


def covers(spec: PreregSpec, config: dict) -> bool:
    """True iff ``config`` selects one point of the pre-registered grid.

    ``config`` is the parameter dict (or a dict with a ``params`` mapping). Every grid key must be present with
    a value from its grid list; any other key (except seed / function / _provenance) is a parameter OUTSIDE the
    pre-registration and fails; the grid must not expand beyond ``max_grid_size``.
    """
    params = config["params"] if isinstance(config.get("params"), dict) else config
    grid = spec.param_grid
    if not grid or grid_size(grid) > spec.max_grid_size:
        return False
    for k in params:
        if k not in grid and k not in _NON_PARAM_KEYS:
            return False
    for k, allowed in grid.items():
        if k not in params:
            return False
        try:
            if not any(params[k] == a and type(params[k]) is type(a) for a in allowed):
                return False
        except Exception:  # noqa: BLE001 - e.g. unhashable / incomparable config values
            return False
    return True


# ------------------------------------------------------------------ index

def load_index(path: Path | None = None) -> list[IndexEntry]:
    p = Path(path) if path else DEFAULT_INDEX
    d = yaml.safe_load(p.read_text())
    rows = d.get("entries") if isinstance(d, dict) else None
    if not isinstance(rows, list):
        raise ValueError(f"{p}: expected a mapping with an 'entries' list")  # noqa: TRY004
    names = {f.name for f in fields(IndexEntry)}
    out = []
    for r in rows:
        unknown = set(r) - names
        if unknown:
            raise ValueError(f"{p}: unknown IndexEntry fields {sorted(unknown)}")
        r = {"notes": "", **r}
        if r.get("kind") not in _KINDS or r.get("status") not in _STATUSES or r.get("fingerprint_source") not in _FP_SOURCES:
            raise ValueError(f"{p}: bad kind/status/fingerprint_source in entry {r.get('family')!r}")
        out.append(IndexEntry(**r))
    fams = [e.family for e in out]
    if len(set(fams)) != len(fams):
        raise ValueError(f"{p}: duplicate family in index")
    return out


_FP_CODE = (
    "import importlib, sys\n"
    "sys.path[:0] = sys.argv[2:]\n"
    "m = importlib.import_module(sys.argv[1])\n"
    "print('FP:' + m.bars_fingerprint())\n"
)


def _fingerprint_in_subprocess(stem: str, paths: list[str], cwd: Path) -> str:
    r = subprocess.run(
        [sys.executable, "-c", _FP_CODE, stem, *paths], cwd=cwd, capture_output=True, text=True, timeout=180, check=False,
        env={"PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1"},
    )
    lines = [ln for ln in r.stdout.splitlines() if ln.startswith("FP:")]
    if r.returncode != 0 or not lines:
        raise RuntimeError(f"cannot recompute fingerprint of {stem}: {r.stderr.strip()[-400:]}")
    return lines[-1][3:]


def recompute_fingerprint(module_path: str, repo_dir: Path | None = None) -> str:
    """Import the frozen module in a subprocess (no side effects on the parent) and call ``bars_fingerprint()``."""
    repo = Path(repo_dir or REPO_DIR)
    mp = (repo / module_path).resolve()
    if not mp.is_file():
        raise FileNotFoundError(mp)
    return _fingerprint_in_subprocess(mp.stem, [str(mp.parent), str(repo / "src")], repo)


def _fingerprint_at_commit(module_path: str, commit: str, repo: Path) -> str:
    src = _git(repo, "show", f"{commit}:{module_path}")
    with tempfile.TemporaryDirectory() as td:
        stem = Path(module_path).stem
        (Path(td) / f"{stem}.py").write_text(src)
        # temp copy first so the old content is imported; siblings still resolve from the repo scripts dir
        return _fingerprint_in_subprocess(stem, [td, str(repo / "scripts"), str(repo / "src")], repo)


def find_freeze_commit(module_path: str, repo_dir: Path | None = None) -> str:
    """Earliest commit after which ``bars_fingerprint()`` equals its HEAD value and never changed again."""
    repo = Path(repo_dir or REPO_DIR)
    head_fp = recompute_fingerprint(module_path, repo)
    commits = _git(repo, "log", "--format=%H", "--follow", "--", module_path).split()
    if not commits:
        raise RuntimeError(f"{module_path} has no git history")
    freeze = commits[0]
    for c in commits[1:]:
        try:
            fp = _fingerprint_at_commit(module_path, c, repo)
        except RuntimeError:
            break  # module did not import at that commit: the stable run starts above it
        if fp != head_fp:
            break
        freeze = c
    return freeze


def _ledger_fingerprints(ledger: Path) -> list[str]:
    d = json.loads(ledger.read_text())
    entries = d.get("entries", []) if isinstance(d, dict) else d
    return [e["fingerprint"] for e in entries if isinstance(e, dict) and e.get("fingerprint")]


def verify_index(index: list[IndexEntry], repo_dir: Path | None = None) -> list[str]:
    """Human-readable problems (empty = consistent). Never raises for a bad entry."""
    repo = Path(repo_dir or REPO_DIR)
    problems: list[str] = []
    for e in index:
        tag = f"[{e.family}]"
        if e.kind == "yaml":
            if not e.yaml_path or not (repo / e.yaml_path).is_file():
                problems.append(f"{tag} yaml_path missing or not a file: {e.yaml_path}")
                continue
            try:
                spec = load_spec(repo / e.yaml_path)
            except (ValueError, OSError, yaml.YAMLError) as exc:
                problems.append(f"{tag} unreadable spec: {exc}")
                continue
            if spec_hash(spec) != e.fingerprint:
                problems.append(f"{tag} spec_hash {spec_hash(spec)[:12]} != recorded {e.fingerprint[:12]}")
            if e.status == "APPROVED" and spec.gates_hash != gates_hash(repo):
                problems.append(f"{tag} gates_hash does not match current config/gates.yaml")
            continue
        if not e.module_path or not (repo / e.module_path).is_file():
            problems.append(f"{tag} module missing: {e.module_path}")
            continue
        try:
            head_fp = recompute_fingerprint(e.module_path, repo)
        except (RuntimeError, OSError, subprocess.SubprocessError) as exc:
            problems.append(f"{tag} {exc}")
            continue
        if head_fp != e.fingerprint:
            problems.append(f"{tag} recomputed fingerprint {head_fp[:12]} != recorded {e.fingerprint[:12]} (file changed after freeze?)")
        try:
            anc = subprocess.run(
                ["git", "-c", "safe.directory=*", "merge-base", "--is-ancestor", e.freeze_commit, "HEAD"],
                cwd=repo, capture_output=True, timeout=60, check=False,
            ).returncode
            if anc != 0:
                problems.append(f"{tag} freeze commit {e.freeze_commit[:10]} is not an ancestor of HEAD")
            elif freeze_time_utc(e.freeze_commit, repo) != e.freeze_time_utc:
                problems.append(f"{tag} freeze_time_utc differs from git committer time of {e.freeze_commit[:10]}")
        except (OSError, subprocess.SubprocessError):
            problems.append(f"{tag} cannot check freeze commit {e.freeze_commit[:10]}")
        if e.ledger_path:
            lp = repo / e.ledger_path
            if not lp.is_file():
                problems.append(f"{tag} ledger_path missing: {e.ledger_path}")
            elif e.fingerprint_source == "trial_history":
                fps = _ledger_fingerprints(lp)
                if not fps:
                    problems.append(f"{tag} trial_history source but {e.ledger_path} records no fingerprint")
                elif e.fingerprint not in fps:
                    problems.append(f"{tag} fingerprint not among those in {e.ledger_path}")
        elif e.fingerprint_source == "trial_history":
            problems.append(f"{tag} fingerprint_source=trial_history but no ledger_path")
        if e.fingerprint_source == "freeze_commit":
            try:
                at_freeze = _fingerprint_at_commit(e.module_path, e.freeze_commit, repo)
                if at_freeze != head_fp:
                    problems.append(f"{tag} freeze-commit fingerprint {at_freeze[:12]} != HEAD {head_fp[:12]}: drift")
                if at_freeze != e.fingerprint:
                    problems.append(f"{tag} recorded fingerprint != freeze-commit recomputation")
            except (RuntimeError, subprocess.SubprocessError) as exc:
                problems.append(f"{tag} cannot recompute at freeze commit: {exc}")
        if e.family == "futures_trend" and e.status not in ("DRAFT", "SUPERSEDED"):
            problems.append(f"{tag} futures_trend must be DRAFT or SUPERSEDED (OD-13)")
    return problems


# ------------------------------------------------------------------ approval

def _charter_commit_time(repo: Path, rel: str) -> datetime | None:
    try:
        out = _git(repo, "log", "-1", "--format=%cI", "--", rel).strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return datetime.fromisoformat(out).astimezone(UTC) if out else None


def is_approved(
    prereg_id: str, config: dict, *, index_path: Path | None = None, repo_dir: Path | None = None
) -> bool:
    """APPROVED yaml entry, unchanged spec hash, current gates hash, config inside the grid, charter older than approval.

    ``prereg_id`` is the index family name or the yaml file stem. Any failure (including unreadable files)
    returns False.
    """
    repo = Path(repo_dir or REPO_DIR)
    try:
        index = load_index(index_path)
        entry = next(
            (e for e in index if e.kind == "yaml" and (e.family == prereg_id or (e.yaml_path and Path(e.yaml_path).stem == prereg_id))),
            None,
        )
        if entry is None or entry.status != "APPROVED" or not entry.yaml_path:
            return False
        spec = load_spec(repo / entry.yaml_path)
        if validate_spec(spec) or spec_hash(spec) != entry.fingerprint:
            return False
        if spec.gates_hash != gates_hash(repo) or not covers(spec, config):
            return False
        if not (repo / spec.charter_ref).is_file():
            return False
        ct = _charter_commit_time(repo, spec.charter_ref)
        return ct is not None and ct < _parse_utc(spec.approved_at_utc)
    except Exception:  # fail closed: any unreadable/invalid state means "not approved"
        log.warning("prereg: is_approved(%r) failed closed", prereg_id, exc_info=True)
        return False
