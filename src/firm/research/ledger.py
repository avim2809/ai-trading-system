"""Host-level append-only, hash-chained trial ledger (ticket P1-01, decisions OD-09 / OD-10).

Every backtest leaves one immutable :class:`TrialRecord` so that N in the Deflated Sharpe Ratio
counts every configuration ever tried, failures included. The canonical file lives OUTSIDE every
worktree (``/local/store/research-ledger/trials.jsonl``) and is written only through
:func:`record_trial` under ``fcntl.flock``. Row format, one JSON object per line::

    {"seq": n, "prev_hash": h_{n-1}, "row": {...TrialRecord...}, "row_hash": h_n}
    h_n = sha256(h_{n-1} + canonical_json(row)),  h_0 = "0" * 64

There is deliberately no update / delete / exclude API. Importing this package must never happen
from a live module (``tests/test_live_import_isolation.py``).
"""

from __future__ import annotations

import dataclasses
import fcntl
import functools
import hashlib
import inspect
import json
import logging
import os
import subprocess
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import pandas as pd

log = logging.getLogger(__name__)

__all__ = [
    "LEDGER_ROOT_ENV",
    "REPO_DIR_ENV",
    "ChainReport",
    "DirtyTreeError",
    "LedgerCorruptError",
    "LedgerNotProvisionedError",
    "Mode",
    "TrialHandle",
    "TrialRecord",
    "UnapprovedPreregError",
    "backtest_logged",
    "canonical_json",
    "config_hash",
    "record_trial",
    "run_trial",
    "trials",
    "verify_chain",
]

LEDGER_ROOT_ENV = "FIRM_RESEARCH_LEDGER_ROOT"  # tests override; default /local/store/research-ledger
REPO_DIR_ENV = "FIRM_LEDGER_REPO_DIR"  # git worktree whose HEAD / dirtiness is recorded (tests override)
ALLOW_DIRTY_ENV = "FIRM_LEDGER_ALLOW_DIRTY"
DEFAULT_ROOT = Path("/local/store/research-ledger")
GENESIS = "0" * 64
LEDGER_FILE = "trials.jsonl"
_DIRTY_IGNORED = ("runs/", "data/", "review/", "research/ledger/")  # outputs, not code
_ERR_MAX = 2000

Mode = Literal["registered", "exploratory", "unregistered", "legacy"]
_MODES = ("registered", "exploratory", "unregistered", "legacy")


class DirtyTreeError(RuntimeError):
    """A ``registered`` trial was attempted on a dirty (or unknown) git tree."""


class UnapprovedPreregError(RuntimeError):
    """A ``registered`` trial whose preregistration is not approved or does not cover the config (P1-09)."""


class LedgerNotProvisionedError(RuntimeError):
    """The ledger root is missing, not writable, or is the host root under pytest."""


class LedgerCorruptError(RuntimeError):
    """The ledger file has a partial / unparseable tail; appending is refused."""


@dataclass(frozen=True)
class TrialRecord:
    trial_id: str
    family: str
    mode: Mode
    config: dict
    config_hash: str
    code_commit: str
    data_snapshot_id: str | None
    seed: int | None
    start: str | None
    end: str | None
    returns_path: str | None
    gross_sharpe: float | None  # per-period (NOT annualised)
    net_sharpe: float | None
    periods_per_year: int | None
    sharpe_conversion: str | None
    n_obs: int | None
    skew: float | None
    kurt: float | None  # raw (non-excess)
    preregistration_id: str | None
    touched_holdout: bool
    status: Literal["completed", "failed"]
    error: str | None
    count_is_estimate: bool = False
    n_variants: int = 1
    source_file: str | None = None
    source_entry_index: int | None = None
    created_at: str = ""


@dataclass(frozen=True)
class ChainReport:
    ok: bool
    n_rows: int
    first_bad_row: int | None
    warnings: tuple[str, ...] = ()
    partial_tail: bool = False  # trailing bytes without newline (crashed writer); ok refers to complete rows


# ------------------------------------------------------------------ helpers

def canonical_json(obj: Any) -> str:
    """Sorted keys, compact separators, NaN/inf rejected, NO ``default=`` (unserialisable raises)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def config_hash(config: dict) -> str:
    return hashlib.sha256(canonical_json(config).encode()).hexdigest()


def _chain_hash(prev: str, row: dict) -> str:
    return hashlib.sha256((prev + canonical_json(row)).encode()).hexdigest()


def _root(*, for_write: bool) -> Path:
    env = os.environ.get(LEDGER_ROOT_ENV)
    root = Path(env) if env else DEFAULT_ROOT
    if for_write:
        if not env and os.environ.get("PYTEST_CURRENT_TEST"):
            raise LedgerNotProvisionedError(
                f"refusing the default host ledger root {DEFAULT_ROOT} under pytest; "
                f"set {LEDGER_ROOT_ENV} to a temp dir"
            )
        if not root.is_dir() or not os.access(root, os.W_OK | os.X_OK):
            raise LedgerNotProvisionedError(
                f"ledger root {root} is missing or not writable. The owner must provision it: "
                f"mkdir -p {root}/returns {root}/inbox; chgrp -R research {root}; chmod 2775 {root} "
                f"{root}/returns {root}/inbox"
            )
    return root


def _repo_dir() -> Path:
    env = os.environ.get(REPO_DIR_ENV)
    return Path(env) if env else Path(__file__).resolve().parents[3]


def _git_state(repo: Path) -> tuple[str | None, list[str]]:
    """(HEAD sha or None, porcelain lines outside the ignored output dirs)."""
    try:
        sha = subprocess.run(
            ["git", "-c", "safe.directory=*", "rev-parse", "HEAD"],
            cwd=repo, capture_output=True, text=True, check=True, timeout=30,
        ).stdout.strip()
        out = subprocess.run(
            ["git", "-c", "safe.directory=*", "status", "--porcelain", "--untracked-files=all"],
            cwd=repo, capture_output=True, text=True, check=True, timeout=60,
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        log.warning("ledger: cannot read git state in %s: %r", repo, exc)
        return None, []
    dirty = []
    for ln in out.splitlines():
        path = ln[3:].split(" -> ")[-1].strip('"')
        if not path.startswith(_DIRTY_IGNORED):
            dirty.append(ln)
    return sha, dirty


def _jsonable_dict(d: dict) -> dict:
    return json.loads(canonical_json(d))


# ------------------------------------------------------------------ writer

def _read_last(f) -> tuple[int, str] | None:
    """(seq, row_hash) of the last line, reading from the end only; None if empty."""
    f.seek(0, os.SEEK_END)
    size = f.tell()
    if size == 0:
        return None
    f.seek(size - 1)
    if f.read(1) != b"\n":
        raise LedgerCorruptError("ledger tail has no trailing newline (partial write); refusing to append")
    pos, buf = size, b""
    while pos > 0:
        step = min(65536, pos)
        pos -= step
        f.seek(pos)
        buf = f.read(step) + buf
        body = buf[:-1] if buf.endswith(b"\n") else buf
        if b"\n" in body:
            break
    last = buf[:-1].rsplit(b"\n", 1)[-1]
    try:
        d = json.loads(last)
        return int(d["seq"]), str(d["row_hash"])
    except (ValueError, KeyError, TypeError) as exc:
        raise LedgerCorruptError(f"unparseable last ledger line: {exc!r}") from exc


def _append_locked(path: Path, row: dict) -> None:
    with open(path, "a+b") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            last = _read_last(f)
            seq, prev = (0, GENESIS) if last is None else (last[0] + 1, last[1])
            line = canonical_json(
                {"seq": seq, "prev_hash": prev, "row": row, "row_hash": _chain_hash(prev, row)}
            ).encode() + b"\n"
            f.seek(0, os.SEEK_END)
            f.write(line)
            f.flush()
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def _validate(rec: TrialRecord) -> None:
    if rec.mode not in _MODES:
        raise ValueError(f"unknown mode {rec.mode!r}")
    if rec.status not in ("completed", "failed"):
        raise ValueError(f"unknown status {rec.status!r}")
    if rec.mode == "registered" and not rec.preregistration_id:
        raise ValueError("mode='registered' requires preregistration_id")
    if rec.mode in ("exploratory", "unregistered") and rec.preregistration_id is not None:
        raise ValueError(f"mode={rec.mode!r} must have preregistration_id=None")
    if (rec.gross_sharpe is not None or rec.net_sharpe is not None) and rec.periods_per_year is None:
        raise ValueError("a Sharpe is present but periods_per_year is None")
    if rec.config_hash != config_hash(rec.config):
        raise ValueError("config_hash does not match canonical_json(config)")
    if rec.n_variants < 0:
        raise ValueError("n_variants must be >= 0")


def record_trial(rec: TrialRecord, returns: pd.Series | None = None) -> str:
    """Append ``rec`` (and its returns parquet) to the ledger; return the trial_id."""
    _validate(rec)
    root = _root(for_write=True)
    config = _jsonable_dict(rec.config)
    mode, prereg, commit = rec.mode, rec.preregistration_id, rec.code_commit

    if mode != "legacy":
        sha, dirty = _git_state(_repo_dir())
        if mode == "registered" and (sha is None):
            raise DirtyTreeError("cannot determine git HEAD for a registered trial")
        commit = sha or "unknown"
        if dirty:
            digest = hashlib.sha256("\n".join(sorted(dirty)).encode()).hexdigest()
            prov = dict(config.get("_provenance", {}))
            prov["dirty_paths_digest"] = digest
            commit = f"{commit}+dirty"
            if mode == "registered":
                if os.environ.get(ALLOW_DIRTY_ENV) != "1":
                    raise DirtyTreeError(f"registered trial on a dirty tree ({len(dirty)} paths); commit first")
                prov["downgraded_from_prereg"] = prereg
                mode, prereg = "exploratory", None
                log.warning("ledger: registered trial downgraded to exploratory (dirty tree, %s=1)", ALLOW_DIRTY_ENV)
            config["_provenance"] = prov

    returns_path = rec.returns_path
    if returns is not None:
        expected = f"returns/{rec.trial_id}.parquet"
        if returns_path is not None and returns_path != expected:
            raise ValueError(f"returns_path must be {expected!r}")
        returns_path = expected
        (root / "returns").mkdir(exist_ok=True)
        final = root / returns_path
        tmp = root / "returns" / f".{rec.trial_id}.{os.getpid()}.tmp"
        frame = returns.rename("returns").to_frame()
        frame.attrs = {}  # inherited attrs (e.g. DataFrames from benchmark frames) are not JSON-serialisable; values are unaffected
        frame.to_parquet(tmp)
        os.replace(tmp, final)
    elif returns_path is not None and not (root / returns_path).exists() and rec.mode != "legacy":
        raise ValueError(f"returns_path {returns_path!r} given without returns and file is missing")

    err = rec.error[:_ERR_MAX] if rec.error else rec.error
    row = dataclasses.asdict(
        dataclasses.replace(
            rec, mode=mode, preregistration_id=prereg, code_commit=commit, config=config,
            returns_path=returns_path, error=err, created_at=datetime.now(UTC).isoformat(),
        )
    )
    _append_locked(root / LEDGER_FILE, row)  # the append is the commit point
    log.info("ledger: recorded trial %s family=%s mode=%s status=%s", rec.trial_id, rec.family, mode, rec.status)
    return rec.trial_id


# ------------------------------------------------------------------ reader / verifier

def verify_chain(path: Path | None = None) -> ChainReport:
    p = Path(path) if path else _root(for_write=False) / LEDGER_FILE
    if not p.exists():
        return ChainReport(True, 0, None)
    data = p.read_bytes()
    partial = bool(data) and not data.endswith(b"\n")
    raw = data.split(b"\n")
    tail = raw.pop()  # b"" if file ends with newline, else the partial line
    prev, n = GENESIS, 0
    for i, ln in enumerate(raw):
        try:
            d = json.loads(ln)
            ok = (
                d["seq"] == i and d["prev_hash"] == prev and d["row_hash"] == _chain_hash(prev, d["row"])
                and set(d) == {"seq", "prev_hash", "row", "row_hash"}
            )
        except (ValueError, KeyError, TypeError):
            ok = False
        if not ok:
            return ChainReport(False, i, i, partial_tail=partial)
        prev, n = d["row_hash"], i + 1
    warnings: list[str] = []
    if partial:
        warnings.append(f"partial trailing line of {len(tail)} bytes (crashed writer)")
    rdir = p.parent / "returns"
    if rdir.is_dir():
        known = set()
        for ln in raw:
            rp = json.loads(ln)["row"].get("returns_path")
            if rp:
                known.add(Path(rp).name)
        orphans = sorted(x.name for x in rdir.glob("*.parquet") if x.name not in known)
        if orphans:
            warnings.append(f"{len(orphans)} orphan returns parquet file(s) without a row, e.g. {orphans[0]}")
    return ChainReport(True, n, None, tuple(warnings), partial)


def trials(family: str | None = None, mode: str | None = None) -> pd.DataFrame:
    """Read-only copy of the ledger rows (one row per TrialRecord), plus ``seq``."""
    p = _root(for_write=False) / LEDGER_FILE
    cols = [f.name for f in dataclasses.fields(TrialRecord)]
    rows: list[dict] = []
    if p.exists():
        for ln in p.read_bytes().split(b"\n"):
            if not ln:
                continue
            try:
                d = json.loads(ln)
            except ValueError:
                break  # partial tail of a crashed writer
            rows.append({"seq": d["seq"], **d["row"]})
    df = pd.DataFrame(rows, columns=["seq", *cols])
    if family is not None:
        df = df[df["family"] == family]
    if mode is not None:
        df = df[df["mode"] == mode]
    return df.reset_index(drop=True).copy()


# ------------------------------------------------------------------ run_trial / backtest_logged

_SETTABLE = {
    "data_snapshot_id", "seed", "start", "end", "gross_sharpe", "net_sharpe", "periods_per_year",
    "sharpe_conversion", "n_obs", "skew", "kurt", "touched_holdout", "n_variants",
}


@dataclass
class TrialHandle:
    """Mutable bag the body of :func:`run_trial` fills in; turned into a TrialRecord on exit."""

    fields: dict[str, Any] = field(default_factory=dict)
    returns: pd.Series | None = None

    def set(self, **kw: Any) -> None:
        bad = set(kw) - _SETTABLE
        if bad:
            raise TypeError(f"cannot set {sorted(bad)}; allowed: {sorted(_SETTABLE)}")
        self.fields.update(kw)


@contextmanager
def run_trial(
    family: str, config: dict, *, mode: Mode = "exploratory", prereg: str | None = None,
    prereg_config: dict | None = None,
) -> Iterator[TrialHandle]:
    """Record a trial in ``finally``; on exception write ``status='failed'`` and re-raise.

    Failed trials count toward N (copies the pattern at ``experiments/runner.py:62-82``).

    ``mode="registered"`` first requires ``firm.research.prereg.is_approved(prereg, prereg_config or config)``;
    otherwise :class:`UnapprovedPreregError` is raised BEFORE the body runs and nothing is recorded (nothing ran).
    ``prereg_config`` is the parameter dict checked against the pre-registered grid (default: ``config``).
    """
    if mode == "registered":
        from firm.research import prereg as _prereg  # lazy: keeps ledger import light

        if not prereg or not _prereg.is_approved(prereg, prereg_config if prereg_config is not None else config):
            raise UnapprovedPreregError(
                f"preregistration {prereg!r} is not approved or does not cover this config; "
                "use mode='exploratory' (counted, never promotable)"
            )
    h = TrialHandle()
    status: Literal["completed", "failed"] = "completed"
    error: str | None = None
    try:
        yield h
    except BaseException as exc:
        status, error = "failed", repr(exc)[:_ERR_MAX]
        raise
    finally:
        fields = dict(h.fields)
        if h.returns is not None and "n_obs" not in fields:
            fields["n_obs"] = int(h.returns.notna().sum())
        tid = uuid.uuid4().hex
        rec = TrialRecord(
            trial_id=tid, family=family, mode=mode, config=config, config_hash=config_hash(config),
            code_commit="", data_snapshot_id=fields.pop("data_snapshot_id", None),
            seed=fields.pop("seed", config.get("seed") if isinstance(config.get("seed"), int) else None),
            start=fields.pop("start", None), end=fields.pop("end", None), returns_path=None,
            gross_sharpe=fields.pop("gross_sharpe", None), net_sharpe=fields.pop("net_sharpe", None),
            periods_per_year=fields.pop("periods_per_year", None),
            sharpe_conversion=fields.pop("sharpe_conversion", None), n_obs=fields.pop("n_obs", None),
            skew=fields.pop("skew", None), kurt=fields.pop("kurt", None),
            preregistration_id=prereg, touched_holdout=bool(fields.pop("touched_holdout", False)),
            status=status, error=error, n_variants=int(fields.pop("n_variants", 1)),
        )
        try:
            record_trial(rec, returns=h.returns)
        except DirtyTreeError:
            raise
        except Exception:
            if status == "failed":  # never mask the original exception
                log.exception("ledger: could not record failed trial %s", tid)
            else:
                raise


def backtest_logged(
    family: str, *, mode: Mode = "exploratory", prereg: str | None = None, fixed_args: tuple[str, ...] = ()
):
    """Decorator: log every call of the wrapped backtest function as one trial.

    ``config`` = the bound call arguments (strict canonical JSON; if an argument is not
    serialisable the trial is still recorded with ``args_repr`` and ``config_is_lossy=True`` rather
    than dropped). A ``pd.Series`` result is stored as the trial's returns.

    With ``mode="registered"`` the bound arguments, minus the names in ``fixed_args`` (data inputs that are
    not strategy parameters), must lie inside the approved pre-registration's grid (P1-09).
    """

    def deco(fn):
        sig = inspect.signature(fn)

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            cfg: dict = {"function": f"{fn.__module__}.{fn.__qualname__}", "args": dict(bound.arguments)}
            try:
                canonical_json(cfg)
            except (TypeError, ValueError):
                cfg = {
                    "function": cfg["function"], "config_is_lossy": True,
                    "args_repr": {k: repr(v)[:500] for k, v in bound.arguments.items()},
                }
            params = {k: v for k, v in bound.arguments.items() if k not in fixed_args}
            with run_trial(family, cfg, mode=mode, prereg=prereg, prereg_config=params) as h:
                result = fn(*args, **kwargs)
                if isinstance(result, pd.Series):
                    h.returns = result
                return result

        return wrapper

    return deco
