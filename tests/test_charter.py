"""Charter front-matter parser, validator and ordering check (P5-02). Synthetic fixtures and tmp git repos only."""

from __future__ import annotations

import copy
import os
import subprocess
from pathlib import Path

import pandas as pd
import pytest
import yaml

from firm.research import charter as C

REPO = Path(__file__).resolve().parents[1]
CH_DIR = "research/charters"
GATES_SHA = "a" * 64


def _front(**over) -> dict:
    d = {
        "family": "etf_trend", "charter_version": 1, "evidence_base": "Hurst, Ooi, Pedersen 2017",
        "rebalance_frequency": "weekly review", "expected_turnover": "about 2x a year one-way",
        "expected_annual_cost_bps": "about 15", "expected_worst_year": "a loss in the low teens",
        "expected_longest_flat_months": "up to 48", "correlation_expectations": "0.5 vs SPY",
        "falsification": "no net Sharpe edge over 60/40 in most decades", "gates_yaml_sha256": GATES_SHA,
        "mechanism_committed_at_utc": "2026-10-10T00:00:00Z", "tau": 0.09,
        "tau_derivation": "plan/drafts/P5-02/tau_note.md", "gross_cap": 1.0, "gross_cap_bound_ceiling": 0.20,
        "max_dd_procedure": {
            "method": "stationary_bootstrap", "block_length": "politis_white", "draws": 10000, "quantile": 0.95,
            "seed": 20261004, "path_length_days": 2520, "path_length_rationale": "fixed by gates.yaml",
            "applied_to": "selected_config_1x_cost",
        },
        "max_dd_analytic_2p5_tau": True, "survival_reference": "max(bootstrap_p95, 2.5*tau)",
        "stress_reference": "bootstrap_p95_at_episode_length", "expected_sharpe_haircut_min": 0.5,
        "combined_forecast_avg_abs_expected": 5, "speed_weights_rule": "equal_within_group_over_surviving_speeds",
        "decommission": {"soft_dd_mult": 1.0, "hard_dd_mult": 1.5, "triggers": ["dd", "cusum", "fidelity_2_months", "mechanism"]},
        "approved_by": "owner", "approved_commit": "deadbeef",
    }
    d.update(over)
    return d


BODY = "# Charter\n\n## 1. Mechanism\n\ntext\n\n## 7b. Falsification\n\nwhat would kill it\n"


def _doc(front: dict, body: str = BODY) -> str:
    return "---\n" + yaml.safe_dump(front, sort_keys=False) + "---\n" + body


# ------------------------------------------------------------------ parsing

def test_parse_roundtrip():
    fm, body = C.parse_charter(_doc(_front()))
    assert fm["family"] == "etf_trend" and fm["max_dd_procedure"]["draws"] == 10000
    assert body.startswith("# Charter")


@pytest.mark.parametrize("text", ["no front matter", "---\nfamily: [unclosed\n---\nbody", "---\n- a\n- b\n---\nbody"])
def test_parse_rejects_bad_front_matter(text):
    with pytest.raises(C.CharterError):
        C.parse_charter(text)


# ------------------------------------------------------------------ front-matter validation

def test_valid_front_matter_has_no_problems():
    assert C.validate_front_matter(_front(), gates_sha256=GATES_SHA) == []


@pytest.mark.parametrize("key", C.REQUIRED_KEYS)
def test_every_required_key_is_enforced(key):
    fm = _front()
    del fm[key]
    assert any(key in p for p in C.validate_front_matter(fm, gates_sha256=GATES_SHA))


def test_missing_path_length_days():
    fm = _front()
    del fm["max_dd_procedure"]["path_length_days"]
    assert any("path_length_days" in p for p in C.validate_front_matter(fm))


@pytest.mark.parametrize(
    "over",
    [
        {"survival_reference": "bootstrap_p95"},
        {"survival_reference": "max(bootstrap_p95, 2*tau)"},
        {"gates_yaml_sha256": "b" * 64},
        {"max_dd_analytic_2p5_tau": False},
        {"expected_sharpe_haircut_min": 0.4},
        {"gross_cap_bound_ceiling": 0.5},
        {"gross_cap": 1.5},
        {"tau": 0.0},
        {"tau": 9},  # percent typed as a fraction
        {"evidence_base": "<fill in>"},
        {"decommission": {"soft_dd_mult": 1.0, "hard_dd_mult": 1.5, "triggers": ["dd"]}},
    ],
)
def test_rejects_bad_values(over):
    assert C.validate_front_matter(_front(**over), gates_sha256=GATES_SHA)


@pytest.mark.parametrize(
    "key,val",
    [("draws", 9999), ("quantile", 0.99), ("block_length", 10), ("method", "iid"), ("path_length_days", 0),
     ("path_length_days", 1260)],
)
def test_rejects_bad_procedure(key, val):
    fm = _front()
    fm["max_dd_procedure"][key] = val
    assert C.validate_front_matter(fm, gates_path_length_days=2520)


def test_survival_reference_whitespace_tolerant():
    assert C.validate_front_matter(_front(survival_reference="max( bootstrap_p95 , 2.5 * tau )")) == []


def test_approval_fields_may_be_blank_unless_strict():
    fm = _front(approved_by="<owner>", approved_commit="<sha>", mechanism_committed_at_utc="<from git>")
    assert C.validate_front_matter(fm) == []
    assert C.validate_front_matter(fm, strict=True)


def test_validation_does_not_mutate():
    fm = _front()
    snap = copy.deepcopy(fm)
    C.validate_front_matter(fm, gates_sha256=GATES_SHA)
    assert fm == snap


# ------------------------------------------------------------------ body

def test_body_requires_falsification_and_mechanism_sections():
    assert C.validate_body(BODY) == []
    assert any("Falsification" in p for p in C.validate_body("# C\n\n## Mechanism\n\nx\n"))
    assert any("Mechanism" in p for p in C.validate_body("# C\n\n## Falsification\n\nx\n"))


@pytest.mark.parametrize(
    "line",
    ["Max drawdown: 23%", "max-DD = 21.5 %", "survival_dd is 18%", "The maximum drawdown of -31.2% is expected", "stress_dd at 25%"],
)
def test_body_rejects_numeric_max_dd_literal(line):
    assert any("max-drawdown" in p for p in C.validate_body(BODY + "\n" + line + "\n"))


@pytest.mark.parametrize(
    "line",
    ["The SG Trend Index max drawdown 20.61% is a published figure.", "Hard decommission DD = 1.5 x survival_dd.",
     "tau = 9% with 2.5 x tau as a rule of thumb"],
)
def test_body_allows_references_and_multiples(line):
    assert C.validate_body(BODY + "\n" + line + "\n") == []


# ------------------------------------------------------------------ files + ordering (synthetic git repo)

def _git(d: Path, *args: str, when: str) -> None:
    env = {**os.environ, "GIT_COMMITTER_DATE": when, "GIT_AUTHOR_DATE": when}
    subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", "-c", "commit.gpgsign=false", *args],
                   cwd=d, check=True, capture_output=True, env=env)


def _repo(tmp_path: Path, when: str = "2026-11-01T00:00:00+00:00") -> Path:
    _git(tmp_path, "init", "-q", when=when)
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "gates.yaml").write_text("x: 1\n")
    (tmp_path / CH_DIR).mkdir(parents=True)
    (tmp_path / CH_DIR / ".gitkeep").write_text("")
    _git(tmp_path, "add", "-A", when=when)
    _git(tmp_path, "commit", "-qm", "base", when=when)
    return tmp_path


def _commit_charter(repo: Path, name: str, text: str, when: str) -> str:
    rel = f"{CH_DIR}/{name}"
    (repo / rel).write_text(text)
    _git(repo, "add", rel, when=when)
    _git(repo, "commit", "-qm", f"add {name}", when=when)
    return rel


def _ledger(*rows: tuple[str, str]) -> pd.DataFrame:
    return pd.DataFrame([{"family": f, "created_at": t} for f, t in rows], columns=["family", "created_at"])


def test_charter_added_time_uses_first_add_not_later_edit(tmp_path):
    repo = _repo(tmp_path)
    rel = _commit_charter(repo, "etf_trend.md", _doc(_front()), "2026-11-02T00:00:00+00:00")
    (repo / rel).write_text(_doc(_front()) + "\nedit\n")
    _git(repo, "commit", "-qam", "edit", when="2026-11-09T00:00:00+00:00")
    assert C.charter_added_time(repo, rel).isoformat().startswith("2026-11-02T00:00:00")


def test_ordering_ok_when_charter_strictly_earlier(tmp_path):
    repo = _repo(tmp_path)
    rel = _commit_charter(repo, "etf_trend.md", _doc(_front()), "2026-11-02T00:00:00+00:00")
    led = _ledger(("etf_trend", "2026-11-03T00:00:00Z"), ("other", "2026-10-01T00:00:00Z"))
    assert C.check_charter_precedes_ledger(repo, rel, "etf_trend", ledger=led) == []


@pytest.mark.parametrize("ledger_ts", ["2026-11-02T00:00:00Z", "2026-11-01T12:00:00Z"])
def test_ordering_fails_when_equal_or_ledger_earlier(tmp_path, ledger_ts):
    repo = _repo(tmp_path)
    rel = _commit_charter(repo, "etf_trend.md", _doc(_front()), "2026-11-02T00:00:00+00:00")
    led = _ledger(("etf_trend", "2026-11-05T00:00:00Z"), ("etf_trend", ledger_ts))
    assert C.check_charter_precedes_ledger(repo, rel, "etf_trend", ledger=led)


def test_ordering_no_ledger_rows_is_fine_but_uncommitted_charter_with_rows_fails(tmp_path):
    repo = _repo(tmp_path)
    missing = f"{CH_DIR}/none.md"
    assert C.check_charter_precedes_ledger(repo, missing, "etf_trend", ledger=_ledger()) == []
    assert C.check_charter_precedes_ledger(
        repo, missing, "etf_trend", ledger=_ledger(("etf_trend", "2026-11-05T00:00:00Z")))


def test_validate_charter_file_end_to_end(tmp_path):
    repo = _repo(tmp_path)
    sha = C.gates_sha256(repo)
    rel = _commit_charter(repo, "etf_trend.md", _doc(_front(gates_yaml_sha256=sha)), "2026-11-02T00:00:00+00:00")
    assert C.validate_charter_file(repo / rel, repo_dir=repo, ledger=_ledger()) == []
    bad = _commit_charter(repo, "etf_breakout.md", _doc(_front(family="etf_breakout")), "2026-11-02T00:00:00+00:00")
    assert any("gates_yaml_sha256" in p for p in C.validate_charter_file(repo / bad, repo_dir=repo, ledger=_ledger()))


def test_iter_charters_skips_template(tmp_path):
    d = tmp_path / CH_DIR
    d.mkdir(parents=True)
    for n in ("TEMPLATE.md", "a.md", "b.md", "notes.txt"):
        (d / n).write_text("x")
    assert [p.name for p in C.iter_charters(tmp_path)] == ["a.md", "b.md"]


# ------------------------------------------------------------------ the draft template parses and mirrors the schema

def test_template_draft_front_matter_has_every_required_key():
    tpl = REPO / CH_DIR / "TEMPLATE.md"
    if not tpl.exists():
        tpl = REPO / "plan" / "drafts" / "P5-02" / "TEMPLATE.md"
    fm, body = C.parse_charter(tpl.read_text())
    assert set(C.REQUIRED_KEYS) <= set(fm)
    assert C.validate_body(body) == []
    # unfilled template placeholders must be reported as problems, never silently accepted
    assert C.validate_front_matter(fm)
