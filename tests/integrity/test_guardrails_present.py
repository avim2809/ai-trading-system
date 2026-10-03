"""Agent guardrails must exist, say what the plan says, and stay in sync (ticket P0-04).

Reads COMMITTED files only (never ``.claude/settings.local.json``). CODEOWNERS-protected: if a test
here looks wrong, report it; do not change it to make the files pass.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
HOOKS = REPO_ROOT / "deploy" / "claude-research-hooks"
DENY_HOLDOUT = HOOKS / "deny_holdout.py"
STOP_HOOK = HOOKS / "stop_integrity.sh"

SEALED_READ_TOKENS = ("data_alpaca", "s2_forward", "forward_monitors")


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _freeze() -> dict:
    return yaml.safe_load((REPO_ROOT / "config" / "research_freeze.yaml").read_text())


def _deny_rules(settings: dict) -> list[str]:
    return list(settings.get("permissions", {}).get("deny", []))


def _read_denies(settings: dict) -> list[str]:
    return [r for r in _deny_rules(settings) if r.startswith("Read(")]


def _assert_ops_safe(settings: dict, label: str) -> None:
    """Rules that bind root ops sessions too: no sealed-data Read denies, no sandbox, no hook gating."""
    for rule in _read_denies(settings):
        assert rule == "Read(./.env)", f"{label}: ops-incompatible Read deny {rule}"
    for rule in _deny_rules(settings):
        assert not any(tok in rule for tok in SEALED_READ_TOKENS), f"{label}: {rule}"
    for key in ("sandbox", "allowManagedHooksOnly"):
        assert key not in settings, f"{label}: must not set {key}"
    assert "disableBypassPermissionsMode" not in settings.get("permissions", {}), label
    assert "disableBypassPermissionsMode" not in settings, label


# ----------------------------------------------------------------------------- committed settings

def test_project_settings_deny_guardrail_edits_and_force_push():
    settings = _json(REPO_ROOT / ".claude" / "settings.json")
    deny = _deny_rules(settings)
    for required in (
        "Edit(./.claude/settings.json)",
        "Write(./.claude/settings.json)",
        "Edit(./.claude/settings.local.json)",
        "Edit(./.claude/hooks/**)",
        "Edit(./tests/integrity/**)",
        "Write(./tests/integrity/**)",
        "Edit(./tests/test_live_import_isolation.py)",
        "Edit(./config/gates.yaml)",
        "Edit(./config/research_freeze.yaml)",
        "Edit(./research/preregistration/**)",
        "Edit(./research/charters/**)",
        "Edit(./research/approvals/**)",
        "Edit(./research/monitoring_sealed/**)",
        "Bash(git push --force*)",
        "Bash(git push -f*)",
    ):
        assert required in deny, required
    assert not any("worktrees" in r for r in deny), ".claude/worktrees/** must stay editable"
    assert "hooks" not in settings, "the committed project settings must register no hooks"
    _assert_ops_safe(settings, ".claude/settings.json")
    assert "Bash(git *)" not in deny and not any(r.startswith("Bash(systemctl") for r in deny)
    assert not any(r.startswith("Bash(curl") for r in deny)


def test_settings_local_json_is_not_committed():
    tracked = subprocess.run(
        ["git", "ls-files", ".claude/settings.local.json"], cwd=REPO_ROOT, capture_output=True, text=True, check=False
    )
    if tracked.returncode == 0:
        assert tracked.stdout.strip() == ""
    assert ".claude/settings.local.json" in (REPO_ROOT / ".gitignore").read_text()


def test_managed_settings_template_is_ops_compatible():
    settings = _json(REPO_ROOT / "deploy" / "claude-managed-settings.json")
    _assert_ops_safe(settings, "managed-settings")
    assert "hooks" not in settings
    deny = _deny_rules(settings)
    for required in ("Edit(./.claude/settings.json)", "Edit(./tests/integrity/**)", "Edit(./config/gates.yaml)",
                     "Bash(git push --force*)"):
        assert required in deny, required


# ----------------------------------------------------------------------------- research profile

def test_research_user_settings_template():
    settings = _json(REPO_ROOT / "deploy" / "claude-research-user-settings.json")
    freeze = _freeze()
    assert settings["permissions"]["disableBypassPermissionsMode"] == "disable"
    allow = settings.get("permissions", {}).get("allow", [])
    assert not any(a.startswith("Bash(systemctl") for a in allow)
    assert "Bash(systemctl *)" in _deny_rules(settings)

    live = "/local/store/git/ai-trading-system/"

    def relative(path: str) -> str:
        for prefix in ("./", "/" + live, live):  # project-relative, or the absolute live-checkout location
            if path.startswith(prefix):
                path = path[len(prefix):]
                break
        return path[:-3] if path.endswith("/**") else path

    read_denied = {relative(rule[len("Read("):-1]) for rule in _read_denies(settings)}
    read_denied.discard(".env")
    assert read_denied == set(freeze["deny_paths"])
    # both spellings are present: the research clone's relative paths AND the live checkout's absolute paths
    assert {r for r in _read_denies(settings) if r.startswith("Read(/")} , "absolute live-checkout Read denies missing"

    assert settings["sandbox"]["enabled"] is True
    assert settings["sandbox"]["allowUnsandboxedCommands"] is False
    sandbox_denied = {relative(p) for p in settings["sandbox"]["filesystem"]["denyRead"]}
    assert sandbox_denied == set(freeze["deny_paths"])

    commands = {
        hook["command"]
        for event in settings["hooks"].values()
        for group in event
        for hook in group["hooks"]
    }
    assert commands == {"/etc/claude-code/hooks/deny_holdout.py", "/etc/claude-code/hooks/stop_integrity.sh"}
    assert not any("CLAUDE_PROJECT_DIR" in c for c in commands), "hooks must not run from the project dir"


def test_deny_json_template_equals_freeze_config():
    deny = _json(HOOKS / "research_freeze.deny.json")
    freeze = _freeze()
    assert deny["seal_date"] == freeze["seal_date"].isoformat()
    assert deny["allow_roots"] == freeze["allow_roots"]
    assert deny["deny_paths"] == freeze["deny_paths"]


@pytest.mark.parametrize("path", [DENY_HOLDOUT, STOP_HOOK])
def test_hook_templates_exist_and_are_executable(path):
    assert path.is_file(), path
    assert path.stat().st_mode & stat.S_IXUSR, f"{path} must be executable"
    assert not (REPO_ROOT / ".claude" / "hooks").exists(), "hooks are templates under deploy/, not in .claude/hooks"


# ----------------------------------------------------------------------------- CODEOWNERS / CI / docs

def test_codeowners_cover_protected_paths():
    text = (REPO_ROOT / ".github" / "CODEOWNERS").read_text()
    owned = {ln.split()[0] for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")}
    for required in (
        "/AGENTS.md", "/CLAUDE.md", "/.claude/", "/deploy/claude-managed-settings.json",
        "/deploy/claude-research-user-settings.json", "/deploy/claude-research-hooks/", "/tests/integrity/",
        "/tests/test_live_import_isolation.py", "/config/gates.yaml", "/config/research_freeze.yaml",
        "/research/preregistration/", "/research/charters/", "/research/approvals/",
        "/scripts/*_preregistered*.py", "/docs/*trial_history.json", "/.github/",
    ):
        assert required in owned, f"CODEOWNERS missing {required}"
    assert "advisory" in text.lower(), "CODEOWNERS must say it is advisory (OD-06, no second identity)"


def test_ci_has_push_trigger_and_full_history_for_integrity():
    ci = yaml.safe_load((REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text())
    triggers = ci.get("on") or ci.get(True)  # PyYAML reads the bare key `on` as True
    assert "pull_request" in triggers and "push" in triggers
    assert triggers["push"]["branches"] == ["main"]
    backend = ci["jobs"]["backend"]["steps"]
    checkout = next(s for s in backend if str(s.get("uses", "")).startswith("actions/checkout"))
    assert checkout["with"]["fetch-depth"] == 0

    integ = yaml.safe_load((REPO_ROOT / ".github" / "workflows" / "integrity.yml").read_text())
    itriggers = integ.get("on") or integ.get(True)
    assert "push" in itriggers and "pull_request" in itriggers
    text = (REPO_ROOT / ".github" / "workflows" / "integrity.yml").read_text()
    assert "FIRM_INTEGRITY_CI" in text and "fetch-depth: 0" in text and "tests/integrity" in text


def test_agents_md_rules_and_scope():
    text = (REPO_ROOT / "AGENTS.md").read_text()
    hard = text.split("## Hard rules", 1)[1].split("\n## ", 1)[0]
    numbers = [int(m.group(1)) for m in re.finditer(r"^(\d+)\. ", hard, flags=re.M)]
    assert numbers == list(range(1, 14)), numbers
    for heading in ("## Session types", "## Mission", "## Rules of thumb", "## Workflow per ticket",
                    "## Definition of done"):
        assert heading in text, heading
    assert "OPS session" in text and "RESEARCH session" in text

    # rule 1 lists exactly the freeze deny_paths and does not forbid the pre-seal inputs
    rule1 = hard.split("2. **Ledger", 1)[0]
    forbidden_part = rule1.split("Pre-seal research inputs", 1)[0]
    for entry in _freeze()["deny_paths"]:
        assert entry.rstrip("/") in forbidden_part.replace("`", ""), f"rule 1 must list {entry}"
    assert "data/research/eodhd" not in forbidden_part
    for allowed in _freeze()["allow_roots"][:3]:
        assert allowed in rule1

    # rule 8 and 10 scoped to lifecycle-managed strategies, legacy pipeline grandfathered (OD-08)
    assert "grandfathered" in text
    # CLAUDE.md "Rules of thumb" were folded in verbatim in substance
    for phrase in ("resolve_live_startup()", "IBKRBroker.connect()", "asyncio.to_thread()", "bare `except: pass`",
                   "frontend/src/api/{types,client}.ts", "update_news_guard"):
        assert phrase in text, phrase


def test_claude_md_imports_agents_and_has_no_stale_text():
    claude = (REPO_ROOT / "CLAUDE.md").read_text()
    assert claude.splitlines()[0].strip() == "@AGENTS.md"
    assert "Pipeline" in claude, "DEPLOY.md cites the pipeline overview in CLAUDE.md"
    for name in ("AGENTS.md", "CLAUDE.md"):
        text = (REPO_ROOT / name).read_text().lower()
        assert "sleeved capital" not in text, name
        assert "not live yet" not in text, name
    assert "allocation" in claude.lower() and "2026-09-30" in claude


def test_research_integrity_cursor_rule_and_redteam_doc_exist():
    rule = (REPO_ROOT / ".cursor" / "rules" / "research-integrity.mdc").read_text()
    assert "research_freeze.yaml" in rule and "alwaysApply" in rule
    red = (REPO_ROOT / "docs" / "GUARDRAIL_REDTEAM.md").read_text()
    for needle in ("disableAllHooks", "settings.local.json", "tests/integrity", "config/gates.yaml",
                   "data/research/eodhd", "kill_switch_state.json", "systemctl status", "hotfix"):
        assert needle in red, needle
    assert "GUARDRAIL_REDTEAM.md" in (REPO_ROOT / "docs" / "README.md").read_text()


# ----------------------------------------------------------------------------- deny_holdout hook

def _load_hook():
    spec = importlib.util.spec_from_file_location("deny_holdout_template", DENY_HOLDOUT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def hook():
    return _load_hook()


@pytest.fixture(scope="module")
def deny_cfg():
    return _json(HOOKS / "research_freeze.deny.json")


def _decide(hook, deny_cfg, tool, **tool_input):
    return hook.decide({"tool_name": tool, "tool_input": tool_input, "cwd": "/x"}, config=deny_cfg)


def _is_deny(decision) -> bool:
    out = decision["hookSpecificOutput"]
    return out["hookEventName"] == "PreToolUse" and out["permissionDecision"] == "deny" and bool(
        out["permissionDecisionReason"]
    )


@pytest.mark.parametrize(
    ("tool", "tool_input"),
    [
        ("Read", {"file_path": "data/forward_monitors/x"}),
        ("Read", {"file_path": "/local/store/git/ai-trading-system/data_alpaca/kill_switch_state.json"}),
        ("Grep", {"pattern": "fill", "path": "data/research/s2_forward"}),
        ("Read", {"file_path": "docs/s2_forward_snapshot.json"}),
        ("Read", {"file_path": "data/live_state.db"}),
        ("Read", {"file_path": "data/llm_cache_ab_llm.db"}),
        ("Bash", {"command": "cat data/cycle_history.json | head"}),
        ("Bash", {"command": "ls research/monitoring_sealed/allocation_forward"}),
        ("Bash", {"command": "echo $HOLDOUT_UNSEAL_TOKEN"}),
        ("Bash", {"command": "echo x > .claude/settings.local.json"}),
        ("Bash", {"command": "echo '{}' >> .claude/hooks/deny_holdout.py"}),
        ("Bash", {"command": "echo x > tests/integrity/x"}),
        ("Bash", {"command": "echo x > config/gates.yaml"}),
        ("Bash", {"command": "tee config/research_freeze.yaml < /dev/null"}),
        ("Bash", {"command": "sed -i 's/a/b/' tests/integrity/test_holdout_guard.py"}),
        ("Bash", {"command": "cp /tmp/x research/preregistration/INDEX.yaml"}),
        ("Bash", {"command": "mv /tmp/x research/charters/a.md"}),
        ("Bash", {"command": "git checkout HEAD~1 -- config/gates.yaml"}),
        ("Bash", {"command": "git restore tests/integrity/test_guardrails_present.py"}),
        ("Bash", {"command": "python -c \"open('tests/test_live_import_isolation.py','w').write('')\""}),
        ("Bash", {"command": "cat data/research/eodhd/etfs/2026-10-05.csv"}),
        ("Edit", {"file_path": "tests/integrity/test_holdout_guard.py", "old_string": "a", "new_string": "b"}),
        ("Write", {"file_path": "/work/.claude/settings.json", "content": "{}"}),
        ("Write", {"file_path": "research/approvals/x.yaml", "content": "a"}),
    ],
)
def test_deny_holdout_denies(hook, deny_cfg, tool, tool_input):
    decision = _decide(hook, deny_cfg, tool, **tool_input)
    assert decision is not None and _is_deny(decision), (tool, tool_input, decision)


@pytest.mark.parametrize(
    ("tool", "tool_input"),
    [
        ("Bash", {"command": "ls docs/"}),
        ("Bash", {"command": "git status"}),
        ("Bash", {"command": "grep -rn foo tests/integrity/ 2>&1 | head"}),  # read-only use of a protected path
        ("Bash", {"command": "cat tests/integrity/test_holdout_guard.py"}),
        ("Bash", {"command": "pytest -q tests/integrity"}),
        ("Read", {"file_path": "data/research/eodhd/x"}),
        ("Read", {"file_path": "data/research/eodhd/etfs_full/SPY.parquet"}),
        ("Read", {"file_path": "src/firm/data/cache.py"}),
        ("Read", {"file_path": "config/research_freeze.yaml"}),
        ("Read", {"file_path": "docs/HOLDOUT_POLICY.md"}),
        ("Edit", {"file_path": "src/firm/validation/sharpe_stats.py", "old_string": "a", "new_string": "b"}),
        ("Write", {"file_path": ".claude/worktrees/P0-04/notes.md", "content": "x"}),
        ("Bash", {"command": "cat data/research/eodhd/etfs/2026-09-30.csv"}),
    ],
)
def test_deny_holdout_passes_everything_else_with_no_decision(hook, deny_cfg, tool, tool_input):
    assert _decide(hook, deny_cfg, tool, **tool_input) is None


def test_deny_holdout_script_protocol(tmp_path):
    """stdout is empty and exit 0 for a non-matching call; JSON deny for a matching one."""
    deny_file = tmp_path / "deny.json"
    shutil.copy(HOOKS / "research_freeze.deny.json", deny_file)
    script = tmp_path / "deny_holdout.py"
    script.write_text(DENY_HOLDOUT.read_text().replace("/etc/claude-code/research_freeze.deny.json", str(deny_file)))

    def run(payload: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run([sys.executable, str(script)], input=payload, capture_output=True, text=True, check=False)

    ok = run(json.dumps({"tool_name": "Bash", "tool_input": {"command": "ls docs/"}, "cwd": "/x"}))
    assert ok.returncode == 0 and ok.stdout == ""
    bad = run(json.dumps({"tool_name": "Read", "tool_input": {"file_path": "data/forward_monitors/x"}, "cwd": "/x"}))
    assert bad.returncode == 0
    assert json.loads(bad.stdout)["hookSpecificOutput"]["permissionDecision"] == "deny"
    garbled = run("{not json")
    assert json.loads(garbled.stdout)["hookSpecificOutput"]["permissionDecision"] == "deny"  # fail closed
    deny_file.unlink()
    missing = run(json.dumps({"tool_name": "Bash", "tool_input": {"command": "ls"}, "cwd": "/x"}))
    assert json.loads(missing.stdout)["hookSpecificOutput"]["permissionDecision"] == "deny"  # fail closed


# ----------------------------------------------------------------------------- stop_integrity hook

class _Stop:
    """A hermetic copy of stop_integrity.sh with its absolute roots pointed into tmp_path."""

    def __init__(self, tmp_path: Path, *, user: str | None = None):
        self.tmp = tmp_path
        self.wt_root = tmp_path / "research" / ".claude" / "worktrees"
        self.venv_root = tmp_path / "venvs"
        self.live_venv = tmp_path / "live" / ".venv"
        self.skip_log = tmp_path / "skips.log"
        self.wt = self.wt_root / "P0-04"
        (self.wt / "src" / "firm").mkdir(parents=True)
        (self.wt / "src" / "firm" / "__init__.py").write_text("")
        self.main_checkout = tmp_path / "live" / "ai-trading-system"
        self.main_checkout.mkdir(parents=True)
        me = subprocess.run(["id", "-un"], capture_output=True, text=True, check=True).stdout.strip()
        text = STOP_HOOK.read_text()
        subs = {
            "/local/store/research/ai-trading-system/.claude/worktrees": str(self.wt_root),
            "/local/store/research-venvs": str(self.venv_root),
            "/local/store/git/ai-trading-system/.venv": str(self.live_venv),
            "/var/log/claude-research/stop_hook_skips.log": str(self.skip_log),
            "RESEARCH_USER=research": f"RESEARCH_USER={user or me}",
        }
        for old, new in subs.items():
            assert old in text, f"stop_integrity.sh must define {old!r} as a top-level constant"
            text = text.replace(old, new)
        self.script = tmp_path / "stop_integrity.sh"
        self.script.write_text(text)
        self.script.chmod(0o755)
        self.me = me

    def fake_python(self, path: Path, *, pytest_rc: int, verify_rc: int = 0) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "#!/bin/sh\n"
            'case "$*" in\n'
            f'  *"-m pytest"*) echo "FAKE PYTEST SUMMARY: rc={pytest_rc}"; touch "{self.tmp}/pytest_ran"; exit {pytest_rc};;\n'
            f"  *) exit {verify_rc};;\n"
            "esac\n"
        )
        path.chmod(0o755)
        return path

    def run(self, *, project_dir: Path | None = None, stdin: str = "{}") -> subprocess.CompletedProcess[str]:
        env = {**os.environ, "CLAUDE_PROJECT_DIR": str(project_dir or self.wt)}
        return subprocess.run(["bash", str(self.script)], input=stdin, env=env, capture_output=True, text=True,
                              timeout=120, check=False)

    @property
    def pytest_ran(self) -> bool:
        return (self.tmp / "pytest_ran").exists()


def test_stop_hook_failing_pytest_blocks_with_exit_2(tmp_path):
    stop = _Stop(tmp_path)
    stop.fake_python(stop.wt / ".venv" / "bin" / "python", pytest_rc=1)
    out = stop.run()
    assert out.returncode == 2, out.stderr
    assert "FAKE PYTEST SUMMARY" in out.stderr


def test_stop_hook_passing_pytest_allows_stop(tmp_path):
    stop = _Stop(tmp_path)
    stop.fake_python(stop.wt / ".venv" / "bin" / "python", pytest_rc=0)
    out = stop.run()
    assert out.returncode == 0, out.stderr
    assert stop.pytest_ran


def test_stop_hook_uses_shared_research_venv_by_depset(tmp_path):
    stop = _Stop(tmp_path)
    (stop.wt / ".research-depset").write_text("ml\n")
    stop.fake_python(stop.venv_root / "ml" / "bin" / "python", pytest_rc=0)
    assert stop.run().returncode == 0 and stop.pytest_ran


def test_stop_hook_active_flag_never_blocks_twice(tmp_path):
    stop = _Stop(tmp_path)
    stop.fake_python(stop.wt / ".venv" / "bin" / "python", pytest_rc=1)
    out = stop.run(stdin=json.dumps({"stop_hook_active": True}))
    assert out.returncode == 0 and not stop.pytest_ran


def test_stop_hook_does_nothing_in_the_main_checkout(tmp_path):
    stop = _Stop(tmp_path)
    stop.fake_python(stop.wt / ".venv" / "bin" / "python", pytest_rc=1)
    out = stop.run(project_dir=stop.main_checkout)
    assert out.returncode == 0 and not stop.pytest_ran


def test_stop_hook_refuses_the_live_venv_interpreter(tmp_path):
    stop = _Stop(tmp_path)
    live_python = stop.fake_python(stop.live_venv / "bin" / "python", pytest_rc=0)
    (stop.wt / ".venv" / "bin").mkdir(parents=True)
    (stop.wt / ".venv" / "bin" / "python").symlink_to(live_python)
    out = stop.run()
    assert out.returncode == 2 and not stop.pytest_ran
    assert "no safe interpreter" in out.stderr.lower()


def test_stop_hook_refuses_interpreter_that_does_not_import_the_worktree(tmp_path):
    stop = _Stop(tmp_path)
    stop.fake_python(stop.wt / ".venv" / "bin" / "python", pytest_rc=0, verify_rc=1)
    out = stop.run()
    assert out.returncode == 2 and not stop.pytest_ran


def test_stop_hook_without_interpreter_fails_closed_for_research_user(tmp_path):
    stop = _Stop(tmp_path)
    out = stop.run()
    assert out.returncode == 2 and "no safe interpreter" in out.stderr.lower()
    assert not stop.skip_log.exists()


def test_stop_hook_without_interpreter_skips_and_logs_for_ops_user(tmp_path):
    stop = _Stop(tmp_path, user="someone-else")
    stop.skip_log.parent.mkdir(parents=True, exist_ok=True)
    out = stop.run()
    assert out.returncode == 0
    assert stop.skip_log.exists() and "P0-04" in stop.skip_log.read_text()
