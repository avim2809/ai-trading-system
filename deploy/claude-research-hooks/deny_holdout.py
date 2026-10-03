#!/usr/bin/env python3
"""PreToolUse hook for RESEARCH sessions only (ticket P0-04; owner decisions OD-05, OD-07).

TEMPLATE: the owner installs this file root-owned (mode 0755) at ``/etc/claude-code/hooks/deny_holdout.py``
together with ``research_freeze.deny.json`` at ``/etc/claude-code/research_freeze.deny.json`` (mode 0644), and
registers it only in the research unix user's own root-owned ``~/.claude/settings.json``. It is never registered
in the committed project settings or in the host-wide managed settings, which also bind root ops sessions.

Behaviour. Reads the PreToolUse JSON on stdin. Denies a call when

* any checked string contains ``HOLDOUT_UNSEAL``;
* a checked string names a path under the deny list (``deny_paths`` of ``config/research_freeze.yaml``);
* a checked string names a guardrail file (AGENTS.md rule 5) together with a write verb (redirect, ``tee``,
  ``sed -i``, ``cp``/``mv``/``rm``, ``python -c ... open(..., 'w')``, ``git checkout``/``git restore`` ...), or a native
  Edit/Write tool targets one;
* a ``data/`` path carries an ISO date on or after ``seal_date``.

Every other call: exit 0 with NO output, so the normal permission flow and deny/ask rules still apply
(emitting ``allow`` would skip the permission prompt and loosen research sessions). Malformed input or a missing
deny list FAILS CLOSED (deny).

Checked strings: for ``Bash`` the command; for ``Edit``/``Write``/``MultiEdit``/``NotebookEdit`` only the path fields
(file content may legitimately mention sealed paths, e.g. in docs); for every other tool all string values (path,
pattern, glob). Limits, recorded in docs/GUARDRAIL_REDTEAM.md: it matches command TEXT only. ``grep -r`` or a Python script
that walks a directory without naming a sealed path is stopped only by the file ACL and the sandbox
(GitHub issues #24846 and #61208 report enforcement gaps), not by this hook.
"""

from __future__ import annotations

import json
import re
import sys

DENY_JSON = "/etc/claude-code/research_freeze.deny.json"

# AGENTS.md rule 5: files research sessions never write.
PROTECTED = (
    ".claude/settings.json",
    ".claude/settings.local.json",
    ".claude/hooks",
    "tests/integrity",
    "tests/test_live_import_isolation.py",
    "config/gates.yaml",
    "config/research_freeze.yaml",
    "research/preregistration",
    "research/charters",
    "research/approvals",
    "deploy/claude-managed-settings.json",
    "deploy/claude-research-user-settings.json",
    "deploy/claude-research-hooks",
    ".github/CODEOWNERS",
)
WRITE_TOOLS = frozenset({"Edit", "Write", "MultiEdit", "NotebookEdit"})
PATH_FIELDS = ("file_path", "notebook_path", "path")

_LEFT = r"(?<![A-Za-z0-9_.-])"
_RIGHT = r"(?![A-Za-z0-9_])(?!\.[A-Za-z])"
_NOT_SEP = r"[^|&;\n]*"


def _path_regex(entry: str) -> re.Pattern[str]:
    body = re.escape(entry.rstrip("/")).replace(r"\*", r"[^\s'\"/]*")
    return re.compile(_LEFT + body + _RIGHT)


def _verbs_regex(path_rx: str) -> list[re.Pattern[str]]:
    p = path_rx
    return [
        re.compile(r">>?\s*['\"]?[^\s;|&<>'\"]*" + p),  # redirect into the path
        re.compile(r"\btee\b" + _NOT_SEP + p),
        re.compile(r"\b(?:sed|perl)\b" + _NOT_SEP + r"\s-[A-Za-z-]*i" + _NOT_SEP + p),
        re.compile(r"\b(?:cp|mv|rm|ln|truncate|chmod|chown|install|dd|rsync|unlink)\b" + _NOT_SEP + p),
        re.compile(r"\bgit\s+(?:checkout|restore|apply|reset|rm|stash)\b" + _NOT_SEP + p),
    ]


def _deny(reason: str) -> dict:
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        }
    }


def load_config(path: str = DENY_JSON) -> dict:
    with open(path, encoding="utf-8") as fh:
        cfg = json.load(fh)
    if not isinstance(cfg.get("deny_paths"), list) or not cfg.get("seal_date"):
        raise ValueError("deny list is missing deny_paths or seal_date")
    return cfg


def _strings(payload: dict) -> list[str]:
    tool = payload.get("tool_name", "")
    tool_input = payload.get("tool_input") or {}
    if tool == "Bash":
        return [str(tool_input.get("command", ""))]
    if tool in WRITE_TOOLS:
        return [str(tool_input[f]) for f in PATH_FIELDS if isinstance(tool_input.get(f), str)]
    out: list[str] = []

    def walk(value: object) -> None:
        if isinstance(value, str):
            out.append(value)
        elif isinstance(value, dict):
            for v in value.values():
                walk(v)
        elif isinstance(value, list):
            for v in value:
                walk(v)

    walk(tool_input)
    return out


def _writes_protected(command: str, tool: str, text: str) -> str | None:
    for entry in PROTECTED:
        rx = _path_regex(entry)
        if not rx.search(text):
            continue
        if tool in WRITE_TOOLS:
            return entry
        if tool != "Bash":
            continue
        path_rx = rx.pattern
        if any(v.search(command) for v in _verbs_regex(path_rx)):
            return entry
        py_write = re.search(r"open\(|write_text\(|write_bytes\(", command) and re.search(
            r"""['"](?:w|a|wb|ab|w\+|r\+)['"]|write_text\(|write_bytes\(""", command
        )
        if py_write:
            return entry
    return None


def decide(payload: dict, config: dict | None = None) -> dict | None:
    """Return a PreToolUse deny decision, or ``None`` (no output) for every non-matching call."""
    cfg = config if config is not None else load_config()
    tool = str(payload.get("tool_name", ""))
    seal = str(cfg["seal_date"])
    deny_rx = [(entry, _path_regex(entry)) for entry in cfg["deny_paths"]]
    for text in _strings(payload):
        if "HOLDOUT_UNSEAL" in text:
            return _deny("HOLDOUT_UNSEAL is human-held; research sessions never reference the unseal token")
        for entry, rx in deny_rx:
            if rx.search(text):
                return _deny(f"{entry} holds post-seal data (seal_date {seal}); research sessions may not touch it")
        guarded = _writes_protected(text if tool == "Bash" else "", tool, text)
        if guarded:
            return _deny(f"{guarded} is a protected guardrail file (AGENTS.md rule 5); report instead of editing")
        if "data/" not in text:
            continue
        for token in re.split(r"\s+", text):
            if "data/" not in token:
                continue
            for iso in re.findall(r"\d{4}-\d{2}-\d{2}", token):
                if iso >= seal:
                    return _deny(f"data path references {iso} >= seal_date {seal}")
    return None


def main() -> int:
    try:
        payload = json.load(sys.stdin)
        decision = decide(payload)
    except Exception as exc:  # fail closed: the hook cannot judge this call
        decision = _deny(f"deny_holdout hook could not evaluate the call ({type(exc).__name__}: {exc}); failing closed")
    if decision is not None:
        sys.stdout.write(json.dumps(decision))
    return 0


if __name__ == "__main__":
    sys.exit(main())
