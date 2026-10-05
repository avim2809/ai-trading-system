"""Draft of the owner-committed docs-index integrity test (agents may not write the protected suite)."""

from pathlib import Path

DOCS = Path(__file__).resolve().parents[2] / "docs"  # protected suite dir -> repo root


def test_llm_policy_exists_and_is_indexed():
    assert (DOCS / "LLM_POLICY.md").exists()
    assert "LLM_POLICY.md" in (DOCS / "README.md").read_text()


def test_llm_policy_required_clauses():
    text = (DOCS / "LLM_POLICY.md").read_text()
    assert "post-cutoff evaluation" in text
    assert "G-RESEARCH and G-PAPER" in text
