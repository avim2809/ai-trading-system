"""RAG-quality evaluation: the RAG Triad.

Retrieval grounding is *necessary but not sufficient* — LLMs introduce
unsupported or contradictory content even with relevant context — so RAG
output must be measured, not assumed correct. This module implements the
TruLens RAG Triad:

* **context_relevance** — are the retrieved chunks relevant to the question?
  (irrelevant context is what gets woven into hallucinations)
* **groundedness** — is each claim in the answer supported by the context?
* **answer_relevance** — does the answer actually address the question?

Two backends:

* a deterministic **lexical** scorer (default) — needs no network, so it runs
  in CI and gives a stable regression signal;
* an optional **LLM-as-judge** (:func:`make_llm_judge`) for higher-fidelity
  scoring when a provider is configured.

Known limitation (documented in the literature): groundedness checks whether
claims are *supported* by context, not whether the inference is *logically
correct* — a "grounded but wrong" answer can still score high.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

# Minimal stopword set for lexical overlap; intentionally small and dependency-free.
_STOPWORDS = frozenset(
    "the a an and or of to in for on at by is are was were be been being with "
    "as it its this that these those from into over under than then so such no "
    "not what which who whom how when where why did do does done has have had "
    "you your i we they he she them his her our their me my".split()
)

# A judge scores one triad metric in [0, 1] given (question, contexts, answer).
Judge = Callable[[str, str, list[str], str], float]


def _tokens(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", (text or "").lower())
    return {w for w in words if w not in _STOPWORDS and len(w) > 1}


def _overlap(a: set[str], b: set[str]) -> float:
    """Fraction of *a* covered by *b* (0 when *a* is empty)."""
    if not a:
        return 0.0
    return len(a & b) / len(a)


@dataclass
class TriadScores:
    """The three RAG-Triad metrics, each in [0, 1]."""

    context_relevance: float
    groundedness: float
    answer_relevance: float

    def mean(self) -> float:
        return (self.context_relevance + self.groundedness + self.answer_relevance) / 3.0

    def as_dict(self) -> dict[str, float]:
        return {
            "context_relevance": round(self.context_relevance, 4),
            "groundedness": round(self.groundedness, 4),
            "answer_relevance": round(self.answer_relevance, 4),
            "mean": round(self.mean(), 4),
        }


@dataclass
class RagEvalCase:
    """One evaluation example: a question, the retrieved contexts, the answer."""

    question: str
    contexts: list[str] = field(default_factory=list)
    answer: str = ""


class RagTriadEvaluator:
    """Scores RAG answers on the three RAG-Triad metrics."""

    def __init__(self, judge: Judge | None = None) -> None:
        self._judge = judge

    # ── lexical backend ─────────────────────────────────────────────

    @staticmethod
    def _lexical_context_relevance(question: str, contexts: list[str]) -> float:
        if not contexts:
            return 0.0
        q = _tokens(question)
        return sum(_overlap(q, _tokens(c)) for c in contexts) / len(contexts)

    @staticmethod
    def _lexical_groundedness(answer: str, contexts: list[str]) -> float:
        ctx = set().union(*(_tokens(c) for c in contexts)) if contexts else set()
        return _overlap(_tokens(answer), ctx)

    @staticmethod
    def _lexical_answer_relevance(question: str, answer: str) -> float:
        return _overlap(_tokens(question), _tokens(answer))

    # ── public scoring ──────────────────────────────────────────────

    def evaluate(self, case: RagEvalCase) -> TriadScores:
        if self._judge is not None:
            return TriadScores(
                context_relevance=self._judge(
                    "context_relevance", case.question, case.contexts, case.answer
                ),
                groundedness=self._judge(
                    "groundedness", case.question, case.contexts, case.answer
                ),
                answer_relevance=self._judge(
                    "answer_relevance", case.question, case.contexts, case.answer
                ),
            )
        return TriadScores(
            context_relevance=self._lexical_context_relevance(case.question, case.contexts),
            groundedness=self._lexical_groundedness(case.answer, case.contexts),
            answer_relevance=self._lexical_answer_relevance(case.question, case.answer),
        )

    def evaluate_many(self, cases: list[RagEvalCase]) -> dict[str, float]:
        """Mean triad scores across *cases* (empty → all zeros)."""
        if not cases:
            return {"context_relevance": 0.0, "groundedness": 0.0,
                    "answer_relevance": 0.0, "mean": 0.0, "n": 0}
        scored = [self.evaluate(c) for c in cases]
        n = len(scored)
        agg = {
            "context_relevance": sum(s.context_relevance for s in scored) / n,
            "groundedness": sum(s.groundedness for s in scored) / n,
            "answer_relevance": sum(s.answer_relevance for s in scored) / n,
        }
        agg["mean"] = sum(agg.values()) / 3.0
        agg = {k: round(v, 4) for k, v in agg.items()}
        agg["n"] = n
        return agg


def make_llm_judge(llm: object, model: str | None = None) -> Judge:
    """Build an LLM-as-judge scorer backed by an ``LLMService``-like object.

    The returned callable prompts the model for a single 0–1 score per metric
    and parses the first float in the reply (defaulting to 0.0 on failure).
    """
    _metric_prompts = {
        "context_relevance": (
            "Score from 0 to 1 how relevant the CONTEXT is to the QUESTION."
        ),
        "groundedness": (
            "Score from 0 to 1 how fully the ANSWER is supported by the CONTEXT "
            "(1 = every claim supported, 0 = unsupported/contradicted)."
        ),
        "answer_relevance": (
            "Score from 0 to 1 how well the ANSWER addresses the QUESTION."
        ),
    }

    def judge(metric: str, question: str, contexts: list[str], answer: str) -> float:
        instruction = _metric_prompts.get(metric, "Score from 0 to 1.")
        ctx = "\n".join(contexts)
        messages = [
            {"role": "system", "content":
                f"{instruction} Reply with ONLY a number between 0 and 1."},
            {"role": "user", "content":
                f"QUESTION:\n{question}\n\nCONTEXT:\n{ctx}\n\nANSWER:\n{answer}"},
        ]
        try:
            raw = llm.chat(messages, model=model, temperature=0.0)
            m = re.search(r"[01](?:\.\d+)?", raw)
            return max(0.0, min(1.0, float(m.group(0)))) if m else 0.0
        except Exception:
            return 0.0

    return judge
