"""Phase 2b: RAG-Triad evaluation harness."""

from __future__ import annotations

from firm.eval.rag_eval import (
    RagEvalCase,
    RagTriadEvaluator,
    make_llm_judge,
)


class TestLexicalTriad:
    def test_grounded_answer_scores_high(self):
        case = RagEvalCase(
            question="What was the Sharpe ratio of run_a?",
            contexts=["Run run_a achieved a Sharpe ratio of 1.5 over 2021."],
            answer="Run run_a had a Sharpe ratio of 1.5.",
        )
        s = RagTriadEvaluator().evaluate(case)
        assert s.groundedness > 0.6
        assert s.answer_relevance > 0.3
        assert s.context_relevance > 0.3

    def test_hallucinated_answer_scores_low_groundedness(self):
        case = RagEvalCase(
            question="What was the Sharpe ratio of run_a?",
            contexts=["Run run_a achieved a Sharpe ratio of 1.5 over 2021."],
            answer="The portfolio invested heavily in cryptocurrency derivatives "
                   "and lunar mining futures.",
        )
        s = RagTriadEvaluator().evaluate(case)
        assert s.groundedness < 0.34

    def test_no_context_zero_relevance(self):
        case = RagEvalCase(question="anything?", contexts=[], answer="something")
        s = RagTriadEvaluator().evaluate(case)
        assert s.context_relevance == 0.0
        assert s.groundedness == 0.0

    def test_evaluate_many_aggregates(self):
        cases = [
            RagEvalCase("q one", ["q one context"], "q one answer"),
            RagEvalCase("q two", ["q two context"], "q two answer"),
        ]
        agg = RagTriadEvaluator().evaluate_many(cases)
        assert agg["n"] == 2
        assert set(agg) >= {"context_relevance", "groundedness",
                            "answer_relevance", "mean"}
        assert 0.0 <= agg["mean"] <= 1.0

    def test_scores_serialisable(self):
        s = RagTriadEvaluator().evaluate(
            RagEvalCase("q", ["q ctx"], "q ans")
        )
        d = s.as_dict()
        assert set(d) == {"context_relevance", "groundedness",
                          "answer_relevance", "mean"}


class TestLLMJudge:
    def test_judge_parses_score_from_reply(self):
        class FakeLLM:
            def chat(self, messages, **kwargs):
                return "I would rate this 0.85 out of 1."
        judge = make_llm_judge(FakeLLM())
        evaluator = RagTriadEvaluator(judge=judge)
        s = evaluator.evaluate(RagEvalCase("q", ["ctx"], "ans"))
        assert s.groundedness == 0.85
        assert s.context_relevance == 0.85

    def test_judge_handles_bad_reply(self):
        class FakeLLM:
            def chat(self, messages, **kwargs):
                return "no number here"
        judge = make_llm_judge(FakeLLM())
        s = RagTriadEvaluator(judge=judge).evaluate(RagEvalCase("q", ["c"], "a"))
        assert s.groundedness == 0.0
