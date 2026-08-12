"""
Day 14 — AI Evaluation & Benchmarking Pipeline
AICB-P1: AI Practical Competency Program, Phase 1

Key concepts from lecture:
    - Evaluation = Scientific Method for AI (Hypothesis → Experiment → Measure → Conclude → Iterate)
    - 4 nhóm metrics: Task Completion, Answer Quality, RAG-Specific, Business
    - RAG pipeline metrics: Context Recall → Context Precision → Faithfulness → Answer Relevancy
    - LLM-as-Judge: rubric scoring 1-5, detect bias (positional, verbosity, self-preference)
    - Golden dataset: stratified sampling (5 Easy + 7 Medium + 5 Hard + 3 Adversarial)
    - Failure taxonomy: hallucination, irrelevant, incomplete, off_topic, refusal
    - 5 Whys method for root cause analysis
    - CI/CD integration: eval as quality gate (score < threshold = block deploy)
    - Continuous Improvement Loop: Evaluate → Analyze → Improve → Augment → Repeat

Instructions:
    1. Fill in every required section marked with TODO.
    2. Do NOT change class/function signatures. The optional ``contexts``
       parameter in ``run_full_eval`` is part of the required interface.
    3. Copy this file to solution/solution.py when done.
    4. Run: pytest tests/ -v

The reranking helper is an optional bonus exercise and may remain unimplemented.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable


# ---------------------------------------------------------------------------
# Task 1 — Data Models (Golden Dataset + Evaluation Results)
# ---------------------------------------------------------------------------

@dataclass
class QAPair:
    """
    A question-answer pair for evaluation (part of the Golden Dataset).

    From lecture: Golden dataset cần có:
        - question: câu hỏi user
        - ground_truth (expected_answer): expert-written expected answer
        - context: source documents cần retrieve
        - metadata: difficulty (easy/medium/hard), category, source_docs

    Fields:
        question:        The question to answer.
        expected_answer: The reference/ground-truth answer (expert-written).
        context:            Source context (may be empty string if not applicable).
        metadata:           Optional metadata dict (difficulty, category, etc.).
        retrieved_contexts: List of retrieved chunks (ORDER = retriever rank).
                            Used by the retrieval-side metrics (Task 2b).
    """
    question: str
    expected_answer: str
    context: str = ""
    metadata: dict = field(default_factory=dict)
    retrieved_contexts: list = field(default_factory=list)


@dataclass
class EvalResult:
    """
    Evaluation result for a single Q&A pair.

    From lecture - RAG metrics pipeline:
        Question → Retriever → Context → Generator → Answer
        Each step has a metric: Context Recall, Context Precision, Faithfulness, Answer Relevancy

    From lecture - Score interpretation:
        0.8-1.0: Good (Monitor, maintain)
        0.6-0.8: Needs work (Analyze failures, iterate)
        < 0.6: Significant issues (Deep investigation required)

    Fields:
        qa_pair:        The original QAPair.
        actual_answer:  What the agent actually returned.
        faithfulness:   Float 0-1, how grounded the answer is in context.
        relevance:      Float 0-1, how relevant the answer is to the question.
        completeness:   Float 0-1, how complete the answer is vs expected.
        passed:         True if all three scores >= 0.5.
        failure_type:   None if passed, otherwise one of:
                        "hallucination", "irrelevant", "incomplete", "off_topic".
        context_precision: Float 0-1 or None — quality of retrieval ranking.
        context_recall:    Float 0-1 or None — coverage of expected by context.
                        (Both stay None unless retrieved chunks are supplied;
                         they are NOT part of overall_score().)
    """
    qa_pair: QAPair
    actual_answer: str
    faithfulness: float
    relevance: float
    completeness: float
    passed: bool
    failure_type: str | None = None
    context_precision: float | None = None
    context_recall: float | None = None

    def overall_score(self) -> float:
        """Compute the average of faithfulness, relevance, and completeness.

        Returns:
            (faithfulness + relevance + completeness) / 3.0
        """
        return (self.faithfulness + self.relevance + self.completeness) / 3.0


# ---------------------------------------------------------------------------
# Task 2 — RAGAS Evaluator (Simplified word-overlap heuristic)
# ---------------------------------------------------------------------------
# In production, replace with actual RAGAS framework:
#   from ragas import evaluate
#   from ragas.metrics import Faithfulness, AnswerRelevancy, ContextRecall, ContextPrecision
#
# Or DeepEval:
#   from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
#   assert_test(test_case, [faithfulness, hallucination])
#
# Or TruLens:
#   from trulens.core import Feedback
#   f_groundedness = Feedback(provider.groundedness_measure_with_cot_reasons)
# ---------------------------------------------------------------------------

# Common English stopwords are ignored so overlap reflects *content* words,
# not filler (otherwise "is"/"a"/"the" inflate every score).
STOPWORDS: set[str] = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "of", "in", "on", "at", "to", "for", "with", "as", "by", "and", "or",
    "it", "its", "this", "that", "these", "those", "from", "into", "than",
}


def _tokenize(text: str) -> set[str]:
    """Lowercase word tokenization, ignoring punctuation and stopwords."""
    if not text:
        return set()
    tokens = re.findall(r"\b\w+\b", text.lower())
    return {t for t in tokens if t not in STOPWORDS}


class RAGASEvaluator:
    """
    Evaluates RAG pipeline outputs using RAGAS-inspired heuristics.

    All metrics use word overlap rather than LLM calls for simplicity.
    Replace with actual LLM-based evaluation in production.
    """

    def evaluate_faithfulness(self, answer: str, context: str) -> float:
        """
        Measure how grounded the answer is in the context.

        Heuristic:
            answer_tokens = _tokenize(answer)
            context_tokens = _tokenize(context)
            faithfulness = |answer_tokens ∩ context_tokens| / |answer_tokens|
            Clamp to [0.0, 1.0]. Return 1.0 if answer is empty.

        Returns:
            float in [0.0, 1.0] — 1.0 = fully grounded in context.
        """
        answer_tokens = _tokenize(answer)
        if not answer_tokens:
            return 1.0
        context_tokens = _tokenize(context)
        faithfulness = len(answer_tokens & context_tokens) / len(answer_tokens)
        return min(max(faithfulness, 0.0), 1.0)

    def evaluate_relevance(self, answer: str, question: str) -> float:
        """
        Measure how relevant the answer is to the question.

        Heuristic:
            relevance = |answer_tokens ∩ question_tokens| / |question_tokens|
            Clamp to [0.0, 1.0]. Return 1.0 if question is empty.

        Returns:
            float in [0.0, 1.0]
        """
        question_tokens = _tokenize(question)
        if not question_tokens:
            return 1.0
        answer_tokens = _tokenize(answer)
        relevance = len(answer_tokens & question_tokens) / len(question_tokens)
        return min(max(relevance, 0.0), 1.0)

    def evaluate_completeness(self, answer: str, expected: str) -> float:
        """
        Measure how well the answer covers the expected answer.

        Heuristic:
            completeness = |answer_tokens ∩ expected_tokens| / |expected_tokens|
            Clamp to [0.0, 1.0]. Return 1.0 if expected is empty.

        Returns:
            float in [0.0, 1.0]
        """
        expected_tokens = _tokenize(expected)
        if not expected_tokens:
            return 1.0
        answer_tokens = _tokenize(answer)
        completeness = len(answer_tokens & expected_tokens) / len(expected_tokens)
        return min(max(completeness, 0.0), 1.0)

    # -----------------------------------------------------------------------
    # Task 2b — Retrieval-side metrics (evaluate the GET-CONTEXT step)
    # -----------------------------------------------------------------------
    # From lecture (RAG pipeline): Context Recall → Context Precision →
    #   Faithfulness → Answer Relevancy. The two below score the RETRIEVER,
    #   operating on a LIST of chunks (order = retriever rank).
    # -----------------------------------------------------------------------

    def evaluate_context_recall(self, contexts: list[str], expected: str) -> float:
        """Context Recall — how much of the expected answer is covered by the
        UNION of retrieved chunks.

        Heuristic:
            union_tokens = ⋃ _tokenize(chunk) for chunk in contexts
            recall = |expected_tokens ∩ union_tokens| / |expected_tokens|
            Clamp to [0.0, 1.0]. Return 1.0 if expected is empty.

        Low recall => retriever missed evidence the answer needs.
        """
        expected_tokens = _tokenize(expected)
        if not expected_tokens:
            return 1.0
        union_tokens: set[str] = set()
        for chunk in contexts:
            union_tokens.update(_tokenize(chunk))
        recall = len(expected_tokens & union_tokens) / len(expected_tokens)
        return min(max(recall, 0.0), 1.0)

    def evaluate_context_precision(
        self,
        contexts: list[str],
        expected: str,
        relevance_threshold: float = 0.1,
    ) -> float:
        """Context Precision — RANK-AWARE Average Precision (AP@K), like RAGAS.
        Rewards retrievers that place RELEVANT chunks BEFORE noise.

        Steps:
            1. A chunk is "relevant" if it covers >= relevance_threshold of the
               expected tokens:  |chunk ∩ expected| / |expected| >= threshold
            2. Precision@k = (#relevant in top-k) / k
            3. AP@K = (1 / #relevant) * Σ_k [ Precision@k · relevant_k ]

        Return 1.0 if expected empty; 0.0 if no chunks or none relevant.
        Reordering relevant chunks earlier (reranking) raises this score.
        """
        expected_tokens = _tokenize(expected)
        if not expected_tokens:
            return 1.0
        if not contexts:
            return 0.0

        rel_flags: list[int] = []
        for chunk in contexts:
            chunk_tokens = _tokenize(chunk)
            coverage = len(chunk_tokens & expected_tokens) / len(expected_tokens)
            rel_flags.append(1 if coverage >= relevance_threshold else 0)

        num_relevant = sum(rel_flags)
        if num_relevant == 0:
            return 0.0

        sum_precision = 0.0
        running_rel_count = 0
        for k, is_rel in enumerate(rel_flags, start=1):
            if is_rel:
                running_rel_count += 1
                precision_at_k = running_rel_count / k
                sum_precision += precision_at_k

        ap_at_k = sum_precision / num_relevant
        return min(max(ap_at_k, 0.0), 1.0)

    def run_full_eval(
        self,
        answer: str,
        question: str,
        context: str,
        expected: str,
        contexts: list[str] | None = None,
    ) -> EvalResult:
        """
        Run the three answer-side evaluations and, when ``contexts`` is
        supplied, both retrieval-side evaluations.

        passed = True if all three scores >= 0.5.

        failure_type determination (first match wins):
            faithfulness < 0.3  → "hallucination"
            relevance < 0.3     → "irrelevant"
            completeness < 0.3  → "incomplete"
            otherwise if failed → "off_topic"

        Retrieval wiring:
            contexts is None → context_recall and context_precision stay None
            contexts provided → evaluate and store both retrieval metrics

        The two retrieval metrics diagnose the retriever and do not change the
        three-metric ``passed`` rule or ``overall_score()``.

        Returns:
            EvalResult with all fields populated.
        """
        faithfulness = self.evaluate_faithfulness(answer, context)
        relevance = self.evaluate_relevance(answer, question)
        completeness = self.evaluate_completeness(answer, expected)

        passed = faithfulness >= 0.5 and relevance >= 0.5 and completeness >= 0.5

        if faithfulness < 0.3:
            failure_type: str | None = "hallucination"
        elif relevance < 0.3:
            failure_type = "irrelevant"
        elif completeness < 0.3:
            failure_type = "incomplete"
        elif not passed:
            failure_type = "off_topic"
        else:
            failure_type = None

        if contexts is None:
            c_recall: float | None = None
            c_precision: float | None = None
        else:
            c_recall = self.evaluate_context_recall(contexts, expected)
            c_precision = self.evaluate_context_precision(contexts, expected)

        qa_pair = QAPair(
            question=question,
            expected_answer=expected,
            context=context,
            retrieved_contexts=contexts if contexts is not None else [],
        )

        return EvalResult(
            qa_pair=qa_pair,
            actual_answer=answer,
            faithfulness=faithfulness,
            relevance=relevance,
            completeness=completeness,
            passed=passed,
            failure_type=failure_type,
            context_precision=c_precision,
            context_recall=c_recall,
        )


# ---------------------------------------------------------------------------
# Reranking helper (used by Exercise 3.5 — boosting Context Precision)
# ---------------------------------------------------------------------------

def rerank_by_overlap(contexts: list[str], query: str) -> list[str]:
    """A minimal lexical reranker: sort chunks by word overlap with the query,
    most-overlapping first. Stand-in for a real cross-encoder reranker.
    """
    query_tokens = set(_tokenize(query))
    if not query_tokens:
        return list(contexts)

    return sorted(
        contexts,
        key=lambda c: len(_tokenize(c) & query_tokens),
        reverse=True,
    )


# ---------------------------------------------------------------------------
# Task 3 — LLM Judge
# ---------------------------------------------------------------------------

class LLMJudge:
    """
    Uses an LLM to score AI responses according to a rubric.
    """

    def __init__(self, judge_llm_fn: Callable[[str], str]) -> None:
        self.judge_llm_fn = judge_llm_fn

    def score_response(
        self,
        question: str,
        answer: str,
        rubric: dict[str, Any],
    ) -> dict[str, Any]:
        prompt = (
            f"Question: {question}\n"
            f"Answer: {answer}\n"
            f"Rubric: {rubric}\n"
            "Please evaluate the answer based on the rubric and return JSON mapping each criterion to a score between 0.0 and 1.0."
        )
        raw_response = self.judge_llm_fn(prompt)

        scores: dict[str, float] = {}
        try:
            import json
            parsed = json.loads(raw_response)
            if isinstance(parsed, dict):
                if "scores" in parsed and isinstance(parsed["scores"], dict):
                    parsed_scores = parsed["scores"]
                else:
                    parsed_scores = parsed

                for criterion in rubric:
                    if criterion in parsed_scores and isinstance(parsed_scores[criterion], (int, float)):
                        scores[criterion] = float(parsed_scores[criterion])
                    else:
                        scores[criterion] = 0.5
            else:
                scores = {criterion: 0.5 for criterion in rubric}
        except Exception:
            scores = {criterion: 0.5 for criterion in rubric}

        return {
            "scores": scores,
            "reasoning": raw_response,
        }

    def detect_bias(self, scores_batch: list[dict[str, Any]]) -> dict[str, Any]:
        if not scores_batch:
            return {
                "positional_bias": False,
                "leniency_bias": False,
                "severity_bias": False,
            }

        all_scores: list[float] = []
        item_averages: list[float] = []

        for item in scores_batch:
            scores_dict = item.get("scores", {})
            if scores_dict:
                vals = [float(v) for v in scores_dict.values() if isinstance(v, (int, float))]
                all_scores.extend(vals)
                if vals:
                    item_averages.append(sum(vals) / len(vals))

        if all_scores:
            avg_all = sum(all_scores) / len(all_scores)
            leniency_bias = avg_all > 0.8
            severity_bias = avg_all < 0.3
        else:
            leniency_bias = False
            severity_bias = False

        if len(item_averages) >= 2:
            first_avg = item_averages[0]
            rest_avg = sum(item_averages[1:]) / len(item_averages[1:])
            positional_bias = first_avg > rest_avg + 0.1
        else:
            positional_bias = False

        return {
            "positional_bias": positional_bias,
            "leniency_bias": leniency_bias,
            "severity_bias": severity_bias,
        }


# ---------------------------------------------------------------------------
# Task 4 — Benchmark Runner
# ---------------------------------------------------------------------------

class BenchmarkRunner:
    """
    Runs a full evaluation benchmark.
    """

    def run(
        self,
        qa_pairs: list[QAPair],
        agent_fn: Callable[[str], str],
        evaluator: RAGASEvaluator,
    ) -> list[EvalResult]:
        results: list[EvalResult] = []
        for pair in qa_pairs:
            actual_answer = agent_fn(pair.question)
            eval_res = evaluator.run_full_eval(
                answer=actual_answer,
                question=pair.question,
                context=pair.context,
                expected=pair.expected_answer,
                contexts=pair.retrieved_contexts if pair.retrieved_contexts else None,
            )
            eval_res.qa_pair = pair
            results.append(eval_res)
        return results

    def generate_report(self, results: list[EvalResult]) -> dict[str, Any]:
        if not results:
            return {
                "total": 0,
                "passed": 0,
                "pass_rate": 0.0,
                "avg_faithfulness": 0.0,
                "avg_relevance": 0.0,
                "avg_completeness": 0.0,
                "avg_context_recall": None,
                "avg_context_precision": None,
                "failure_types": {},
            }

        total = len(results)
        passed_count = sum(1 for r in results if r.passed)
        pass_rate = passed_count / total

        avg_faithfulness = sum(r.faithfulness for r in results) / total
        avg_relevance = sum(r.relevance for r in results) / total
        avg_completeness = sum(r.completeness for r in results) / total

        rec_scores = [r.context_recall for r in results if r.context_recall is not None]
        avg_context_recall = sum(rec_scores) / len(rec_scores) if rec_scores else None

        prec_scores = [r.context_precision for r in results if r.context_precision is not None]
        avg_context_precision = sum(prec_scores) / len(prec_scores) if prec_scores else None

        failure_types: dict[str, int] = {}
        for r in results:
            if r.failure_type:
                failure_types[r.failure_type] = failure_types.get(r.failure_type, 0) + 1

        return {
            "total": total,
            "passed": passed_count,
            "pass_rate": pass_rate,
            "avg_faithfulness": avg_faithfulness,
            "avg_relevance": avg_relevance,
            "avg_completeness": avg_completeness,
            "avg_context_recall": avg_context_recall,
            "avg_context_precision": avg_context_precision,
            "failure_types": failure_types,
        }

    def run_regression(self, new_results: list[EvalResult], baseline_results: list[EvalResult]) -> dict[str, Any]:
        new_f = sum(r.faithfulness for r in new_results) / len(new_results) if new_results else 0.0
        new_r = sum(r.relevance for r in new_results) / len(new_results) if new_results else 0.0
        new_c = sum(r.completeness for r in new_results) / len(new_results) if new_results else 0.0

        base_f = sum(r.faithfulness for r in baseline_results) / len(baseline_results) if baseline_results else 0.0
        base_r = sum(r.relevance for r in baseline_results) / len(baseline_results) if baseline_results else 0.0
        base_c = sum(r.completeness for r in baseline_results) / len(baseline_results) if baseline_results else 0.0

        regressions: list[str] = []
        if base_f - new_f > 0.05:
            regressions.append("faithfulness")
        if base_r - new_r > 0.05:
            regressions.append("relevance")
        if base_c - new_c > 0.05:
            regressions.append("completeness")

        return {
            "new_avg_faithfulness": new_f,
            "new_avg_relevance": new_r,
            "new_avg_completeness": new_c,
            "baseline_avg_faithfulness": base_f,
            "baseline_avg_relevance": base_r,
            "baseline_avg_completeness": base_c,
            "regressions": regressions,
            "passed": len(regressions) == 0,
        }

    def identify_failures(
        self,
        results: list[EvalResult],
        threshold: float = 0.5,
    ) -> list[EvalResult]:
        return [
            r for r in results
            if r.faithfulness < threshold or r.relevance < threshold or r.completeness < threshold
        ]


# ---------------------------------------------------------------------------
# Task 5 — Failure Analyzer
# ---------------------------------------------------------------------------

class FailureAnalyzer:
    """
    Analyzes failed evaluation results to identify patterns and suggest fixes.
    """

    def categorize_failures(
        self, failures: list[EvalResult]
    ) -> dict[str, int]:
        counts: dict[str, int] = {}
        for f in failures:
            if f.failure_type:
                counts[f.failure_type] = counts.get(f.failure_type, 0) + 1
        return counts

    def find_root_cause(self, failure: EvalResult) -> str:
        scores = {
            "faithfulness": failure.faithfulness,
            "relevance": failure.relevance,
            "completeness": failure.completeness,
        }
        min_val = min(scores.values())
        lowest_keys = [k for k, v in scores.items() if v == min_val]

        if len(lowest_keys) > 1:
            return "Multiple issues detected — review full pipeline"

        lowest_metric = lowest_keys[0]
        if lowest_metric == "faithfulness":
            return "Context is missing or irrelevant — improve retrieval"
        elif lowest_metric == "relevance":
            return "Answer does not address the question — improve prompt clarity"
        elif lowest_metric == "completeness":
            return "Answer is missing key information — increase context window or improve generation"
        else:
            return "Multiple issues detected — review full pipeline"

    def generate_improvement_log(self, failures: list[EvalResult], suggestions: list[str]) -> str:
        lines = [
            "| Failure ID | Type | Root Cause | Suggested Fix | Status |",
            "|------------|------|------------|---------------|--------|",
        ]
        for idx, f in enumerate(failures, start=1):
            f_id = f"F{idx:03d}"
            f_type = f.failure_type if f.failure_type else "unknown"
            root_cause = self.find_root_cause(f)
            suggested_fix = suggestions[idx - 1] if idx - 1 < len(suggestions) else "Review pipeline configuration"
            lines.append(f"| {f_id} | {f_type} | {root_cause} | {suggested_fix} | Open |")

        return "\n".join(lines)

    def generate_improvement_suggestions(
        self, failures: list[EvalResult]
    ) -> list[str]:
        if not failures:
            return []

        categories = self.categorize_failures(failures)
        suggestions: list[str] = []

        category_map = {
            "hallucination": "Implement hallucination checker to filter unsupported claims",
            "irrelevant": "Refine prompt and instructions to keep generator focused on question intent",
            "incomplete": "Add few-shot examples showing complete answers to improve completeness",
            "off_topic": "Improve intent classification and routing before context retrieval",
            "refusal": "Calibrate safety guardrails to avoid unnecessary refusals",
        }

        # Sort failure categories by count descending to truly prioritize most frequent failures
        sorted_categories = sorted(categories.items(), key=lambda item: item[1], reverse=True)

        for cat_type, count in sorted_categories:
            if count > 0 and cat_type in category_map:
                sug = category_map[cat_type]
                if sug not in suggestions:
                    suggestions.append(sug)

        defaults = [
            "Increase chunk size in RAG pipeline to reduce context fragmentation",
            "Implement a cross-encoder reranker to elevate relevant evidence in top chunks",
            "Add few-shot examples showing complete answers to improve completeness",
            "Implement hallucination checker to filter unsupported claims",
            "Refine system prompt and grounding instructions for better answer quality",
        ]

        for s in defaults:
            if s not in suggestions:
                suggestions.append(s)
            if len(suggestions) >= 3:
                break

        return suggestions
