# Findings Log — Research Paper Assistant (RAG project)

## Finding 1: Ambiguous questions retrieve the wrong statistic

**Vague question:**
"What hallucination rate did RAG reach on TruthfulQA?"

**Answer given (wrong context):**
4.8% — cited from paper 2609.22043v1, page 10-11.

**More specific question:**
"What hallucination rate did RAG reach on TruthfulQA when the memory contains both correct answers and misconceptions?"

**Answer given (correct):**
53.0%, versus 23.0% for the memory-free baseline — matches the source text exactly, cited from the same paper, page 2-3.

## Cause

The source paper reports multiple hallucination rates measured under different experimental conditions. A vague question retrieves a passage that is topically relevant but answers a different sub-question. The language model then states that number confidently instead of flagging the ambiguity or asking for clarification.

## Takeaway

Retrieval precision depends heavily on how specific the question is. A production system would benefit from per-claim verification, or from detecting when multiple candidate answers exist in the retrieved context and asking a clarifying question rather than picking one silently.

## Why this matters

This is a real, reproducible example of a known RAG failure mode (confident generation from ambiguous retrieval), not a bug in the pipeline itself. Documenting it, along with the before/after fix, demonstrates evaluation-driven development rather than assuming the system works because outputs look fluent.

## Finding 2: Automated answer evaluation results (v5)

**Setup:** Ran all 8 answerable evaluation questions through the full pipeline
(retrieval + Groq LLM generation), then had the LLM grade each answer against
the expected answer as CORRECT, PARTIAL, or WRONG.

**Result:** 5/8 correct (62%), 1 partial, 2 wrong.

**On the 2 wrong answers:** Both were questions about paper 2609.19601v1
(VisRAG and NotebookRAG). In both cases the model said "I don't know based on
the provided papers" instead of guessing — this is a retrieval miss (the right
chunk wasn't in the top 5 results), not a hallucination. The system failed safely.

**On refusal detection:** Tested 2 "unanswerable" questions (e.g. "What is the
capital of France?"). Both scored above the 0.45 similarity threshold used to
trigger "I don't know," so the app would have answered anyway. Relevant-question
scores ranged 0.62-0.76, and irrelevant-question scores ranged 0.47-0.63 — the
two ranges overlap, so a single similarity cutoff can't fully separate them.

**Takeaway:** A simple top-score threshold is a reasonable first pass for
detecting out-of-scope questions, but it isn't reliable on its own. A stronger
approach would combine retrieval score with an explicit LLM check ("does this
context actually answer the question?") before generating a response.

**Update:** In manual testing through the web interface, "What is the capital
of France?" was correctly refused ("I don't know based on the provided
papers"), scoring right at the threshold boundary. This confirms the
threshold is workable but sits close to the edge — a small buffer (e.g.
raising it to 0.5) would make refusals more reliable and consistent.