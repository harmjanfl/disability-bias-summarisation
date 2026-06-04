from __future__ import annotations

from typing import Any
from typing_extensions import TypedDict


class FeedbackItem(TypedDict, total=False):
    id: str
    date: str
    region: str
    sex: str
    age: str
    other_factors: str
    feedback: str
    topic: str


class SentenceAssignment(TypedDict, total=False):
    summary_sentence_index: int
    summary_sentence: str
    best_topic: str
    best_score: float
    second_topic: str
    second_score: float
    margin: float
    assigned_topic: str   # topic name, or "UNASSIGNED"
    assignment_reason: str
    theta: float
    delta: float


class EquityMetrics(TypedDict, total=False):
    embedding_model: str
    summary_sentences: list[str]
    sentence_assignments: list[SentenceAssignment]
    topic_affinity_per_summary_sentence: dict[str, list[float]]
    assigned_counts: dict[str, int]       # hard-assigned summary sentences per topic
    input_counts: dict[str, int]          # input feedback items per topic
    invisible_topics: list[str]           # present in input, no assigned sentence
    visible_topics: list[str]             # present in input, >=1 assigned sentence
    visibility_rate_when_present: float   # len(visible) / len(present)
    # Minority-aware breakdown (DIS-prefixed topics treated as the minority class)
    minority_visible_topics: list[str]
    minority_invisible_topics: list[str]
    minority_visibility_rate: float       # 1.0 when no minority topics in input
    majority_visible_topics: list[str]
    majority_invisible_topics: list[str]
    majority_visibility_rate: float       # 1.0 when no majority topics in input
    min_minority_visibility_rate: float   # threshold used for this run
    min_majority_visibility_rate: float   # threshold used for this run
    unassigned_summary_count: int
    unassigned_summary_share: float
    within_range: bool                    # both class thresholds satisfied
    theta: float                          # min_evidence_threshold used
    delta: float                          # ambiguity_margin used
    warnings: list[str]


class AgentState(TypedDict, total=False):
    # Minimal user-facing inputs. These are what you will primarily provide in Studio.
    input_messages: list[dict[str, str]]
    max_iterations: int
    metadata: dict[str, Any]

    # Derived request context.
    prompt: str
    source_text: str

    # Workflow data.
    feedback_items: list[FeedbackItem]
    labeled_feedback: list[FeedbackItem]
    generated_response: str
    final_response: str
    summary_sentences_structured: list[str]


    # Review / iteration control.
    equity_metrics: EquityMetrics
    fairness_feedback: str
    review_feedback: str
    rewrite_instructions: str
    iteration: int
    done: bool
    # Minority-aware acceptance thresholds (set from Studio Input panel).
    # Defaults are applied in representation_analyzer_node if unset.
    min_minority_visibility_rate: float
    min_majority_visibility_rate: float

    # Debugging / observability.
    raw_topic_labeler_output: str
    raw_fairness_output: str
    errors: list[str]
