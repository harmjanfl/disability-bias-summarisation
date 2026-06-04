from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .nodes import (
    fairness_reviewer_node,
    finalize_node,
    initialize_node,
    parse_feedback_node,
    representation_analyzer_node,
    route_after_review,
    summarizer_node,
    topic_labeler_node,
)
from .state import AgentState

builder = StateGraph(AgentState)

builder.add_node("initialize", initialize_node)
builder.add_node("parse_feedback", parse_feedback_node)
builder.add_node("topic_labeler", topic_labeler_node)
builder.add_node("summarizer", summarizer_node)
builder.add_node("representation_analyzer", representation_analyzer_node)
builder.add_node("fairness_reviewer", fairness_reviewer_node)
builder.add_node("finalize", finalize_node)

builder.add_edge(START, "initialize")
builder.add_edge("initialize", "parse_feedback")
builder.add_edge("parse_feedback", "topic_labeler")
builder.add_edge("topic_labeler", "summarizer")
builder.add_edge("summarizer", "representation_analyzer")
builder.add_edge("representation_analyzer", "fairness_reviewer")
builder.add_conditional_edges(
    "fairness_reviewer",
    route_after_review,
    {
        "summarizer": "summarizer",
        "finalize": "finalize",
    },
)
builder.add_edge("finalize", END)

graph = builder.compile(name="lgtrx_bias_mitigation_graph")
