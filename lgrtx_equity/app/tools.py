from __future__ import annotations

import math
import os
import re
from collections import Counter, defaultdict
from functools import lru_cache
from typing import Any

import numpy as np
from langchain_core.tools import tool
from sklearn.metrics.pairwise import cosine_similarity

TOPIC_TAXONOMY = [
    "DIS accessibility",
    "DIS communication",
    "DIS eligibility",
    "DIS information",
    "DIS organization",
    "DIS prioritization",
    "DIS services",
    "DIS specialized_support",
    "DIS supplies",
    "GEN accessibility",
    "GEN communication",
    "GEN crowding",
    "GEN eligibility",
    "GEN information",
    "GEN organization",
    "GEN prioritization",
    "GEN registration",
    "GEN services",
    "GEN supplies",
]

THESIS_EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "has", "have", "i", "in", "is", "it", "its", "of", "on", "or", "our",
    "that", "the", "their", "there", "this", "to", "was", "we", "were", "with",
    "would", "could", "should", "more", "about", "during", "people", "community",
}


def split_sentences(text: str) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    text = re.sub(r"\r\n?", "\n", text)
    parts = re.split(r"(?<=[.!?])\s+|\n+|(?=\d+[.)]\s)", text)
    return [p.strip(" -\t") for p in parts if p and p.strip(" -\t")]


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def tokenize(text: str) -> list[str]:
    words = re.findall(r"[a-zA-Z0-9_\-']+", text.lower())
    return [w for w in words if w not in STOPWORDS and len(w) > 2]


@lru_cache(maxsize=1)
def _load_sentence_transformer_model():
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore

        return SentenceTransformer(THESIS_EMBEDDING_MODEL)
    except Exception:
        return None


@lru_cache(maxsize=1)
def _load_azure_embeddings_client():
    try:
        from openai import AzureOpenAI  # type: ignore
    except Exception:
        return None

    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION")
    deployment = (
        os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
        or os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT")
        or os.getenv("AZURE_OPENAI_EMBEDDING_MODEL")
    )

    if not all([endpoint, api_key, api_version, deployment]):
        return None

    try:
        client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version,
        )
        return client, deployment
    except Exception:
        return None


def embed_texts(texts: list[str]) -> tuple[np.ndarray, str]:
    clean_texts = [normalize_text(t) for t in texts]
    if not clean_texts:
        return np.zeros((0, 0), dtype=float), "none"

    st_model = _load_sentence_transformer_model()
    if st_model is not None:
        vectors = st_model.encode(
            clean_texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.asarray(vectors, dtype=float), THESIS_EMBEDDING_MODEL

    azure_client = _load_azure_embeddings_client()
    if azure_client is not None:
        client, deployment = azure_client
        response = client.embeddings.create(model=deployment, input=clean_texts)
        vectors = np.asarray([item.embedding for item in response.data], dtype=float)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms, f"azure:{deployment}"

    vocab = sorted({token for text in clean_texts for token in tokenize(text)})
    if not vocab:
        return np.zeros((len(clean_texts), 1), dtype=float), "lexical-fallback"

    vocab_index = {token: idx for idx, token in enumerate(vocab)}
    matrix = np.zeros((len(clean_texts), len(vocab)), dtype=float)
    for row_idx, text in enumerate(clean_texts):
        counts = Counter(tokenize(text))
        for token, count in counts.items():
            matrix[row_idx, vocab_index[token]] = float(count)

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms, "lexical-fallback"


def _safe_topic(item: dict[str, Any]) -> str:
    topic = str(item.get("topic") or "other").strip().lower()
    return topic if topic else "other"


def _feedback_text(item: dict[str, Any]) -> str:
    return str(item.get("feedback") or item.get("text") or "").strip()


def _is_minority(topic: str) -> bool:
    """A topic is considered minority-group when its identifier starts with 'dis'.

    The pipeline's primary equity goal is representation of disability-related
    feedback, so topics with the 'DIS' / 'dis' prefix are treated as the
    minority class throughout analysis and review.
    """
    return str(topic).strip().lower().startswith("dis")





def _mean_topic_affinity(
    similarity_matrix: np.ndarray,
    summary_idx: int,
    input_indices: list[int],
) -> float:
    """A_tj: mean cosine similarity between summary sentence j and all inputs in topic t."""
    if not input_indices:
        return float("-inf")
    return float(similarity_matrix[input_indices, summary_idx].mean())


def _gated_hard_assignment(
    topic_scores: dict[str, float],
    theta: float,
    delta: float,
) -> tuple[str | None, str, float]:
    """
    Apply the gated hard mention rule (thesis formulation).

    A summary sentence is assigned to a topic only when:
      1. best affinity score >= theta  (MIN_EVIDENCE_THRESHOLD)
      2. best score exceeds second-best score by >= delta  (AMBIGUITY_MARGIN)

    Returns: (assigned_topic | None, reason, margin)
    """
    valid = sorted(
        [(t, s) for t, s in topic_scores.items() if s != float("-inf")],
        key=lambda x: x[1],
        reverse=True,
    )
    if not valid:
        return None, "no_topic_present_in_input", float("-inf")

    best_topic, best_score = valid[0]
    second_score = valid[1][1] if len(valid) > 1 else float("-inf")
    margin = best_score - second_score if math.isfinite(second_score) else float("inf")

    if best_score < theta:
        return None, "below_min_evidence_threshold", margin
    if math.isfinite(margin) and margin < delta:
        return None, "below_ambiguity_margin", margin

    return best_topic, "assigned", margin


@tool
def analyze_equity(
    labeled_feedback: list[dict[str, Any]],
    summary: str,
    min_evidence_threshold: float = 0.40,
    ambiguity_margin: float = 0.02,
    pre_split_sentences: list[str] | None = None,
    min_minority_visibility_rate: float = 1.0,
    min_majority_visibility_rate: float = 0.50,
) -> dict[str, Any]:
    """
    Analyze equity of topic coverage using the gated hard mention rule.

    For each summary sentence s_j and topic t, compute the topic affinity score
    A_tj (mean cosine similarity between s_j and all input sentences in topic t).
    A sentence is hard-assigned to its best topic only when:
      - best A_tj >= min_evidence_threshold (theta), AND
      - best A_tj exceeds second-best A_tj by >= ambiguity_margin (delta).

    The primary equity signal is visibility_rate_when_present: among topics that
    appear in the input, what fraction of them receive at least one assigned
    summary sentence? A topic is invisible when it is present in the input but
    receives zero hard-assigned summary sentences.
    """
    theta = min_evidence_threshold
    delta = ambiguity_margin

    _empty = {
        "embedding_model": THESIS_EMBEDDING_MODEL,
        "summary_sentences": [],
        "sentence_assignments": [],
        "topic_affinity_per_summary_sentence": {},
        "assigned_counts": {},
        "input_counts": {},
        "invisible_topics": [],
        "visible_topics": [],
        "visibility_rate_when_present": 0.0,
        "minority_visible_topics": [],
        "minority_invisible_topics": [],
        "minority_visibility_rate": 1.0,
        "majority_visible_topics": [],
        "majority_invisible_topics": [],
        "majority_visibility_rate": 1.0,
        "min_minority_visibility_rate": min_minority_visibility_rate,
        "min_majority_visibility_rate": min_majority_visibility_rate,
        "unassigned_summary_count": 0,
        "unassigned_summary_share": 0.0,
        "within_range": False,
        "theta": theta,
        "delta": delta,
        "warnings": [],
    }

    if not labeled_feedback:
        return {**_empty, "warnings": ["No labeled feedback provided."]}

    input_sentences = [normalize_text(_feedback_text(item)) for item in labeled_feedback]
    if pre_split_sentences:
        summary_sentences = [
            normalize_text(s) for s in pre_split_sentences if s and s.strip()
        ]
    else:
        summary_sentences = split_sentences(summary)

    if not summary_sentences:
        topics_present = sorted({_safe_topic(item) for item in labeled_feedback})
        return {
            **_empty,
            "invisible_topics": topics_present,
            "warnings": ["Summary was empty after sentence splitting."],
        }

    all_texts = input_sentences + summary_sentences
    embeddings, embedding_model_name = embed_texts(all_texts)

    num_inputs = len(input_sentences)
    similarity_matrix = cosine_similarity(embeddings[:num_inputs], embeddings[num_inputs:])
    # shape: (num_inputs, num_summary_sentences) — rows=inputs, cols=summary

    topics = sorted({_safe_topic(item) for item in labeled_feedback})
    topic_to_indices: dict[str, list[int]] = defaultdict(list)
    for idx, item in enumerate(labeled_feedback):
        topic_to_indices[_safe_topic(item)].append(idx)

    input_counts = {topic: len(topic_to_indices[topic]) for topic in topics}

    # Build topic affinity matrix: topic -> [A_t0, A_t1, ..., A_t(J-1)]
    topic_affinity_matrix: dict[str, list[float]] = {}
    for topic in topics:
        indices = topic_to_indices[topic]
        affinities = [
            round(_mean_topic_affinity(similarity_matrix, j, indices), 6)
            for j in range(len(summary_sentences))
        ]
        topic_affinity_matrix[topic] = affinities

    # Assign each summary sentence via the gated hard mention rule
    assigned_counts: dict[str, int] = {topic: 0 for topic in topics}
    sentence_assignments: list[dict[str, Any]] = []
    unassigned_summary_count = 0

    for j, summary_sentence in enumerate(summary_sentences):
        topic_scores = {topic: topic_affinity_matrix[topic][j] for topic in topics}

        assigned_topic, reason, margin = _gated_hard_assignment(topic_scores, theta, delta)

        best_valid = sorted(
            [(t, s) for t, s in topic_scores.items() if s != float("-inf")],
            key=lambda x: x[1],
            reverse=True,
        )
        best_topic = best_valid[0][0] if best_valid else None
        best_score = best_valid[0][1] if best_valid else float("-inf")
        second_topic = best_valid[1][0] if len(best_valid) > 1 else None
        second_score = best_valid[1][1] if len(best_valid) > 1 else float("-inf")

        if assigned_topic is None:
            unassigned_summary_count += 1
        else:
            assigned_counts[assigned_topic] += 1

        sentence_assignments.append({
            "summary_sentence_index": j,
            "summary_sentence": summary_sentence,
            "best_topic": best_topic,
            "best_score": round(best_score, 6) if math.isfinite(best_score) else best_score,
            "second_topic": second_topic,
            "second_score": round(second_score, 6) if math.isfinite(second_score) else second_score,
            "margin": round(margin, 6) if math.isfinite(margin) else margin,
            "assigned_topic": assigned_topic if assigned_topic is not None else "UNASSIGNED",
            "assignment_reason": reason,
            "theta": theta,
            "delta": delta,
        })

    # Equity metrics: visibility when present
    topics_present_in_input = [t for t in topics if input_counts[t] > 0]
    visible_topics = [t for t in topics_present_in_input if assigned_counts[t] > 0]
    invisible_topics = [t for t in topics_present_in_input if assigned_counts[t] == 0]

    n_present = len(topics_present_in_input)
    visibility_rate = len(visible_topics) / n_present if n_present > 0 else 0.0

    total_summary = len(summary_sentences)
    unassigned_share = unassigned_summary_count / total_summary if total_summary > 0 else 0.0

    # Minority-aware equity: split topics by minority-class membership.
    # The pipeline's primary objective is full coverage of the minority class
    # (DIS-prefixed topics). The majority class (everything else, typically
    # GEN-prefixed) is held to a softer threshold so the loop can terminate
    # gracefully once minority coverage is satisfied.
    minority_present = [t for t in topics_present_in_input if _is_minority(t)]
    majority_present = [t for t in topics_present_in_input if not _is_minority(t)]

    minority_visible = [t for t in minority_present if t in visible_topics]
    minority_invisible = [t for t in minority_present if t in invisible_topics]
    majority_visible = [t for t in majority_present if t in visible_topics]
    majority_invisible = [t for t in majority_present if t in invisible_topics]

    # When a class has no representation in the input, treat its visibility
    # rate as 1.0 so it cannot block acceptance ("nothing to fail on").
    minority_visibility_rate = (
        len(minority_visible) / len(minority_present) if minority_present else 1.0
    )
    majority_visibility_rate = (
        len(majority_visible) / len(majority_present) if majority_present else 1.0
    )

    # within_range now reflects the asymmetric, minority-first equity goal.
    # Both class-level thresholds must be satisfied AND at least one topic
    # must be present in the input (an empty input cannot pass).
    within_range = (
        n_present > 0
        and minority_visibility_rate >= min_minority_visibility_rate
        and majority_visibility_rate >= min_majority_visibility_rate
    )

    warnings: list[str] = []
    if embedding_model_name != THESIS_EMBEDDING_MODEL:
        warnings.append(
            f"Requested thesis model '{THESIS_EMBEDDING_MODEL}' was unavailable; "
            f"used '{embedding_model_name}' instead."
        )

    return {
        "embedding_model": embedding_model_name,
        "summary_sentences": summary_sentences,
        "sentence_assignments": sentence_assignments,
        "topic_affinity_per_summary_sentence": topic_affinity_matrix,
        "assigned_counts": assigned_counts,
        "input_counts": input_counts,
        "invisible_topics": invisible_topics,
        "visible_topics": visible_topics,
        "visibility_rate_when_present": round(visibility_rate, 6),
        "minority_visible_topics": minority_visible,
        "minority_invisible_topics": minority_invisible,
        "minority_visibility_rate": round(minority_visibility_rate, 6),
        "majority_visible_topics": majority_visible,
        "majority_invisible_topics": majority_invisible,
        "majority_visibility_rate": round(majority_visibility_rate, 6),
        "min_minority_visibility_rate": min_minority_visibility_rate,
        "min_majority_visibility_rate": min_majority_visibility_rate,
        "unassigned_summary_count": unassigned_summary_count,
        "unassigned_summary_share": round(unassigned_share, 6),
        "within_range": within_range,
        "theta": theta,
        "delta": delta,
        "warnings": warnings,
    }
