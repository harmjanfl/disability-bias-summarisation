import os
from dotenv import load_dotenv

from langchain_openai import AzureChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

from app.state import AgentState

load_dotenv()

endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
api_key = os.getenv("AZURE_OPENAI_API_KEY")
api_version = os.getenv("AZURE_OPENAI_API_VERSION")

missing = [
    name for name, value in {
        "AZURE_OPENAI_ENDPOINT": endpoint,
        "AZURE_OPENAI_DEPLOYMENT": deployment,
        "AZURE_OPENAI_API_KEY": api_key,
        "AZURE_OPENAI_API_VERSION": api_version,
    }.items()
    if not value
]

if missing:
    raise RuntimeError(f"Missing environment variables: {', '.join(missing)}")

model = AzureChatOpenAI(
    azure_endpoint=endpoint,
    azure_deployment=deployment,
    api_key=api_key,
    api_version=api_version,
    temperature=0,
)


def generator_node(state: AgentState):
    prompt = state["prompt"]
    input_messages = state.get("input_messages", [])

    if input_messages:
        messages = input_messages
    else:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful summarization assistant. Write a clear, accurate summary grounded only in the user's input. "
                    "Prioritize equitable representation rather than equal mention frequency. "
                    "If the input mentions vulnerable, marginalized, at-risk, underserved, or otherwise disproportionately affected groups, "
                    "increase the likelihood that they appear in the summary when they are relevant to the main content. "
                    "Do not invent groups, harms, or claims that are not supported by the input. "
                    "When such groups are mentioned, explicitly preserve their needs, risks, barriers, or perspectives in the summary instead of collapsing them into generic language. "
                    "Use salience-weighting: details about vulnerable groups should receive extra attention relative to baseline summarization, even if they are mentioned briefly, provided they are supported by the input. "
                    "If multiple groups are mentioned, give additional space to the most vulnerable or disproportionately impacted groups. "
                    "Be specific, concise, and natural."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]

    response = model.invoke(messages)
    return {"generated_response": response.content}


def reviewer_node(state: AgentState):
    prompt = state["prompt"]
    generated_response = state["generated_response"]

    messages = [
        {
            "role": "system",
            "content": (
                "You are a bias, fairness, and equity reviewer evaluating an LLM-generated summary. "
                "Your job is to review the summary against the original input and then write revision instructions for an LLM rewriter. "
                "Optimize for measurable equity gain over a baseline summary. "
                "Check whether vulnerable, marginalized, at-risk, underserved, or disproportionately affected groups mentioned in the original input are missing, minimized, generalized away, or framed less prominently than warranted. "
                "Prefer supported amplification over equal distribution: when the input mentions such groups, recommend increasing their visibility in the output if that would better reflect equity concerns in the source. "
                "Do not ask the rewriter to add any group, harm, or perspective that is not grounded in the original input. "
                "Flag overcorrection too: if the summary adds unsupported equity claims or overstates a group's role, say so. "
                "Use a structured review with concise bullet points under these headings: Missing Equity Signals, Harmful or Flattening Language, Overcorrections/Unsupported Additions, and Rewrite Instructions for the LLM Rewriter. "
                "In the rewrite instructions, name the specific sentence or omission to fix and explicitly tell the LLM rewriter how to change prominence, wording, or detail."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Original prompt:\n{prompt}\n\n"
                f"Generated summary:\n{generated_response}\n\n"
                "Review this summary for bias, fairness, and equity concerns."
            ),
        },
    ]

    response = model.invoke(messages)
    return {"review_feedback": response.content}


def rewriter_node(state: AgentState):
    prompt = state["prompt"]
    generated_response = state["generated_response"]
    review_feedback = state["review_feedback"]

    messages = [
        {
            "role": "system",
            "content": (
                "You are a rewriting assistant. Rewrite an LLM-generated summary using review feedback written for an LLM rewriter. "
                "Keep the answer accurate, natural, and fully grounded in the original input. "
                "Your goal is to increase equity in the summary, not merely maintain equal representation. "
                "If the original input mentions vulnerable, marginalized, at-risk, underserved, or disproportionately affected groups, make sure they are explicitly represented in the rewritten summary when relevant. "
                "Apply supported emphasis: increase the prominence, specificity, and retention of those groups' needs, risks, barriers, or perspectives when the source supports it. "
                "Do not invent groups, impacts, or causal claims, and do not use generic boilerplate equity language unsupported by the input. "
                "Follow the review instructions precisely, resolve omissions first, then remove harmful or flattening phrasing, then tighten for clarity. "
                "Return only the rewritten response and do not mention the review or rewriting process."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Original prompt:\n{prompt}\n\n"
                f"Original generated response:\n{generated_response}\n\n"
                f"Reviewer feedback:\n{review_feedback}\n\n"
                "Please rewrite the response accordingly."
            ),
        },
    ]

    response = model.invoke(messages)
    return {"final_response": response.content}


builder = StateGraph(AgentState)

builder.add_node("generator", generator_node)
builder.add_node("reviewer", reviewer_node)
builder.add_node("rewriter", rewriter_node)

builder.add_edge(START, "generator")
builder.add_edge("generator", "reviewer")
builder.add_edge("reviewer", "rewriter")
builder.add_edge("rewriter", END)

graph = builder.compile(checkpointer=InMemorySaver())
