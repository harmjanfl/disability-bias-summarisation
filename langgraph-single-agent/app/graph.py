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
    temperature=0.2,
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
                    "You are a helpful assistant. "
                    "Answer the user's prompt clearly, accurately, and naturally. "
                    "Base your response only on the information provided in the prompt."
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
                "You are a strict bias, fairness, and evidence-grounding reviewer.\n\n"
                "Your task is NOT to lightly polish the draft. "
                "Your task is to identify the most important substantive problems.\n\n"
                "Check for:\n"
                "- bias, stereotyping, unfair assumptions, exclusion, or harmful framing\n"
                "- overrepresentation or underrepresentation of topics/groups\n"
                "- unsupported claims or claims that go beyond the original prompt\n"
                "- missing caveats, nuance, or uncertainty\n"
                "- weak structure or emphasis that could distort the message\n\n"
                "Use the original prompt as the source of truth. "
                "Do not introduce new groups or topics that are not present there.\n\n"
                "Return exactly these sections:\n\n"
                "MAJOR_ISSUES:\n"
                "- 3 to 5 bullets describing the highest-impact problems\n\n"
                "KEEP:\n"
                "- bullets for what is already solid and should be preserved\n\n"
                "REWRITE_STRATEGY:\n"
                "- bullets explaining how the response should be restructured, rebalanced, or rewritten\n\n"
                "Be concrete and direct. Prefer high-impact criticism over minor wording suggestions."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Original prompt:\n{prompt}\n\n"
                f"Generated summary:\n{generated_response}\n\n"
                "Review this summary for bias, fairness, grounding, and structural problems."
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
                "You are a senior rewriting assistant.\n\n"
                "Rewrite the answer from scratch using:\n"
                "1. the original prompt as the source of truth\n"
                "2. the reviewer feedback as guidance\n"
                "3. the original generated response only as a reference, not as a template\n\n"
                "Important:\n"
                "- Do NOT make only minimal edits.\n"
                "- If the draft has meaningful problems, produce a substantially revised answer.\n"
                "- You may reorganize, shorten, expand, merge, or replace content entirely.\n"
                "- Prefer a clearly better final answer over preserving the original wording.\n"
                "- Keep all claims grounded in the original prompt.\n"
                "- Remove unsupported, biased, or overstated claims.\n"
                "- Do not introduce topics or groups not present in the original prompt.\n"
                "- Keep the final response natural, clear, neutral, and helpful.\n\n"
                "Return only the rewritten response."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Original prompt:\n{prompt}\n\n"
                f"Original generated response:\n{generated_response}\n\n"
                f"Reviewer feedback:\n{review_feedback}\n\n"
                "Write the best final response. "
                "Treat the original draft as disposable if needed."
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