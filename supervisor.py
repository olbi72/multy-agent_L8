import json

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver

from agents import plan, research, critique
from config import settings, SUPERVISOR_SYSTEM_PROMPT
from tools import save_report


llm = ChatOpenAI(
    model=settings.model_name,
    api_key=settings.api_key.get_secret_value(),
)

checkpointer = InMemorySaver()

supervisor = create_agent(
    model=llm,
    tools=[plan, research, critique, save_report],
    system_prompt=SUPERVISOR_SYSTEM_PROMPT,
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={"save_report": True}
        )
    ],
    checkpointer=checkpointer,
)


def format_critique_result(critique_json: str) -> str:
    try:
        data = json.loads(critique_json)
    except Exception:
        return critique_json

    parts = [f"Verdict: {data.get('verdict')}"]

    if "is_fresh" in data:
        parts.extend([
            f"Fresh: {data.get('is_fresh')}",
            f"Complete: {data.get('is_complete')}",
            f"Well structured: {data.get('is_well_structured')}",
        ])

    if "is_grounded_in_kb" in data:
        parts.extend([
            f"Grounded in KB: {data.get('is_grounded_in_kb')}",
            f"Covers main points: {data.get('covers_main_points')}",
            f"Well structured: {data.get('is_well_structured')}",
        ])

    strengths = data.get("strengths", [])
    if strengths:
        parts.append("Strengths:\n- " + "\n- ".join(strengths))

    gaps = data.get("gaps", [])
    if gaps:
        parts.append("Gaps:\n- " + "\n- ".join(gaps))

    revision_requests = data.get("revision_requests", [])
    if revision_requests:
        parts.append("Revision requests:\n- " + "\n- ".join(revision_requests))

    return "\n\n".join(parts)