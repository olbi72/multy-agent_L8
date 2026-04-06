from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI

from config import settings, PLANNER_SYSTEM_PROMPT
from schemas import ResearchPlan
from tools import web_search, knowledge_search


llm = ChatOpenAI(
    model=settings.model_name,
    api_key=settings.api_key.get_secret_value(),
)

planner_agent = create_agent(
    model=llm,
    tools=[web_search, knowledge_search],
    system_prompt=PLANNER_SYSTEM_PROMPT,
    response_format=ResearchPlan,
)


@tool
def plan(request: str) -> str:
    """
    Create a structured research plan for the user's request.
    Returns a validated ResearchPlan as text.
    """
    seen_tool_calls = set()
    seen_tool_results = set()
    final_step = None

    print("\n[Planner] Starting planning...")

    for step in planner_agent.stream(
        {"messages": [("user", request)]},
        stream_mode="values",
    ):
        final_step = step
        messages = step.get("messages", [])
        if not messages:
            continue

        last_msg = messages[-1]

        tool_calls = getattr(last_msg, "tool_calls", None)
        if tool_calls:
            for call in tool_calls:
                tool_name = call.get("name", "unknown_tool")
                tool_args = str(call.get("args", {}))
                key = (tool_name, tool_args)
                if key not in seen_tool_calls:
                    seen_tool_calls.add(key)
                    print(f"[Planner] 🔧 Tool call: {tool_name}({tool_args})")

        if getattr(last_msg, "type", "") == "tool":
            content = getattr(last_msg, "content", "")
            if content not in seen_tool_results:
                seen_tool_results.add(content)
                preview = content[:500] + ("..." if len(content) > 500 else "")
                print(f"[Planner] 📎 Result: {preview}")

    if not final_step:
        return (
            '{'
            '"goal":"Planning failed",'
            '"search_queries":[],'
            '"sources_to_check":["web"],'
            '"output_format":"short report"'
            '}'
        )

    structured = final_step["structured_response"]
    print("[Planner] Finished planning.")
    return structured.model_dump_json(indent=2, ensure_ascii=False)