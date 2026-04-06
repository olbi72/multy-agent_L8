from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI

from config import settings, RESEARCH_SYSTEM_PROMPT
from tools import web_search, read_url, knowledge_search


llm = ChatOpenAI(
    model=settings.model_name,
    api_key=settings.api_key.get_secret_value(),
)

research_agent = create_agent(
    model=llm,
    tools=[web_search, read_url, knowledge_search],
    system_prompt=RESEARCH_SYSTEM_PROMPT,
)


@tool
def research(request: str) -> str:
    """
    Execute research based on the given plan or revision request.
    Returns research findings in markdown form.
    """
    seen_tool_calls = set()
    seen_tool_results = set()
    final_step = None

    print("\n[Researcher] Starting research...")

    for step in research_agent.stream(
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
                    print(f"[Researcher] 🔧 Tool call: {tool_name}({tool_args})")

        if getattr(last_msg, "type", "") == "tool":
            content = getattr(last_msg, "content", "")
            if content not in seen_tool_results:
                seen_tool_results.add(content)
                preview = content[:500] + ("..." if len(content) > 500 else "")
                print(f"[Researcher] 📎 Result: {preview}")

    if not final_step:
        return "No research findings generated."

    messages = final_step.get("messages", [])
    if not messages:
        return "No research findings generated."

    final_message = messages[-1]
    print("[Researcher] Finished research.")
    return getattr(final_message, "content", str(final_message))