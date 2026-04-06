import json
import uuid

from langgraph.types import Command

from config import settings
from supervisor import supervisor


CURRENT_THREAD_ID = None


def new_thread_id() -> str:
    return f"interactive-session-{uuid.uuid4()}"


def print_interrupt(interrupt_value) -> None:
    print("\n" + "=" * 60)
    print("⏸️  ACTION REQUIRES APPROVAL")
    print("=" * 60)

    if isinstance(interrupt_value, dict):
        action_requests = interrupt_value.get("action_requests", [])
        if action_requests:
            for action in action_requests:
                print(f"Tool:  {action.get('name', 'unknown')}")
                print(
                    f"Args:  {json.dumps(action.get('args', {}), ensure_ascii=False, indent=2)}"
                )

                content = action.get("args", {}).get("content", "")
                if content:
                    preview = content[:800]
                    if len(content) > 800:
                        preview += "..."
                    print("\nPreview:\n")
                    print(preview)
        else:
            print(json.dumps(interrupt_value, ensure_ascii=False, indent=2))
    else:
        print(str(interrupt_value))

    print("=" * 60)


def extract_interrupt(step_or_result):
    interrupts = step_or_result.get("__interrupt__", [])
    if interrupts:
        first_interrupt = interrupts[0]
        return getattr(first_interrupt, "value", first_interrupt)
    return None


def print_stream_step(step, seen_tool_calls, seen_tool_results):
    messages = step.get("messages", [])
    if not messages:
        return

    last_msg = messages[-1]

    tool_calls = getattr(last_msg, "tool_calls", None)
    if tool_calls:
        for call in tool_calls:
            tool_name = call.get("name", "unknown_tool")
            tool_args = str(call.get("args", {}))
            key = (tool_name, tool_args)
            if key not in seen_tool_calls:
                seen_tool_calls.add(key)
                print(f"\n🔧 Tool call: {tool_name}({tool_args})")

    if getattr(last_msg, "type", "") == "tool":
        content = getattr(last_msg, "content", "")
        if content not in seen_tool_results:
            seen_tool_results.add(content)
            preview = content[:500] + ("..." if len(content) > 500 else "")
            print(f"\n📎 Result: {preview}")


def run_supervisor(user_input: str):
    config = {"configurable": {"thread_id": CURRENT_THREAD_ID}}
    final_answer = None
    final_step = None

    seen_tool_calls = set()
    seen_tool_results = set()

    for step in supervisor.stream(
        {"messages": [("user", user_input)]},
        config=config,
        stream_mode="values",
    ):
        final_step = step
        print_stream_step(step, seen_tool_calls, seen_tool_results)

        messages = step.get("messages", [])
        if messages:
            last_message = messages[-1]
            if (
                getattr(last_message, "type", "") == "ai"
                and getattr(last_message, "content", "")
            ):
                final_answer = last_message.content

    return final_step or {}, final_answer


def resume_supervisor(decision_payload: dict):
    config = {"configurable": {"thread_id": CURRENT_THREAD_ID}}
    final_answer = None
    final_step = None

    seen_tool_calls = set()
    seen_tool_results = set()

    for step in supervisor.stream(
        Command(resume=decision_payload),
        config=config,
        stream_mode="values",
    ):
        final_step = step
        print_stream_step(step, seen_tool_calls, seen_tool_results)

        messages = step.get("messages", [])
        if messages:
            last_message = messages[-1]
            if (
                getattr(last_message, "type", "") == "ai"
                and getattr(last_message, "content", "")
            ):
                final_answer = last_message.content

    return final_step or {}, final_answer


def handle_interrupt(interrupt_value):
    while True:
        action = input("\n👉 approve / edit / reject: ").strip().lower()

        if action == "approve":
            return {"decisions": [{"type": "approve"}]}

        if action == "edit":
            user_feedback = input("✏️  Your feedback: ").strip()

            if isinstance(interrupt_value, dict):
                action_requests = interrupt_value.get("action_requests", [])
                if action_requests:
                    original_action = action_requests[0]
                    original_args = original_action.get("args", {}).copy()

                    original_content = original_args.get("content", "")
                    edited_content = (
                        original_content
                        + f"\n\n## User requested changes\n{user_feedback}"
                    )

                    return {
                        "decisions": [
                            {
                                "type": "edit",
                                "edited_action": {
                                    "name": original_action.get("name", "save_report"),
                                    "args": {
                                        **original_args,
                                        "content": edited_content,
                                    },
                                },
                            }
                        ]
                    }

            print("Could not prepare edited action. Try approve or reject.")
            continue

        if action == "reject":
            reason = input("Reason: ").strip()
            return {
                "decisions": [
                    {
                        "type": "reject",
                        "message": reason or "User rejected the write action.",
                    }
                ]
            }

        print("Please type: approve, edit, or reject.")


def main():
    global CURRENT_THREAD_ID

    print("Multi-Agent Research System with HITL (type 'exit' to quit)")
    print("-" * 60)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        CURRENT_THREAD_ID = new_thread_id()
        result, final_answer = run_supervisor(user_input)

        revision_count = 0
        max_revisions = settings.max_revision_rounds
        stopped_due_to_revision_limit = False

        while True:
            interrupt_value = extract_interrupt(result)
            if not interrupt_value:
                break

            if not final_answer:
                revision_count += 1
                if revision_count > max_revisions:
                    print(
                        f"\nAgent:\nStopped after {max_revisions} revision rounds without reaching approval."
                    )
                    stopped_due_to_revision_limit = True
                    break

            print_interrupt(interrupt_value)
            decision_payload = handle_interrupt(interrupt_value)
            result, final_answer = resume_supervisor(decision_payload)

        if stopped_due_to_revision_limit:
            continue

        if final_answer:
            print(f"\nAgent:\n{final_answer}")
        else:
            print("\nAgent: No final response generated.")


if __name__ == "__main__":
    main()