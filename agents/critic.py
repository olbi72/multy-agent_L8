import json

from langchain_openai import ChatOpenAI
from langchain.tools import tool

from config import settings
from schemas import CritiqueResult
from tools import web_search, knowledge_search


def extract_keywords(text: str) -> set[str]:
    words = []
    for word in text.lower().replace("/", " ").replace("-", " ").split():
        clean = "".join(ch for ch in word if ch.isalnum())
        if len(clean) >= 4:
            words.append(clean)
    return set(words)


def kb_result_looks_relevant(request: str, kb_result: str) -> bool:
    if not kb_result:
        return False

    kb_lower = kb_result.lower()

    if "no relevant documents found" in kb_lower:
        return False

    request_keywords = extract_keywords(request)
    if not request_keywords:
        return True

    overlap_count = sum(1 for kw in request_keywords if kw in kb_lower)
    return overlap_count >= 2


def build_verification_query(payload) -> str:
    if isinstance(payload, dict):
        original_request = str(payload.get("request", "")).strip()
        plan = payload.get("plan", {})
        findings = str(payload.get("findings", "")).strip()

        if isinstance(plan, dict):
            search_queries = plan.get("search_queries", [])
            if search_queries:
                return str(search_queries[0])

            goal = str(plan.get("goal", "")).strip()
            if goal:
                return goal

        if original_request:
            return original_request

        if findings:
            return findings[:300]

        return "exchange rate forecast 2026"

    if isinstance(payload, str):
        return payload.strip()

    return "exchange rate forecast 2026"


llm = ChatOpenAI(
    model=settings.model_name,
    api_key=settings.api_key.get_secret_value(),
)

structured_critic_llm = llm.with_structured_output(CritiqueResult)


@tool
def critique(request) -> str:
    """
    Critically evaluate research findings and return a structured critique.
    """
    payload = request
    request_text = json.dumps(payload, ensure_ascii=False) if isinstance(payload, dict) else str(payload)
    request_lower = request_text.lower()

    is_local_only = (
        (
            "local knowledge base" in request_lower
            or "knowledge base documents" in request_lower
            or "ingested local knowledge base" in request_lower
            or "documents in the local knowledge base" in request_lower
            or "documents already ingested" in request_lower
            or "local documents" in request_lower
            or "documents available in the local knowledge base" in request_lower
            or "the needed material is already available via the local knowledge base tool" in request_lower
        )
        and ("web" not in request_lower and "external" not in request_lower)
    )

    if is_local_only:
        print("\n[Critic] Starting local-only critique...")
        kb_result = knowledge_search.invoke(
            {"query": "main ideas from local knowledge base documents"}
        )
        preview = kb_result[:500] + ("..." if len(kb_result) > 500 else "")
        print(f"[Critic] 📎 Single KB check: {preview}")

        structured = structured_critic_llm.invoke(
            f"""
You are evaluating a local-knowledge-base-only research summary.

Return CritiqueResult.

Rules:
- This is a narrow validation task.
- Do not expand scope beyond the local documents.
- Do not require external freshness.
- Approve if the findings are grounded in the local corpus, cover the main ideas reasonably well, and are structured clearly.
- Revise only if something important is missing, unsupported, or poorly structured.
- verdict must be exactly APPROVE or REVISE.

User request:
{request_text}

Local knowledge base verification evidence:
{kb_result}
"""
        )
        print("[Critic] Finished local-only critique.")
        return structured.model_dump_json(indent=2, ensure_ascii=False)

    print("\n[Critic] Starting external/mixed critique...")

    verification_query = build_verification_query(payload)
    print(f"[Critic] Verification query: {verification_query}")

    kb_result = knowledge_search.invoke({"query": verification_query})
    kb_is_relevant = kb_result_looks_relevant(verification_query, kb_result)

    if kb_is_relevant:
        kb_preview = kb_result[:500] + ("..." if len(kb_result) > 500 else "")
        print(f"[Critic] 📎 Single KB check: {kb_preview}")
    else:
        kb_result = "KB evidence not relevant to this request."
        print("[Critic] 📎 KB check: skipped as not relevant.")

    web_result = web_search.invoke({"query": verification_query})
    web_preview = web_result[:500] + ("..." if len(web_result) > 500 else "")
    print(f"[Critic] 📎 Single web check: {web_preview}")

    structured = structured_critic_llm.invoke(
        f"""
You are evaluating external or mixed-source research findings.

Return CritiqueResult.

Rules:
- This is a verification task, not a new research task.
- Use the provided evidence only.
- Do not expand into new searches, adjacent topics, or broad literature review.
- Check freshness only based on the provided web evidence.
- Check completeness, structure, and whether the findings appear supported.
- verdict must be exactly APPROVE or REVISE.

User request:
{request_text}

Verification query used:
{verification_query}

Local knowledge base verification evidence:
{kb_result}

Web verification evidence:
{web_result}
"""
    )

    print("[Critic] Finished external/mixed critique.")
    return structured.model_dump_json(indent=2, ensure_ascii=False)