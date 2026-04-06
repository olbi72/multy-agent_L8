import os
from typing import List, Dict

import httpx
import trafilatura
from ddgs import DDGS
from langchain.tools import tool

from retriever import search_knowledge_base
from config import settings


def web_search_raw(query: str) -> List[Dict]:
    try:
        results = DDGS().text(query, max_results=settings.max_search_results)

        formatted_results = [
            {
                "title": item.get("title", "")[:200],
                "url": item.get("href", ""),
                "snippet": item.get("body", "")[:500],
            }
            for item in results
        ]

        return formatted_results

    except Exception as e:
        return [
            {
                "title": "Search error",
                "url": "",
                "snippet": f"web_search failed: {str(e)}",
            }
        ]


@tool
def web_search(query: str) -> str:
    """
    Search the web for relevant information about a topic.
    Returns a short formatted list of search results with title, URL, and snippet.
    """
    results = web_search_raw(query)

    lines = []
    for i, item in enumerate(results, start=1):
        lines.append(
            f"{i}. {item['title']}\n"
            f"URL: {item['url']}\n"
            f"Snippet: {item['snippet']}"
        )

    return "\n\n".join(lines)


def read_url_raw(url: str) -> str:
    try:
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            text = trafilatura.extract(downloaded)
            if text:
                return text[: settings.max_url_content_length]
    except Exception:
        pass

    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
            )
        }
        response = httpx.get(url, headers=headers, timeout=20.0, follow_redirects=True)
        response.raise_for_status()

        text = trafilatura.extract(response.text)
        if text:
            return text[: settings.max_url_content_length]

        return "Error: page downloaded but main text could not be extracted."
    except Exception as e:
        return f"Error: could not read page content. Details: {str(e)}"


@tool
def read_url(url: str) -> str:
    """
    Read the main text content from a web page URL.
    Returns extracted page content or an error message if the page cannot be read.
    """
    return read_url_raw(url)


def save_report_raw(filename: str, content: str) -> str:
    try:
        os.makedirs(settings.output_dir, exist_ok=True)

        path = os.path.join(settings.output_dir, filename)

        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

        return f"Report saved successfully to: {path}"
    except Exception as e:
        return f"Error: could not write report. Details: {str(e)}"


@tool
def save_report(filename: str, content: str) -> str:
    """
    Save a Markdown research report to a file in the output directory.
    Returns a confirmation message with the saved file path.
    """
    return save_report_raw(filename, content)


@tool
def write_report(filename: str, content: str) -> str:
    """
    Backward-compatible alias for save_report.
    """
    return save_report_raw(filename, content)


@tool
def knowledge_search(query: str) -> str:
    """
    Search the local knowledge base. Use for questions about ingested documents.
    Returns relevant passages from the local vector database.
    """
    try:
        result = search_knowledge_base(query)
        return f"[knowledge_search]\n{result}"
    except Exception as e:
        return f"Knowledge search error: {str(e)}"