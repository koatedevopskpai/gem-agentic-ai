"""
WEB SEARCH TOOL
===============
Wraps the duckduckgo_search library as a LangChain-compatible tool.

WHY duckduckgo_search directly instead of langchain-community?
  The langchain-community package (which provides DuckDuckGoSearchRun)
  is deprecated and emits runtime warnings. By calling the underlying
  library directly, we eliminate the deprecation noise and one
  dependency. The @tool decorator from langchain_core makes the
  function fully compatible with both agent implementations.

WHY import inside the function?
  Deferred (lazy) import: if duckduckgo_search is not installed or
  the network is unavailable, only this tool breaks — the rest of
  the agent still works. A top-level import would crash the entire
  application at startup.
"""

import logging

from langchain_core.tools import tool

logger = logging.getLogger(__name__)


@tool
def web_search(query: str, max_results: int = 5) -> str:
    """Search the web for current events, news, or general information.

    Uses DuckDuckGo's search API — no API key required, no rate
    limiting for moderate usage. This makes it ideal for demos and
    portfolio pieces where the reviewer can run the code without
    signing up for yet another service.

    LIMITATIONS
      DuckDuckGo results are less comprehensive than Google or Bing.
      For production, swap this implementation for a paid search API
      (SerpAPI, Tavily, etc.) — the tool interface stays the same.

    Args:
        query: The search query string (e.g., "latest AI news 2026").
        max_results: Maximum number of results to return (default 5, max 20).

    Returns:
        Formatted search results as a single string, or an error message
        explaining why the search failed.
    """
    logger.info("web_search called: query='%s', max_results=%d", query, max_results)

    # ── Lazy import ──────────────────────────────────────────
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        logger.error("duckduckgo_search library is not installed.")
        return (
            "Error: The web search library is not available. "
            "Install it with: pip install duckduckgo-search"
        )

    # ── Execute search ───────────────────────────────────────
    try:
        # DDGS is a context manager that handles connection lifecycle.
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
    except Exception as e:
        logger.exception("DuckDuckGo search failed for query '%s': %s", query, e)
        return f"Search failed: {e}"

    # ── Format results ───────────────────────────────────────
    if not results:
        logger.info("No search results found for query: %s", query)
        return "No results found."

    # Each result is a dict with keys: title, href, body
    formatted: list[str] = []
    for i, r in enumerate(results, 1):
        title = r.get("title", "Untitled")
        href = r.get("href", "")
        body = r.get("body", "")
        formatted.append(f"{i}. {title}\n   {href}\n   {body}")

    return "\n\n".join(formatted)
