"""arxiv MCP server.

Exposes arxiv search and paper-fetch tools so that the Researcher and RA
personas can ground their work in recent literature before drafting or
critiquing theory.

Transport: stdio (default for `mcp.run()`).

Tools:
    search_arxiv(query, max_results=5, sort_by="relevance")
    get_paper(arxiv_id)
    download_paper_text(arxiv_id, max_chars=40000)
"""
from __future__ import annotations

import io
from typing import Any

import arxiv
import httpx
from mcp.server.fastmcp import FastMCP
from pypdf import PdfReader

mcp = FastMCP("arxiv")

_client = arxiv.Client(page_size=25, delay_seconds=3.0, num_retries=3)

_SORT_MAP = {
    "relevance": arxiv.SortCriterion.Relevance,
    "submitted": arxiv.SortCriterion.SubmittedDate,
    "last_updated": arxiv.SortCriterion.LastUpdatedDate,
}


def _format_paper(result: arxiv.Result) -> dict[str, Any]:
    return {
        "arxiv_id": result.get_short_id(),
        "title": result.title.strip(),
        "authors": [a.name for a in result.authors],
        "published": result.published.isoformat() if result.published else None,
        "updated": result.updated.isoformat() if result.updated else None,
        "primary_category": result.primary_category,
        "categories": list(result.categories),
        "summary": " ".join(result.summary.split()),
        "pdf_url": result.pdf_url,
        "entry_id": result.entry_id,
        "doi": result.doi,
        "journal_ref": result.journal_ref,
    }


@mcp.tool()
def search_arxiv(
    query: str,
    max_results: int = 5,
    sort_by: str = "relevance",
) -> list[dict[str, Any]]:
    """Search arxiv for papers matching the given query.

    Args:
        query: Search string. Supports arxiv field prefixes
            (e.g. `ti:"tensor network"`, `au:del_maestro`, `cat:cond-mat.str-el`).
        max_results: Maximum number of results to return (1-25). Default 5.
        sort_by: One of "relevance", "submitted", or "last_updated".
            Use "submitted" to get the most recent papers.

    Returns:
        A list of paper metadata dictionaries with arxiv_id, title, authors,
        summary (abstract), published date, pdf_url, and more.
    """
    capped = max(1, min(int(max_results), 25))
    sort_criterion = _SORT_MAP.get(sort_by, arxiv.SortCriterion.Relevance)
    search = arxiv.Search(
        query=query,
        max_results=capped,
        sort_by=sort_criterion,
        sort_order=arxiv.SortOrder.Descending,
    )
    return [_format_paper(r) for r in _client.results(search)]


@mcp.tool()
def get_paper(arxiv_id: str) -> dict[str, Any]:
    """Fetch full metadata and abstract for a single arxiv paper.

    Args:
        arxiv_id: Short arxiv identifier, e.g. "2401.12345" or
            "hep-th/0101001". Version suffixes like "v2" are accepted.
    """
    search = arxiv.Search(id_list=[arxiv_id])
    try:
        result = next(_client.results(search))
    except StopIteration as exc:
        raise ValueError(f"No arxiv paper found for id {arxiv_id!r}") from exc
    return _format_paper(result)


@mcp.tool()
def download_paper_text(arxiv_id: str, max_chars: int = 40000) -> dict[str, Any]:
    """Download a paper's PDF and return extracted plain text.

    Use this when the abstract is not enough and you need to read methodology,
    derivations, or specific sections. The text is truncated to `max_chars`
    to keep the model context reasonable.

    Args:
        arxiv_id: Short arxiv identifier.
        max_chars: Maximum number of characters to return (default 40000).

    Returns:
        Dict with title, num_pages, pages_extracted, truncated flag, and text.
    """
    meta = get_paper(arxiv_id)
    with httpx.Client(follow_redirects=True, timeout=60.0) as client:
        resp = client.get(meta["pdf_url"])
        resp.raise_for_status()
        reader = PdfReader(io.BytesIO(resp.content))

    pages_text: list[str] = []
    total = 0
    for page in reader.pages:
        text = page.extract_text() or ""
        pages_text.append(text)
        total += len(text)
        if total >= max_chars:
            break

    joined = "\n\n".join(pages_text)
    return {
        "arxiv_id": meta["arxiv_id"],
        "title": meta["title"],
        "num_pages": len(reader.pages),
        "pages_extracted": len(pages_text),
        "truncated": len(joined) > max_chars,
        "text": joined[:max_chars],
    }


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
