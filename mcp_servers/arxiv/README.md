# arxiv MCP server

A minimal MCP server that gives Claude Code three tools:

- `search_arxiv(query, max_results=5, sort_by="relevance")`
- `get_paper(arxiv_id)`
- `download_paper_text(arxiv_id, max_chars=40000)`

## Register it

From the repo root:

```bash
./scripts/register-arxiv-mcp.sh
```

This registers the server at **user scope** (`claude mcp add -s user ...`), so
it is available in every Claude Code session for your user. The helper runs
the server via `uv run --with ...`, so only `uv` is required — no separate
virtualenv or install step.

Verify with:

```bash
claude mcp list
```

## Use it inside the research container

The container bakes in `uv` and pre-warms the dependency cache. After
starting a container with `~/.claude` bind-mounted, run the same script
inside it to register the server for the container environment:

```bash
./scripts/register-arxiv-mcp.sh
```
