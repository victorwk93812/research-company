#!/usr/bin/env bash
# Register the arxiv MCP server at user scope for the current Claude Code
# install. Idempotent — safe to re-run.
#
# Because the server is registered via `uv run --with ...`, the only
# runtime requirement is `uv` on PATH. Dependencies are resolved on demand
# and cached per-environment, so the same script works identically on the
# host and inside the research container (each environment registers the
# absolute path it sees locally).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SERVER_PATH="${REPO_ROOT}/mcp_servers/arxiv/arxiv_mcp_server.py"

if [[ ! -f "${SERVER_PATH}" ]]; then
  echo "error: arxiv MCP server not found at ${SERVER_PATH}" >&2
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "error: 'uv' is not on PATH. Install it from https://astral.sh/uv" >&2
  exit 1
fi

if ! command -v claude >/dev/null 2>&1; then
  echo "error: 'claude' (Claude Code CLI) is not on PATH." >&2
  exit 1
fi

UV_BIN="$(command -v uv)"

# Wipe any prior registration so re-runs pick up path changes cleanly.
claude mcp remove -s user arxiv >/dev/null 2>&1 || true

claude mcp add -s user arxiv -- \
  "${UV_BIN}" run --quiet \
    --with "mcp[cli]" \
    --with arxiv \
    --with pypdf \
    --with httpx \
    python "${SERVER_PATH}"

echo "Registered arxiv MCP server at user scope."
echo "  Server: ${SERVER_PATH}"
echo "Verify with: claude mcp list"
