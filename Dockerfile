# syntax=docker/dockerfile:1.7
# Overnight Research Company container.
# Works with both Docker and Podman (and their composes).
FROM ubuntu:24.04

ARG DEBIAN_FRONTEND=noninteractive
ARG NODE_MAJOR=20

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    TZ=UTC \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# ---------------------------------------------------------------------------
# Base OS toolchain: shell essentials, C/C++, Python, Rust (all native apt).
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl wget gnupg git sudo \
        less vim nano tmux htop procps file unzip zip jq ripgrep \
        locales tzdata \
        build-essential cmake pkg-config autoconf automake libtool \
        gcc g++ gdb make clang lldb \
        python3 python3-pip python3-venv python3-dev python-is-python3 \
        rustc cargo rustfmt \
        openssl libssl-dev zlib1g-dev \
    && locale-gen en_US.UTF-8 \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Node.js (for Claude Code CLI) + Claude Code itself.
# ---------------------------------------------------------------------------
RUN curl -fsSL https://deb.nodesource.com/setup_${NODE_MAJOR}.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

RUN npm install -g @anthropic-ai/claude-code \
    && npm cache clean --force

# ---------------------------------------------------------------------------
# uv (fast Python package manager used by the Python Engineer persona
# and by the arxiv MCP server).
# ---------------------------------------------------------------------------
RUN curl -LsSf https://astral.sh/uv/install.sh | INSTALLER_NO_MODIFY_PATH=1 sh \
    && mv /root/.local/bin/uv /usr/local/bin/uv \
    && mv /root/.local/bin/uvx /usr/local/bin/uvx

# ---------------------------------------------------------------------------
# Full TeX Live distribution for the LaTeX Writer persona.
# This is a very large layer (multi-GB) but we cache it late so iterating on
# earlier layers doesn't force a reinstall.
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        texlive-full \
        latexmk \
        biber \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Pre-warm the uv cache with the arxiv MCP server's dependencies so the
# first `claude mcp` call inside a fresh container is fast.
# ---------------------------------------------------------------------------
RUN uv run --quiet \
        --with "mcp[cli]" \
        --with arxiv \
        --with pypdf \
        --with httpx \
        python -c "import mcp, arxiv, pypdf, httpx; print('arxiv MCP deps prewarmed')"

WORKDIR /workspace

# Default HOME is /root; we bind-mount the host ~/.claude onto /root/.claude
# at run time so auth, settings, and MCP registrations persist.
CMD ["/bin/bash"]
