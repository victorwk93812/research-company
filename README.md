# 🔬 The Nightly Research Company: Autonomous Lab Framework

This repository houses an autonomous, multi-agent AI research framework driven by Claude Code. It is designed to run unsupervised overnight on a Linux lab server, grinding through complex theoretical physics derivations, algorithm design, and high-performance computing implementations.

By the time you wake up, the framework will have produced theoretical drafts, rigorous mathematical critiques, optimized Python simulations, and a compiled XeLaTeX report of the findings.

## 🏗️ Architecture Overview

The system bypasses standard LLM context-collapse and hallucination loops by utilizing **File-Based Memory** and a **Static Persona State Machine**. Claude Code sequentially adopts specialized personas, reading intermediate `.md`, `.py`, and `.tex` files to anchor its context before executing the next phase.

### Directory Structure
```text
/research-company
├── README.md                  <-- This file
├── CLAUDE.md                  <-- Master directives auto-loaded by Claude
├── Dockerfile                 <-- Container image (docker + podman compatible)
├── docker-compose.yml         <-- Compose file (docker compose / podman-compose)
├── .claude/
│   └── settings.json          <-- Enforces "auto" permission mode
├── /personas/                 <-- Static behavioral anchors
│   ├── 01_Researcher.md       (Theory & Ideation, grounded in arxiv)
│   ├── 02_RA_Skeptic.md       (Rigorous Critique, cross-checks arxiv)
│   ├── 03_Python_Engineer.md  (HPC Implementation)
│   ├── 04_LaTeX_Writer.md     (XeLaTeX Typesetting)
│   └── 05_Review_Board.md     (Final Evaluation)
├── /mcp_servers/
│   └── /arxiv/                <-- arxiv MCP server (search + PDF fetch)
├── /scripts/
│   └── register-arxiv-mcp.sh  <-- Register arxiv MCP at user scope
└── /runs/                     <-- Isolated daily execution workspaces
    ├── /run_001
    └── /run_002
```

## ⚙️ Prerequisites

You can run the framework **either** directly on a lab server **or** inside the provided container. The container is the recommended path because it pins every toolchain (TeX Live, compilers, Python, Rust, Node, Claude Code) to a known-good version.

### Option A — Run on bare metal
* **Claude Code CLI:** Authenticated and ready.
* **uv:** For reproducible Python project and dependency management.
* **XeLaTeX / TeX Live:** Required by the LaTeX Writer.
* **Standard build tools:** `make`, `cmake`, `gcc` (plus Rust if needed).

### Option B — Run in the container (recommended)
* **Docker** ≥ 24 with Compose v2, **or** **Podman** ≥ 4 with `podman-compose`.
* Your host `~/.claude` directory, already authenticated with Claude Code (`claude login` on the host once).

## 📦 The Container

The image is a single `Dockerfile` that works for both Docker and Podman (and their respective composes). It bundles:

* **TeX Live full** + `latexmk` + `biber` (for the LaTeX Writer)
* **C/C++:** `gcc`, `g++`, `clang`, `gdb`, `cmake`, `build-essential`
* **Python 3** + `uv`
* **Rust:** `rustc`, `cargo`, `rustfmt` (native apt packages)
* **Node.js 20** + the **Claude Code CLI** (`claude`)
* The **arxiv MCP server** dependencies pre-warmed in the `uv` cache

### Build the image

```bash
# Docker
docker compose build

# Podman
podman-compose build
# or equivalently:
# podman build -t research-company:latest .
```

> The TeX Live layer is large (multi-GB). The first build will take a while; subsequent builds reuse the cached layer.

### Start a research session

The container is designed to be invoked interactively, with the repo and your `~/.claude` bind-mounted. Use `run --rm` rather than `up` so you get a clean TTY per session:

```bash
# Docker
docker compose run --rm research

# Podman
podman-compose run --rm research
```

Or without compose:

```bash
# Docker
docker run --rm -it \
  -v "$PWD":/workspace \
  -v "$HOME/.claude":/root/.claude \
  -w /workspace \
  research-company:latest

# Podman (rootless: host UID maps to root-in-container, so bind mounts just work)
podman run --rm -it \
  -v "$PWD":/workspace \
  -v "$HOME/.claude":/root/.claude \
  -w /workspace \
  research-company:latest
```

> If you also want the host's `~/.claude.json` visible inside the container (e.g. to share host-scope MCP registrations), add `-v "$HOME/.claude.json":/root/.claude.json` **after** running `touch ~/.claude.json` on the host. Without the `touch`, the daemon will create it as a directory and Claude Code will fail to start.

You now have a shell inside `/workspace` with `claude` on the `PATH` and your host auth already mounted.

### A note on file ownership

- **Podman rootless** maps your host UID to `root` inside the container automatically, so files created during the run appear on the host owned by you. Nothing else needed.
- **Docker** runs as real root by default, so files the container creates will be owned by `root` on the host. Two ways to avoid that:
  - Run with `-u "$(id -u):$(id -g)" -e HOME=/workspace` (Claude Code auth mount still needs to be at `$HOME/.claude` — you'd also need to pass that through), or
  - `sudo chown -R "$USER" runs/` after the run.

## 📚 arxiv MCP server

The Researcher and RA personas are required to ground their work in recent literature. They do this through a small MCP server at `mcp_servers/arxiv/` that exposes three tools:

| Tool | Purpose |
| --- | --- |
| `search_arxiv(query, max_results, sort_by)` | Keyword / field search against arxiv |
| `get_paper(arxiv_id)` | Full metadata + abstract for one paper |
| `download_paper_text(arxiv_id, max_chars)` | PDF → extracted plain text |

### Register the MCP server (user scope)

Run this **once per environment** where you want the server available — i.e. once on the host for your normal Claude Code usage, and once inside the container for containerized runs:

```bash
./scripts/register-arxiv-mcp.sh
```

The script:

1. Registers at **user scope** (`claude mcp add -s user arxiv ...`), so every Claude Code session for your user sees the server.
2. Wires up the server via `uv run --with mcp[cli] --with arxiv --with pypdf --with httpx`, so the only runtime dependency is `uv` — no virtualenv to manage.
3. Resolves the absolute path of `arxiv_mcp_server.py` at call time, so the same script produces the right registration on the host (host path) and inside the container (`/workspace/mcp_servers/arxiv/...`).

Confirm it registered:

```bash
claude mcp list
# arxiv: uv run --quiet --with mcp[cli] --with arxiv --with pypdf --with httpx python /.../arxiv_mcp_server.py
```

> Because `~/.claude.json` is **not** typically mounted by the default container setup (only `~/.claude` is), host and container maintain separate MCP registrations — that is intentional, since the absolute paths differ between environments. If you mount `~/.claude.json` into the container as well (the default `docker-compose.yml` does), re-running `register-arxiv-mcp.sh` inside the container will simply overwrite the host entry with the container-local path. Re-run it on the host afterwards to switch back.

## 🚀 How to Run a Nightly Experiment

It is recommended to run this under a tmux session or a nohup session for easier progress checking and result harvesting.

### 1. Create a new run directory
```bash
mkdir -p runs/run_001
cd runs/run_001
```

### 2. Write the instruction file
Create an `instruction.md` file in this new directory. This is the seed prompt for the entire night. Be as mathematically and physically specific as possible.

**Example `instruction.md`:**
> **Objective:** Propose a method to utilize SU(2) symmetries to optimize tensor network simulations, specifically focusing on reducing memory consumption during a DMRG step for a Heisenberg chain.
>
> **Deliverables:**
> 1. Derive the block-sparse tensor contraction mathematically.
> 2. Implement a Python prototype to benchmark the memory consumption of the SU(2) symmetric approach against a dense U(1) baseline.
> 3. Summarize the memory scaling results in a formal report.

### 3. Launch the engine

With `.claude/settings.json` enforcing auto mode, execute Claude Code with a master prompt that triggers the workflow defined in `CLAUDE.md`:

```bash
claude "Read instruction.md and begin Phase 1 of the research cycle. Read your roles from ../../personas/ as needed. Do not stop until Phase 5 yields a final_review.md."
```

The same command works identically on the host and inside the container.

## 🛡️ Failsafes and Security

* **Resource Limiting:** The Python Engineer persona is hardcoded to enforce `resource.setrlimit` (RAM capping) and C-library thread limiting (`OMP_NUM_THREADS`, etc.) in all scripts. This prevents runaway tensor contractions from crashing the shared lab server.
* **Context Anchoring:** Do not delete the files in `/personas/`. Claude relies on physically reading these via `cat` to context-switch cleanly and reset its behavior during long sessions.
* **Modularity:** The Python Engineer will automatically build its environment in `./src/`, and the LaTeX Writer will generate its files in `./report/`.
