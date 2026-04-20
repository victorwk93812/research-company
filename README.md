# The Nightly Research Company

An autonomous, multi-agent research framework driven by Claude Code. Given a single `instruction.md`, it cycles through five specialized personas — Researcher, RA Skeptic, LaTeX Writer, Python Engineer, Review Board — to produce a grounded theoretical draft, a rigorous critique, a benchmarked Python prototype, and a compiled XeLaTeX report, all unattended.

The system avoids LLM context collapse by using **file-based memory** and a **static persona state machine**: Claude physically reads each persona file before adopting its role, then writes its output to disk before the next persona reads it back.

---

## Architecture

```text
research-company/
├── README.md                   This file
├── CLAUDE.md                   Master directives auto-loaded by Claude Code
├── Dockerfile                  Container image (works with docker + podman)
├── docker-compose.yml          Compose entrypoint (docker compose + podman-compose)
├── .claude/
│   └── settings.json           Pins Claude Code to "auto" permission mode
├── personas/                   Static behavioral anchors — read before each phase
│   ├── 01_Researcher.md        Theory & ideation, grounded in arxiv literature
│   ├── 02_RA_Skeptic.md        Rigorous critique, cross-checks arxiv literature
│   ├── 03_Python_Engineer.md   HPC implementation
│   ├── 04_LaTeX_Writer.md      XeLaTeX typesetting
│   └── 05_Review_Board.md      Final evaluation
├── mcp_servers/
│   └── arxiv/                  MCP server: search_arxiv, get_paper, download_paper_text
├── scripts/
│   └── register-arxiv-mcp.sh   One-shot user-scope MCP registration (host or container)
└── runs/
    └── run_NNN/                One isolated workspace per nightly run
        ├── instruction.md      Seed prompt (you write this)
        ├── theory_draft.md     Researcher output
        ├── ra_critique.md      RA Skeptic output
        ├── src/                Python Engineer workspace
        ├── report/             LaTeX Writer workspace
        └── final_review.md     Review Board output
```

---

## Requirements

Pick one of:

- **Container (recommended)** — Docker ≥ 24 or Podman ≥ 4. Compose (`docker compose` / `podman-compose`) is optional; the walkthrough below shows both paths. Nothing else needs to be installed; the image ships TeX Live, compilers, Python, Rust, Node, Claude Code, and `uv`.
- **Bare metal** — Linux with Claude Code, `uv`, TeX Live (XeLaTeX + `latexmk`), `gcc`/`g++`/`cmake`, and a recent Rust toolchain already installed.

Either way, you need a Claude Code account already authenticated on the host (`claude` login flow completed once).

---

## How to Run a Nightly Experiment

This is the complete walkthrough for a first-time user. Each step has a verification command — run it and check the expected output before moving on. The whole path takes about 20–40 minutes the first time (most of that is the TeX Live layer downloading).

> Assumes you've cloned this repo and your working directory is the repo root.

### Step 1 — Install and verify your container runtime

Install Docker or Podman. Compose (`docker compose` / `podman-compose`) is recommended but optional.

```bash
# Docker
docker --version
docker compose version     # optional — skip if you don't have Compose

# OR Podman
podman --version
podman-compose --version   # optional — skip if you don't have podman-compose
```

Expected: each installed command prints a version, no errors. If Compose is missing, the walkthrough will show you the equivalent plain `docker run` / `podman run` commands; nothing else changes.

The rest of this guide uses `docker` / `docker compose` in examples. Substitute `podman` / `podman-compose` throughout if you're on Podman — the flags are identical.

### Step 2 — Authenticate Claude Code on the host

Even if you plan to run everything from inside the container, you must log in to Claude Code **once on the host**. The container bind-mounts the host's `~/.claude` directory to inherit that authentication.

```bash
# If Claude Code isn't installed on the host yet:
npm install -g @anthropic-ai/claude-code

claude    # interactive login flow, complete it in your browser
```

Verify:

```bash
ls ~/.claude
# Expected: at least a .credentials.json file (plus settings.json, projects/, etc.)
```

If `~/.claude` is empty, the login didn't complete — retry before continuing.

### Step 3 — Build the container image

**With Compose:**

```bash
docker compose build
```

**Without Compose (plain `docker` / `podman`):**

```bash
docker build -t research-company:latest .
# Or with Podman:
# podman build -t research-company:latest .
```

Expected: a successful build ending in something like `naming to docker.io/library/research-company:latest`.

The TeX Live layer is several GB; expect this step to take 10–25 minutes on a first build depending on your link speed. Subsequent builds reuse the cached layers.

Verify:

```bash
docker image ls research-company
# Expected: a row with REPOSITORY=research-company and TAG=latest.
```

### Step 4 — Enter the container

**With Compose:**

```bash
docker compose run --rm research
```

**Without Compose (plain `docker` / `podman`):**

```bash
# Docker
docker run --rm -it \
  -v "$PWD":/workspace:z \
  -v "$HOME/.claude":/root/.claude:z \
  -w /workspace \
  research-company:latest

# Podman (rootless: host UID maps to root-in-container, so bind mounts just work)
podman run --rm -it \
  -v "$PWD":/workspace:z \
  -v "$HOME/.claude":/root/.claude:z \
  -w /workspace \
  research-company:latest
```

The two bind mounts are the important part:
- `$PWD → /workspace` makes this repository visible inside the container.
- `$HOME/.claude → /root/.claude` forwards your Claude Code auth (and any settings / MCP registrations) into the container.

The `:z` suffix on each mount tells the runtime to relabel the source for SELinux so the container can actually read/write it. On SELinux-enforcing hosts (RHEL, Fedora, Rocky, Alma, CentOS Stream) this is what prevents `ls: Permission denied` errors. On non-SELinux hosts it's a harmless no-op, so leave it on.

> If you also want the host's `~/.claude.json` visible inside the container (e.g. to inherit host-scope MCP registrations), add `-v "$HOME/.claude.json":/root/.claude.json:z` to the command — **but** run `touch ~/.claude.json` on the host first, or the daemon will create it as a directory and Claude Code will fail to start.

You should land in an interactive shell at `/workspace`. Verify that the toolchain is present:

```bash
claude --version            # Claude Code version
uv --version                # uv version
xelatex --version | head -1 # XeLaTeX version line
gcc --version | head -1     # gcc version line
rustc --version             # rustc version
python3 --version           # Python 3.x
ls ~/.claude                # must show .credentials.json (from the host mount)
```

If `~/.claude` is empty inside the container, your bind mount didn't land — exit, check `docker-compose.yml`, and make sure `$HOME/.claude` exists on the host.

**Keep this shell open** for the remaining steps.

### Step 5 — Register the arxiv MCP server (inside the container)

The Researcher and RA personas are required to ground their work in recent literature through an MCP server that exposes three tools: `search_arxiv`, `get_paper`, `download_paper_text`. Register it at user scope once per environment — so once inside the container now, and (separately) once on your host if you want it available outside the container too.

From `/workspace` in the container:

```bash
./scripts/register-arxiv-mcp.sh
# Expected final line: "Registered arxiv MCP server at user scope."
```

Verify:

```bash
claude mcp list
# Expected: a line beginning with "arxiv:" pointing at a uv run ... command
#           that ends in .../mcp_servers/arxiv/arxiv_mcp_server.py
```

Smoke-test it by asking Claude a trivial question that forces it to hit the server:

```bash
claude -p "Use the arxiv MCP server to search for one paper about 'DMRG'. Return just the arxiv id and title."
# Expected: a single arxiv id and paper title. If Claude says the tool isn't available,
# re-run register-arxiv-mcp.sh and re-check claude mcp list.
```

> **If you also want the MCP server available outside the container**, run the same script once from the repo root on the host. Each environment registers its own absolute path; the two don't conflict.

### Step 6 — Create a run directory and write the instruction

From `/workspace` inside the container (or the repo root on the host):

```bash
mkdir -p runs/run_001
cd runs/run_001
```

Create `runs/run_001/instruction.md`. This single file is the seed prompt for the entire night, so be mathematically and physically specific. Minimal example:

```markdown
# Objective
Propose a method to exploit SU(2) symmetries to reduce memory consumption
during a DMRG step on a Heisenberg chain.

# Deliverables
1. A mathematical derivation of the block-sparse tensor contraction.
2. A Python prototype benchmarking memory consumption of the SU(2) symmetric
   approach against a dense U(1) baseline.
3. A formal report summarizing the memory scaling results.
```

Verify:

```bash
cat instruction.md
# Expected: your objective and deliverables, printed back to you.
```

### Step 7 — Launch the engine

Still inside `runs/run_001/`:

```bash
claude "Read instruction.md and begin Phase 1 of the research cycle. Read your roles from ../../personas/ as needed. Do not stop until Phase 5 yields a final_review.md."
```

Because `.claude/settings.json` pins the permission mode to `auto`, Claude will execute tools without prompting. The run will proceed through all five phases, writing intermediate files as it goes. Expect it to take from tens of minutes to several hours depending on the complexity of the instruction.

For long runs, wrap the command in `tmux` (or `nohup`) so you can disconnect and reattach:

```bash
tmux new -s run_001
# inside tmux:
claude "Read instruction.md and begin Phase 1..."
# detach with Ctrl-b d; reattach with: tmux attach -t run_001
```

### Step 8 — Harvest the output

When the engine terminates, your run directory will contain the full trail:

```text
runs/run_001/
├── instruction.md        Your seed prompt
├── theory_draft.md       Researcher's proposal (with a Literature Review section citing arxiv ids)
├── ra_critique.md        RA's critique (with an independent literature cross-check)
├── src/                  Python Engineer's uv project + simulation.log
├── report/
│   ├── main.tex          LaTeX source
│   └── main.pdf          Compiled report (if XeLaTeX succeeded)
└── final_review.md       Review Board's verdict
```

Verify:

```bash
ls runs/run_001/
test -f runs/run_001/final_review.md && echo "cycle complete" || echo "cycle did not finish — inspect the last persona's output"
```

If `final_review.md` is missing, read the most recently modified file in the run directory to see where the cycle stopped, then re-launch with a follow-up prompt pointing Claude at the next unfinished phase.

---

## Running on bare metal (alternative to the container)

If you'd rather skip the container, the only changes to the walkthrough above are:

- Skip steps 1, 3, 4. Install TeX Live, `uv`, and a C/C++/Rust toolchain yourself.
- In step 5, run `./scripts/register-arxiv-mcp.sh` directly from the host repo root.
- In step 6–7, run everything from a normal host shell.

Everything else is identical — the same personas, the same `claude` command, the same outputs.

---

## File ownership: Docker vs Podman

Bind mounts cross the host/container boundary, so UID mapping matters:

- **Podman rootless** maps your host UID to `root` inside the container automatically. Files created during the run appear on the host owned by you. No action required.
- **Docker** runs as real root by default, so files the container creates will be owned by `root` on the host. Either pass `-u "$(id -u):$(id -g)"` at run time, or run `sudo chown -R "$USER" runs/` after the run.

## Optional: sharing `~/.claude.json` with the container

`~/.claude/` (mounted by default) holds auth, settings, and MCP registrations. `~/.claude.json` is a separate host file that holds some global Claude Code config. If you want to share it with the container too, uncomment the relevant line in `docker-compose.yml` — **but first** run `touch ~/.claude.json` on the host. Otherwise the daemon will create it as a directory and Claude Code will fail to start.

## SELinux note

On SELinux-enforcing distributions (RHEL, Fedora, Rocky, Alma, CentOS Stream) the container is not allowed to touch bind-mounted host files unless the source has a matching label. Both `docker-compose.yml` and the plain `docker run` / `podman run` examples above already pass `:z` on every bind mount, which asks the runtime to apply a shared label. On non-SELinux hosts `:z` is a harmless no-op. If you see `ls: cannot access …: Permission denied` on `/workspace` or `/root/.claude` inside the container, double-check that the `:z` suffix is still on every mount.

---

## Failsafes and operational notes

- **Resource limiting.** The Python Engineer persona enforces `resource.setrlimit` (RAM capping) and C-library thread limits (`OMP_NUM_THREADS`, etc.) in every script, preventing runaway tensor contractions from crashing shared lab servers.
- **Context anchoring.** Do not delete or rename files in `personas/`. Claude relies on physically reading them (`cat ../../personas/0X_*.md`) to reset its context cleanly between phases.
- **Modularity.** The Python Engineer always initializes its workspace under `runs/run_NNN/src/` via `uv init`; the LaTeX Writer always writes to `runs/run_NNN/report/`. Do not move these.
- **MCP scope.** The arxiv server is registered at **user scope**, not project scope. That means a host registration and a container registration are independent — one does not leak into the other, even though both are pinned to your user.
