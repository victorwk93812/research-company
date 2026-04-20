# 🔬 The Nightly Research Company: Autonomous Lab Framework

This repository houses an autonomous, multi-agent AI research framework driven by Claude Code. It is designed to run unsupervised overnight on a Linux lab server, grinding through complex theoretical physics derivations, algorithm design, and high-performance computing implementations.

By the time you wake up, the framework will have produced theoretical drafts, rigorous mathematical critiques, optimized Python simulations, and a compiled XeLaTeX report of the findings.

## 🏗️ Architecture Overview

The system bypasses standard LLM context-collapse and hallucination loops by utilizing **File-Based Memory** and a **Static Persona State Machine**. Claude Code sequentially adopts specialized personas, reading intermediate `.md`, `.py`, and `.tex` files to anchor its context before executing the next phase.

### Directory Structure
```text
/research-company
├── README.md              <-- This file
├── CLAUDE.md                  <-- Master directives auto-loaded by Claude
├── .claude/
│   └── settings.json          <-- Enforces "auto" permission mode
├── /personas/                 <-- Static behavioral anchors
│   ├── 01_Researcher.md       (Theory & Ideation)
│   ├── 02_RA_Skeptic.md       (Rigorous Critique)
│   ├── 03_Python_Engineer.md  (HPC Implementation)
│   ├── 04_LaTeX_Writer.md     (XeLaTeX Typesetting)
│   └── 05_Review_Board.md     (Final Evaluation)
└── /runs/                     <-- Isolated daily execution workspaces
    ├── /run_001
    └── /run_002
```

## ⚙️ Prerequisites

Before launching a nightly run, ensure your lab server has the following installed:
* **Claude Code CLI:** Authenticated and ready.
* **uv:** For blazing-fast, reproducible Python project and dependency management.
* **XeLaTeX / TeX Live:** Required by the LaTeX Writer to compile physics reports.
* **Standard build tools:** `make`, `cmake`, `gcc` (if the Python Engineer requires compiling C-extensions or using OpenBLAS).

## 🚀 How to Run a Nightly Experiment

Do not run Claude Code directly from the project root, as this will pollute the directory. 

It is recommended to run this under a tmux session or a nohup session for easier progress checking and result harvesting.  

Follow these steps to initiate a cycle:

### 1. Create a New Run Directory
Initialize a new, sequentially numbered directory inside `/runs/`. 
```bash
mkdir -p runs/run_001
cd runs/run_001
```

### 2. Write the Instruction File
Create an `instruction.md` file in this new directory. This is the seed prompt for the entire night. Be as mathematically and physically specific as possible.

**Example `instruction.md`:**
> **Objective:** Propose a method to utilize SU(2) symmetries to optimize tensor network simulations, specifically focusing on reducing memory consumption during a DMRG step for a Heisenberg chain. 
> 
> **Deliverables:**
> 1. Derive the block-sparse tensor contraction mathematically.
> 2. Implement a Python prototype to benchmark the memory consumption of the SU(2) symmetric approach against a dense U(1) baseline.
> 3. Summarize the memory scaling results in a formal report.

### 3. Launch the Engine
With the `.claude/settings.json` enforcing auto-mode, you simply execute Claude Code with a master prompt that triggers the workflow defined in `CLAUDE.md`.

```bash
claude "Read instruction.md and begin Phase 1 of the research cycle. Read your roles from ../../personas/ as needed. Do not stop until Phase 5 yields a final_review.md."
```

## 🛡️ Failsafes and Security

* **Resource Limiting:** The Python Engineer persona is hardcoded to enforce `resource.setrlimit` (RAM capping) and C-library thread limiting (`OMP_NUM_THREADS`, etc.) in all scripts. This prevents runaway tensor contractions from crashing the shared lab server.
* **Context Anchoring:** Do not delete the files in `/personas/`. Claude relies on physically reading these via `cat` to context-switch cleanly and reset its behavior during long sessions.
* **Modularity:** The Python Engineer will automatically build its environment in `./src/`, and the LaTeX Writer will generate its files in `./report/`. 
