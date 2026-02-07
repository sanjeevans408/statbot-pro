**StatBot Pro - Autonomous CSV Data Analyst**
Project 3: Operations - Autonomous CSV Data Analyst Agent

**Overview**
StatBot Pro empowers business users who have messy CSV/Excel files but lack Python skills. The user uploads a file and asks a complex question (for example, "What is the sales trend per region and how does it correlate with marketing spend?"). The agent writes pandas code, executes it inside a restricted sandbox, and returns text and/or a chart.

**Product Brand**
StatBot Pro

**Core Capabilities**
- LLM-powered dataframe Q&A via LangChain's Pandas DataFrame Agent.
- Self-correction on transient failures with configurable retries and backoff.
- Security sandbox that restricts imports, file access, and dangerous `os` operations.
- Safe local plotting with matplotlib for `hist`, `bar`, `line`, `scatter`, and `box`.
- Chart output saved to `charts/` and returned with a local URL.
- Security audit command that attempts blocked actions to validate sandbox isolation.

**Tech Stack**
- LangChain Experimental Pandas Agent
- Pandas and NumPy
- Matplotlib
- NVIDIA ChatNVIDIA endpoint
- Docker for production-grade isolation (recommended; not implemented in this repo)

**How It Works**
- `sandbox.py` loads `CSV_PATH` into pandas and prints basic dataset info.
- The LLM agent is created only when `ALLOW_LLM=true`, `ALLOW_DANGEROUS_CODE=true`, and `SANDBOX_ENABLED=true`.
- The sandbox overrides `open` and selected `os` APIs, restricts imports to an allowlist, and enforces `SANDBOX_DIR` boundaries.
- `plot_data` generates charts locally without invoking the LLM.

**Quickstart**
1. Install dependencies: `pip install -r requre.txt`
2. Set required env vars: `NVIDIA_API_KEY=...` `ALLOW_LLM=true` `ALLOW_DANGEROUS_CODE=true` `SANDBOX_ENABLED=true`
3. Run: `python sandbox.py`
4. Optional: serve charts with `python -m http.server 8000 -d charts`

**CLI Commands**
- `ask <your question>`
- `plot bar <x> <y> [mean|sum|count|median] <filename.png>`
- `plot hist <x> <bins> <filename.png>`
- `plot line <x> <y> <filename.png>`
- `plot scatter <x> <y> <filename.png>`
- `plot box <x> <y> <filename.png>`
- `audit`
- `quit`

**Environment Variables**
- `NVIDIA_API_KEY` required for ChatNVIDIA and must start with `nvapi-`.
- `NVIDIA_MODEL` default `deepseek-ai/deepseek-v3.2`.
- `CSV_PATH` default `sample.csv`.
- `CHART_DIR` default `charts`.
- `SANDBOX_ENABLED` default `true`.
- `SANDBOX_DIR` default `.sandbox`.
- `ALLOW_LLM` default `true`.
- `ALLOW_DANGEROUS_CODE` default `true`.
- `NVIDIA_REQUEST_TIMEOUT` default `90`.
- `NVIDIA_MAX_COMPLETION_TOKENS` default `512`.
- `NVIDIA_RETRIES` default `2`.
- `NVIDIA_BACKOFF_BASE` default `1.5`.

**Security Audit**
Run `audit` to test that the sandbox blocks outside file access and dangerous `os` calls. The audit creates a canary file inside the sandbox and tries to delete or read files outside the sandbox. Results are printed to the console.

**Implementation Plan**
| Week | Goal | Key Tasks and Deliverables | Review and Testing Focus |
| --- | --- | --- | --- |
| 1 | The Pandas Agent | Set up LangChain Pandas DataFrame agent, load a sample CSV, test basic inspection questions. | Accuracy check against manual pandas results. |
| 2 | Graphing Capability | Prompt the agent to use matplotlib, capture chart output, save `.png`, return URL. | Verify rolling-average plot correctness. |
| 3 | Safety and Sandboxing | Restrict libraries and file system access, enforce isolated working directory. | Security audit with prompt-injection attempts. |
| 4 | UI and Persistence | Stream intermediate steps to UI, final polish, end-to-end test. | Verify transparency and stability. |

**Scope Notes**
- Local fallback answers support simple stats like mean, median, sum, min, and max when the LLM is disabled.
- For production, run execution inside Docker and apply resource limits and network isolation.
