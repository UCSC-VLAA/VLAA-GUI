# VLAA-GUI

VLAA-GUI is a Python framework for building and evaluating GUI agents that act through desktop screenshots, grounding models, and `pyautogui`-style actions. The current codebase centers on a manager-worker agent architecture, optional memory and retrieval, and benchmark integrations for OSWorld and WindowsAgentArena.

## What is in this repo

- An interactive local desktop agent exposed as the `agent` CLI.
- A planner-executor stack with explicit grounding, reflection, and token tracking.
- Optional retrieval, episodic/narrative memory, zoom grounding, and code execution.
- Integration code for OSWorld and WindowsAgentArena evaluations.
- Unit tests around grounding, hover actions, and platform-specific behavior.

## Core architecture

The main runtime lives under `vlaa_gui/agent_core/`:

- `agents/agent.py`: top-level orchestrator.
- `agents/manager.py`: high-level planner that decomposes tasks into subtasks.
- `agents/worker.py`: executor that generates grounded GUI actions and reflections.
- `agents/grounding.py`: `OSWorldACI`, the screenshot-based action interface used by the local agent and benchmark runners.
- `core/engine.py`: model/provider backends.
- `core/knowledge.py`: retrieval and memory support.
- `memory/procedural_memory.py`: system prompts and action-space instructions.

The agent is model/provider-agnostic at the framework level. The config supports providers such as OpenAI, Anthropic, Gemini, Ark/Volcengine, Azure OpenAI, Qwen, and others implemented in `engine.py`.

## Supported workflows

### 1. Local interactive agent

Runs on your current machine, captures the live desktop, and executes actions locally.

### 2. OSWorld evaluation

VLAA-GUI includes runner glue for the OSWorld benchmark. Use this path when you want to evaluate inside OSWorld's VM-backed `DesktopEnv`, not when you want to control your current desktop directly.

The OSWorld-specific setup and runner notes live in [`osworld_setup/README.md`](/Users/sergiu/research/VLAA-GUI/osworld_setup/README.md).

### 3. WindowsAgentArena evaluation

Scripts in [`waa_setup/README.md`](/Users/sergiu/research/VLAA-GUI/waa_setup/README.md) adapt the agent to Microsoft's WindowsAgentArena benchmark.

## Requirements

- Python `>=3.12`
- `uv` recommended for environment management
- A configured model provider and API credentials
- For local runs:
  - macOS: grant Accessibility and Screen Recording permissions to the terminal/app you use
  - Linux/Windows: run in an environment where `pyautogui` can control the desktop

## Installation

```bash
git clone <your-fork-or-upstream-url>
cd VLAA-GUI
uv sync
```

If you prefer editable install without syncing the lockfile:

```bash
uv pip install -e .
```

## Configuration

Start from the template:

```bash
cp config/template.toml config/config.toml
```

Then fill in the sections you actually need:

- `[model]`: main planning/execution model
- `[grounding]`: grounding model and screen-resize settings
- `[coding]`: model used by the optional coding agent
- `[embedding]`: embedding backend for memory retrieval
- `[perception]`: observation type such as `screenshot` or `a11y_tree`
- `[planning]`: reflection and planner depth settings
- `[context_management]`: retrieval backend and memory mode
- `[action_space]`: `pyautogui` or `pyautogui_coding`

Important runtime notes:

- The local CLI defaults to `config/config.toml`.
- On first run, the agent downloads seed knowledge-base files into `agent_memory/` from the upstream Agent-S release assets.
- Logs and token-usage summaries are written under `logs/`.

## Running the local agent

```bash
agent --config-path config/config.toml # You need to source the uv environment first
```

Or:

```bash
uv run vlaa_gui.agent_core.run_agent --config-path config/config.toml
```

This starts an interactive loop that prompts for:

```text
Query:
```

After each task, the run writes logs plus token usage summaries to `logs/`.

## OSWorld quick start

OSWorld is a separate benchmark environment. The flow is:

1. Set up OSWorld itself, including its VM images and `DesktopEnv`.
2. Install this package into the same Python environment OSWorld uses.
3. Copy or mount the contents of [`osworld_setup/`](/Users/sergiu/research/VLAA-GUI/osworld_setup/) into your OSWorld workspace.
4. Run the OSWorld adapter scripts from inside that environment.

Minimal install step inside the OSWorld environment:

```bash
cd /path/to/VLAA-GUI
uv pip install .
```

Typical entry points:

```bash
uv run python osworld_setup/run_locally.py
uv run python osworld_setup/run_multienv_vlaa.py
```

Use the benchmark-specific README before running either script:

- [`osworld_setup/README.md`](/Users/sergiu/research/VLAA-GUI/osworld_setup/README.md)

Practical notes:

- OSWorld runs inside benchmark VMs; the local `agent` CLI does not.
- For OSWorld, treat `osworld_setup/run_locally.py` as the main single-VM entry point.
- Keep your config aligned with the environment: use screenshot observations unless you have explicitly verified another perception mode is supported by that runner.
- Result artifacts, logs, and benchmark outputs are written by the OSWorld runners, not by the local interactive loop.

## Full Bedrock local run

A ready-to-edit Bedrock profile is included at [`config/full-bedrock.toml`](/Users/sergiu/research/VLAA-GUI/config/full-bedrock.toml). It configures the local interactive agent to use Bedrock for planning, grounding, coding, and search.

Launcher script:

```bash
./scripts/run-agent-full-bedrock.sh
```

The script:

- uses [`config/full-bedrock.toml`](/Users/sergiu/research/VLAA-GUI/config/full-bedrock.toml) by default
- checks that AWS credentials are available through `AWS_PROFILE` or standard AWS env vars
- forwards any extra CLI args to the `agent` command

Examples:

```bash
AWS_PROFILE=my-bedrock-profile ./scripts/run-agent-full-bedrock.sh
AWS_PROFILE=my-bedrock-profile ./scripts/run-agent-full-bedrock.sh --config-path config/full-bedrock.toml
```

## Useful config switches

- `grounding.enable_zoom_grounding = true`
  - Run a coarse grounding pass and then refine on a zoomed crop.
- `context_management.memory_type = "episodic"` or `"mixed"`
  - Turn on memory retrieval and memory updates.
- `context_management.search_engine = "llm"` or `"perplexica"`
  - Use retrieval-backed knowledge instead of pure local reasoning.
- `context_management.search_engine = "search_agent"`
  - Use the dedicated search agent path instead of the legacy KB retrieval flow.
- `action_space.engine = "pyautogui_coding"`
  - Exposes a coding agent that can execute local code.
  - This is powerful and unsafe if you do not trust the task or model.

## Evaluation entry points

Top-level evaluation scripts are included, but each benchmark has its own environment and setup steps:

- OSWorld local runner: [`osworld_setup/run_locally.py`](/Users/sergiu/research/VLAA-GUI/osworld_setup/run_locally.py)
- OSWorld multi-environment runner: [`osworld_setup/run_multienv_vlaa.py`](/Users/sergiu/research/VLAA-GUI/osworld_setup/run_multienv_vlaa.py)
- WAA sequential runner: [`waa_setup/run.py`](/Users/sergiu/research/VLAA-GUI/waa_setup/run.py)
- WAA multi-VM runner: [`waa_setup/run_multienv_vlaa.py`](/Users/sergiu/research/VLAA-GUI/waa_setup/run_multienv_vlaa.py)

Use the benchmark-specific READMEs before running those scripts:

- [`osworld_setup/README.md`](/Users/sergiu/research/VLAA-GUI/osworld_setup/README.md)
- [`waa_setup/README.md`](/Users/sergiu/research/VLAA-GUI/waa_setup/README.md)

## Acknowledgements
This project builds on top of the open-source [Agent-S](https://github.com/simular-ai/Agent-S) codebase and is inspired by the growing ecosystem of GUI agents and benchmarks. Thanks to the communities around [Agent-S](https://github.com/simular-ai/Agent-S), [OSWorld](https://github.com/xlang-ai/OSWorld), [WindowsAgentArena](https://github.com/microsoft/WindowsAgentArena), and others for their contributions to this exciting space!
