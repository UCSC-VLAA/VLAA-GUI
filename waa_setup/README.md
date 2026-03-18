# WindowsAgentArena (WAA) Integration for vlaa_gui

This directory contains scripts to run `vlaa_gui` as an agent inside [WindowsAgentArena](https://github.com/microsoft/WindowsAgentArena) (WAA), Microsoft's benchmark for evaluating GUI agents on Windows 11 desktop tasks.

## Prerequisites

1. **WAA repository** cloned and set up (Docker + Windows 11 golden image). Follow the [WAA setup guide](https://github.com/microsoft/WindowsAgentArena#setup).
2. **vlaa_gui** installed (`uv sync` or `pip install -e .` from the repo root).
3. A valid `config.toml` in the WAA working directory (copy from `config/template.toml` and fill in your API keys).

## Directory Structure

```
waa_setup/
  vlaa_gui/
    __init__.py
    agent.py          # ZooAgent adapter (WAA interface -> vlaa_gui Agent)
  run.py              # Main evaluation runner
  lib_run_single.py   # Single-task execution loop
  show_results.py     # Result analysis utility
  README.md           # This file
```

## Setup

### 1. Copy agent code into WAA

WAA expects agent code under its `mm_agents/` directory. Copy the `vlaa_gui/` package:

```bash
cp -r waa_setup/vlaa_gui /path/to/WindowsAgentArena/src/mm_agents/vlaa_gui
```

### 2. Copy runner scripts

Copy the runner and helper scripts into your WAA working directory (or wherever you run WAA from):

```bash
cp waa_setup/run.py /path/to/WindowsAgentArena/src/run_vlaa.py
cp waa_setup/lib_run_single.py /path/to/WindowsAgentArena/src/lib_run_single.py
```

### 3. Configuration

Place a `config.toml` in the WAA working directory. Use `config/template.toml` from the `vlaa_gui` repo as a starting point:

```bash
cp config/template.toml /path/to/WindowsAgentArena/src/config.toml
# Edit config.toml with your API keys, model choices, etc.
```

## Running

### Local execution

```bash
cd /path/to/WindowsAgentArena/src

python run_vlaa.py \
    --model gpt-4o \
    --test_all_meta_path evaluation_examples_windows/test_all.json \
    --max_steps 15 \
    --result_dir ./results
```

### Running on AWS

The runner supports cloud VM providers via `--provider_name`. For AWS, WAA's `desktop_env` manages EC2 instances automatically:

```bash
python run_vlaa.py \
    --provider_name aws \
    --region us-east-1 \
    --model gpt-4o \
    --test_all_meta_path evaluation_examples_windows/test_all.json \
    --max_steps 15 \
    --result_dir ./results
```

This uses WAA's `desktop_env.providers.aws.manager` to launch and manage EC2 instances with the correct Windows 11 AMI for your region. No manual VM provisioning is needed — the runner handles the full lifecycle.

Other supported providers: `azure`, `vmware`, `virtualbox`, `docker` (default).

### Multi-worker execution

Split tasks across workers (e.g. 4 workers on separate machines or containers):

```bash
python run_vlaa.py --worker_id 0 --num_workers 4 ...
python run_vlaa.py --worker_id 1 --num_workers 4 ...
python run_vlaa.py --worker_id 2 --num_workers 4 ...
python run_vlaa.py --worker_id 3 --num_workers 4 ...
```

### View results

```bash
python show_results.py --result_dir ./results --model gpt-4o
```

## How It Works

`ZooAgent` (in `vlaa_gui/agent.py`) wraps `vlaa_gui`'s `Agent` class and adapts its interface:

| WAA expects | `vlaa_gui` provides | Adapter logic |
|---|---|---|
| `predict() -> (response, actions, logs, computer_update_args)` | `predict() -> (info_dict, List[actions])` | Unpacks 2-tuple into 4-tuple |
| `obs["screenshot"]` as `BytesIO` | `obs["screenshot"]` as `bytes` | `.getvalue()` conversion |
| Windows 11 VM tasks | Platform-agnostic pyautogui actions | Sets `platform="windows"` on OSWorldACI |

The grounding agent (`OSWorldACI`) works for remote VM screenshots, which is exactly what WAA provides. Setting `platform="windows"` ensures Windows-specific prompts and action patterns are used.

## Cloud Provider Support (Azure-to-AWS Migration)

WAA was originally designed for Azure VMs, but `run.py` supports multiple cloud providers via the `--provider_name` flag, following the same pattern as `osworld_setup/run_multienv_zoo.py`. This removes the hard Azure dependency and allows running WAA evaluations on AWS or other providers.

### Supported providers

| Provider | `--provider_name` | Notes |
|---|---|---|
| Docker (local) | `docker` (default) | Original WAA setup; runs Windows 11 inside Docker |
| AWS EC2 | `aws` | Auto-resolves AMI via `desktop_env.providers.aws.manager.IMAGE_ID_MAP` |
| Azure | `azure` | Original WAA cloud target |
| VMware | `vmware` | Requires `--path_to_vm` pointing to the `.vmx` file |
| VirtualBox | `virtualbox` | Requires `--path_to_vm` pointing to the VM image |

### Provider-specific arguments

| Argument | Applies to | Description |
|---|---|---|
| `--provider_name` | All | Selects the virtualization backend |
| `--region` | `aws`, `azure` | Cloud region (e.g. `us-east-1`, `eastus`) |
| `--path_to_vm` | `vmware`, `virtualbox` | Path to local VM image |
| `--headless` | All | Run without GUI (default: true) |

### AWS AMI note

When using `--provider_name aws`, the runner calls `desktop_env.providers.aws.manager.IMAGE_ID_MAP` to look up the correct Windows 11 AMI for your region and screen size. If WAA's `IMAGE_ID_MAP` does not yet include Windows 11 AMIs for AWS (since WAA was originally Azure-focused), you will need to:

1. Build a custom Windows 11 AMI from the WAA golden image
2. Register it in `IMAGE_ID_MAP` under your target region and screen size tuple

The runner will log a warning if it cannot resolve a snapshot name and proceed with `snapshot_name=None`, which may cause `DesktopEnv` to fail at initialization.
