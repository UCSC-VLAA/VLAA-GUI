"""Script to run end-to-end evaluation on the benchmark.
Utils and basic architecture credit to https://github.com/web-arena-x/webarena/blob/main/run.py.
"""

"""
an example runs.json file and how to use it:                                                                                                      
                                                                                                                                                         
  runs.json:                                                                                                                                               
  [
    {"result_dir": "traj/baseline", "config_path": "config/baseline-bedrock.toml"},                                                                        
    {"result_dir": "traj/ablation-no-search", "config_path": "config/wo-done-gemini.toml"}                                                               
  ]

  Multi-run (shared VM pool):
  uv run run_multienv_zoo.py \
      --runs-file runs.json \
      --provider_name aws \
      --region us-east-1 \
      --num_envs 30 \
      --max_steps 100

  Both configs share the same 30 VMs. Tasks are interleaved round-robin so both runs progress evenly. Results go to traj/baseline/... and
  traj/ablation-no-search/... respectively.

  Single-run (backwards compatible, unchanged):
  uv run run_multienv_zoo.py \
      --result_dir traj/baseline \
      --config-path config/baseline-bedrock.toml \
      --provider_name aws \
      --region us-east-1 \
      --num_envs 30 \
      --max_steps 100
"""

import argparse
import copy
import time
import datetime
import json
import logging
import os
import signal
import sys
import tomllib
from dataclasses import dataclass
from multiprocessing import Process, Manager, current_process, Queue
from desktop_env.desktop_env import DesktopEnv
import lib_run_single
from typing import Any, Dict, List


#  Logger Configs {{{ #
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

datetime_str: str = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")

file_handler = logging.FileHandler(
    os.path.join("logs", "normal-{:}.log".format(datetime_str)), encoding="utf-8"
)
stdout_handler = logging.StreamHandler(sys.stdout)

stdout_handler.setLevel(logging.INFO)
file_handler.setLevel(logging.INFO)


formatter = logging.Formatter(
    fmt="\x1b[1;33m[%(asctime)s \x1b[31m%(levelname)s \x1b[32m%(module)s/%(lineno)d-%(processName)s\x1b[1;33m] \x1b[0m%(message)s"
)

stdout_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

stdout_handler.addFilter(logging.Filter("desktopenv"))
# file_handler.addFilter(logging.Filter("vlaa_gui"))

logger.addHandler(stdout_handler)
logger.addHandler(file_handler)
#  }}} Logger Configs #

logger = logging.getLogger("desktopenv.experiment")

# Global variables for signal handling
active_environments = []
processes = []
is_terminating = False


def distribute_tasks(test_all_meta: dict) -> list:
    all_tasks = []
    for domain, examples in test_all_meta.items():
        for example_id in examples:
            all_tasks.append((domain, example_id))
    return all_tasks


def process_signal_handler(signum, frame, env_idx):
    logger.info(f"Process {env_idx + 1} received signal {signum}. Shutting down...")
    local_vars = frame.f_locals
    active_environments = local_vars.get("active_environments", [])
    for env in active_environments:
        if env is not None:
            try:
                logger.info(f"Process {env_idx + 1} closing environment...")
                env.close()
                logger.info(f"Process {env_idx + 1} environment closed successfully")
            except Exception as e:
                logger.error(f"Process {env_idx + 1} error closing environment: {e}")
    logger.info(f"Process {env_idx + 1} shutdown complete. Exiting.")
    sys.exit(0)


def signal_handler(signum, frame):
    global is_terminating, active_environments, processes
    if is_terminating:
        return
    is_terminating = True
    logger.info(f"Received signal {signum}. Gracefully shutting down...")
    for env in active_environments:
        try:
            logger.info("Closing environment...")
            env.close()
            logger.info("Environment closed successfully")
        except Exception as e:
            logger.error(f"Error closing environment: {e}")
    for p in processes:
        if p.is_alive():
            try:
                logger.info(f"Sending termination signal to process {p.name}...")
                p.terminate()
            except Exception as e:
                logger.error(f"Error sending termination signal to process: {e}")
    time.sleep(1)
    for p in processes:
        if p.is_alive():
            try:
                logger.info(f"Forcefully terminating process {p.name}...")
                import signal as sig

                os.kill(p.pid, sig.SIGKILL)
            except Exception as e:
                logger.error(f"Error forcefully terminating process: {e}")
    logger.info("Shutdown complete. Exiting.")
    sys.exit(0)


def _strip_model_prefix(model_name: str) -> str:
    """Strip regional and brand prefixes from Bedrock model names.

    E.g. 'global.anthropic.claude-sonnet-4-6' -> 'claude-sonnet-4-6'
    """
    return model_name.rsplit(".", 1)[-1] if "." in model_name else model_name


def _get_toml_value(config: Dict, path: List[str], default: Any = None) -> Any:
    current = config
    try:
        for key in path:
            current = current[key]
        return current
    except (KeyError, TypeError):
        return default


def _format_instruction(instruction: str, max_len: int = 120) -> str:
    cleaned = " ".join((instruction or "").split())
    if len(cleaned) > max_len:
        return cleaned[: max_len - 3] + "..."
    return cleaned


def _token_summaries_json_path(output_dir: str) -> str:
    """Return the path to the persistent token-summaries JSON file."""
    return os.path.join(output_dir, "token_usage_summaries.json")


def load_previous_token_summaries(output_dir: str) -> List[Dict]:
    """Load token summaries saved by a previous (paused) run, if any."""
    json_path = _token_summaries_json_path(output_dir)
    if not os.path.exists(json_path):
        return []
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            logger.info(
                "Loaded %d token summaries from previous run: %s",
                len(data),
                json_path,
            )
            return data
    except Exception as e:
        logger.warning(
            "Failed to load previous token summaries from %s: %s", json_path, e
        )
    return []


def _save_token_summaries_json(token_summaries: List[Dict], output_dir: str) -> str:
    """Persist token summaries as JSON so they survive pause-rerun."""
    os.makedirs(output_dir, exist_ok=True)
    json_path = _token_summaries_json_path(output_dir)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(token_summaries, f, ensure_ascii=False, indent=2)
    return json_path


def write_token_usage_markdown(token_summaries: List[Dict], output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "token_usage_summary.md")

    # Also persist the raw data so it survives pause-rerun
    _save_token_summaries_json(token_summaries, output_dir)

    # Group summaries by domain
    domain_summaries: Dict[str, List[Dict]] = {}
    for summary in token_summaries:
        domain = summary.get("domain", "unknown")
        domain_summaries.setdefault(domain, []).append(summary)

    # Compute overall totals
    total_tasks = len(token_summaries)
    total_output_tokens = sum(s.get("total_output_tokens", 0) for s in token_summaries)
    total_all_tokens = sum(s.get("total_tokens", 0) for s in token_summaries)

    # Compute per-domain totals
    domain_stats: Dict[str, Dict[str, int]] = {}
    for domain, summaries in sorted(domain_summaries.items()):
        domain_stats[domain] = {
            "tasks": len(summaries),
            "output_tokens": sum(s.get("total_output_tokens", 0) for s in summaries),
            "total_tokens": sum(s.get("total_tokens", 0) for s in summaries),
        }

    # Compute agent-level totals across all tasks
    agent_totals: Dict[str, Dict[str, int]] = {}
    for summary in token_summaries:
        for agent_type, stats in summary.get("agent_breakdown", {}).items():
            agent_totals.setdefault(
                agent_type,
                {"output_tokens": 0, "total_tokens": 0, "calls": 0},
            )
            agent_totals[agent_type]["output_tokens"] += stats.get("output_tokens", 0)
            agent_totals[agent_type]["total_tokens"] += stats.get("total_tokens", 0)
            agent_totals[agent_type]["calls"] += stats.get("calls", 0)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Token Usage Summary\n\n")
        f.write(f"Generated: {datetime.datetime.now().isoformat()}\n\n")

        # Overall statistics
        f.write("## Overall\n\n")
        f.write("| Metric | Value |\n")
        f.write("| --- | ---: |\n")
        f.write(f"| Total Tasks | {total_tasks} |\n")
        f.write(f"| Total Output Tokens | {total_output_tokens:,} |\n")
        f.write(f"| Total Tokens | {total_all_tokens:,} |\n")
        total_llm_calls = sum(
            stats.get("calls", 0)
            for s in token_summaries
            for stats in s.get("agent_breakdown", {}).values()
        )
        avg_output = total_output_tokens / total_tasks if total_tasks else 0
        avg_total = total_all_tokens / total_tasks if total_tasks else 0
        avg_llm_calls = total_llm_calls / total_tasks if total_tasks else 0
        f.write(f"| Avg Output Tokens / Task | {avg_output:,.0f} |\n")
        f.write(f"| Avg Total Tokens / Task | {avg_total:,.0f} |\n")
        f.write(f"| Total LLM Calls | {total_llm_calls:,} |\n")
        f.write(f"| Avg LLM Calls / Task | {avg_llm_calls:,.1f} |\n")

        # Per-domain breakdown
        f.write("\n## Per-domain token usage\n\n")
        f.write(
            "| Domain | Tasks | Output Tokens | Total Tokens | Avg Output / Task |\n"
        )
        f.write("| --- | ---: | ---: | ---: | ---: |\n")
        for domain in sorted(domain_stats.keys()):
            ds = domain_stats[domain]
            avg = ds["output_tokens"] / ds["tasks"] if ds["tasks"] else 0
            f.write(
                f"| {domain} | {ds['tasks']} | {ds['output_tokens']:,} "
                f"| {ds['total_tokens']:,} | {avg:,.0f} |\n"
            )

        # Agent breakdown (all tasks)
        f.write("\n## Agent token usage (all tasks)\n\n")
        f.write("| Agent | Output Tokens | Total Tokens | Calls |\n")
        f.write("| --- | ---: | ---: | ---: |\n")
        for agent_type in sorted(agent_totals.keys()):
            stats = agent_totals[agent_type]
            f.write(
                f"| {agent_type} | {stats['output_tokens']:,} | {stats['total_tokens']:,} | {stats['calls']} |\n"
            )

        # Per-task details grouped by domain
        f.write("\n## Per-task details\n\n")
        for domain in sorted(domain_summaries.keys()):
            f.write(f"### {domain}\n\n")
            f.write("| Task ID | Output Tokens | Total Tokens | Instruction |\n")
            f.write("| --- | ---: | ---: | --- |\n")
            for summary in sorted(
                domain_summaries[domain], key=lambda x: x.get("task_id", "")
            ):
                task_id = summary.get("task_id", "")
                output_tokens = summary.get("total_output_tokens", 0)
                all_tokens = summary.get("total_tokens", 0)
                instruction = _format_instruction(summary.get("task_instruction", ""))
                f.write(
                    f"| {task_id} | {output_tokens:,} | {all_tokens:,} | {instruction} |\n"
                )
            f.write("\n")

    return output_path


def write_env_status_file(
    shared_env_status, output_dir: str, completed: int, total: int
):
    """Write a live markdown status file showing all active environments."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "live_status.md")
    try:
        # Snapshot the shared dict to avoid mutation during iteration
        status_snapshot = dict(shared_env_status)
    except Exception:
        return

    now_str = datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    )
    lines = [
        f"# Live Environment Status\n",
        f"Updated: {now_str} | Completed: {completed}/{total}\n\n",
        "| Process | Status | VM Instance | VNC URL | Domain | Task ID | Instruction |\n",
        "|---------|--------|-------------|---------|--------|---------|-------------|\n",
    ]

    for proc_name in sorted(status_snapshot.keys()):
        entry = status_snapshot[proc_name]
        status = entry.get("status", "UNKNOWN")
        vm_id = entry.get("vm_id", "") or ""
        public_ip = entry.get("public_ip", "")
        if public_ip:
            vnc_url = f"http://{public_ip}:5910/vnc.html"
        else:
            vm_ip = entry.get("vm_ip", "")
            vnc_url = f"http://{vm_ip}:5910/vnc.html" if vm_ip else ""
        domain = entry.get("domain", "") or "\u2014"
        task_id = entry.get("task_id", "") or "\u2014"
        instruction = (
            _format_instruction(entry.get("instruction", ""), max_len=80) or "\u2014"
        )

        # Truncate VM ID for readability
        vm_id_short = vm_id if len(vm_id) <= 22 else vm_id[:22] + "..."

        lines.append(
            f"| {proc_name} | {status} | {vm_id_short} | {vnc_url} | {domain} | {task_id} | {instruction} |\n"
        )

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.writelines(lines)
    except Exception:
        pass


@dataclass
class RunConfig:
    """Per-run configuration bundle for multi-run support."""

    run_idx: int
    result_dir: str
    config_path: str
    args: argparse.Namespace
    engine_params: dict
    engine_params_for_grounding: dict
    engine_params_for_coding: dict
    engine_params_for_searcher: dict


def _apply_toml_config(args: argparse.Namespace, config_path: str) -> None:
    """Load a TOML config file and apply its values to *args* in-place."""
    with open(config_path, "rb") as f:
        toml_cfg = tomllib.load(f)

    # Main Model Config
    args.model_provider = _get_toml_value(toml_cfg, ["model", "provider"], "openai")
    args.model = _get_toml_value(toml_cfg, ["model", "name"], "")
    args.model_dir_name = _strip_model_prefix(args.model)
    args.model_url = _get_toml_value(toml_cfg, ["model", "url"], "")
    args.model_api_key = _get_toml_value(toml_cfg, ["model", "api_key"], "")
    args.temperature = _get_toml_value(toml_cfg, ["model", "temperature"], None)
    args.top_p = _get_toml_value(toml_cfg, ["model", "top_p"], None)
    args.max_tokens = _get_toml_value(toml_cfg, ["model", "max_tokens"], 6400)
    args.model_project_id = _get_toml_value(toml_cfg, ["model", "project_id"], "")
    args.model_region = _get_toml_value(toml_cfg, ["model", "region"], "")
    args.model_aws_keys = _get_toml_value(toml_cfg, ["model", "aws_keys"], [])
    args.api_version = _get_toml_value(toml_cfg, ["model", "api_version"], "")
    args.model_thinking = _get_toml_value(
        toml_cfg, ["model", "thinking"], args.model_provider == "anthropic_bedrock"
    )

    # Grounding Model Config
    args.grounding_model_provider = _get_toml_value(
        toml_cfg, ["grounding", "provider"], "openai"
    )
    args.grounding_model = _get_toml_value(
        toml_cfg, ["grounding", "grounding_model"], ""
    )
    args.grounding_model_url = _get_toml_value(toml_cfg, ["grounding", "url"], "")
    args.grounding_model_api_key = _get_toml_value(
        toml_cfg, ["grounding", "api_key"], ""
    )
    args.grounding_width = _get_toml_value(
        toml_cfg, ["grounding", "grounding_width"], None
    )
    args.grounding_height = _get_toml_value(
        toml_cfg, ["grounding", "grounding_height"], None
    )
    args.resize_width = _get_toml_value(toml_cfg, ["grounding", "resize_width"], None)
    args.grounding_model_region = _get_toml_value(
        toml_cfg, ["grounding", "region"], args.model_region
    )
    args.enable_zoom_grounding = _get_toml_value(
        toml_cfg, ["grounding", "enable_zoom_grounding"], False
    )
    args.zoom_grounding_crop_ratio = _get_toml_value(
        toml_cfg, ["grounding", "zoom_grounding_crop_ratio"], 0.5
    )
    args.grounding_model_type = _get_toml_value(
        toml_cfg, ["grounding", "type"], "single"
    )
    args.grounding_temperature = _get_toml_value(
        toml_cfg, ["grounding", "temperature"], None
    )
    args.grounding_top_p = _get_toml_value(toml_cfg, ["grounding", "top_p"], None)

    # Self-hosted Endpoint Config
    args.endpoint_provider = _get_toml_value(
        toml_cfg, ["grounding_endpoint", "provider"], ""
    )
    args.endpoint_url = _get_toml_value(toml_cfg, ["grounding_endpoint", "url"], "")
    args.endpoint_api_key = _get_toml_value(
        toml_cfg, ["grounding_endpoint", "api_key"], ""
    )

    # Coding model config
    args.coding_model_provider = _get_toml_value(
        toml_cfg, ["coding", "provider"], "openai"
    )
    args.coding_model = _get_toml_value(toml_cfg, ["coding", "name"], "")
    args.coding_model_url = _get_toml_value(toml_cfg, ["coding", "url"], "")
    args.coding_model_api_key = _get_toml_value(toml_cfg, ["coding", "api_key"], "")
    args.coding_model_api_keys = _get_toml_value(toml_cfg, ["coding", "api_keys"], [])
    args.coding_model_temperature = _get_toml_value(
        toml_cfg, ["coding", "temperature"], None
    )
    args.coding_model_top_p = _get_toml_value(toml_cfg, ["coding", "top_p"], None)
    args.coding_model_thinking = _get_toml_value(
        toml_cfg, ["coding", "thinking"], False
    )
    args.coding_model_thinking_budget = _get_toml_value(
        toml_cfg, ["coding", "thinking_budget"], None
    )
    args.coding_model_thinking_level = _get_toml_value(
        toml_cfg, ["coding", "thinking_level"], None
    )
    args.coding_model_include_thoughts = _get_toml_value(
        toml_cfg, ["coding", "include_thoughts"], False
    )

    # Embedding Config
    args.embedding_engine_type = _get_toml_value(
        toml_cfg, ["embedding", "engine_type"], "openai"
    )

    # Perception
    args.observation_type = _get_toml_value(
        toml_cfg, ["perception", "observation_type"], "screenshot"
    )

    # Planning
    args.planner_hierarchical_depth = _get_toml_value(
        toml_cfg, ["planning", "hierarchical_depth"], 1
    )
    args.with_reflection = _get_toml_value(
        toml_cfg, ["planning", "with_reflection"], False
    )
    args.use_recon = _get_toml_value(toml_cfg, ["planning", "use_recon"], False)

    # Context Management
    args.search_engine = _get_toml_value(
        toml_cfg, ["context_management", "search_engine"], None
    )
    args.kb_name = _get_toml_value(
        toml_cfg, ["context_management", "kb_name"], "agent_memory"
    )
    args.memory_type = _get_toml_value(
        toml_cfg, ["context_management", "memory_type"], "mixed"
    )
    args.memory_representation = _get_toml_value(
        toml_cfg, ["context_management", "memory_representation"], "vector"
    )
    args.knowledge_storage = _get_toml_value(
        toml_cfg, ["context_management", "knowledge_storage"], "db"
    )

    # Searcher agent config
    args.searcher_type = _get_toml_value(toml_cfg, ["searcher", "type"], "llm")
    args.searcher_provider = _get_toml_value(
        toml_cfg, ["searcher", "provider"], "gemini"
    )
    args.searcher_model = _get_toml_value(
        toml_cfg, ["searcher", "model"], "gemini-3-flash-preview"
    )
    args.searcher_api_key = _get_toml_value(toml_cfg, ["searcher", "api_key"], "")
    args.searcher_temperature = _get_toml_value(
        toml_cfg, ["searcher", "temperature"], None
    )
    args.searcher_top_p = _get_toml_value(toml_cfg, ["searcher", "top_p"], None)
    args.searcher_url = _get_toml_value(toml_cfg, ["searcher", "url"], "")
    args.searcher_budget = _get_toml_value(toml_cfg, ["searcher", "budget"], 20)

    # Gating Strategies
    args.enable_gate = _get_toml_value(toml_cfg, ["gate", "enable_gate"], False)
    args.loop_detection = _get_toml_value(toml_cfg, ["gate", "loop_detection"], False)
    args.feasibility_check = _get_toml_value(
        toml_cfg, ["gate", "feasibility_check"], False
    )

    # Action Space
    args.action_space = _get_toml_value(
        toml_cfg, ["action_space", "engine"], "pyautogui"
    )

    # TTS
    args.action_tts_num = _get_toml_value(toml_cfg, ["tts", "action_tts_num"], 1)
    args.use_verifier = _get_toml_value(toml_cfg, ["tts", "use_verifier"], False)


def _build_engine_params(args: argparse.Namespace) -> tuple:
    """Build the four engine-parameter dicts from a fully-resolved *args*."""
    if args.search_engine == "None" or args.search_engine == "":
        args.search_engine = None

    engine_params = {
        "engine_type": args.model_provider,
        "model": args.model,
        "base_url": getattr(args, "model_url", ""),
        "api_key": getattr(args, "model_api_key", ""),
        "api_version": getattr(args, "api_version", ""),
        "project_id": getattr(args, "model_project_id", ""),
        "region": getattr(args, "model_region", ""),
        "aws_keys": getattr(args, "model_aws_keys", []),
        "thinking": getattr(args, "model_thinking", False),
        "temperature": getattr(args, "temperature", None),
        "top_p": getattr(args, "top_p", None),
    }

    if args.endpoint_url:
        engine_params_for_grounding = {
            "engine_type": args.endpoint_provider,
            "base_url": args.endpoint_url,
            "api_key": args.endpoint_api_key,
            "temperature": getattr(args, "grounding_temperature", None),
            "top_p": getattr(args, "grounding_top_p", None),
        }
    else:
        grounding_height = args.grounding_height
        grounding_width = args.grounding_width
        if grounding_width is None:
            grounding_width = args.screen_width
        if grounding_height is None:
            grounding_height = args.screen_height * grounding_width / args.screen_width

        engine_params_for_grounding = {
            "engine_type": args.grounding_model_provider,
            "model": args.grounding_model,
            "grounding_width": grounding_width,
            "grounding_height": grounding_height,
            "temperature": getattr(args, "grounding_temperature", None),
            "top_p": getattr(args, "grounding_top_p", None),
        }

        if args.grounding_model_provider == "anthropic_bedrock":
            engine_params_for_grounding["region"] = args.grounding_model_region
            engine_params_for_grounding["thinking"] = False
            if getattr(args, "model_aws_keys", None):
                engine_params_for_grounding["aws_keys"] = args.model_aws_keys
        else:
            engine_params_for_grounding["base_url"] = args.grounding_model_url
            engine_params_for_grounding["api_key"] = args.grounding_model_api_key

    engine_params_for_coding = {
        "engine_type": args.coding_model_provider,
        "base_url": args.coding_model_url,
        "api_key": args.coding_model_api_key,
        "api_keys": getattr(args, "coding_model_api_keys", []),
        "model": args.coding_model,
        "project_id": getattr(args, "model_project_id", ""),
        "region": getattr(args, "model_region", ""),
        "aws_keys": getattr(args, "model_aws_keys", []),
        "thinking": getattr(args, "coding_model_thinking", False),
        "thinking_budget": getattr(args, "coding_model_thinking_budget", None),
        "thinking_level": getattr(args, "coding_model_thinking_level", None),
        "temperature": getattr(args, "coding_model_temperature", None),
        "top_p": getattr(args, "coding_model_top_p", None),
        "include_thoughts": getattr(args, "coding_model_include_thoughts", False),
    }

    engine_params_for_searcher = {
        "engine_type": args.searcher_provider,
        "type": args.searcher_type,
        "model": args.searcher_model,
        "api_key": args.searcher_api_key,
        "base_url": getattr(args, "searcher_url", ""),
        "budget": args.searcher_budget,
        "temperature": getattr(args, "searcher_temperature", None),
        "top_p": getattr(args, "searcher_top_p", None),
        "project_id": getattr(args, "model_project_id", ""),
        "region": getattr(args, "model_region", ""),
        "aws_keys": getattr(args, "model_aws_keys", []),
    }

    return (
        engine_params,
        engine_params_for_grounding,
        engine_params_for_coding,
        engine_params_for_searcher,
    )


def build_run_configs(shared_args: argparse.Namespace) -> list:
    """Build a list of RunConfig from shared CLI args.

    When ``--runs-file`` is provided, each entry in the JSON array becomes a
    separate RunConfig with its own TOML config.  Otherwise, a single RunConfig
    is created from the existing ``--result_dir`` / ``--config-path`` pair.
    """
    if shared_args.runs_file:
        with open(shared_args.runs_file, "r", encoding="utf-8") as f:
            runs = json.load(f)
        if not isinstance(runs, list) or not runs:
            raise ValueError("--runs-file must contain a non-empty JSON array")
        run_configs: list[RunConfig] = []
        for i, entry in enumerate(runs):
            if "result_dir" not in entry or "config_path" not in entry:
                raise ValueError(
                    f"Each entry in --runs-file must have 'result_dir' and "
                    f"'config_path' keys (entry {i})"
                )
            run_args = copy.deepcopy(shared_args)
            run_args.result_dir = entry["result_dir"]
            run_args.config_path = entry["config_path"]
            if not os.path.exists(run_args.config_path):
                raise FileNotFoundError(
                    f"Config file not found for run {i}: {run_args.config_path}"
                )
            _apply_toml_config(run_args, run_args.config_path)
            ep, epg, epc, eps = _build_engine_params(run_args)
            run_configs.append(
                RunConfig(
                    run_idx=i,
                    result_dir=entry["result_dir"],
                    config_path=entry["config_path"],
                    args=run_args,
                    engine_params=ep,
                    engine_params_for_grounding=epg,
                    engine_params_for_coding=epc,
                    engine_params_for_searcher=eps,
                )
            )
        return run_configs
    else:
        ep, epg, epc, eps = _build_engine_params(shared_args)
        return [
            RunConfig(
                run_idx=0,
                result_dir=shared_args.result_dir,
                config_path=shared_args.config_path,
                args=shared_args,
                engine_params=ep,
                engine_params_for_grounding=epg,
                engine_params_for_coding=epc,
                engine_params_for_searcher=eps,
            )
        ]


def run_env_tasks(
    task_queue: Queue,
    run_configs: list,
    shared_args: argparse.Namespace,
    per_run_scores: dict,
    per_run_token_summaries: dict,
    shared_env_status=None,
):
    active_environments = []
    env = None
    proc_name = current_process().name
    # Index run_configs by run_idx for fast lookup
    rc_by_idx = {rc.run_idx: rc for rc in run_configs}
    try:
        # Use IMAGE_ID_MAP for AWS provider to get snapshot_name
        snapshot_name = None
        region = getattr(shared_args, "region", None)
        if shared_args.provider_name == "aws" and region is not None:
            try:
                from desktop_env.providers.aws.manager import IMAGE_ID_MAP

                screen_size = (shared_args.screen_width, shared_args.screen_height)
                snapshot_name = IMAGE_ID_MAP[region].get(
                    screen_size, IMAGE_ID_MAP[region][(1920, 1080)]
                )
            except Exception as e:
                logger.error(f"Failed to get snapshot_name from IMAGE_ID_MAP: {e}")
                snapshot_name = None
        from vlaa_gui.agent_core.agents.agent import Agent
        from vlaa_gui.agent_core.agents.grounding import OSWorldACI
        from vlaa_gui.agent_core.agents.verifier_agent import VerifierAgent
        from vlaa_gui.agent_core.agents.recon_agent import ReconAgent

        # The first run config determines require_a11y_tree for the VM.
        # All configs that need a11y_tree will work because the VM enables
        # it at creation time; configs that don't need it simply ignore it.
        any_needs_a11y = any(
            rc.args.observation_type in ["a11y_tree", "screenshot_a11y_tree", "som"]
            for rc in run_configs
        )

        env = DesktopEnv(
            path_to_vm=shared_args.path_to_vm,
            action_space="pyautogui",
            provider_name=shared_args.provider_name,
            region=region,
            snapshot_name=snapshot_name,
            screen_size=(shared_args.screen_width, shared_args.screen_height),
            headless=shared_args.headless,
            os_type="Ubuntu",
            require_a11y_tree=any_needs_a11y,
            enable_proxy=True,
            client_password=getattr(shared_args, "client_password", ""),
        )

        # Lazy agent cache: run_idx -> (agent, verifier_agent, recon_agent)
        agent_cache: Dict[int, tuple] = {}

        def _get_or_create_agents(run_idx):
            if run_idx in agent_cache:
                return agent_cache[run_idx]
            rc = rc_by_idx[run_idx]
            a = rc.args

            coding_agent_flag = a.action_space == "pyautogui_coding"

            grounding_agent = OSWorldACI(
                env=env,
                platform="linux",
                engine_params_for_generation=rc.engine_params,
                engine_params_for_grounding=rc.engine_params_for_grounding,
                engine_params_for_searcher=rc.engine_params_for_searcher,
                width=shared_args.screen_width,
                height=shared_args.screen_height,
                resize_width=a.resize_width,
                grounding_model_type=a.grounding_model_type,
                code_agent_engine_params=rc.engine_params_for_coding,
                code_agent_budget=20,
            )
            agent = Agent(
                rc.engine_params,
                grounding_agent,
                platform="linux",
                action_space="pyautogui",
                observation_type=a.observation_type,
                with_reflection=a.with_reflection,
                search_engine=a.search_engine,
                memory_root_path="./agent_memory",
                memory_folder_name=a.kb_name,
                embedding_engine_type=a.embedding_engine_type,
                memory_type=a.memory_type,
                coding_agent_flag=coding_agent_flag,
                action_tts_num=a.action_tts_num,
                debug=a.debug,
                enable_gate=a.enable_gate,
                loop_detection=a.loop_detection,
                feasibility_check=a.feasibility_check,
            )

            verifier = None
            if a.use_verifier:
                verifier = VerifierAgent(rc.engine_params, platform="linux")

            recon = None
            if a.use_recon:
                recon = ReconAgent(rc.engine_params, platform="linux")

            agent_cache[run_idx] = (agent, verifier, recon)
            logger.info(
                f"[{proc_name}] Created agents for run {run_idx} ({rc.config_path})"
            )
            return agent_cache[run_idx]

        active_environments.append(env)
        proc_name = current_process().name
        if shared_env_status is not None:
            shared_env_status[proc_name] = {
                "vm_id": env.path_to_vm or "",
                "vm_ip": getattr(env, "vm_ip", ""),
                "public_ip": getattr(env, "public_ip", ""),
                "domain": "",
                "task_id": "",
                "instruction": "",
                "status": "IDLE",
            }
        logger.info(f"Process {proc_name} started.")
        while True:
            try:
                item = task_queue.get(timeout=5)
            except Exception:
                break
            run_idx, domain, example_id = item
            rc = rc_by_idx[run_idx]
            run_args = rc.args
            agent, verifier_agent, recon_agent = _get_or_create_agents(run_idx)
            try:
                config_file = os.path.join(
                    shared_args.test_config_base_dir,
                    f"examples/{domain}/{example_id}.json",
                )
                with open(config_file, "r", encoding="utf-8") as f:
                    example = json.load(f)
                instruction = example["instruction"]
                if shared_env_status is not None:
                    shared_env_status[proc_name] = {
                        **shared_env_status.get(proc_name, {}),
                        "domain": domain,
                        "task_id": example_id,
                        "instruction": instruction,
                        "status": "RUNNING",
                    }
                example_result_dir = os.path.join(
                    rc.result_dir,
                    run_args.action_space,
                    run_args.observation_type,
                    run_args.model_dir_name,
                    domain,
                    example_id,
                )
                os.makedirs(example_result_dir, exist_ok=True)
                logger.info(f"[{proc_name}][Run {run_idx}][Domain]: {domain}")
                logger.info(f"[{proc_name}][Run {run_idx}][Example ID]: {example_id}")
                logger.info(f"[{proc_name}][Run {run_idx}][Instruction]: {instruction}")
                try:
                    lib_run_single.run_single_example(
                        agent,
                        env,
                        example,
                        run_args.max_steps,
                        instruction,
                        run_args,
                        example_result_dir,
                        per_run_scores[run_idx],
                        verifier_agent=verifier_agent,
                        recon_agent=recon_agent,
                        shared_token_summaries=per_run_token_summaries[run_idx],
                        domain=domain,
                        shared_env_status=shared_env_status,
                    )
                except Exception as e:
                    import traceback

                    logger.error(
                        f"Exception in {proc_name} run {run_idx} {domain}/{example_id}: {e}"
                    )
                    logger.error(traceback.format_exc())
                    try:
                        env.controller.end_recording(
                            os.path.join(example_result_dir, "recording.mp4")
                        )
                    except Exception as rec_e:
                        logger.error(f"Failed to end recording: {rec_e}")
                    with open(os.path.join(example_result_dir, "traj.jsonl"), "a") as f:
                        f.write(json.dumps({"Error": f"{domain}/{example_id} - {e}"}))
                        f.write("\n")
                finally:
                    # Reset status to IDLE after task completes or errors
                    if shared_env_status is not None:
                        shared_env_status[proc_name] = {
                            **shared_env_status.get(proc_name, {}),
                            "domain": "",
                            "task_id": "",
                            "instruction": "",
                            "status": "IDLE",
                        }
            except Exception as e:
                logger.error(f"Task-level error in {proc_name}: {e}")
                import traceback

                logger.error(traceback.format_exc())
    except Exception as e:
        logger.error(f"Process-level error in {proc_name}: {e}")
        import traceback

        logger.error(traceback.format_exc())
    finally:
        if shared_env_status is not None:
            try:
                shared_env_status[proc_name] = {
                    **shared_env_status.get(proc_name, {}),
                    "status": "STOPPED",
                }
            except Exception:
                pass
        logger.info(f"{proc_name} cleaning up environment...")
        try:
            if env:
                env.close()
                logger.info(f"{proc_name} environment closed successfully")
        except Exception as e:
            logger.error(f"{proc_name} error during environment cleanup: {e}")


def config() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run end-to-end evaluation on the benchmark"
    )

    # =======================================================================
    # 1. environment configurations
    # =======================================================================

    parser.add_argument(
        "--result_dir",
        type=str,
        default="results",
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=50,
        help="Number of environments to run in parallel",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default="config/config.toml",
        help="Path to the agent model configuration TOML file.",
    )
    parser.add_argument(
        "--test_all_meta_path",
        type=str,
        default="evaluation_examples/test_nogdrive.json",
    )
    parser.add_argument("--max_steps", type=int, default=100)

    # logging
    parser.add_argument(
        "--headless", type=bool, default=True, help="Run in headless mode"
    )
    parser.add_argument(
        "--debug",
        type=bool,
        default=False,
        help="Enable debug mode with verbose logging.",
    )

    # Environment config
    parser.add_argument(
        "--provider_name",
        type=str,
        default="aws",
        help="Virtualization provider (vmware, docker, aws, azure, gcp, virtualbox)",
    )
    parser.add_argument(
        "--region", type=str, default="us-east-1", help="AWS region for the VM"
    )

    parser.add_argument(
        "--client_password", type=str, default="", help="Client password"
    )
    parser.add_argument("--path_to_vm", type=str, default=None)
    parser.add_argument("--screen_width", type=int, default=1920)
    parser.add_argument("--screen_height", type=int, default=1080)
    parser.add_argument("--sleep_after_execution", type=float, default=1.0)

    # Data config
    parser.add_argument("--domain", type=str, default="all")

    parser.add_argument(
        "--test_config_base_dir", type=str, default="evaluation_examples"
    )
    parser.add_argument("--max_trajectory_length", type=int, default=8)

    # =======================================================================
    # 2. model configuration file path (new core parameter)
    # =======================================================================

    # Multi-run support: --runs-file overrides --result_dir and --config-path
    parser.add_argument(
        "--runs-file",
        type=str,
        default=None,
        help=(
            "Path to a JSON file listing multiple (result_dir, config_path) "
            "pairs to run on a shared VM pool.  When provided, --result_dir "
            "and --config-path are ignored."
        ),
    )

    args = parser.parse_args()

    # Default AWS password per README: osworld-public-evaluation
    if args.provider_name == "aws" and not args.client_password:
        args.client_password = "osworld-public-evaluation"

    # When --runs-file is given, TOML loading is deferred to build_run_configs().
    if not args.runs_file:
        if not os.path.exists(args.config_path):
            raise FileNotFoundError(
                f"Configuration file not found at: {args.config_path}"
            )
        print(f"Loading model configuration from: {args.config_path}")
        _apply_toml_config(args, args.config_path)

    return args


def _result_base_dir(rc: RunConfig) -> str:
    """Compute the result base directory for a RunConfig."""
    return os.path.join(
        rc.result_dir,
        rc.args.action_space,
        rc.args.observation_type,
        rc.args.model_dir_name,
    )


def test(
    run_configs: list,
    shared_args: argparse.Namespace,
    all_tasks: list,
    total_task_count: int = 0,
) -> None:
    global processes
    for rc in run_configs:
        logger.info("Run %d args: %s", rc.run_idx, rc.args)
    logger.info(f"Total tasks across all runs: {len(all_tasks)}")

    # Email milestone tracking
    from tools.email_notifier import send_milestone_email, MILESTONES

    # from tools.notion_publisher import publish_to_notion
    from tools.wandb_tracker import (
        init_wandb_run,
        log_wandb_metrics,
        log_wandb_tables,
        finish_wandb_run,
    )

    # Initialize wandb run with first run's config (aggregate tracking)
    first_rc = run_configs[0]
    init_wandb_run(
        {
            "model": first_rc.args.model,
            "model_dir_name": first_rc.args.model_dir_name,
            "model_provider": first_rc.args.model_provider,
            "action_space": first_rc.args.action_space,
            "observation_type": first_rc.args.observation_type,
            "max_steps": first_rc.args.max_steps,
            "num_envs": shared_args.num_envs,
            "provider_name": shared_args.provider_name,
            "region": shared_args.region,
            "total_tasks": total_task_count,
            "config_path": first_rc.config_path,
            "result_dir": first_rc.result_dir,
            "num_runs": len(run_configs),
        }
    )

    with Manager() as manager:
        # Per-run shared data structures
        per_run_scores: Dict[int, Any] = {
            rc.run_idx: manager.list() for rc in run_configs
        }
        per_run_token_summaries: Dict[int, Any] = {
            rc.run_idx: manager.list() for rc in run_configs
        }

        # Restore token summaries from previous (paused) runs
        for rc in run_configs:
            rbd = _result_base_dir(rc)
            previous_summaries = load_previous_token_summaries(rbd)
            for s in previous_summaries:
                per_run_token_summaries[rc.run_idx].append(s)

        shared_env_status = manager.dict()
        task_queue = manager.Queue()
        for item in all_tasks:
            task_queue.put(item)

        num_envs = shared_args.num_envs
        processes = []
        # Stagger process launches to avoid AWS RequestLimitExceeded errors.
        launch_delay = 2 if num_envs > 10 else 0
        worker_args = (
            task_queue,
            run_configs,
            shared_args,
            per_run_scores,
            per_run_token_summaries,
            shared_env_status,
        )
        for i in range(num_envs):
            p = Process(
                target=run_env_tasks,
                args=worker_args,
                name=f"EnvProcess-{i + 1}",
            )
            p.daemon = True
            p.start()
            processes.append(p)
            logger.info(f"Started process {p.name} with PID {p.pid}")
            if launch_delay and i < num_envs - 1:
                time.sleep(launch_delay)

        # Track which milestones have already been notified
        notified_milestones = set()

        def _total_completed() -> int:
            return sum(len(per_run_scores[rc.run_idx]) for rc in run_configs)

        def _all_token_summaries() -> list:
            out = []
            for rc in run_configs:
                out.extend(list(per_run_token_summaries[rc.run_idx]))
            return out

        try:
            while True:
                alive_count = 0
                for idx, p in enumerate(processes):
                    if not p.is_alive():
                        logger.warning(f"Process {p.name} died, restarting...")
                        new_p = Process(
                            target=run_env_tasks,
                            args=worker_args,
                            name=f"EnvProcess-Restart-{idx + 1}",
                        )
                        new_p.daemon = True
                        new_p.start()
                        processes[idx] = new_p
                        logger.info(
                            f"Restarted process {new_p.name} with PID {new_p.pid}"
                        )
                    else:
                        alive_count += 1

                # Check email milestones (exclude 100% which is sent after join)
                completed = _total_completed()
                for milestone in MILESTONES:
                    if milestone == 100:
                        continue
                    if milestone not in notified_milestones:
                        threshold = int(total_task_count * milestone / 100)
                        if threshold > 0 and completed >= threshold:
                            logger.info(
                                f"Milestone {milestone}% reached "
                                f"({completed}/{total_task_count}). Sending email..."
                            )
                            send_milestone_email(milestone, completed, total_task_count)
                            log_wandb_tables(_all_token_summaries())
                            notified_milestones.add(milestone)

                # Update live_status.md and token summaries per run
                for rc in run_configs:
                    rbd = _result_base_dir(rc)
                    try:
                        write_env_status_file(
                            shared_env_status, rbd, completed, total_task_count
                        )
                    except Exception as e:
                        logger.debug(f"Failed to update env status file: {e}")

                    run_ts = list(per_run_token_summaries[rc.run_idx])
                    if run_ts:
                        try:
                            write_token_usage_markdown(run_ts, rbd)
                        except Exception as e:
                            logger.debug(
                                f"Failed to update token summary for run {rc.run_idx}: {e}"
                            )

                # wandb metrics use aggregate across all runs
                log_wandb_metrics(_all_token_summaries(), completed, total_task_count)

                if task_queue.empty():
                    logger.info("All tasks finished.")
                    break
                if alive_count == 0:
                    logger.error("All processes died, exiting.")
                    break
                time.sleep(5)
            for p in processes:
                p.join()

            # Send 100% milestone email after all processes finish
            if 100 not in notified_milestones:
                completed = _total_completed()
                logger.info(
                    f"Milestone 100% reached "
                    f"({completed}/{total_task_count}). Sending email..."
                )
                send_milestone_email(100, completed, total_task_count)
                all_ts = _all_token_summaries()
                log_wandb_metrics(all_ts, completed, total_task_count)
                log_wandb_tables(all_ts)
                notified_milestones.add(100)
        except KeyboardInterrupt:
            logger.info(
                "Main process received KeyboardInterrupt. Initiating graceful shutdown..."
            )
            raise
        except Exception as e:
            logger.error(
                f"Unexpected error while waiting for processes: {e}", exc_info=True
            )
            for p in processes:
                if p.is_alive():
                    try:
                        logger.info(f"Terminating process {p.name} due to error...")
                        p.terminate()
                    except Exception as term_e:
                        logger.error(f"Error terminating process {p.name}: {term_e}")
            raise

        # Final per-run summaries
        for rc in run_configs:
            scores = list(per_run_scores[rc.run_idx])
            token_summaries = list(per_run_token_summaries[rc.run_idx])
            avg = sum(scores) / len(scores) if scores else 0
            logger.info(f"[Run {rc.run_idx}] Average score: {avg}")
            rbd = _result_base_dir(rc)
            if token_summaries:
                output_path = write_token_usage_markdown(token_summaries, rbd)
                logger.info(
                    f"[Run {rc.run_idx}] Token usage summary saved to: {output_path}"
                )
            else:
                logger.info(
                    f"[Run {rc.run_idx}] No token usage summaries were collected."
                )

    finish_wandb_run()


def get_unfinished(
    action_space, use_model, observation_type, result_dir, total_file_json
):
    target_dir = os.path.join(result_dir, action_space, observation_type, use_model)

    if not os.path.exists(target_dir):
        return total_file_json

    finished = {}
    for domain in os.listdir(target_dir):
        finished[domain] = []
        domain_path = os.path.join(target_dir, domain)
        if os.path.isdir(domain_path):
            for example_id in os.listdir(domain_path):
                if example_id == "onboard":
                    continue
                example_path = os.path.join(domain_path, example_id)
                if os.path.isdir(example_path):
                    if "result.txt" not in os.listdir(example_path):
                        # empty all files under example_id
                        for file in os.listdir(example_path):
                            os.remove(os.path.join(example_path, file))
                    else:
                        finished[domain].append(example_id)

    if not finished:
        return total_file_json

    for domain, examples in finished.items():
        if domain in total_file_json:
            total_file_json[domain] = [
                x for x in total_file_json[domain] if x not in examples
            ]

    return total_file_json


def get_result(action_space, use_model, observation_type, result_dir, total_file_json):
    target_dir = os.path.join(result_dir, action_space, observation_type, use_model)
    if not os.path.exists(target_dir):
        print("New experiment, no result yet.")
        return None

    all_result = []

    for domain in os.listdir(target_dir):
        domain_path = os.path.join(target_dir, domain)
        if os.path.isdir(domain_path):
            for example_id in os.listdir(domain_path):
                example_path = os.path.join(domain_path, example_id)
                if os.path.isdir(example_path):
                    if "result.txt" in os.listdir(example_path):
                        # empty all files under example_id
                        try:
                            all_result.append(
                                float(
                                    open(
                                        os.path.join(example_path, "result.txt"), "r"
                                    ).read()
                                )
                            )
                        except Exception as e:
                            print(
                                f"Error reading result for {domain}/{example_id}: {e}"
                            )
                            all_result.append(0.0)

    if not all_result:
        print("New experiment, no result yet.")
        return None
    else:
        print("Current Success Rate:", sum(all_result) / len(all_result) * 100, "%")
        return all_result


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    ####### The complete version of the list of examples #######
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = config()

    # Build RunConfig list (single entry for legacy CLI, N entries for --runs-file)
    run_configs = build_run_configs(args)
    logger.info(f"Number of run configs: {len(run_configs)}")
    for rc in run_configs:
        logger.info(
            f"  Run {rc.run_idx}: config={rc.config_path} result_dir={rc.result_dir}"
        )

    # Save per-run args.json
    for rc in run_configs:
        path_to_args = os.path.join(
            rc.result_dir,
            rc.args.action_space,
            rc.args.observation_type,
            rc.args.model_dir_name,
            "args.json",
        )
        os.makedirs(os.path.dirname(path_to_args), exist_ok=True)
        with open(path_to_args, "w", encoding="utf-8") as f:
            json.dump(vars(rc.args), f, indent=4)

    # Load task metadata
    with open(args.test_all_meta_path, "r", encoding="utf-8") as f:
        test_all_meta = json.load(f)

    if args.domain != "all":
        test_all_meta = {args.domain: test_all_meta[args.domain]}

    # Total task count per run (all tasks, including finished) for milestones
    tasks_per_run = sum(len(v) for v in test_all_meta.values())
    total_task_count = tasks_per_run * len(run_configs)

    # Build per-run unfinished task lists and interleave them round-robin
    per_run_tasks: Dict[int, list] = {}
    for rc in run_configs:
        # Deep-copy so get_unfinished can mutate freely
        meta_copy = {d: list(ids) for d, ids in test_all_meta.items()}
        unfinished = get_unfinished(
            rc.args.action_space,
            rc.args.model_dir_name,
            rc.args.observation_type,
            rc.result_dir,
            meta_copy,
        )
        per_run_tasks[rc.run_idx] = distribute_tasks(unfinished)

        left_info = ""
        for domain in unfinished:
            left_info += f"{domain}: {len(unfinished[domain])}\n"
        logger.info(f"[Run {rc.run_idx} ({rc.config_path})] Left tasks:\n{left_info}")

        get_result(
            rc.args.action_space,
            rc.args.model_dir_name,
            rc.args.observation_type,
            rc.result_dir,
            meta_copy,
        )

    # Interleave tasks round-robin across runs for even progress
    all_tasks: list = []
    max_task_len = max((len(tasks) for tasks in per_run_tasks.values()), default=0)
    for i in range(max_task_len):
        for run_idx in sorted(per_run_tasks.keys()):
            tasks = per_run_tasks[run_idx]
            if i < len(tasks):
                domain, example_id = tasks[i]
                all_tasks.append((run_idx, domain, example_id))

    logger.info(f"Total interleaved tasks to run: {len(all_tasks)}")
    test(run_configs, args, all_tasks, total_task_count=total_task_count)
