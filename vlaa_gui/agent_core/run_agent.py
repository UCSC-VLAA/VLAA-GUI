import argparse
import datetime
import io
import json
import logging
import os
import platform
import pyautogui
import sys
import signal
import time
import tomllib
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from PIL import Image

from vlaa_gui.agent_core.agents.grounding import OSWorldACI
from vlaa_gui.agent_core.agents.MacOSACI import MacOSACI
from vlaa_gui.agent_core.agents.agent import Agent
from vlaa_gui.agent_core.agents.verifier_agent import VerifierAgent
from vlaa_gui.agent_core.utils.local_env import LocalEnv
from vlaa_gui.agent_core.utils.token_tracker import (
    reset_global_tracker,
)


current_platform = platform.system().lower()

# Global flag to track pause state for debugging
paused = False

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

datetime_str: str = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)


def get_char():
    """Get a single character from stdin without pressing Enter"""
    try:
        # Import termios and tty on Unix-like systems
        if platform.system() in ["Darwin", "Linux"]:
            import termios
            import tty

            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                tty.setraw(sys.stdin.fileno())
                ch = sys.stdin.read(1)
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            return ch
        else:
            # Windows fallback
            import msvcrt

            return msvcrt.getch().decode("utf-8", errors="ignore")
    except Exception:
        return input()  # Fallback for non-terminal environments


def signal_handler(signum, frame):
    """Handle Ctrl+C signal for debugging during agent execution"""
    global paused

    if not paused:
        print("\n\n🔸 GUI Agent Workflow Paused 🔸")
        print("=" * 50)
        print("Options:")
        print("  • Press Ctrl+C again to quit")
        print("  • Press Esc to resume workflow")
        print("=" * 50)

        paused = True

        while paused:
            try:
                print("\n[PAUSED] Waiting for input... ", end="", flush=True)
                char = get_char()

                if ord(char) == 3:  # Ctrl+C
                    print("\n\n🛑 Exiting Agent...")
                    sys.exit(0)
                elif ord(char) == 27:  # Esc
                    print("\n\n▶️  Resuming Agent workflow...")
                    paused = False
                    break
                else:
                    print(f"\n   Unknown command: '{char}' (ord: {ord(char)})")

            except KeyboardInterrupt:
                print("\n\n🛑 Exiting Agent...")
                sys.exit(0)
    else:
        # Already paused, second Ctrl+C means quit
        print("\n\n🛑 Exiting Agent...")
        sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)

file_handler = logging.FileHandler(
    os.path.join("logs", "normal-{:}.log".format(datetime_str)), encoding="utf-8"
)
debug_handler = logging.FileHandler(
    os.path.join("logs", "debug-{:}.log".format(datetime_str)), encoding="utf-8"
)
stdout_handler = logging.StreamHandler(sys.stdout)
sdebug_handler = logging.FileHandler(
    os.path.join("logs", "sdebug-{:}.log".format(datetime_str)), encoding="utf-8"
)

file_handler.setLevel(logging.INFO)
debug_handler.setLevel(logging.DEBUG)
stdout_handler.setLevel(logging.INFO)
sdebug_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter(
    fmt="\x1b[1;33m[%(asctime)s \x1b[31m%(levelname)s \x1b[32m%(module)s/%(lineno)d-%(processName)s\x1b[1;33m] \x1b[0m%(message)s"
)
file_handler.setFormatter(formatter)
debug_handler.setFormatter(formatter)
stdout_handler.setFormatter(formatter)
sdebug_handler.setFormatter(formatter)

stdout_handler.addFilter(logging.Filter("desktopenv"))
sdebug_handler.addFilter(logging.Filter("desktopenv"))

logger.addHandler(file_handler)
logger.addHandler(debug_handler)
logger.addHandler(stdout_handler)
logger.addHandler(sdebug_handler)

platform_os = platform.system()


def _is_macos_platform(platform_name: str) -> bool:
    return str(platform_name).lower() in {"darwin", "macos", "mac"}


def _normalize_observation_type_for_platform(
    platform_name: str, observation_type: str
) -> str:
    if _is_macos_platform(platform_name) and observation_type in {
        "a11y_tree",
        "screenshot_a11y_tree",
        "som",
    }:
        logger.warning(
            "macOS ACI uses screenshot grounding only; forcing observation_type from %s to screenshot.",
            observation_type,
        )
        return "screenshot"
    return observation_type


def _select_grounding_agent_class(platform_name: str):
    return MacOSACI if _is_macos_platform(platform_name) else OSWorldACI


def show_permission_dialog(code: str, action_description: str):
    """Show a platform-specific permission dialog and return True if approved."""
    if platform.system() == "Darwin":
        result = os.system(
            f'osascript -e \'display dialog "Do you want to execute this action?\n\n{code} which will try to {action_description}" with title "Action Permission" buttons {{"Cancel", "OK"}} default button "OK" cancel button "Cancel"\''
        )
        return result == 0
    elif platform.system() == "Linux":
        result = os.system(
            f'zenity --question --title="Action Permission" --text="Do you want to execute this action?\n\n{code}" --width=400 --height=200'
        )
        return result == 0
    return False


def _get_toml_value(config, path, default=None):
    """Safely read a nested value from the TOML config."""
    current = config
    try:
        for key in path:
            current = current[key]
        return current
    except (KeyError, TypeError):
        return default


def _strip_model_prefix(model_name: str) -> str:
    """Strip regional and provider prefixes from model names."""
    return model_name.rsplit(".", 1)[-1] if "." in model_name else model_name


def _token_summaries_json_path(output_dir: str) -> str:
    return os.path.join(output_dir, "token_usage_summaries.json")


def load_previous_token_summaries(output_dir: str) -> List[Dict]:
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
        logger.warning("Failed to load token summaries from %s: %s", json_path, e)
    return []


def _save_token_summaries_json(token_summaries: List[Dict], output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    json_path = _token_summaries_json_path(output_dir)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(token_summaries, f, ensure_ascii=False, indent=2)
    return json_path


def _format_instruction(instruction: str, max_len: int = 120) -> str:
    cleaned = " ".join((instruction or "").split())
    if len(cleaned) > max_len:
        return cleaned[: max_len - 3] + "..."
    return cleaned


def write_token_usage_markdown(token_summaries: List[Dict], output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "token_usage_summary.md")

    _save_token_summaries_json(token_summaries, output_dir)

    grouped: Dict[str, List[Dict]] = {"interactive": token_summaries}
    total_tasks = len(token_summaries)
    total_output_tokens = sum(s.get("total_output_tokens", 0) for s in token_summaries)
    total_all_tokens = sum(s.get("total_tokens", 0) for s in token_summaries)

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

        f.write("\n## Agent token usage (all tasks)\n\n")
        f.write("| Agent | Output Tokens | Total Tokens | Calls |\n")
        f.write("| --- | ---: | ---: | ---: |\n")
        for agent_type in sorted(agent_totals.keys()):
            stats = agent_totals[agent_type]
            f.write(
                f"| {agent_type} | {stats['output_tokens']:,} | {stats['total_tokens']:,} | {stats['calls']} |\n"
            )

        f.write("\n## Per-task details\n\n")
        for domain in sorted(grouped.keys()):
            f.write(f"### {domain}\n\n")
            f.write("| Task ID | Output Tokens | Total Tokens | Instruction |\n")
            f.write("| --- | ---: | ---: | --- |\n")
            for summary in sorted(grouped[domain], key=lambda x: x.get("task_id", "")):
                task_id = summary.get("task_id", "")
                output_tokens = summary.get("total_output_tokens", 0)
                all_tokens = summary.get("total_tokens", 0)
                instruction = _format_instruction(summary.get("task_instruction", ""))
                f.write(
                    f"| {task_id} | {output_tokens:,} | {all_tokens:,} | {instruction} |\n"
                )
            f.write("\n")

    return output_path


def _apply_toml_config(args: argparse.Namespace, config_path: str) -> Dict[str, Any]:
    """Load a TOML config file and apply values to args in-place."""
    with open(config_path, "rb") as f:
        toml_cfg = tomllib.load(f)

    args.model_provider = _get_toml_value(toml_cfg, ["model", "provider"], "openai")
    args.model = _get_toml_value(toml_cfg, ["model", "name"], "")
    args.model_dir_name = _strip_model_prefix(args.model)
    args.model_url = _get_toml_value(toml_cfg, ["model", "url"], "")
    args.model_api_key = _get_toml_value(toml_cfg, ["model", "api_key"], "")
    args.model_api_keys = _get_toml_value(toml_cfg, ["model", "api_keys"], [])
    args.model_temperature = _get_toml_value(toml_cfg, ["model", "temperature"], None)
    args.model_top_p = _get_toml_value(toml_cfg, ["model", "top_p"], None)
    args.max_tokens = _get_toml_value(toml_cfg, ["model", "max_tokens"], None)
    args.model_project_id = _get_toml_value(toml_cfg, ["model", "project_id"], "")
    args.model_region = _get_toml_value(toml_cfg, ["model", "region"], "")
    args.model_aws_keys = _get_toml_value(toml_cfg, ["model", "aws_keys"], [])
    args.api_version = _get_toml_value(toml_cfg, ["model", "api_version"], "")
    args.model_thinking = _get_toml_value(toml_cfg, ["model", "thinking"], False)
    args.model_thinking_budget = _get_toml_value(
        toml_cfg, ["model", "thinking_budget"], None
    )
    args.model_thinking_level = _get_toml_value(
        toml_cfg, ["model", "thinking_level"], None
    )
    args.model_reasoning_effort = _get_toml_value(
        toml_cfg, ["model", "reasoning_effort"], args.model_thinking_level
    )
    args.model_include_thoughts = _get_toml_value(
        toml_cfg, ["model", "include_thoughts"], False
    )

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
    args.grounding_model_api_version = _get_toml_value(
        toml_cfg, ["grounding", "api_version"], ""
    )
    args.grounding_thinking = _get_toml_value(
        toml_cfg, ["grounding", "thinking"], False
    )
    args.grounding_temperature = _get_toml_value(
        toml_cfg, ["grounding", "temperature"], None
    )
    args.grounding_top_p = _get_toml_value(toml_cfg, ["grounding", "top_p"], None)
    args.grounding_thinking_budget = _get_toml_value(
        toml_cfg, ["grounding", "thinking_budget"], None
    )
    args.grounding_thinking_level = _get_toml_value(
        toml_cfg, ["grounding", "thinking_level"], None
    )
    args.grounding_reasoning_effort = _get_toml_value(
        toml_cfg,
        ["grounding", "reasoning_effort"],
        args.grounding_thinking_level,
    )
    args.grounding_include_thoughts = _get_toml_value(
        toml_cfg, ["grounding", "include_thoughts"], False
    )

    args.endpoint_provider = _get_toml_value(
        toml_cfg, ["grounding_endpoint", "provider"], ""
    )
    args.endpoint_url = _get_toml_value(toml_cfg, ["grounding_endpoint", "url"], "")
    args.endpoint_api_key = _get_toml_value(
        toml_cfg, ["grounding_endpoint", "api_key"], ""
    )

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
        toml_cfg, ["coding", "include_thoughts"], True
    )

    args.embedding_engine_type = _get_toml_value(
        toml_cfg, ["embedding", "engine_type"], "openai"
    )
    args.embedding_model = _get_toml_value(toml_cfg, ["embedding", "model"], "")

    args.observation_type = _get_toml_value(
        toml_cfg, ["perception", "observation_type"], "screenshot"
    )
    args.planner_mode = _get_toml_value(toml_cfg, ["planning", "mode"], "proactive")
    args.planner_hierarchical_depth = _get_toml_value(
        toml_cfg, ["planning", "hierarchical_depth"], 1
    )
    args.with_reflection = _get_toml_value(
        toml_cfg, ["planning", "with_reflection"], True
    )
    args.reflection_thinking = _get_toml_value(
        toml_cfg, ["planning", "reflection_thinking"], args.model_thinking
    )
    args.use_recon = _get_toml_value(toml_cfg, ["planning", "use_recon"], False)

    args.search_engine = _get_toml_value(
        toml_cfg, ["context_management", "search_engine"], None
    )
    args.kb_name = _get_toml_value(toml_cfg, ["context_management", "kb_name"], "agent_memory")
    args.memory_type = _get_toml_value(
        toml_cfg, ["context_management", "memory_type"], "mixed"
    )
    args.lexical_weight = _get_toml_value(
        toml_cfg, ["context_management", "lexical_weight"], 0.0
    )
    args.memory_representation = _get_toml_value(
        toml_cfg, ["context_management", "memory_representation"], "vector"
    )
    args.knowledge_storage = _get_toml_value(
        toml_cfg, ["context_management", "knowledge_storage"], "db"
    )

    args.searcher_type = _get_toml_value(toml_cfg, ["searcher", "type"], "llm")
    args.searcher_provider = _get_toml_value(
        toml_cfg, ["searcher", "provider"], "gemini"
    )
    args.searcher_model = _get_toml_value(
        toml_cfg, ["searcher", "model"], "gemini-3.1-pro-preview"
    )
    args.searcher_api_key = _get_toml_value(toml_cfg, ["searcher", "api_key"], "")
    args.searcher_api_keys = _get_toml_value(toml_cfg, ["searcher", "api_keys"], [])
    args.searcher_url = _get_toml_value(toml_cfg, ["searcher", "url"], "")
    args.searcher_budget = _get_toml_value(toml_cfg, ["searcher", "budget"], 20)
    args.searcher_temperature = _get_toml_value(
        toml_cfg, ["searcher", "temperature"], None
    )
    args.searcher_top_p = _get_toml_value(toml_cfg, ["searcher", "top_p"], None)
    args.searcher_reasoning_effort = _get_toml_value(
        toml_cfg, ["searcher", "reasoning_effort"], None
    )

    args.enable_gate = _get_toml_value(toml_cfg, ["gate", "enable_gate"], False)
    args.loop_detection = _get_toml_value(toml_cfg, ["gate", "loop_detection"], False)
    args.feasibility_check = _get_toml_value(
        toml_cfg, ["gate", "feasibility_check"], False
    )

    args.action_space = _get_toml_value(
        toml_cfg, ["action_space", "engine"], "pyautogui_coding"
    )
    args.action_tts_num = _get_toml_value(toml_cfg, ["tts", "action_tts_num"], 1)
    args.use_verifier = _get_toml_value(toml_cfg, ["verifier", "enabled"], True)

    args.enable_click_validation = _get_toml_value(
        toml_cfg, ["click_validation", "enabled"], False
    )
    args.click_validation_max_retries = _get_toml_value(
        toml_cfg, ["click_validation", "max_retries"], 3
    )
    args.click_validation_zoom_refine_on_failure = _get_toml_value(
        toml_cfg, ["click_validation", "zoom_refine_on_failure"], False
    )
    args.click_validation_zoom_crop_ratio = _get_toml_value(
        toml_cfg, ["click_validation", "zoom_crop_ratio"], 0.5
    )
    args.click_validation_provider = _get_toml_value(
        toml_cfg, ["click_validation", "provider"], None
    )
    args.click_validation_model = _get_toml_value(
        toml_cfg, ["click_validation", "model"], None
    )
    args.click_validation_url = _get_toml_value(
        toml_cfg, ["click_validation", "url"], None
    )
    args.click_validation_api_key = _get_toml_value(
        toml_cfg, ["click_validation", "api_key"], None
    )
    args.click_validation_temperature = _get_toml_value(
        toml_cfg, ["click_validation", "temperature"], None
    )
    args.click_validation_top_p = _get_toml_value(
        toml_cfg, ["click_validation", "top_p"], None
    )

    return toml_cfg


def _build_engine_params(
    args: argparse.Namespace,
    screen_width: int,
    screen_height: int,
) -> tuple:
    """Build engine parameter dictionaries from fully resolved args."""
    if args.search_engine in {"None", ""}:
        args.search_engine = None

    engine_params = {
        "engine_type": args.model_provider,
        "model": args.model,
        "base_url": getattr(args, "model_url", ""),
        "api_key": getattr(args, "model_api_key", ""),
        "api_keys": getattr(args, "model_api_keys", []),
        "api_version": getattr(args, "api_version", ""),
        "project_id": getattr(args, "model_project_id", ""),
        "region": getattr(args, "model_region", ""),
        "aws_keys": getattr(args, "model_aws_keys", []),
        "thinking": getattr(args, "model_thinking", False),
        "thinking_budget": getattr(args, "model_thinking_budget", None),
        "thinking_level": getattr(args, "model_thinking_level", None),
        "reasoning_effort": getattr(args, "model_reasoning_effort", None),
        "temperature": getattr(args, "model_temperature", None),
        "top_p": getattr(args, "model_top_p", None),
        "max_tokens": getattr(args, "max_tokens", None),
        "include_thoughts": getattr(args, "model_include_thoughts", False),
    }

    reflection_engine_params = dict(engine_params)
    reflection_engine_params["thinking"] = getattr(args, "reflection_thinking", False)

    if args.endpoint_url:
        engine_params_for_grounding = {
            "engine_type": args.endpoint_provider,
            "base_url": args.endpoint_url,
            "api_key": args.endpoint_api_key,
            "api_version": getattr(args, "api_version", ""),
            "temperature": getattr(args, "grounding_temperature", None),
            "top_p": getattr(args, "grounding_top_p", None),
        }
    else:
        grounding_height = args.grounding_height
        grounding_width = args.grounding_width
        if grounding_width is None:
            grounding_width = screen_width
        if grounding_height is None:
            grounding_height = int(screen_height * grounding_width / screen_width)

        engine_params_for_grounding = {
            "engine_type": args.grounding_model_provider,
            "base_url": args.grounding_model_url,
            "api_key": args.grounding_model_api_key,
            "model": args.grounding_model,
            "api_version": getattr(args, "grounding_model_api_version", ""),
            "thinking": getattr(args, "grounding_thinking", False),
            "thinking_budget": getattr(args, "grounding_thinking_budget", None),
            "thinking_level": getattr(args, "grounding_thinking_level", None),
            "reasoning_effort": getattr(args, "grounding_reasoning_effort", None),
            "include_thoughts": getattr(args, "grounding_include_thoughts", False),
            "temperature": getattr(args, "grounding_temperature", None),
            "top_p": getattr(args, "grounding_top_p", None),
            "grounding_width": grounding_width,
            "grounding_height": grounding_height,
        }

        if args.grounding_model_provider == "anthropic_bedrock":
            engine_params_for_grounding["region"] = args.grounding_model_region
            if getattr(args, "model_aws_keys", None):
                engine_params_for_grounding["aws_keys"] = args.model_aws_keys

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
        "api_keys": getattr(args, "searcher_api_keys", []),
        "base_url": getattr(args, "searcher_url", ""),
        "budget": args.searcher_budget,
        "temperature": getattr(args, "searcher_temperature", None),
        "top_p": getattr(args, "searcher_top_p", None),
        "reasoning_effort": getattr(args, "searcher_reasoning_effort", None),
        "project_id": getattr(args, "model_project_id", ""),
        "region": getattr(args, "model_region", ""),
        "aws_keys": getattr(args, "model_aws_keys", []),
    }

    return (
        engine_params,
        reflection_engine_params,
        engine_params_for_grounding,
        engine_params_for_coding,
        engine_params_for_searcher,
    )


def scale_screen_dimensions(width: int, height: int, max_dim_size: int):
    scale_factor = min(max_dim_size / width, max_dim_size / height, 1)
    safe_width = int(width * scale_factor)
    safe_height = int(height * scale_factor)
    return safe_width, safe_height


def run_agent(
    agent,
    max_steps: int,
    instruction: str,
    scaled_width: int,
    scaled_height: int,
    observation_type: str,
    action_space: str,
    with_reflection: bool = True,
    planner_mode: str = "proactive",
    verifier_agent: Optional[VerifierAgent] = None,
) -> Optional[Dict[str, Any]]:
    global paused

    # Initialize token tracker for this task
    token_tracker = reset_global_tracker()
    token_tracker.set_task_instruction(instruction)

    obs = {}
    traj = "Task:\n" + instruction
    subtask_traj = ""
    traj_lines = [f"Task: {instruction}"]
    for step in range(max_steps):
        # Check if we're in paused state and wait
        while paused:
            time.sleep(0.1)
        # Get screen shot using pyautogui
        screenshot = pyautogui.screenshot()
        screenshot = screenshot.resize((scaled_width, scaled_height), Image.LANCZOS)

        # Save the screenshot to a BytesIO object
        buffered = io.BytesIO()
        screenshot.save(buffered, format="PNG")

        # Get the byte value of the screenshot
        screenshot_bytes = buffered.getvalue()
        # Convert to base64 string.
        obs["screenshot"] = screenshot_bytes

        # Check again for pause state before prediction
        while paused:
            time.sleep(0.1)

        print(f"\n🔄 Step {step + 1}/{max_steps}: Getting next action from agent...")

        # Get next action code from the agent
        info, code = agent.predict(instruction=instruction, observation=obs)

        is_done = "done" in code[0].lower()
        is_fail = "fail" in code[0].lower()

        if is_done:
            if verifier_agent:
                verdict = verifier_agent.verify_completion(
                    instruction=instruction,
                    observation=obs,
                    trajectory="\n".join(traj_lines),
                    executor_plan=info.get("executor_plan"),
                )
                if verdict.get("complete"):
                    if platform.system() == "Darwin":
                        os.system(
                            'osascript -e \'display dialog "Task Completed" with title "OpenACI Agent" buttons "OK" default button "OK"\''
                        )
                    elif platform.system() == "Linux":
                        os.system(
                            'zenity --info --title="OpenACI Agent" --text="Task Completed" --width=200 --height=100'
                        )

                    agent.update_narrative_memory(traj)
                    break

                print(
                    f"Verifier rejected completion: {verdict.get('reason')} | Missing: {verdict.get('missing_steps')}"
                )
                traj_lines.append(
                    f"Verifier rejected DONE: reason={verdict.get('reason')}, missing_steps={verdict.get('missing_steps')}"
                )
                agent.restart_after_failed_verification(
                    reason=verdict.get("reason", ""),
                    missing_steps=verdict.get("missing_steps", ""),
                )
                subtask_traj = ""
                continue
            else:
                if platform.system() == "Darwin":
                    os.system(
                        'osascript -e \'display dialog "Task Completed" with title "OpenACI Agent" buttons "OK" default button "OK"\''
                    )
                elif platform.system() == "Linux":
                    os.system(
                        'zenity --info --title="OpenACI Agent" --text="Task Completed" --width=200 --height=100'
                    )

                agent.update_narrative_memory(traj)
                break

        if is_fail:
            if platform.system() == "Darwin":
                os.system(
                    'osascript -e \'display dialog "Task Completed" with title "OpenACI Agent" buttons "OK" default button "OK"\''
                )
            elif platform.system() == "Linux":
                os.system(
                    'zenity --info --title="OpenACI Agent" --text="Task Completed" --width=200 --height=100'
                )

            agent.update_narrative_memory(traj)
            break

        if "next" in code[0].lower():
            continue

        if "wait" in code[0].lower():
            time.sleep(5)
            continue

        else:
            time.sleep(1.0)
            print("EXECUTING CODE:", code[0])

            # Check for pause state before execution
            while paused:
                time.sleep(0.1)

            # Ask for permission before executing
            exec(code[0])
            print("Action executed.")
            time.sleep(1.0)
            if with_reflection:
                # Update task and subtask trajectories and optionally the episodic memory
                traj += (
                    "\n\nReflection:\n"
                    + str(info["reflection"])
                    + "\n\n----------------------\n\nPlan:\n"
                    + info["executor_plan"]
                )
            if planner_mode != "iterative":
                subtask_traj = agent.update_episodic_memory(info, subtask_traj)

    # Write token usage to file after task completion
    try:
        csv_path, json_path = token_tracker.write_both()
        token_tracker.log_summary()
        print(f"\n📊 Token usage saved to: {csv_path} and {json_path}")
    except Exception as e:
        logger.error(f"Failed to write token usage: {e}")
    try:
        return asdict(token_tracker.get_summary())
    except Exception as e:
        logger.error(f"Failed to build token usage summary: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Run Agent with specified configurations."
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default="config/config.toml",
        help="Path to the agent model configuration TOML file.",
    )
    parser.add_argument(
        "--pricing_config_path",
        type=str,
        default="config/pricing.toml",
        help="Path to the pricing configuration file.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with verbose logging.",
    )

    args = parser.parse_args()

    if not os.path.exists(args.config_path):
        raise FileNotFoundError(f"Configuration file not found at: {args.config_path}")

    print(f"Loading model configuration from: {args.config_path}")
    toml_cfg = _apply_toml_config(args, args.config_path)

    # print args
    print("Agent args:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")

    assert (
        args.grounding_model_provider and args.grounding_model
    ) or args.endpoint_url, (
        "Error: No grounding model was provided. Either provide an API based model, or a self-hosted HuggingFace endpoint"
    )

    # Re-scales screenshot size to ensure it fits in UI-TARS context limit
    screen_width, screen_height = pyautogui.size()
    print(f"Screen size: {screen_width}x{screen_height}")
    scaled_width, scaled_height = scale_screen_dimensions(
        screen_width, screen_height, max_dim_size=2400
    )

    (
        engine_params,
        reflection_engine_params,
        engine_params_for_grounding,
        engine_params_for_coding,
        engine_params_for_searcher,
    ) = _build_engine_params(
        args, screen_width=screen_width, screen_height=screen_height
    )

    engine_params_for_embedding = {
        "base_url": _get_toml_value(toml_cfg, ["embedding", "base_url"], ""),
        "api_key": _get_toml_value(toml_cfg, ["embedding", "api_key"], ""),
        "embedding_model": _get_toml_value(toml_cfg, ["embedding", "model"], ""),
    }

    if args.action_space == "pyautogui_coding":
        coding_agent_flag = True
        print(
            "⚠️  WARNING: Local coding environment enabled. This will execute arbitrary code locally!"
        )
        local_env = LocalEnv()
    else:
        coding_agent_flag = False
        local_env = None

    click_validation_engine_params = None
    if args.enable_click_validation and args.click_validation_provider:
        click_validation_engine_params = {
            "engine_type": args.click_validation_provider,
            "model": args.click_validation_model,
            "base_url": args.click_validation_url or "",
            "api_key": args.click_validation_api_key or "",
            "temperature": getattr(args, "click_validation_temperature", None),
            "top_p": getattr(args, "click_validation_top_p", None),
        }

    args.observation_type = _normalize_observation_type_for_platform(
        current_platform, args.observation_type
    )

    grounding_agent_cls = _select_grounding_agent_class(current_platform)
    grounding_agent = grounding_agent_cls(
        platform=current_platform,
        env=local_env,
        engine_params_for_generation=engine_params,
        engine_params_for_grounding=engine_params_for_grounding,
        engine_params_for_searcher=engine_params_for_searcher,
        width=screen_width,
        height=screen_height,
        resize_width=args.resize_width,
        grounding_model_type=args.grounding_model_type,
        code_agent_engine_params=engine_params_for_coding,
        code_agent_budget=20,
        enable_click_validation=args.enable_click_validation,
        click_validation_engine_params=click_validation_engine_params,
        click_validation_max_retries=args.click_validation_max_retries,
        click_validation_zoom_refine_on_failure=args.click_validation_zoom_refine_on_failure,
        click_validation_zoom_crop_ratio=args.click_validation_zoom_crop_ratio,
        enable_zoom_grounding=args.enable_zoom_grounding,
        zoom_grounding_crop_ratio=args.zoom_grounding_crop_ratio,
        debug=args.debug,
    )

    verifier_agent = None
    if args.use_verifier:
        verifier_agent = VerifierAgent(engine_params, platform=current_platform)

    agent = Agent(
        engine_params,
        grounding_agent,
        platform=current_platform,
        pricing_config_path=args.pricing_config_path,
        action_space=args.action_space,
        observation_type=args.observation_type,
        search_engine=args.search_engine,
        embedding_engine_type=args.embedding_engine_type,
        embedding_engine_params=engine_params_for_embedding,
        planning_mode=args.planner_mode,
        with_reflection=args.with_reflection,
        reflection_engine_params=reflection_engine_params,
        memory_type=args.memory_type,
        lexical_weight=args.lexical_weight,
        coding_agent_flag=coding_agent_flag,
        action_tts_num=args.action_tts_num,
        debug=args.debug,
        enable_gate=args.enable_gate,
        loop_detection=args.loop_detection,
        feasibility_check=args.feasibility_check,
    )

    token_summaries = load_previous_token_summaries(log_dir)

    while True:
        query = input("Query: ")

        agent.reset()
        time.sleep(1.0)

        # Run the agent on your own device
        summary = run_agent(
            agent=agent,
            max_steps=30,
            instruction=query,
            scaled_width=scaled_width,
            scaled_height=scaled_height,
            observation_type=args.observation_type,
            action_space=args.action_space,
            with_reflection=args.with_reflection,
            planner_mode=args.planner_mode,
            verifier_agent=verifier_agent,
        )

        if summary:
            token_summaries.append(summary)
            summary_path = write_token_usage_markdown(token_summaries, log_dir)
            print(f"Updated aggregate token summary: {summary_path}")

        response = input("Would you like to provide another query? (y/n): ")
        if response.lower() != "y":
            break


if __name__ == "__main__":
    main()
