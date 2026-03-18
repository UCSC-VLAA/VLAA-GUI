"""Main evaluation runner for WindowsAgentArena (WAA) with vlaa_gui.

Modelled after WAA's ``run.py`` and ``osworld_setup/run_locally.py``.
Loads tasks from WAA's ``evaluation_examples_windows/test_all.json``,
distributes them across workers, and collects results.
"""

import argparse
import datetime
import json
import logging
import os
import sys

import tomllib
from tqdm import tqdm

try:
    from vlaa_gui.agent import ZooAgent
except ImportError:
    from waa_setup.vlaa_gui.agent import ZooAgent

# Lazily imported inside functions that need them so the script can still
# parse arguments and show help even when WAA's desktop_env is not installed.
# from desktop_env.desktop_env import DesktopEnv  # noqa: E402

import lib_run_single

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

datetime_str: str = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")
os.makedirs("logs", exist_ok=True)

file_handler = logging.FileHandler(
    os.path.join("logs", f"normal-{datetime_str}.log"), encoding="utf-8"
)
debug_handler = logging.FileHandler(
    os.path.join("logs", f"debug-{datetime_str}.log"), encoding="utf-8"
)
stdout_handler = logging.StreamHandler(sys.stdout)

file_handler.setLevel(logging.INFO)
debug_handler.setLevel(logging.DEBUG)
stdout_handler.setLevel(logging.INFO)

formatter = logging.Formatter(
    fmt="[%(asctime)s %(levelname)s %(module)s/%(lineno)d-%(processName)s] %(message)s"
)
file_handler.setFormatter(formatter)
debug_handler.setFormatter(formatter)
stdout_handler.setFormatter(formatter)

stdout_handler.addFilter(logging.Filter("desktopenv"))

logger.addHandler(file_handler)
logger.addHandler(debug_handler)
logger.addHandler(stdout_handler)

logger = logging.getLogger("desktopenv.experiment")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
def config() -> argparse.Namespace:
    """Parse CLI arguments, merging defaults from ``config.toml``."""
    cfg: dict = {}
    try:
        with open("config.toml", "rb") as f:
            cfg = tomllib.load(f)
    except FileNotFoundError:
        raise SystemExit("ERROR: config.toml not found. Please create one.")

    def _tv(section: str, key: str, default=None):
        """Get a value from the TOML config with a fallback."""
        try:
            return cfg[section][key]
        except (KeyError, TypeError):
            return default

    parser = argparse.ArgumentParser(
        description="Run vlaa_gui evaluation on WindowsAgentArena"
    )

    # VM / cloud provider
    parser.add_argument(
        "--provider_name",
        type=str,
        default="docker",
        help="Virtualization provider (docker, aws, azure, vmware, virtualbox).",
    )
    parser.add_argument(
        "--region",
        type=str,
        default=None,
        help="Cloud region for the VM (e.g. us-east-1 for AWS, eastus for Azure).",
    )
    parser.add_argument(
        "--path_to_vm",
        type=str,
        default=None,
        help="Path to local VM image (for vmware/virtualbox providers).",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=True,
        help="Run in headless mode.",
    )

    # WAA-specific
    parser.add_argument(
        "--emulator_ip",
        type=str,
        default="localhost",
        help="IP address of the Windows 11 VM / emulator (docker provider only).",
    )
    parser.add_argument("--screen_width", type=int, default=1920)
    parser.add_argument("--screen_height", type=int, default=1080)
    parser.add_argument("--sleep_after_execution", type=float, default=0.0)
    parser.add_argument("--max_steps", type=int, default=15)
    parser.add_argument("--max_trajectory_length", type=int, default=3)

    # Task selection
    parser.add_argument(
        "--test_all_meta_path",
        type=str,
        default="evaluation_examples_windows/test_all.json",
        help="Path to the JSON file listing all tasks.",
    )
    parser.add_argument(
        "--test_config_base_dir",
        type=str,
        default="evaluation_examples_windows",
    )
    parser.add_argument("--domain", type=str, default="all")
    parser.add_argument(
        "--diff_lvl",
        type=str,
        default="all",
        help="Filter tasks by difficulty level (all, easy, medium, hard).",
    )

    # Multi-worker
    parser.add_argument(
        "--worker_id", type=int, default=0, help="This worker's ID (0-indexed)."
    )
    parser.add_argument(
        "--num_workers", type=int, default=1, help="Total number of workers."
    )

    # Output
    parser.add_argument("--result_dir", type=str, default="./results")

    # Model
    parser.add_argument(
        "--model_provider",
        type=str,
        default=_tv("model", "provider", "openai"),
    )
    parser.add_argument(
        "--model",
        type=str,
        default=_tv("model", "name", "gpt-4o"),
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max_tokens", type=int, default=1500)

    # Observation
    parser.add_argument(
        "--observation_type",
        choices=["screenshot", "a11y_tree", "screenshot_a11y_tree", "som"],
        default=_tv("perception", "observation_type", "screenshot"),
    )

    # Misc
    parser.add_argument(
        "--config_path",
        type=str,
        default="config.toml",
        help="Path to TOML config file.",
    )

    args = parser.parse_args()

    # Read verifier flag from TOML (not a CLI arg — matches OSWorld pattern)
    args.use_verifier = _tv("tts", "use_verifier", False)

    return args


# ---------------------------------------------------------------------------
# Result helpers (mirror osworld_setup/run_locally.py)
# ---------------------------------------------------------------------------
def get_unfinished(
    model: str, observation_type: str, result_dir: str, total_file_json: dict
):
    """Return the subset of tasks that have not yet produced a result.txt."""
    target_dir = os.path.join(result_dir, "pyautogui", observation_type, model)

    if not os.path.exists(target_dir):
        return total_file_json

    finished: dict = {}
    for domain in os.listdir(target_dir):
        finished[domain] = []
        domain_path = os.path.join(target_dir, domain)
        if os.path.isdir(domain_path):
            for example_id in os.listdir(domain_path):
                example_path = os.path.join(domain_path, example_id)
                if os.path.isdir(example_path):
                    if "result.txt" not in os.listdir(example_path):
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


def get_result(
    model: str, observation_type: str, result_dir: str, total_file_json: dict
):
    """Print current success rate across completed tasks."""
    target_dir = os.path.join(result_dir, "pyautogui", observation_type, model)
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
                    result_file = os.path.join(example_path, "result.txt")
                    if os.path.isfile(result_file):
                        try:
                            all_result.append(float(open(result_file).read()))
                        except Exception:
                            all_result.append(0.0)

    if not all_result:
        print("New experiment, no result yet.")
        return None
    print(f"Current Success Rate: {sum(all_result) / len(all_result) * 100:.2f}%")
    return all_result


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------
def test(args: argparse.Namespace, test_all_meta: dict) -> None:
    from desktop_env.envs.desktop_env import DesktopEnv
    from vlaa_gui.agent_core.agents.verifier_agent import VerifierAgent

    scores: list = []

    logger.info("Args: %s", args)

    # Build agent
    agent = ZooAgent(
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        observation_type=args.observation_type,
        screen_width=args.screen_width,
        screen_height=args.screen_height,
        config_path=args.config_path,
    )

    # Build verifier agent if enabled in config
    verifier_agent = None
    if args.use_verifier:
        verifier_agent = VerifierAgent(agent.engine_params, platform="windows")
        logger.info("Verifier agent enabled.")

    # Build WAA environment
    env = DesktopEnv(
        action_space="pyautogui",
        screen_size=(args.screen_width, args.screen_height),
        headless=args.headless,
        require_a11y_tree=args.observation_type
        in ["a11y_tree", "screenshot_a11y_tree", "som"],
        emulator_ip=args.emulator_ip,
    )

    for domain in tqdm(test_all_meta, desc="Domain"):
        for example_id in tqdm(test_all_meta[domain], desc="Example", leave=False):
            config_file = os.path.join(
                args.test_config_base_dir,
                f"examples/{domain}/{example_id}.json",
            )
            with open(config_file, "r", encoding="utf-8") as f:
                example = json.load(f)

            logger.info("[Domain]: %s", domain)
            logger.info("[Example ID]: %s", example_id)

            instruction = example["instruction"]
            logger.info("[Instruction]: %s", instruction)

            example_result_dir = os.path.join(
                args.result_dir,
                "pyautogui",
                args.observation_type,
                args.model,
                domain,
                example_id,
            )
            os.makedirs(example_result_dir, exist_ok=True)

            try:
                lib_run_single.run_single_example(
                    agent,
                    env,
                    example,
                    args.max_steps,
                    instruction,
                    args,
                    example_result_dir,
                    scores,
                    verifier_agent=verifier_agent,
                )
            except Exception as e:
                logger.error("Exception in %s/%s: %s", domain, example_id, e)
                with open(os.path.join(example_result_dir, "traj.jsonl"), "a") as f:
                    f.write(
                        json.dumps(
                            {"Error": f"Exception in {domain}/{example_id}: {e}"}
                        )
                    )
                    f.write("\n")

    env.close()
    if scores:
        logger.info("Average score: %.4f", sum(scores) / len(scores))


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    args = config()

    # Save args
    path_to_args = os.path.join(
        args.result_dir,
        "pyautogui",
        args.observation_type,
        args.model,
        "args.json",
    )
    os.makedirs(os.path.dirname(path_to_args), exist_ok=True)
    with open(path_to_args, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=4)

    # Load task list
    with open(args.test_all_meta_path, "r", encoding="utf-8") as f:
        test_all_meta = json.load(f)

    # Filter by domain
    if args.domain != "all":
        test_all_meta = {args.domain: test_all_meta[args.domain]}

    # Multi-worker task distribution
    if args.num_workers > 1:
        all_tasks = []
        for domain, examples in test_all_meta.items():
            for ex in examples:
                all_tasks.append((domain, ex))
        worker_tasks = all_tasks[args.worker_id :: args.num_workers]
        test_all_meta = {}
        for domain, ex in worker_tasks:
            test_all_meta.setdefault(domain, []).append(ex)

    # Skip already-finished tasks
    test_file_list = get_unfinished(
        args.model, args.observation_type, args.result_dir, test_all_meta
    )

    left_info = ""
    for domain in test_file_list:
        left_info += f"{domain}: {len(test_file_list[domain])}\n"
    logger.info("Left tasks:\n%s", left_info)

    get_result(args.model, args.observation_type, args.result_dir, test_all_meta)
    test(args, test_file_list)
