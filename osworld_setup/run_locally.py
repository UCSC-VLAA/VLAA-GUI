"""Script to run end-to-end evaluation on the benchmark.
Utils and basic architecture credit to https://github.com/web-arena-x/webarena/blob/main/run.py.
"""

import argparse
import datetime
import json
import logging
import os
import sys
import tomllib

from vlaa_gui.agent_core.agents.agent import Agent
from vlaa_gui.agent_core.agents.grounding import OSWorldACI
from tqdm import tqdm

from desktop_env.desktop_env import DesktopEnv
import osworld_setup.lib_run_single_vlaa as lib_run_single_vlaa


#  Logger Configs {{{ #
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

datetime_str: str = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")

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
#  }}} Logger Configs #

logger = logging.getLogger("desktopenv.experiment")


def get_config_value(config, key, default=None, toml_section=None, toml_key=None):
    """Helper function to get config value with precedence: args > toml > default"""
    if toml_section and toml_key:
        default = config[toml_section][toml_key]
    return default


def config() -> argparse.Namespace:
    # Load configuration from TOML file if it exists
    config = {}
    try:
        with open("config.toml", "rb") as f:
            config = tomllib.load(f)
    except FileNotFoundError:
        raise Exception("INFO: config.toml not found, please create one.")

    parser = argparse.ArgumentParser(
        description="Run end-to-end evaluation on the benchmark"
    )

    parser.add_argument("--path_to_vm", type=str, default="path_to_vmx_file")
    parser.add_argument(
        "--headless", action="store_true", help="Run in headless machine"
    )
    parser.add_argument("--screen_width", type=int, default=1920)
    parser.add_argument("--screen_height", type=int, default=1080)
    parser.add_argument("--sleep_after_execution", type=float, default=0.0)
    parser.add_argument("--max_steps", type=int, default=15)

    # agent config
    parser.add_argument("--max_trajectory_length", type=int, default=3)
    parser.add_argument(
        "--test_config_base_dir", type=str, default="evaluation_examples"
    )

    # example config
    parser.add_argument("--domain", type=str, default="all")
    parser.add_argument(
        "--test_all_meta_path", type=str, default="evaluation_examples/test_subset.json"
    )

    # logging related
    parser.add_argument("--result_dir", type=str, default="./results")

    # NEW! =======================================================================

    # Model Configurations

    parser.add_argument(
        "--model_provider",
        type=str,
        default=get_config_value(config, "provider", "openai", "model", "provider"),
        help="Specify the provider to use (e.g., openai, anthropic, etc.)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=get_config_value(config, "model", "gpt-4.1", "model", "name"),
        help="Specify the model to use (e.g., gpt-4o)",
    )
    parser.add_argument(
        "--model_url",
        type=str,
        default=get_config_value(config, "model_url", "", "model", "url"),
        help="The URL of the main generation model API.",
    )
    parser.add_argument(
        "--model_api_key",
        type=str,
        default=get_config_value(config, "model_api_key", "", "model", "api_key"),
        help="The API key of the main generation model.",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max_tokens", type=int, default=1500)

    # Grounding model config
    parser.add_argument(
        "--grounding_model_provider",
        type=str,
        default=get_config_value(
            config, "grounding_model_provider", "openai", "grounding", "provider"
        ),
        help="Provider for the API-based grounding model.",
    )
    parser.add_argument(
        "--grounding_model",
        type=str,
        default=get_config_value(
            config,
            "grounding_model",
            "doubao-1-5-ui-tars-250428",
            "grounding",
            "grounding_model",
        ),
        help="API-based grounding model name.",
    )
    parser.add_argument(
        "--grounding_model_url",
        type=str,
        default=get_config_value(config, "grounding_model_url", "", "grounding", "url"),
        help="URL for the API-based grounding model.",
    )
    parser.add_argument(
        "--grounding_model_api_key",
        type=str,
        default=get_config_value(
            config, "grounding_model_api_key", "", "grounding", "api_key"
        ),
        help="API key for the API-based grounding model.",
    )
    parser.add_argument(
        "--grounding_model_resize_width",
        type=int,
        default=get_config_value(
            config,
            "grounding_model_resize_width",
            None,
            "grounding",
            "resize_width",
        ),
        help="Width of screenshot for grounding model.",
    )
    parser.add_argument(
        "--grounding_model_resize_height",
        type=int,
        default=get_config_value(
            config,
            "grounding_model_resize_height",
            None,
            "grounding",
            "resize_height",
        ),
        help="Height of screenshot for grounding model.",
    )

    # Self-hosted endpoint config
    parser.add_argument(
        "--endpoint_provider",
        type=str,
        default=get_config_value(
            config, "endpoint_provider", "", "grounding_endpoint", "provider"
        ),
        help="Provider for the self-hosted grounding model endpoint.",
    )
    parser.add_argument(
        "--endpoint_url",
        type=str,
        default=get_config_value(
            config, "endpoint_url", "", "grounding_endpoint", "url"
        ),
        help="URL for the self-hosted grounding model endpoint.",
    )
    parser.add_argument(
        "--endpoint_api_key",
        type=str,
        default=get_config_value(
            config, "endpoint_api_key", "", "grounding_endpoint", "api_key"
        ),
        help="API key for the self-hosted grounding model.",
    )

    # Embedding engine
    parser.add_argument(
        "--embedding_engine_type",
        type=str,
        default=get_config_value(
            config, "embedding_engine_type", "openai", "embedding", "engine_type"
        ),
        help="Specify the embedding engine type.",
    )

    # Perception
    parser.add_argument(
        "--observation_type",
        choices=["screenshot", "a11y_tree", "screenshot_a11y_tree", "som"],
        default=get_config_value(
            config, "observation_type", "screenshot", "perception", "observation_type"
        ),
        help="Observation type.",
    )

    # Grounding model type
    parser.add_argument(
        "--grounding_model_type",
        type=str,
        default=get_config_value(
            config, "grounding_model_type", "single", "grounding", "type"
        ),
        help="Specify the grounding model type (supports single, multi).",
    )

    # Planning
    parser.add_argument(
        "--planner_mode",
        type=str,
        default=get_config_value(
            config, "planner_mode", "proactive", "planning", "mode"
        ),
        help="Specify the planner mode.",
    )
    parser.add_argument(
        "--planner_hierarchical_depth",
        type=int,
        default=get_config_value(
            config, "planner_hierarchical_depth", 1, "planning", "hierarchical_depth"
        ),
        help="Hierarchical depth of the planner.",
    )
    parser.add_argument(
        "--with-reflection",
        action="store_true",
        default=get_config_value(
            config, "with_reflection", False, "planning", "with_reflection"
        ),
        help="Enable reflection in the planner.",
    )

    # Context management
    parser.add_argument(
        "--search_engine",
        type=str,
        default=get_config_value(
            config, "search_engine", None, "context_management", "search_engine"
        ),
        help="The search engine to use for web queries.",
    )
    parser.add_argument("--kb_name", default="kb", type=str)
    parser.add_argument(
        "--memory_type",
        type=str,
        default=get_config_value(
            config, "memory_type", "mixed", "context_management", "memory_type"
        ),
        help="The memory type.",
    )
    parser.add_argument(
        "--lexical_weight",
        type=float,
        default=get_config_value(
            config, "lexical_weight", 0.0, "context_management", "lexical_weight"
        ),
        help="Lexical weight for hybrid retrieval (0.0 = pure semantic, 1.0 = pure lexical).",
    )
    parser.add_argument(
        "--memory_representation",
        type=str,
        default=get_config_value(
            config,
            "memory_representation",
            "vector",
            "context_management",
            "memory_representation",
        ),
        help="The memory representation.",
    )
    parser.add_argument(
        "--knowledge_storage",
        type=str,
        default=get_config_value(
            config, "knowledge_storage", "db", "context_management", "knowledge_storage"
        ),
        help="The knowledge storage.",
    )

    # Action space
    parser.add_argument(
        "--action_space",
        type=str,
        default=get_config_value(
            config, "action_space", "pyautogui", "action_space", "engine"
        ),
        help="The action space.",
    )

    # TTS
    parser.add_argument(
        "--action_tts_num",
        type=int,
        default=get_config_value(config, "action_tts", 1, "tts", "action_tts"),
        help="Number of TTS actions to perform per step.",
    )

    # Click Validation
    parser.add_argument(
        "--enable_click_validation",
        type=bool,
        default=get_config_value(
            config, "enable_click_validation", False, "click_validation", "enabled"
        ),
        help="Enable click validation to verify click coordinates.",
    )
    parser.add_argument(
        "--click_validation_max_retries",
        type=int,
        default=get_config_value(
            config, "click_validation_max_retries", 3, "click_validation", "max_retries"
        ),
        help="Maximum retries for click validation.",
    )
    parser.add_argument(
        "--click_validation_provider",
        type=str,
        default=get_config_value(
            config, "click_validation_provider", None, "click_validation", "provider"
        ),
        help="Provider for click validation model.",
    )
    parser.add_argument(
        "--click_validation_model",
        type=str,
        default=get_config_value(
            config, "click_validation_model", None, "click_validation", "model"
        ),
        help="Model for click validation.",
    )
    parser.add_argument(
        "--click_validation_url",
        type=str,
        default=get_config_value(
            config, "click_validation_url", None, "click_validation", "url"
        ),
        help="URL for click validation model.",
    )
    parser.add_argument(
        "--click_validation_api_key",
        type=str,
        default=get_config_value(
            config, "click_validation_api_key", None, "click_validation", "api_key"
        ),
        help="API key for click validation model.",
    )

    args = parser.parse_args()

    # If a config file was found, override args with any command-line values that were explicitly set
    # This ensures command-line arguments have the highest priority
    if config:
        # Create a temporary parser to check which args were provided on the command line
        # This avoids overriding TOML values with argparse defaults
        cmd_line_parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
        for action in parser._actions:
            if action.dest not in ("help", "version"):
                cmd_line_parser.add_argument(
                    action.option_strings[0] if action.option_strings else action.dest
                )

        cmd_line_args, _ = cmd_line_parser.parse_known_args()

        # Update args with command-line values only
        for key, value in vars(cmd_line_args).items():
            setattr(args, key, value)

    return args


def test(args: argparse.Namespace, test_all_meta: dict) -> None:
    scores = []
    max_steps = args.max_steps

    # log args
    logger.info("Args: %s", args)
    cfg_args = {
        "path_to_vm": args.path_to_vm,
        "headless": args.headless,
        "action_space": args.action_space,
        "observation_type": args.observation_type,
        "screen_width": args.screen_width,
        "screen_height": args.screen_height,
        "sleep_after_execution": args.sleep_after_execution,
        "max_steps": args.max_steps,
        "max_trajectory_length": args.max_trajectory_length,
        "model": args.model,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "result_dir": args.result_dir,
    }

    if args.search_engine == "None" or args.search_engine == "":
        args.search_engine = None
    # NEW!
    engine_params = {
        "engine_type": args.model_provider,
        "model": args.model,
        "base_url": args.model_url,
        "api_key": args.model_api_key,
    }

    if args.endpoint_url:
        engine_params_for_grounding = {
            "engine_type": args.endpoint_provider,
            "base_url": args.endpoint_url,
            "api_key": args.endpoint_api_key,
        }
    else:
        grounding_height = args.grounding_model_resize_height
        # If not provided, use the aspect ratio of the screen to compute the height
        if grounding_height is None:
            grounding_height = (
                args.screen_height
                * args.grounding_model_resize_width
                / args.screen_width
            )

        engine_params_for_grounding = {
            "engine_type": args.grounding_model_provider,
            "model": args.grounding_model,
            "grounding_width": args.grounding_model_resize_width,
            "grounding_height": grounding_height,
        }

    # Build click validation engine params if configured
    click_validation_engine_params = None
    if args.enable_click_validation and args.click_validation_provider:
        click_validation_engine_params = {
            "engine_type": args.click_validation_provider,
            "model": args.click_validation_model,
            "base_url": args.click_validation_url or "",
            "api_key": args.click_validation_api_key or "",
        }

    # NEW!
    grounding_agent = OSWorldACI(
        platform="linux",
        engine_params_for_generation=engine_params,
        engine_params_for_grounding=engine_params_for_grounding,
        width=args.screen_width,
        height=args.screen_height,
        grounding_model_type=args.grounding_model_type,
        enable_click_validation=args.enable_click_validation,
        click_validation_engine_params=click_validation_engine_params,
        click_validation_max_retries=args.click_validation_max_retries,
    )

    # NEW!
    agent = Agent(
        engine_params,
        grounding_agent,
        platform="linux",
        action_space="pyautogui",
        observation_type=args.observation_type,
        planning_mode=args.planner_mode,
        # planner_hierarchical_depth=args.planner_hierarchical_depth,
        with_reflection=args.with_reflection,
        lexical_weight=args.lexical_weight,
        # memory_representation=args.memory_representation,
        # knowledge_storage=args.knowledge_storage,
        search_engine=args.search_engine,
        memory_root_path=os.getcwd(),
        memory_folder_name=args.kb_name,
        embedding_engine_type=args.embedding_engine_type,
        memory_type=args.memory_type,
    )

    env = DesktopEnv(
        path_to_vm=args.path_to_vm,
        action_space=agent.action_space,
        screen_size=(args.screen_width, args.screen_height),
        headless=args.headless,
        require_a11y_tree=args.observation_type
        in ["a11y_tree", "screenshot_a11y_tree", "som"],
    )

    for domain in tqdm(test_all_meta, desc="Domain"):
        for example_id in tqdm(test_all_meta[domain], desc="Example", leave=False):
            config_file = os.path.join(
                args.test_config_base_dir, f"examples/{domain}/{example_id}.json"
            )
            with open(config_file, "r", encoding="utf-8") as f:
                example = json.load(f)

            logger.info(f"[Domain]: {domain}")
            logger.info(f"[Example ID]: {example_id}")

            instruction = example["instruction"]

            logger.info(f"[Instruction]: {instruction}")
            # wandb each example config settings
            cfg_args["instruction"] = instruction
            cfg_args["start_time"] = datetime.datetime.now().strftime(
                "%Y:%m:%d-%H:%M:%S"
            )

            example_result_dir = os.path.join(
                args.result_dir,
                args.action_space,
                args.observation_type,
                args.model,
                domain,
                example_id,
            )
            os.makedirs(example_result_dir, exist_ok=True)
            # example start running
            try:
                lib_run_single_vlaa.run_single_example(
                    agent,
                    env,
                    example,
                    max_steps,
                    instruction,
                    args,
                    example_result_dir,
                    scores,
                )
            except Exception as e:
                logger.error(f"Exception in {domain}/{example_id}: {e}")
                env.controller.end_recording(
                    os.path.join(example_result_dir, "recording.mp4")
                )
                with open(os.path.join(example_result_dir, "traj.jsonl"), "a") as f:
                    f.write(
                        json.dumps(
                            {"Error": f"Time limit exceeded in {domain}/{example_id}"}
                        )
                    )
                    f.write("\n")

    env.close()
    logger.info(f"Average score: {sum(scores) / len(scores)}")


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
                        except Exception:
                            all_result.append(0.0)

    if not all_result:
        print("New experiment, no result yet.")
        return None
    else:
        print("Current Success Rate:", sum(all_result) / len(all_result) * 100, "%")
        return all_result


if __name__ == "__main__":
    ####### The complete version of the list of examples #######
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    args = config()
    # save args to json in result_dir/action_space/observation_type/model/args.json
    path_to_args = os.path.join(
        args.result_dir,
        args.action_space,
        args.observation_type,
        args.model,
        "args.json",
    )
    os.makedirs(os.path.dirname(path_to_args), exist_ok=True)
    with open(path_to_args, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=4)

    with open(args.test_all_meta_path, "r", encoding="utf-8") as f:
        test_all_meta = json.load(f)

    if args.domain != "all":
        test_all_meta = {args.domain: test_all_meta[args.domain]}

    test_file_list = get_unfinished(
        args.action_space,
        args.model,
        args.observation_type,
        args.result_dir,
        test_all_meta,
    )
    left_info = ""
    for domain in test_file_list:
        left_info += f"{domain}: {len(test_file_list[domain])}\n"
    logger.info(f"Left tasks:\n{left_info}")

    get_result(
        args.action_space,
        args.model,
        args.observation_type,
        args.result_dir,
        test_all_meta,
    )
    test(args, test_file_list)
