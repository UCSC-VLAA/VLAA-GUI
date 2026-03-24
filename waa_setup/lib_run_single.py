"""Single-task execution loop for WindowsAgentArena.

Adapted from ``osworld_setup/lib_run_single_vlaa.py`` for WAA's environment
and agent interface.  Key differences from OSWorld:
- WAA's ``ZooAgent.predict`` returns a 4-tuple
  ``(response, actions, logs, computer_update_args)``
- ``computer_update_args`` can carry VM rendering state updates
"""

import datetime
import json
import logging
import os
import time
from typing import Optional

from vlaa_gui.agent_core.utils.token_tracker import reset_global_tracker

logger = logging.getLogger("desktopenv.agent")
logger.setLevel(logging.INFO)


def run_single_example(
    agent,
    env,
    example: dict,
    max_steps: int,
    instruction: str,
    args,
    example_result_dir: str,
    scores: list,
    verifier_agent=None,
    shared_token_summaries: Optional[list] = None,
):
    """Execute a single WAA task.

    Parameters
    ----------
    agent : ZooAgent
        The WAA-compatible agent wrapper.
    env : DesktopEnv
        WAA desktop environment instance.
    example : dict
        Task configuration dict (with at least ``id``, ``instruction``).
    max_steps : int
        Maximum number of interaction steps.
    instruction : str
        Natural-language instruction for the task.
    args : argparse.Namespace
        Global CLI arguments.
    example_result_dir : str
        Directory to write per-task outputs (screenshots, trajectories, result).
    scores : list
        Shared list to append the final evaluation score to.
    verifier_agent : VerifierAgent, optional
        If provided, verifies DONE claims before accepting task completion.
    shared_token_summaries : list, optional
        If provided, token usage summaries are appended here.
    """
    example_id = example.get("id", "unknown")
    token_tracker = reset_global_tracker(
        task_id=f"{example_id}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    token_tracker.set_task_instruction(instruction)

    runtime_file_handler = _setup_task_file_handler(example_result_dir)
    logger.addHandler(runtime_file_handler)

    try:
        agent.reset()

        env.reset(task_config=example)
        time.sleep(5)
        obs = env._get_obs()

        # Save initial screenshot
        screenshot_bytes = _extract_screenshot_bytes(obs)
        if screenshot_bytes:
            with open(os.path.join(example_result_dir, "step_0.png"), "wb") as fout:
                fout.write(screenshot_bytes)

        with open(
            os.path.join(example_result_dir, "instruction.txt"),
            "w",
            encoding="utf-8",
        ) as f:
            f.write(instruction)

        done = False
        step_idx = 0
        traj_lines = [f"Task: {instruction}"]

        while not done and step_idx < max_steps:
            response_text, actions, logs, computer_update_args = agent.predict(
                instruction, obs
            )

            # Handle terminal / special actions
            is_done = actions and "done" in actions[0].lower()
            is_fail = actions and "fail" in actions[0].lower()

            if is_done:
                if verifier_agent:
                    # Convert obs for verifier (needs bytes, not BytesIO)
                    verifier_obs = dict(obs)
                    screenshot_bytes = _extract_screenshot_bytes(obs)
                    if screenshot_bytes is not None:
                        verifier_obs["screenshot"] = screenshot_bytes
                    verdict = verifier_agent.verify_completion(
                        instruction=instruction,
                        observation=verifier_obs,
                        trajectory="\n".join(traj_lines),
                        executor_plan=logs.get("executor_plan"),
                    )
                    if verdict.get("complete"):
                        logger.info("Verifier accepted completion.")
                        break

                    logger.info(
                        "Verifier rejected completion: %s | Missing: %s",
                        verdict.get("reason"),
                        verdict.get("missing_steps"),
                    )
                    traj_lines.append(
                        f"Verifier rejected DONE: reason={verdict.get('reason')}, "
                        f"missing_steps={verdict.get('missing_steps')}"
                    )
                    agent.restart_after_failed_verification(
                        reason=verdict.get("reason", ""),
                        missing_steps=verdict.get("missing_steps", ""),
                    )
                    continue
                else:
                    logger.info("Agent signalled DONE.")
                    break

            if is_fail:
                logger.info("Agent signalled FAIL.")
                break

            if actions and "next" in actions[0].lower():
                continue

            if actions and "wait" in actions[0].lower():
                time.sleep(5)
                continue

            # Execute each action
            for action in actions:
                action_timestamp = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")
                logger.info("Step %d: %s", step_idx + 1, action)
                traj_lines.append(f"Step {step_idx + 1}: {action}")

                obs, reward, done, info = env.step(action, args.sleep_after_execution)

                # Apply computer_update_args if WAA env supports it
                if computer_update_args:
                    try:
                        env.update_computer(**computer_update_args)
                    except (AttributeError, TypeError):
                        pass

                logger.info("Reward: %.2f", reward)
                logger.info("Done: %s", done)
                info_str = str(info)
                if len(info_str) > 500:
                    info_str = info_str[:500] + "..."
                traj_lines.append(
                    f"Result: reward={reward:.2f}, done={done}, info={info_str}"
                )

                # Save screenshot
                screenshot_bytes = _extract_screenshot_bytes(obs)
                if screenshot_bytes:
                    with open(
                        os.path.join(
                            example_result_dir,
                            f"step_{step_idx + 1}_{action_timestamp}.png",
                        ),
                        "wb",
                    ) as fout:
                        fout.write(screenshot_bytes)

                # Append trajectory entry
                traj_entry = {
                    "step_num": step_idx + 1,
                    "action_timestamp": action_timestamp,
                    "action": action,
                    "reward": reward,
                    "done": done,
                    "info": info,
                    "response": response_text,
                    "screenshot_file": f"step_{step_idx + 1}_{action_timestamp}.png",
                }
                with open(
                    os.path.join(example_result_dir, "traj.jsonl"),
                    "a",
                    encoding="utf-8",
                ) as f:
                    f.write(json.dumps(traj_entry, ensure_ascii=False))
                    f.write("\n")

                if done:
                    logger.info("The episode is done.")
                    break

            step_idx += 1

        # Evaluate
        time.sleep(5)
        result = env.evaluate()
        logger.info("Result: %.2f", result)
        scores.append(result)
        with open(
            os.path.join(example_result_dir, "result.txt"),
            "w",
            encoding="utf-8",
        ) as f:
            f.write(f"{result}\n")

    finally:
        logger.removeHandler(runtime_file_handler)
        runtime_file_handler.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_screenshot_bytes(obs: dict) -> Optional[bytes]:
    """Extract raw screenshot bytes from an observation dict.

    WAA may provide the screenshot as BytesIO or bytes.
    """
    from io import BytesIO

    screenshot = obs.get("screenshot")
    if screenshot is None:
        return None
    if isinstance(screenshot, bytes):
        return screenshot
    if isinstance(screenshot, BytesIO):
        return screenshot.getvalue()
    try:
        return screenshot.read()
    except Exception:
        return None


def _setup_task_file_handler(example_result_dir: str) -> logging.FileHandler:
    """Create a file handler for task-specific runtime logging."""
    handler = logging.FileHandler(
        os.path.join(example_result_dir, "runtime.log"), encoding="utf-8"
    )
    handler.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        fmt="[%(asctime)s %(levelname)s %(module)s/%(lineno)d-%(processName)s] %(message)s"
    )
    handler.setFormatter(fmt)
    return handler
