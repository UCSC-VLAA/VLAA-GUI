"""WAA-compatible adapter wrapping vlaa_gui's Agent.

This module provides ``ZooAgent``, a thin wrapper that translates between
WindowsAgentArena's expected agent interface and vlaa_gui's
:class:`~vlaa_gui.agent_core.agents.agent.Agent`.

WAA expects:
    predict(instruction, obs) -> (response, actions, logs, computer_update_args)

vlaa_gui provides:
    predict(instruction, obs) -> (info_dict, List[code_strings])
"""

import logging
import os
from io import BytesIO
from typing import Any, Dict, List, Tuple

import tomllib

from vlaa_gui.agent_core.agents.agent import Agent
from vlaa_gui.agent_core.agents.grounding import OSWorldACI

logger = logging.getLogger("desktopenv.agent.waa")


class ZooAgent:
    """WindowsAgentArena-compatible agent backed by vlaa_gui."""

    def __init__(
        self,
        model: str = "",
        som_origin: str = "oss",
        temperature: float = 1.0,
        top_p: float = 0.9,
        max_tokens: int = 1500,
        action_space: str = "pyautogui",
        observation_type: str = "screenshot",
        max_trajectory_length: int = 3,
        screen_width: int = 1920,
        screen_height: int = 1080,
        config_path: str = "config.toml",
    ):
        """Initialise ZooAgent from WAA-style arguments and a config.toml.

        Parameters
        ----------
        model : str
            Model name override. Falls back to ``config.toml`` value.
        som_origin : str
            SoM origin (unused by Zoo, kept for WAA compat).
        temperature, top_p, max_tokens : float / int
            Generation hyper-parameters (currently forwarded as-is).
        action_space : str
            Action space identifier (``pyautogui``).
        observation_type : str
            Observation type (``screenshot``, ``a11y_tree``, etc.).
        max_trajectory_length : int
            Max trajectory length forwarded to the inner agent.
        screen_width, screen_height : int
            VM screen dimensions.
        config_path : str
            Path to TOML configuration file.
        """
        # Load TOML config --------------------------------------------------
        if os.path.exists(config_path):
            with open(config_path, "rb") as f:
                cfg = tomllib.load(f)
        else:
            cfg = {}

        model_cfg = cfg.get("model", {})
        grounding_cfg = cfg.get("grounding", {})
        grounding_endpoint_cfg = cfg.get("grounding_endpoint", {})
        coding_cfg = cfg.get("coding", {})
        embedding_cfg = cfg.get("embedding", {})
        perception_cfg = cfg.get("perception", {})
        planning_cfg = cfg.get("planning", {})
        ctx_cfg = cfg.get("context_management", {})
        action_cfg = cfg.get("action_space", {})
        searcher_cfg = cfg.get("searcher", {})
        tts_cfg = cfg.get("tts", {})

        # Resolve model name (CLI arg takes precedence) ----------------------
        resolved_model = model or model_cfg.get("name", "gpt-4o")
        provider = model_cfg.get("provider", "openai")

        # Engine params for the main generation model -----------------------
        engine_params = {
            "engine_type": provider,
            "model": resolved_model,
            "base_url": model_cfg.get("url", ""),
            "api_key": model_cfg.get("api_key", ""),
            "project_id": model_cfg.get("project_id", ""),
            "region": model_cfg.get("region", ""),
            "regions": model_cfg.get("regions", []),
            "aws_keys": model_cfg.get("aws_keys", []),
            "temperature": model_cfg.get("temperature", temperature),
            "top_p": model_cfg.get("top_p", top_p),
        }

        # Engine params for grounding model ---------------------------------
        if grounding_endpoint_cfg.get("url"):
            engine_params_for_grounding = {
                "engine_type": grounding_endpoint_cfg.get("provider", ""),
                "base_url": grounding_endpoint_cfg.get("url", ""),
                "api_key": grounding_endpoint_cfg.get("api_key", ""),
                "temperature": grounding_cfg.get("temperature"),
                "top_p": grounding_cfg.get("top_p"),
            }
        else:
            grounding_width = grounding_cfg.get("grounding_width", screen_width)
            grounding_height = grounding_cfg.get("grounding_height")
            if grounding_height is None:
                grounding_height = screen_height * grounding_width / screen_width
            engine_params_for_grounding = {
                "engine_type": grounding_cfg.get("provider", provider),
                "base_url": grounding_cfg.get("url", ""),
                "api_key": grounding_cfg.get("api_key", ""),
                "model": grounding_cfg.get("grounding_model", resolved_model),
                "temperature": grounding_cfg.get("temperature"),
                "top_p": grounding_cfg.get("top_p"),
                "grounding_width": grounding_width,
                "grounding_height": grounding_height,
            }

        # Engine params for searcher ----------------------------------------
        engine_params_for_searcher = {
            "engine_type": searcher_cfg.get("provider", ""),
            "type": searcher_cfg.get("type", "llm"),
            "model": searcher_cfg.get("model", ""),
            "api_key": searcher_cfg.get("api_key", ""),
            "base_url": searcher_cfg.get("url", ""),
            "budget": searcher_cfg.get("budget", 5),
            "temperature": searcher_cfg.get("temperature"),
            "top_p": searcher_cfg.get("top_p"),
        }

        # Engine params for coding agent ------------------------------------
        engine_params_for_coding = {
            "engine_type": coding_cfg.get("provider", ""),
            "model": coding_cfg.get("name", ""),
            "base_url": coding_cfg.get("url", ""),
            "api_key": coding_cfg.get("api_key", ""),
            "api_keys": coding_cfg.get("api_keys", []),
            "temperature": coding_cfg.get("temperature"),
            "top_p": coding_cfg.get("top_p"),
            "thinking": coding_cfg.get("thinking", False),
            "thinking_budget": coding_cfg.get("thinking_budget"),
            "thinking_level": coding_cfg.get("thinking_level"),
            "include_thoughts": coding_cfg.get("include_thoughts", False),
        }

        # Resolved observation / planning config ----------------------------
        obs_type = perception_cfg.get("observation_type", observation_type)
        with_reflection = planning_cfg.get("with_reflection", False)
        search_engine = ctx_cfg.get("search_engine", None)
        if search_engine in ("", "None"):
            search_engine = None
        memory_type = ctx_cfg.get("memory_type", "null")
        lexical_weight = ctx_cfg.get("lexical_weight", 0.0)
        embedding_engine_type = embedding_cfg.get("engine_type", "openai")
        action_tts_num = tts_cfg.get("action_tts_num", 1)
        coding_agent_flag = action_cfg.get("engine", action_space) == "pyautogui_coding"

        # Build grounding agent (OSWorldACI works for remote VMs) -----------
        self.grounding_agent = OSWorldACI(
            platform="windows",
            engine_params_for_generation=engine_params,
            engine_params_for_grounding=engine_params_for_grounding,
            engine_params_for_searcher=engine_params_for_searcher,
            width=screen_width,
            height=screen_height,
            grounding_model_type=grounding_cfg.get("type", "single"),
            code_agent_engine_params=engine_params_for_coding,
        )

        # Build inner Agent -------------------------------------------------
        self.agent = Agent(
            engine_params=engine_params,
            grounding_agent=self.grounding_agent,
            platform="windows",
            with_reflection=with_reflection,
            action_space="pyautogui",
            observation_type=obs_type,
            search_engine=search_engine,
            memory_root_path=os.getcwd(),
            memory_folder_name="kb",
            memory_type=memory_type,
            embedding_engine_type=embedding_engine_type,
            coding_agent_flag=coding_agent_flag,
            action_tts_num=action_tts_num,
            lexical_weight=lexical_weight,
        )

        self.engine_params = engine_params
        self.max_tokens = max_tokens
        self.temperature = engine_params.get("temperature")
        self.top_p = engine_params.get("top_p")

    # ------------------------------------------------------------------
    # WAA interface
    # ------------------------------------------------------------------

    def predict(
        self,
        instruction: str,
        obs: Dict[str, Any],
    ) -> Tuple[str, List[str], Dict[str, Any], Dict[str, Any]]:
        """Generate the next action in WAA's expected return format.

        Parameters
        ----------
        instruction : str
            Natural-language task instruction.
        obs : dict
            WAA observation dict with keys like ``screenshot`` (BytesIO),
            ``accessibility_tree``, ``window_title``, ``window_rect``.

        Returns
        -------
        tuple
            ``(response_text, actions, logs, computer_update_args)``
        """
        # Convert WAA observation format -> vlaa_gui format
        zoo_obs = self._convert_obs(obs)

        # Call inner agent
        info, actions = self.agent.predict(instruction, zoo_obs)

        # Build WAA-style return values
        response_text = info.get("executor_plan", "")
        logs = info
        computer_update_args = {}

        return response_text, actions, logs, computer_update_args

    def reset(self) -> None:
        """Reset the inner agent's state between tasks."""
        self.agent.reset()

    def restart_after_failed_verification(
        self, reason: str = "", missing_steps: str = ""
    ) -> None:
        """Delegate to inner agent's restart after verifier rejection."""
        self.agent.restart_after_failed_verification(
            reason=reason, missing_steps=missing_steps
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _convert_obs(obs: Dict[str, Any]) -> Dict[str, Any]:
        """Translate WAA observation dict to vlaa_gui format.

        WAA provides ``screenshot`` as a :class:`~io.BytesIO` object;
        vlaa_gui expects raw ``bytes``.
        """
        zoo_obs: Dict[str, Any] = {}

        screenshot = obs.get("screenshot")
        if screenshot is not None:
            if isinstance(screenshot, BytesIO):
                zoo_obs["screenshot"] = screenshot.getvalue()
            elif isinstance(screenshot, bytes):
                zoo_obs["screenshot"] = screenshot
            else:
                # Fallback: try reading
                zoo_obs["screenshot"] = screenshot.read()

        a11y_tree = obs.get("accessibility_tree")
        if a11y_tree is not None:
            zoo_obs["accessibility_tree"] = (
                a11y_tree if isinstance(a11y_tree, str) else str(a11y_tree)
            )

        return zoo_obs
