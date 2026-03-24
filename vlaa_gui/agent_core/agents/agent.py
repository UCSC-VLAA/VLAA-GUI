import json
import logging
import os
import platform
from typing import Dict, List, Optional, Tuple

from vlaa_gui.agent_core.agents.grounding import ACI
from vlaa_gui.agent_core.agents.worker import Worker
from vlaa_gui.agent_core.agents.manager import Manager
from vlaa_gui.agent_core.utils.common_utils import Node
from vlaa_gui.agent_core.core.engine import (
    OpenAIEmbeddingEngine,
    GeminiEmbeddingEngine,
    AzureOpenAIEmbeddingEngine,
    ArkEmbeddingEngine,
    QwenEmbeddingEngine,
)

logger = logging.getLogger("desktopenv.agent")


class UIAgent:
    """Base class for UI automation agents"""

    def __init__(
        self,
        engine_params: Dict,
        grounding_agent: ACI,
        platform: str = platform.system().lower(),
        action_space: str = "pyautogui",
        observation_type: str = "screenshot",
        search_engine: str = "perplexica",
    ):
        """Initialize UIAgent

        Args:
            engine_params: Configuration parameters for the LLM engine
            grounding_agent: Instance of ACI class for UI interaction
            platform: Operating system platform (macos, linux, windows)
            action_space: Type of action space to use (pyautogui, aci)
            observation_type: Type of observations to use (a11y_tree, mixed)
            engine: Search engine to use (perplexica, LLM)
        """
        self.engine_params = engine_params
        self.grounding_agent = grounding_agent
        self.platform = platform
        self.action_space = action_space
        self.observation_type = observation_type
        self.engine = search_engine

    def reset(self) -> None:
        """Reset agent state"""
        pass

    def predict(self, instruction: str, observation: Dict) -> Tuple[Dict, List[str]]:
        """Generate next action prediction

        Args:
            instruction: Natural language instruction
            observation: Current UI state observation

        Returns:
            Tuple containing agent info dictionary and list of actions
        """
        pass

    def update_narrative_memory(self, trajectory: str) -> None:
        """Update narrative memory with task trajectory

        Args:
            trajectory: String containing task execution trajectory
        """
        pass

    def update_episodic_memory(self, meta_data: Dict, subtask_trajectory: str) -> str:
        """Update episodic memory with subtask trajectory

        Args:
            meta_data: Metadata about current subtask execution
            subtask_trajectory: String containing subtask execution trajectory

        Returns:
            Updated subtask trajectory
        """
        pass


class Agent(UIAgent):
    """Agent that uses hierarchical planning and directed acyclic graph modeling for UI automation"""

    def __init__(
        self,
        engine_params: Dict,
        grounding_agent: ACI,
        platform: str = platform.system().lower(),
        planning_mode: str = "proactive",
        with_reflection: bool = True,
        action_space: str = "pyautogui",
        observation_type: str = "screenshot",
        search_engine: Optional[str] = None,
        pricing_config_path: Optional[str] = None,
        reflection_engine_params: Optional[Dict] = None,
        memory_root_path: str = os.getcwd(),
        memory_folder_name: str = "agent_memory",
        memory_type: str = "null",
        kb_release_tag: str = "v0.2.2",
        embedding_engine_type: str = "openai",
        embedding_engine_params: Dict = {},
        coding_agent_flag: bool = False,
        action_tts_num: int = 1,
        debug: bool = False,
        lexical_weight: float = 0.0,
        enable_gate: bool = True,
        loop_detection: bool = True,
        feasibility_check: bool = True,
    ):
        """Initialize Agent

        Args:
            engine_params: Configuration parameters for the LLM engine
            grounding_agent: Instance of ACI class for UI interaction
            platform: Operating system platform (darwin, linux, windows)
            planning_mode: Planning strategy. "proactive" replans after every subtask, "reactive" replans only on failure. Defaults to "proactive".
            action_space: Type of action space to use (pyautogui, other)
            observation_type: Type of observations to use (a11y_tree, screenshot, mixed)
            search_engine: Search engine to use (LLM, perplexica)
            memory_root_path: Path to memory directory. Defaults to current working directory.
            memory_folder_name: Name of memory folder. Defaults to "agent_memory".
            kb_release_tag: Release tag for knowledge base. Defaults to "v0.2.2".
            embedding_engine_type: Embedding engine to use for knowledge base. Defaults to "openai". Supports "openai" and "gemini".
            embedding_engine_params: Parameters for embedding engine. Defaults to {}.
            debug: bool = False,
        """
        super().__init__(
            engine_params,
            grounding_agent,
            platform,
            action_space,
            observation_type,
            search_engine,
        )

        self.planning_mode = planning_mode
        self.with_reflection = with_reflection
        self.pricing_config_path = pricing_config_path
        self.reflection_engine_params = reflection_engine_params or engine_params
        self.memory_root_path = memory_root_path
        self.memory_folder_name = memory_folder_name
        self.lexical_weight = lexical_weight
        self.kb_release_tag = kb_release_tag
        self.memory_type = memory_type
        self.local_kb_path = os.path.join(
            self.memory_root_path, self.memory_folder_name
        )

        self.coding_agent_flag = coding_agent_flag
        self.action_tts_num = action_tts_num
        self.enable_gate = enable_gate
        self.loop_detection = loop_detection
        self.feasibility_check = feasibility_check

        if embedding_engine_type == "openai":
            self.embedding_engine = OpenAIEmbeddingEngine(**embedding_engine_params)
        elif embedding_engine_type == "gemini":
            self.embedding_engine = GeminiEmbeddingEngine(**embedding_engine_params)
        elif embedding_engine_type == "azure":
            self.embedding_engine = AzureOpenAIEmbeddingEngine(
                **embedding_engine_params
            )
        elif embedding_engine_type == "ark":
            self.embedding_engine = ArkEmbeddingEngine(**embedding_engine_params)
        elif embedding_engine_type == "qwen":
            self.embedding_engine = QwenEmbeddingEngine(**embedding_engine_params)
        else:
            raise ValueError(
                f"Unsupported embedding engine type: {embedding_engine_type}"
            )
        self.debug = debug
        self.reset()

    def reset(self) -> None:
        """Reset agent state and initialize components"""
        # Initialize core components
        self.planner = Manager(
            engine_params=self.engine_params,
            grounding_agent=self.grounding_agent,
            local_kb_path=self.local_kb_path,
            pricing_config_path=self.pricing_config_path,
            embedding_engine=self.embedding_engine,
            search_engine=self.engine,
            platform=self.platform,
            observation_type=self.observation_type,
            memory_type=self.memory_type,
            debug=self.debug,
            lexical_weight=self.lexical_weight,
        )
        self.executor = Worker(
            engine_params=self.engine_params,
            grounding_agent=self.grounding_agent,
            pricing_config_path=self.pricing_config_path,
            reflection_engine_params=self.reflection_engine_params,
            local_kb_path=self.local_kb_path,
            embedding_engine=self.embedding_engine,
            platform=self.platform,
            enable_reflection=self.with_reflection,
            observation_type=self.observation_type,
            planning_type=self.planning_mode,
            search_engine=self.engine,
            memory_type=self.memory_type,
            use_task_experience=self.memory_type == "mixed",
            coding_agent_flag=self.coding_agent_flag,
            action_tts_num=self.action_tts_num,
            debug=self.debug,
            lexical_weight=self.lexical_weight,
            enable_gate=self.enable_gate,
            loop_detection=self.loop_detection,
            feasibility_check=self.feasibility_check,
        )

        # Reset state variables
        self.requires_replan: bool = True
        self.needs_next_subtask: bool = True
        self.step_count: int = 0
        self.turn_count: int = 0
        self.failure_subtask: Optional[Node] = None
        self.should_send_action: bool = False
        self.completed_tasks: List[Node] = []
        self.current_subtask: Optional[Node] = None
        self.subtasks: List[Node] = []
        self.search_query: str = ""
        self.subtask_status: str = "Start"  # Start, In, Done
        self.verifier_feedback: Optional[Dict] = None

    def reset_executor_state(self) -> None:
        """Reset executor and step counter"""
        self.executor.reset()
        self.step_count = 0

    def restart_after_failed_verification(
        self, reason: str = "", missing_steps: str = ""
    ) -> None:
        """Force a fresh plan when the verifier rejects completion."""
        logger.info("Verifier rejected DONE. Forcing replan from scratch.")
        self.verifier_feedback = {
            "reason": reason,
            "missing_steps": missing_steps,
        }
        self.requires_replan = True
        self.needs_next_subtask = True
        self.failure_subtask = None
        self.current_subtask = None
        self.subtasks = []
        # Preserve already completed subtasks so replanning does not redo work.
        self.completed_tasks = [
            task for task in self.completed_tasks if task is not None
        ]
        self.subtask_status = "Start"
        self.should_send_action = False
        self.search_query = ""
        self.reset_executor_state()


    def predict(self, instruction: str, observation: Dict) -> Tuple[Dict, List[str]]:
        if self.planning_mode == "iterative":
            return self.executor.generate_next_action_iteratively(
                observation, instruction
            )

        # Initialize the three info dictionaries
        planner_info = {}
        executor_info = {}
        evaluator_info = {
            "obs_evaluator_response": "",
            "num_input_tokens_evaluator": 0,
            "num_output_tokens_evaluator": 0,
            "evaluator_cost": 0.0,
        }
        actions = []

        # If the DONE response by the executor is for a subtask, then the agent should continue with the next subtask without sending the action to the environment
        while not self.should_send_action:
            self.subtask_status = "In"
            _verifier_feedback = None
            # If replan is true, generate a new plan. True at start, after a failed plan, or after subtask completion
            if self.requires_replan:
                logger.info("(RE)PLANNING...")
                _verifier_feedback = self.verifier_feedback
                self.verifier_feedback = None  # clear after use

                self.requires_replan = False
                if "search_query" in planner_info:
                    self.search_query = planner_info["search_query"]
                else:
                    self.search_query = ""

            self.step_count += 1

            # set the should_send_action flag to True if the executor returns an action
            self.should_send_action = True

            # replan on failure (common for both proactive and reactive modes)
            if "FAIL" in actions:
                self.requires_replan = True
                self.needs_next_subtask = True
                self.failure_subtask = self.current_subtask
                self.reset_executor_state()

                # if more subtasks are remaining, we don't want to send DONE to the environment but move on to the next subtask
                if self.subtasks:
                    self.should_send_action = False

            # on subtask completion, behavior depends on the planning mode
            # replan on subtask completion
            elif "DONE" in actions:
                if self.planning_mode == "proactive":
                    self.requires_replan = True
                elif self.planning_mode == "reactive":
                    self.requires_replan = False
                self.needs_next_subtask = True
                self.failure_subtask = None
                self.completed_tasks.append(self.current_subtask)

                # reset the step count, executor, and evaluator
                self.reset_executor_state()

                # if more subtasks are remaining, we don't want to send DONE to the environment but move on to the next subtask
                if self.subtasks:
                    self.should_send_action = False
                self.subtask_status = "Done"

            self.turn_count += 1

        # reset the should_send_action flag for next iteration
        self.should_send_action = False

        # concatenate the three info dictionaries
        info = {
            **{
                k: v
                for d in [planner_info or {}, executor_info or {}, evaluator_info or {}]
                for k, v in d.items()
            }
        }
        info.update(
            {
                "subtask": self.current_subtask.name if self.current_subtask else "N/A",
                "subtask_info": self.current_subtask.info
                if self.current_subtask
                else "N/A",
                "subtask_status": self.subtask_status,
            }
        )

        return info, actions

    # TODO: context management: support partial updates to narrative memory
    def update_narrative_memory(self, trajectory: str) -> None:
        """Update narrative memory from task trajectory

        Args:
            trajectory: String containing task execution trajectory
        """
        try:
            reflection_path = os.path.join(
                self.local_kb_path, self.platform, "narrative_memory.json"
            )
            try:
                reflections = json.load(open(reflection_path))
            except Exception:
                reflections = {}

            if self.search_query not in reflections:
                reflection = self.planner.summarize_narrative(trajectory)
                reflections[self.search_query] = reflection

            with open(reflection_path, "w") as f:
                json.dump(reflections, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to update narrative memory: {e}")

    # TODO: context management: support partial episodic memory updates
    def update_episodic_memory(self, meta_data: Dict, subtask_trajectory: str) -> str:
        """Update episodic memory from subtask trajectory

        Args:
            meta_data: Metadata about current subtask execution
            subtask_trajectory: String containing subtask execution trajectory

        Returns:
            Updated subtask trajectory
        """
        subtask = meta_data["subtask"]
        subtask_info = meta_data["subtask_info"]
        subtask_status = meta_data["subtask_status"]
        # Handle subtask trajectory
        if subtask_status == "Start" or subtask_status == "Done":
            # If it's a new subtask start, finalize the previous subtask trajectory if it exists
            if subtask_trajectory:
                subtask_trajectory += "\nSubtask Completed.\n"
                subtask_key = subtask_trajectory.split(
                    "\n----------------------\n\nPlan:\n"
                )[0]
                try:
                    subtask_path = os.path.join(
                        self.local_kb_path, self.platform, "episodic_memory.json"
                    )
                    kb = json.load(open(subtask_path))
                except Exception as e:
                    logger.error(f"Failed to load episodic memory: {e}")
                    kb = {}
                if subtask_key not in kb.keys():
                    subtask_summarization = self.planner.summarize_episode(
                        subtask_trajectory
                    )
                    kb[subtask_key] = subtask_summarization
                else:
                    subtask_summarization = kb[subtask_key]
                logger.info("subtask_key: %s", subtask_key)
                logger.info("subtask_summarization: %s", subtask_summarization)

                ## create the episodic memory folder if it doesn't exist
                os.makedirs(os.path.dirname(subtask_path), exist_ok=True)
                with open(subtask_path, "w") as fout:
                    json.dump(kb, fout, indent=2)
                # Reset for the next subtask
                subtask_trajectory = ""
            # Start a new subtask trajectory
            subtask_trajectory = (
                "Task:\n"
                + self.search_query
                + "\n\nSubtask: "
                + subtask
                + "\nSubtask Instruction: "
                + subtask_info
                + "\n----------------------\n\nPlan:\n"
                + meta_data["executor_plan"]
                + "\n"
            )
        elif subtask_status == "In":
            # Continue appending to the current subtask trajectory if it's still ongoing
            subtask_trajectory += (
                "\n----------------------\n\nPlan:\n"
                + meta_data["executor_plan"]
                + "\n"
            )

        return subtask_trajectory
