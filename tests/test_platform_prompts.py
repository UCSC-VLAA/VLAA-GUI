from vlaa_gui.memory.procedural_memory import PROCEDURAL_MEMORY


class DummyAgent:
    def click(self, element_description: str):
        """Click on an element."""


DummyAgent.click.is_agent_action = True


def test_worker_prompt_linux_and_windows_match_legacy_output():
    legacy_prompt = PROCEDURAL_MEMORY.construct_worker_procedural_memory(
        DummyAgent,
        skipped_actions=[],
        observation_type="screenshot",
        planning_type="iterative",
    )

    linux_prompt = PROCEDURAL_MEMORY.construct_worker_procedural_memory(
        DummyAgent,
        skipped_actions=[],
        observation_type="screenshot",
        planning_type="iterative",
        platform="linux",
    )
    windows_prompt = PROCEDURAL_MEMORY.construct_worker_procedural_memory(
        DummyAgent,
        skipped_actions=[],
        observation_type="screenshot",
        planning_type="iterative",
        platform="windows",
    )

    assert linux_prompt == legacy_prompt
    assert windows_prompt == legacy_prompt


def test_worker_prompt_macos_gets_osworld_context():
    mac_prompt = PROCEDURAL_MEMORY.construct_worker_procedural_memory(
        DummyAgent,
        skipped_actions=[],
        observation_type="screenshot",
        planning_type="iterative",
        platform="darwin",
    )

    assert "OSWorld macOS environment" in mac_prompt
    assert "/Users/user" in mac_prompt
    assert "/home/user" not in mac_prompt
    assert "CURRENT_OS" not in mac_prompt


def test_search_prompt_macos_mentions_osworld():
    system_prompt = PROCEDURAL_MEMORY.construct_llm_searcher_system_prompt("darwin")
    user_prompt = PROCEDURAL_MEMORY.construct_llm_searcher_prompt(
        "How do I change the wallpaper?", "darwin"
    )

    assert "OSWorld" in system_prompt
    assert "/Users/user" in system_prompt
    assert "OSWorld" in user_prompt


def test_code_agent_prompt_macos_is_platform_aware():
    legacy_prompt = PROCEDURAL_MEMORY.construct_code_agent_prompt()
    linux_prompt = PROCEDURAL_MEMORY.construct_code_agent_prompt("linux")
    windows_prompt = PROCEDURAL_MEMORY.construct_code_agent_prompt("windows")
    mac_prompt = PROCEDURAL_MEMORY.construct_code_agent_prompt("darwin")

    assert linux_prompt == legacy_prompt
    assert windows_prompt == legacy_prompt
    assert "OSWorld macOS environment" in mac_prompt
    assert "/Users/user" in mac_prompt
    assert "/home/user" not in mac_prompt
