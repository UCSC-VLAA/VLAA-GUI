from vlaa_gui.agents.MacOSACI import MacOSACI
from vlaa_gui.agents.grounding import OSWorldACI
from vlaa_gui.run_agent import (
    _normalize_observation_type_for_platform,
    _select_grounding_agent_class,
)


def _make_macos_aci() -> MacOSACI:
    agent = MacOSACI.__new__(MacOSACI)
    agent.platform = "darwin"
    agent.coords1 = None
    agent.obs = None
    agent.resize_width = None
    agent.width = 1920
    agent.height = 1080
    agent.debug = False
    agent.resize_coordinates = lambda coords: tuple(coords)
    return agent


def test_run_agent_uses_macos_aci_on_darwin():
    assert _select_grounding_agent_class("darwin") is MacOSACI
    assert _select_grounding_agent_class("macos") is MacOSACI
    assert _select_grounding_agent_class("linux") is OSWorldACI


def test_macos_forces_non_screenshot_observation_modes():
    assert _normalize_observation_type_for_platform("darwin", "a11y_tree") == "screenshot"
    assert (
        _normalize_observation_type_for_platform("darwin", "screenshot_a11y_tree")
        == "screenshot"
    )
    assert _normalize_observation_type_for_platform("darwin", "som") == "screenshot"
    assert _normalize_observation_type_for_platform("darwin", "screenshot") == "screenshot"
    assert _normalize_observation_type_for_platform("linux", "a11y_tree") == "a11y_tree"


def test_macos_hotkey_normalizes_common_aliases():
    agent = _make_macos_aci()

    command = agent.hotkey(["cmd", "control", "alt", "return"])

    assert "'command'" in command
    assert "'ctrl'" in command
    assert "'option'" in command
    assert "'enter'" in command
    assert "'cmd'" not in command
    assert "'control'" not in command
    assert "'alt'" not in command
    assert "'return'" not in command


def test_macos_type_uses_pbcopy_and_command_shortcuts():
    agent = _make_macos_aci()

    command = agent.type(text="hello 世界", overwrite=True, enter=True)

    assert "pbcopy" in command
    assert "pyautogui.hotkey('command', 'a'" in command
    assert "pyautogui.hotkey('command', 'v'" in command
    assert "pyautogui.press('enter')" in command


def test_macos_open_and_switch_do_not_use_accessibility_or_spotlight():
    agent = _make_macos_aci()

    open_command = agent.open("Safari")
    switch_command = agent.switch_applications("Safari")

    assert "AXUIElement" not in open_command
    assert "command', 'space'" not in open_command
    assert "['open', '-a', target]" in open_command

    assert "AXUIElement" not in switch_command
    assert "command', 'space'" not in switch_command
    assert "osascript" in switch_command


def test_macos_hover_at_moves_to_grounded_coordinates():
    agent = _make_macos_aci()
    agent.generate_coords = lambda description, obs: (123, 456)

    command = agent.hover_at("the Safari toolbar")

    assert command == "import pyautogui; pyautogui.moveTo(123, 456)"


def test_macos_double_click_uses_two_clicks_and_normalized_modifiers():
    agent = _make_macos_aci()
    agent.coords1 = (50, 60)

    command = agent.double_click("the Safari toolbar", hold_keys=["cmd", "alt"])

    assert "pyautogui.keyDown('command')" in command
    assert "pyautogui.keyDown('option')" in command
    assert "pyautogui.click(50, 60, clicks=2, button='left')" in command
