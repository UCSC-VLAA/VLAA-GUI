from typing import List, Optional

from vlaa_gui.agents.grounding import OSWorldACI, agent_action


def _normalize_macos_key(key: str) -> str:
    normalized = str(key).strip().lower()
    aliases = {
        "cmd": "command",
        "command": "command",
        "control": "ctrl",
        "ctrl": "ctrl",
        "opt": "option",
        "alt": "option",
        "option": "option",
        "return": "enter",
    }
    return aliases.get(normalized, normalized)


def _normalize_macos_keys(keys: Optional[List[str]]) -> List[str]:
    return [_normalize_macos_key(key) for key in (keys or [])]


class MacOSACI(OSWorldACI):
    """macOS grounding agent that relies on screenshot grounding only."""

    def __init__(self, **kwargs):
        kwargs["platform"] = "darwin"
        super().__init__(**kwargs)

    @agent_action
    def open(self, app_or_filename: str):
        """Open any application or file with name app_or_filename."""
        return (
            "import subprocess\n"
            f"target = {repr(app_or_filename)}\n"
            "result = subprocess.run(['open', '-a', target], check=False)\n"
            "if result.returncode != 0:\n"
            "    subprocess.run(['open', target], check=False)\n"
        )

    @agent_action
    def switch_applications(self, app_code: str):
        """Switch to a different application that is already open."""
        return (
            "import subprocess\n"
            f"app_name = {repr(app_code)}\n"
            "script = f'tell application \"{app_name}\" to activate'\n"
            "result = subprocess.run(['osascript', '-e', script], check=False)\n"
            "if result.returncode != 0:\n"
            "    subprocess.run(['open', '-a', app_name], check=False)\n"
        )

    @agent_action
    def click(
        self,
        element_description: str,
        num_clicks: int = 1,
        button_type: str = "left",
        hold_keys: List = [],
    ):
        """Click on the element."""
        return super().click(
            element_description=element_description,
            num_clicks=num_clicks,
            button_type=button_type,
            hold_keys=_normalize_macos_keys(hold_keys),
        )

    @agent_action
    def hover_at(self, element_description: str):
        """Move the cursor to the element without clicking."""
        coords1 = self.coords1
        if coords1 is None:
            coords1 = self.generate_coords(element_description, self.obs)
        x, y = self.resize_coordinates(coords1)
        return f"import pyautogui; pyautogui.moveTo({x}, {y})"

    @agent_action
    def double_click(
        self,
        element_description: str,
        hold_keys: List = [],
    ):
        """Double-click on the element."""
        return self.click(
            element_description=element_description,
            num_clicks=2,
            button_type="left",
            hold_keys=hold_keys,
        )

    @agent_action
    def type(
        self,
        element_description: Optional[str] = None,
        text: str = "",
        overwrite: bool = False,
        enter: bool = False,
    ):
        """Type text into an element using macOS-native clipboard paste."""
        command = "import pyautogui; import subprocess; "

        if element_description is not None:
            coords1 = self.coords1
            if coords1 is None:
                coords1 = self.generate_coords(element_description, self.obs)
            x, y = self.resize_coordinates(coords1)
            command += f"pyautogui.click({x}, {y}); "

        if overwrite:
            command += (
                "pyautogui.hotkey('command', 'a', interval=0.5); "
                "pyautogui.press('backspace'); "
            )

        if text:
            command += (
                f"subprocess.run(['pbcopy'], input={repr(text.encode('utf-8'))}, check=True); "
                "pyautogui.hotkey('command', 'v', interval=0.5); "
            )

        if enter:
            command += "pyautogui.press('enter'); "

        return command

    @agent_action
    def drag_and_drop(
        self, starting_description: str, ending_description: str, hold_keys: List = []
    ):
        """Drag from the starting description to the ending description."""
        return super().drag_and_drop(
            starting_description=starting_description,
            ending_description=ending_description,
            hold_keys=_normalize_macos_keys(hold_keys),
        )

    @agent_action
    def hotkey(self, keys: List):
        """Press a hotkey combination with macOS key aliases normalized."""
        normalized_keys = [f"'{key}'" for key in _normalize_macos_keys(keys)]
        return (
            "import pyautogui; "
            f"pyautogui.hotkey({', '.join(normalized_keys)}, interval=0.5)"
        )

    @agent_action
    def hold_and_press(self, hold_keys: List, press_keys: List):
        """Hold a list of keys and press a list of keys."""
        hold_keys = _normalize_macos_keys(hold_keys)
        press_keys = _normalize_macos_keys(press_keys)

        press_keys_str = "[" + ", ".join([f"'{key}'" for key in press_keys]) + "]"
        command = "import pyautogui; "
        for key in hold_keys:
            command += f"pyautogui.keyDown({repr(key)}); "
        command += f"pyautogui.press({press_keys_str}); "
        for key in hold_keys:
            command += f"pyautogui.keyUp({repr(key)}); "

        return command
