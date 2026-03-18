import os
import subprocess
import sys
from typing import Dict, Optional


_TRUTHY = {"1", "true", "yes", "y", "on"}


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in _TRUTHY


class LocalController:
    """Minimal controller to execute bash and python code locally.

    WARNING: Executing arbitrary code is dangerous. Only enable/use this in trusted
    environments and with trusted inputs.
    """

    def __init__(self, echo: Optional[bool] = None):
        if echo is None:
            echo = _env_flag("vlaa_gui_LOCAL_ENV_ECHO", default=False) or _env_flag(
                "vlaa_gui_CODE_AGENT_ECHO", default=False
            )
        self.echo = bool(echo)

    def run_bash_script(self, code: str, timeout: int = 30) -> Dict:
        try:
            proc = subprocess.run(
                ["/bin/bash", "-lc", code],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            output = (proc.stdout or "") + (proc.stderr or "")

            if self.echo:
                print("BASH OUTPUT =======================================")
                print(output)
                print("BASH OUTPUT =======================================")

            return {
                "status": "ok" if proc.returncode == 0 else "error",
                "returncode": proc.returncode,
                "output": output,
                "error": "",
            }
        except subprocess.TimeoutExpired as e:
            return {
                "status": "error",
                "returncode": -1,
                "output": e.stdout or "",
                "error": f"TimeoutExpired: {str(e)}",
            }
        except Exception as e:
            return {
                "status": "error",
                "returncode": -1,
                "output": "",
                "error": str(e),
            }

    def run_python_script(self, code: str) -> Dict:
        try:
            proc = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True,
                text=True,
            )
            if self.echo:
                print("PYTHON OUTPUT =======================================")
                print(proc.stdout or "")
                print("PYTHON OUTPUT =======================================")
            return {
                "status": "ok" if proc.returncode == 0 else "error",
                "return_code": proc.returncode,
                "output": proc.stdout or "",
                "error": proc.stderr or "",
            }
        except Exception as e:
            return {
                "status": "error",
                "return_code": -1,
                "output": "",
                "error": str(e),
            }


class LocalEnv:
    """Simple environment that provides a controller compatible with CodeAgent."""

    def __init__(self):
        self.controller = LocalController()
