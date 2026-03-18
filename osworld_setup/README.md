# Evaluating in OSWorld
## Install vlaa_gui in osworld's environment
Assuming you have already set up the OSWorld environment.
```bash
# activate osworld's env
cd <vlaa_gui_dir >
uv pip install -e .
```

## Copying over Run Files

If you haven't already, please follow the [OSWorld environment setup](https://github.com/xlang-ai/OSWorld/blob/main/README.md). We've provided the relevant OSWorld run files for evaluation in this `osworld_setup` folder. Please copy this over to your OSWorld workspace.

## Run Full Tasks
To run the full set of tasks, simply run `uv run run_locally.py` if you wish to run on your local machine, or `uv run run_multienv_vlaa.py` if you wish to run on AWS provisioned instances.


> [!IMPORTANT]
> Make sure all parameters are set correctly in `config.toml` file, as well as the `run_locally.py` or `run_multienv_vlaa.py` files (not all parameters are read from `config.toml`)!