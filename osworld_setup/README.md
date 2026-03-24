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
The OSWorld integration is fully CLI-driven. There are no TOML files in this path anymore.

For a local single-VM run:

```bash
uv run python osworld_setup/run_locally.py \
  --provider_name vmware \
  --path_to_vm /path/to/Ubuntu.vmx \
  --headless \
  --model gpt-5 \
  --model_provider openai \
  --grounding_model UI-TARS-1.5-7B \
  --grounding_model_provider vllm \
  --grounding_model_url http://127.0.0.1:8000/v1 \
  --result_dir ./results_local
```

For a multi-environment run, use the bash wrapper that parses the OSWorld-style arguments and forwards them to the Python runner:

```bash
bash osworld_setup/run_multienv_vlaa.sh \
  --provider_name aws \
  --region us-east-1 \
  --headless \
  --num_envs 10 \
  --max_steps 100 \
  --model gpt-5 \
  --model_provider openai \
  --grounding_model UI-TARS-1.5-7B \
  --grounding_model_provider vllm \
  --grounding_model_url http://127.0.0.1:8000/v1 \
  --result_dir ./results_multi
```

Key points:

- `--client_password` defaults to `osworld-public-evaluation` when `--provider_name aws`.
- `--model_dir_name` is optional; by default the result folder uses the stripped model name.
- All planner, grounding, coding, search, verifier, and memory settings are configured through CLI flags.
