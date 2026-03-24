#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_SCRIPT="osworld_setup/run_multienv_vlaa.py"

usage() {
  cat <<'EOF'
Usage:
  bash osworld_setup/run_multienv_vlaa.sh [options]

Core options:
  --model VALUE
  --model_provider VALUE
  --grounding_model VALUE
  --grounding_model_provider VALUE
  --result_dir VALUE
  --provider_name VALUE
  --region VALUE
  --num_envs VALUE
  --max_steps VALUE
  --test_all_meta_path VALUE
  --client_password VALUE
  --headless

The wrapper accepts the full explicit VLAA/OSWorld argument surface and
normalizes dashed flags to the underscore form expected by the Python runner.
Use --help to print this message.
EOF
}

need_value() {
  local opt="$1"
  if [[ $# -lt 2 || -z "${2:-}" ]]; then
    echo "Missing value for ${opt}" >&2
    exit 1
  fi
}

normalize_opt() {
  local raw="$1"
  local trimmed="${raw#--}"
  printf -- '--%s' "${trimmed//-/_}"
}

PY_ARGS=()

while [[ $# -gt 0 ]]; do
  raw_opt="$1"
  norm_opt="$(normalize_opt "$raw_opt")"

  case "$norm_opt" in
    --help)
      usage
      exit 0
      ;;
    --headless|--debug|--model_thinking|--model_include_thoughts|--grounding_thinking|--grounding_include_thoughts|--enable_zoom_grounding|--coding_model_thinking|--coding_model_include_thoughts|--with_reflection|--reflection_thinking|--no_reflection_thinking|--use_recon|--enable_gate|--loop_detection|--feasibility_check|--use_verifier)
      PY_ARGS+=("$norm_opt")
      shift
      ;;
    --result_dir|--num_envs|--provider_name|--region|--client_password|--path_to_vm|--screen_width|--screen_height|--sleep_after_execution|--max_steps|--max_trajectory_length|--domain|--test_config_base_dir|--test_all_meta_path|--model_provider|--model|--model_dir_name|--model_url|--model_api_key|--model_api_keys|--model_temperature|--model_top_p|--max_tokens|--model_project_id|--model_region|--model_aws_keys|--api_version|--model_thinking_level|--model_thinking_budget|--model_reasoning_effort|--grounding_model_provider|--grounding_model|--grounding_model_url|--grounding_model_api_key|--grounding_width|--grounding_height|--resize_width|--grounding_model_region|--zoom_grounding_crop_ratio|--grounding_model_type|--grounding_temperature|--grounding_top_p|--grounding_thinking_budget|--grounding_thinking_level|--grounding_reasoning_effort|--endpoint_provider|--endpoint_url|--endpoint_api_key|--coding_model_provider|--coding_model|--coding_model_url|--coding_model_api_key|--coding_model_api_keys|--coding_model_temperature|--coding_model_top_p|--coding_model_thinking_budget|--coding_model_thinking_level|--embedding_engine_type|--observation_type|--planner_hierarchical_depth|--search_engine|--kb_name|--memory_type|--memory_representation|--knowledge_storage|--lexical_weight|--searcher_type|--searcher_provider|--searcher_model|--searcher_api_key|--searcher_api_keys|--searcher_url|--searcher_budget|--searcher_temperature|--searcher_top_p|--searcher_reasoning_effort|--action_space|--action_tts_num)
      need_value "$raw_opt" "${2:-}"
      PY_ARGS+=("$norm_opt" "$2")
      shift 2
      ;;
    --)
      shift
      while [[ $# -gt 0 ]]; do
        PY_ARGS+=("$1")
        shift
      done
      ;;
    *)
      echo "Unknown option: $raw_opt" >&2
      usage >&2
      exit 1
      ;;
  esac
done

cd "$ROOT_DIR"
exec python "$PYTHON_SCRIPT" "${PY_ARGS[@]}"
