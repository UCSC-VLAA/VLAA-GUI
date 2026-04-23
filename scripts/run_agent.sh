#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${CONFIG_PATH:-"$ROOT_DIR/config/your-config.toml"}"

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Config not found: $CONFIG_PATH" >&2
  exit 1
fi

if [[ -z "${AWS_PROFILE:-}" && -z "${AWS_ACCESS_KEY_ID:-}" ]]; then
  cat >&2 <<'EOF'
No AWS credentials detected.

Set one of:
  export AWS_PROFILE=your-profile
or:
  export AWS_ACCESS_KEY_ID=...
  export AWS_SECRET_ACCESS_KEY=...
  export AWS_REGION=us-east-1
EOF
  exit 1
fi

cd "$ROOT_DIR"
exec uv run agent --config-path "$CONFIG_PATH" "$@"
