#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
SCRIPT_DIR="$ROOT_DIR/scripts"
ENV_FILE="$SCRIPT_DIR/llm_endpoint.env"
EXAMPLE_ENV_FILE="$SCRIPT_DIR/llm_endpoint.env.example"
LEGACY_ENV_FILES=(
  "$SCRIPT_DIR/label_matcher.env"
  "$SCRIPT_DIR/verifier.env"
)

if [[ -f "$ENV_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$ENV_FILE"
else
  for legacy_file in "${LEGACY_ENV_FILES[@]}"; do
    if [[ -f "$legacy_file" ]]; then
      echo "Using legacy env file: $legacy_file"
      # shellcheck disable=SC1090
      source "$legacy_file"
      break
    fi
  done

  if [[ ! -f "$ENV_FILE" && ! -f "${LEGACY_ENV_FILES[0]}" && ! -f "${LEGACY_ENV_FILES[1]}" ]]; then
    echo "Missing $ENV_FILE"
    echo "Create it from template:"
    echo "  cp \"$EXAMPLE_ENV_FILE\" \"$ENV_FILE\""
    echo "Then edit values and rerun."
    exit 1
  fi
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
LLM_ENDPOINT_HOST="${LLM_ENDPOINT_HOST:-${LABEL_MATCHER_HOST:-${VERIFIER_HOST:-127.0.0.1}}}"
LLM_ENDPOINT_PORT="${LLM_ENDPOINT_PORT:-${LABEL_MATCHER_PORT:-${VERIFIER_PORT:-8081}}}"

if [[ ! -d ".venv" ]]; then
  "$PYTHON_BIN" -m venv .venv
fi
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo "Starting llm_endpoint on ${LLM_ENDPOINT_HOST}:${LLM_ENDPOINT_PORT}..."
python -m uvicorn tools.llm_endpoint_service:app --host "$LLM_ENDPOINT_HOST" --port "$LLM_ENDPOINT_PORT"
