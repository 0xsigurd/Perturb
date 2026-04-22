#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
SCRIPT_DIR="$ROOT_DIR/scripts"
ENV_FILE="$SCRIPT_DIR/validator.env"
EXAMPLE_ENV_FILE="$SCRIPT_DIR/validator.env.example"

if [[ ! -f "$ENV_FILE" ]]; then
  if [[ -f "$EXAMPLE_ENV_FILE" ]]; then
    echo "Missing $ENV_FILE"
    echo "Create it from template:"
    echo "  cp \"$EXAMPLE_ENV_FILE\" \"$ENV_FILE\""
    echo "Then edit wallet/network values and run this command again."
    exit 1
  fi
fi

if [[ -f "$ENV_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$ENV_FILE"
fi

WALLET_NAME="${WALLET_NAME:-}"
WALLET_HOTKEY="${WALLET_HOTKEY:-}"
NETUID="${NETUID:-1}"
NETWORK="${NETWORK:-local}"
VALIDATOR_EXTRA_ARGS="${VALIDATOR_EXTRA_ARGS:-}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ -z "$WALLET_NAME" || -z "$WALLET_HOTKEY" ]]; then
  echo "WALLET_NAME and WALLET_HOTKEY must be set in $ENV_FILE"
  exit 1
fi

if [[ "${1:-}" == "--foreground" ]]; then
  if [[ ! -d ".venv" ]]; then
    "$PYTHON_BIN" -m venv .venv
  fi

  source .venv/bin/activate
  python -m pip install --upgrade pip
  python -m pip install -r requirements.txt

  echo "Starting validator (wallet=$WALLET_NAME hotkey=$WALLET_HOTKEY netuid=$NETUID network=$NETWORK)..."
  python neurons/validator.py \
    --netuid "$NETUID" \
    --network "$NETWORK" \
    --wallet.name "$WALLET_NAME" \
    --wallet.hotkey "$WALLET_HOTKEY" \
    $VALIDATOR_EXTRA_ARGS
  exit 0
fi

echo "Starting validator with PM2..."
if pm2 describe perturb-validator >/dev/null 2>&1; then
  pm2 restart perturb-validator
else
  pm2 start "./scripts/run_validator.sh" --name perturb-validator --interpreter bash -- --foreground
fi
pm2 save
pm2 status perturb-validator
