#!/usr/bin/env bash
# Shared setup for the operational scripts. Let Compose parse dotenv syntax;
# never source .env as executable shell code or print the resolved HF token.
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$REPO_ROOT"

fail() { printf 'Error: %s\n' "$*" >&2; exit 1; }
require_command() { command -v "$1" >/dev/null 2>&1 || fail "$1 is required."; }

require_command docker
docker compose version >/dev/null 2>&1 || fail 'Docker Compose v2 is required.'
[[ -f .env ]] || fail 'Missing .env. Run: cp .env.example .env, then edit it.'
COMPOSE=(docker compose --project-directory "$REPO_ROOT" --env-file "$REPO_ROOT/.env" -f "$REPO_ROOT/docker-compose.yml")

load_config() {
  require_command python3
  local resolved values
  resolved="$("${COMPOSE[@]}" config --format json)" || fail 'Invalid Compose configuration.'
  values="$(printf '%s' "$resolved" | python3 -c '
import json, sys
s = json.load(sys.stdin)["services"]["vllm"]
args = s["command"]
def arg(name):
    return args[args.index(name) + 1]
values = [s["image"], arg("--model"), s["environment"]["NVIDIA_VISIBLE_DEVICES"],
          arg("--tensor-parallel-size"), str(s["ports"][0]["published"]),
          arg("--gpu-memory-utilization"), arg("--max-model-len")]
if any(not isinstance(v, str) or not v.strip() or "\n" in v or "\r" in v for v in values):
    sys.exit("Error: Configuration values must be nonempty single-line strings.")
print("\n".join(values))
')" || fail 'Could not read resolved server settings.'
  local settings
  mapfile -t settings <<< "$values"
  IMAGE="${settings[0]}"
  MODEL="${settings[1]}"
  GPUS="${settings[2]}"
  TP="${settings[3]}"
  SERVER_PORT="${settings[4]}"
  MEMORY="${settings[5]}"
  CONTEXT="${settings[6]}"
}

validate_config() {
  python3 - "$GPUS" "$TP" "$SERVER_PORT" "$MEMORY" "$CONTEXT" <<'PY'
import math
import re
import sys

gpus, tp, port, memory, context = sys.argv[1:]
def require(condition, message):
    if not condition:
        sys.exit("Error: " + message)

require(re.fullmatch(r"(?:0|[1-9][0-9]*)(?:,(?:0|[1-9][0-9]*))*", gpus),
        "GPU_IDS must be comma-separated GPU indices without spaces (e.g. 0,1).")
ids = gpus.split(",")
require(len(ids) == len(set(ids)), "GPU_IDS must not contain duplicate devices.")
require(re.fullmatch(r"[1-9][0-9]*", tp), "TENSOR_PARALLEL_SIZE must be a positive integer.")
require(int(tp) == len(ids), "TENSOR_PARALLEL_SIZE must equal the number of selected GPUs.")
require(port.isascii() and port.isdigit() and 1 <= int(port) <= 65535,
        "PORT must be an integer between 1 and 65535.")
try:
    utilization = float(memory)
except ValueError:
    utilization = float("nan")
require(math.isfinite(utilization) and 0 < utilization <= 1,
        "GPU_MEMORY_UTILIZATION must be greater than 0 and at most 1.")
require(re.fullmatch(r"[1-9][0-9]*", context), "MAX_MODEL_LEN must be a positive integer.")
PY
}
