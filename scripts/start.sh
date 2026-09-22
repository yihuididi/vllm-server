#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"

load_config
validate_config
printf 'Starting vLLM server\n\n'
printf '%-20s %s\n' \
  'Image:' "$IMAGE" \
  'Model:' "$MODEL" \
  'GPU(s):' "$GPUS" \
  'Tensor parallel:' "$TP" \
  'Port:' "$SERVER_PORT" \
  'GPU utilization:' "$MEMORY" \
  'Max model length:' "$CONTEXT"

"${COMPOSE[@]}" up -d
printf '\nContainer started; model loading may take several minutes.\n'
printf 'Check readiness: ./scripts/healthcheck.sh\n'
printf 'Follow logs: docker compose logs -f vllm\n'
