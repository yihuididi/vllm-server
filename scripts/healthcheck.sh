#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"

require_command curl
load_config
BASE_URL="http://127.0.0.1:${SERVER_PORT}"
if ! curl --noproxy '*' --fail --silent --show-error --connect-timeout 3 --max-time 10 \
  "${BASE_URL}/health" >/dev/null; then
  fail 'vLLM is not ready. Model loading may still be in progress; check docker compose logs -f vllm.'
fi

printf 'vLLM server is healthy.\n\nConfigured model:\n%s\n\nEndpoint:\nhttp://localhost:%s/v1\n' \
  "$MODEL" "$SERVER_PORT"
