# Dockerized vLLM server

Run vLLM's native OpenAI-compatible server with the official `vllm/vllm-openai`
image. Model, GPUs, host port, Hugging Face authentication, and runtime settings
are configurable. Chat, completions, streaming, structured outputs, token usage,
and Prometheus metrics come directly from vLLM.

## Prerequisites

- Linux with NVIDIA GPUs, enough free VRAM for the model and KV cache, and a
  driver compatible with the CUDA version in your chosen vLLM image.
- Docker Engine with Docker Compose v2 (validated with Compose 2.21).
- NVIDIA Container Toolkit with the `nvidia` Docker runtime registered.
- Bash 4+, Python 3 (standard library only), and curl for the scripts.
- Disk space and network access for the image and model download.

Follow NVIDIA's [Container Toolkit installation instructions](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
For a standard Docker installation, runtime registration uses
`sudo nvidia-ctk runtime configure --runtime=docker` followed by a Docker daemon
restart; the linked instructions cover rootless Docker separately.

Check host and container GPU access before starting vLLM:

```bash
nvidia-smi
docker run --rm --runtime=nvidia --gpus all \
  nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

## Quick start

```bash
cp .env.example .env
# Edit .env: choose MODEL_NAME, free GPU_IDS, PORT, and other settings.
./scripts/start.sh
./scripts/healthcheck.sh
```

Startup runs `docker compose up -d` and returns while the model is loading.
The first run downloads the image and model and may take several minutes.
Repeat the health check until it succeeds. It exits nonzero when the service
is unavailable or still loading.

```bash
docker compose logs -f vllm
docker compose ps
curl --fail http://localhost:8000/v1/models
./scripts/stop.sh
```

The default API base URL is `http://localhost:8000/v1`. Replace `8000` in the
examples if you change `PORT`. Scripts resolve paths from their own location,
so they also work when invoked from another directory.

## Configuration

| Setting | Default | Meaning |
| --- | --- | --- |
| `VLLM_IMAGE` | `vllm/vllm-openai:latest` | Official image; pin a tested tag or digest for reproducibility |
| `MODEL_NAME` | `Qwen/Qwen3-8B` | Hugging Face model ID |
| `GPU_IDS` | `0` | Comma-separated host GPU indices, without spaces |
| `TENSOR_PARALLEL_SIZE` | `1` | Number of selected GPUs across which to split the model |
| `PORT` | `8000` | Published host TCP port |
| `HF_TOKEN` | empty | Token for private/gated models; requires access to the model |
| `GPU_MEMORY_UTILIZATION` | `0.90` | Fraction of each selected GPU's memory available to this instance |
| `MAX_MODEL_LEN` | `8192` | Maximum total context length in tokens |

`MODEL_NAME` and `GPU_IDS` must be set. `.env.example` supplies the initial
values; Compose supplies defaults for the optional settings. The scripts let
Compose parse `.env`, including quotes and comments, without executing it as
shell code. Exported shell variables take precedence over `.env`, so an
override such as `PORT=8080 ./scripts/start.sh` works. Apply the same override
when running the health check, or save it in `.env`.

General server defaults and additional native vLLM options live in
`configs/vllm.yaml`, mounted read-only into the container. Compose passes model,
memory utilization, context length, and tensor parallelism as command-line
arguments. These override YAML values, even when Compose uses its fallback
defaults. Change memory utilization and context length in `.env`; add settings
such as `dtype` or `max-num-seqs` in YAML. Keep YAML `host: 0.0.0.0` and
`port: 8000` aligned with the internal port mapping and health check.

vLLM documents the [CLI > YAML > defaults precedence](https://docs.vllm.ai/en/stable/configuration/serve_args/).
To change models, edit `MODEL_NAME` and rerun `./scripts/start.sh`. For a YAML-only
change, restart the process with `docker compose restart vllm`. To fetch a newer
image for the configured tag, run `docker compose pull` before starting.

### GPU selection and tensor parallelism

| Host GPUs | `.env` setting | Tensor parallel size |
| --- | --- | --- |
| GPU 0 | `GPU_IDS=0` | `TENSOR_PARALLEL_SIZE=1` |
| GPU 3 | `GPU_IDS=3` | `TENSOR_PARALLEL_SIZE=1` |
| GPUs 0 and 1 | `GPU_IDS=0,1` | `TENSOR_PARALLEL_SIZE=2` |
| GPUs 0–3 | `GPU_IDS=0,1,2,3` | `TENSOR_PARALLEL_SIZE=4` |

Compose uses `runtime: nvidia` and sets `NVIDIA_VISIBLE_DEVICES` to `GPU_IDS`.
The NVIDIA runtime exposes only the selected devices; a single selected host
GPU is available to vLLM as CUDA device 0. This uses NVIDIA's documented
[GPU enumeration mechanism](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/docker-specialized.html#gpu-enumeration).
Do not add `gpus: all`, which would introduce a competing device selection.

For this single-node tensor-parallel setup, the start script requires exactly
as many distinct GPU indices as `TENSOR_PARALLEL_SIZE`. It also checks numeric
settings before starting Docker. Model architecture constraints and available
VRAM can further limit valid tensor parallel sizes; vLLM checks these when
loading. GPU indices are the indices shown by host `nvidia-smi`.

Verify device isolation after startup:

```bash
docker compose exec vllm nvidia-smi -L
```

### Cache and authentication

The host's `~/.cache/huggingface` is mounted at `/root/.cache/huggingface`.
Downloads survive container removal and are reused on subsequent starts,
following the [official vLLM Docker deployment](https://docs.vllm.ai/en/stable/deployment/docker/).
The YAML config uses a separate read-only mount. `ipc: host` supplies shared
memory for PyTorch and tensor-parallel workers.

Set `HF_TOKEN` in `.env` for gated/private models, and first obtain access on
Hugging Face. `.env` is ignored by Git; commit only `.env.example`. Startup
summaries omit the token. Full `docker compose config` output and container
inspection can include it, so use `docker compose config --quiet` for validation.

The published port binds all host interfaces, as in the plan. This initial
setup has no API authentication; `HF_TOKEN` authenticates model downloads,
not inference requests. Use it on a trusted host/network. If you configure
vLLM API authentication later, add the corresponding bearer token to client
requests.

## Exercise the native API

Set these to the port and model you configured (or the model ID reported by
`/v1/models` if you set a custom served model name in YAML):

```bash
BASE_URL=http://localhost:8000
MODEL=Qwen/Qwen3-8B
curl --fail "$BASE_URL/v1/models"
```

These Qwen3 examples disable thinking to keep the smoke checks brief.
Non-streaming chat includes native `usage` token counts:

```bash
curl --fail-with-body "$BASE_URL/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}],\"max_tokens\":256,\"chat_template_kwargs\":{\"enable_thinking\":false},\"stream\":false}"
```

Streaming chat uses server-sent events. `curl -N` disables client buffering;
`stream_options.include_usage` requests usage in the final stream chunk:

```bash
curl --fail-with-body -N "$BASE_URL/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}],\"max_tokens\":256,\"chat_template_kwargs\":{\"enable_thinking\":false},\"stream\":true,\"stream_options\":{\"include_usage\":true}}"
```

Structured output uses the native `response_format` JSON schema field. The
schema constrains output shape; omitting it allows normal generation. See
vLLM's [structured output documentation](https://docs.vllm.ai/en/stable/features/structured_outputs/)
for model/backend support, including reasoning-model considerations.

```bash
curl --fail-with-body "$BASE_URL/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d @- <<JSON
{
  "model": "$MODEL",
  "messages": [{"role": "user", "content": "Extract the person: Alice is 30 years old. Return only JSON."}],
  "max_tokens": 256,
  "chat_template_kwargs": {"enable_thinking": false},
  "stream": false,
  "response_format": {
    "type": "json_schema",
    "json_schema": {
      "name": "person",
      "strict": true,
      "schema": {
        "type": "object",
        "properties": {
          "name": {"type": "string"},
          "age": {"type": "integer"}
        },
        "required": ["name", "age"],
        "additionalProperties": false
      }
    }
  }
}
JSON
```

The structured JSON is in `choices[0].message.content`, inside the normal
OpenAI-compatible response envelope. This example disables Qwen3 thinking with
`chat_template_kwargs.enable_thinking`, as documented in the
[Qwen deployment guide](https://qwen.readthedocs.io/en/stable/deployment/vllm.html);
adapt the reasoning settings when changing models.
`/v1/completions` is also available for models that support text completion;
chat requests require a compatible chat template.

Read the native Prometheus metrics:

```bash
curl --fail "$BASE_URL/metrics"
```

Before calling a deployment verified, check that `/v1/models` reports the
chosen model, non-streaming chat returns a completion and usage, streaming
returns chunks ending with `[DONE]`, structured chat returns content matching
the schema, and `/metrics` returns Prometheus text.

## Troubleshooting

- **GPU runtime errors:** run the prerequisite GPU probe and check that Docker
  lists the `nvidia` runtime with `docker info`. Do this before debugging vLLM.
- **Out of memory:** inspect free VRAM with `nvidia-smi`, choose unused GPUs,
  reduce context length, or choose a smaller/quantized model. Lower memory
  utilization when sharing a GPU, provided enough memory remains for the model.
- **401/403 during model download:** check `HF_TOKEN` and gated-model access.
- **Health check fails during startup:** follow `docker compose logs -f vllm`.
  Docker allows a ten-minute health-check start period, but large downloads can
  take longer; an unhealthy status does not itself stop the container.
- **Port already allocated:** change `PORT` in `.env` and rerun the start script.

For configuration-only validation without a GPU or Docker daemon:

```bash
for script in scripts/*.sh; do bash -n "$script" || exit; done
docker compose --env-file .env.example config --quiet
```
