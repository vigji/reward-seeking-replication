#!/usr/bin/env bash
# GPU machine setup, idempotent. Run from the repo root; reads .env (see .env.example).
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .env ] && { set -a; source .env; set +a; }
scripts/fetch_alphaxiv.sh
[ -d .venv ] || uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e ".[gpu,dev]"
.venv/bin/python -c "import torch, vllm, transformers, peft; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'vllm', vllm.__version__, 'transformers', transformers.__version__, 'peft', peft.__version__)"
.venv/bin/python -m pytest -q
