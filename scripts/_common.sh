# Shared defaults for the paper-table scripts (sourced, not executed).
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO"
PY=${PYTHON_BIN:-.venv/bin/python}
MODELS=${MODELS:-$REPO/models}          # <repo>/models/<name>, see scripts/fetch_models.py
LLAMA2=${LLAMA2:-$MODELS/Llama-2-7b-hf}
LLAMA31=${LLAMA31:-$MODELS/Llama-3.1-8B-Instruct}
OUT_ROOT=${OUT_ROOT:-$REPO/results}
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
