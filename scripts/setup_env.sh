#!/usr/bin/env bash
# Build the repository's own evaluation environment with uv.
#
#   bash scripts/setup_env.sh            # create .venv and install requirements.txt
#   LOCK=1 bash scripts/setup_env.sh     # install requirements.lock.txt (exact versions) instead
#   VENV_DIR=/local/disk/venvs/lrkv LOCK=1 bash scripts/setup_env.sh
#                                        # build on the local NVMe volume and symlink .venv to it
#
# VENV_DIR defaults to ./.venv. The repository sits on CFS (network storage), where
# importing a package tree of ~15k files is slow (lm-eval's TaskManager alone takes
# ~100 s scanning its task YAMLs); an environment on local disk avoids that, and
# .venv in the repository becomes a symlink so every documented command still works.
# The uv cache lives in ./.uv-cache; both are gitignored.
# torch comes from the cu121 index, everything else from PyPI (or PIP_INDEX_URL if set).
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO"

PYTHON=${PYTHON:-/usr/local/python3.10/bin/python3.10}
VENV_DIR=${VENV_DIR:-$REPO/.venv}
export UV_CACHE_DIR=${UV_CACHE_DIR:-$REPO/.uv-cache}
TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cu121}
mkdir -p "$UV_CACHE_DIR"

if ! command -v uv >/dev/null; then
  echo "uv not found; install it (pip install uv) or create the venv manually" >&2; exit 1
fi
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  uv venv "$VENV_DIR" --python "$PYTHON"
fi
if [[ "$(readlink -f "$VENV_DIR")" != "$(readlink -f "$REPO/.venv")" ]]; then
  if [[ -e "$REPO/.venv" && ! -L "$REPO/.venv" ]]; then
    echo ".venv is a real directory; move it aside before pointing it at $VENV_DIR" >&2; exit 1
  fi
  ln -sfn "$VENV_DIR" "$REPO/.venv"
fi
PY="$VENV_DIR/bin/python"

if [[ "${LOCK:-0}" == "1" ]]; then
  uv pip install --python "$PY" --index-url "$TORCH_INDEX" --extra-index-url "${PIP_INDEX_URL:-https://pypi.org/simple}" \
    --index-strategy unsafe-best-match -r requirements.lock.txt
else
  uv pip install --python "$PY" --index-url "$TORCH_INDEX" "torch==2.5.1"
  uv pip install --python "$PY" ${PIP_INDEX_URL:+--index-url "$PIP_INDEX_URL"} -r requirements.txt
fi

$PY - <<'PYEOF'
import torch, transformers, triton, lm_eval, datasets, accelerate
print(f"python {__import__('sys').version.split()[0]}  torch {torch.__version__}  triton {triton.__version__}  "
      f"transformers {transformers.__version__}  lm_eval {lm_eval.__version__}  datasets {datasets.__version__}  "
      f"accelerate {accelerate.__version__}  cuda {torch.cuda.is_available()}")
PYEOF
echo "environment ready: $REPO/.venv"
