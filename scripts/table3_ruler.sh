#!/usr/bin/env bash
# STAR-KV Table 3 protocol: the nine RULER tasks @4096 on Llama-3.1-8B-Instruct (batch 4),
# sharded over GPUS, then a comparison with the paper. ~25 min on 6 GPUs.
#
#   GPUS=2,3,4,5,6,7 bash scripts/table3_ruler.sh
#   CHECKPOINT=ckpt.pt DECODE_MODE=triton NAME=belt_r64 GPUS=2,3,4,5,6,7 bash scripts/table3_ruler.sh
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
NAME=${NAME:-llama31_8b_instruct_baseline}
OUT=${OUT:-$OUT_ROOT/table3/$NAME}
SUITE=ruler MODEL="${MODEL:-$LLAMA31}" CHECKPOINT="${CHECKPOINT:-}" DECODE_MODE="${DECODE_MODE:-reference}" \
  GPUS="${GPUS:-0,1,2,3,4,5}" OUT="$OUT" PYTHON_BIN="$PY" bash scripts/eval_sharded.sh
$PY -m evals.compare "$OUT" --paper "${PAPER:-llama31_8b_instruct}" --markdown "$OUT/compare.md"
