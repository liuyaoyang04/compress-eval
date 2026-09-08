#!/usr/bin/env bash
# STAR-KV Table 2 protocol: the seven LongBench tasks on Llama-3.1-8B-Instruct (batch 4, 31500),
# sharded over GPUS, then a comparison with the paper. ~45 min on 4 GPUs (qmsum is the long pole).
#
#   GPUS=4,5,6,7 bash scripts/table2_longbench.sh
#   CHECKPOINT=ckpt.pt DECODE_MODE=triton NAME=belt_r64 GPUS=4,5,6,7 bash scripts/table2_longbench.sh
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
NAME=${NAME:-llama31_8b_instruct_baseline}
OUT=${OUT:-$OUT_ROOT/table2/$NAME}
# balanced default split for four GPUs: the three 512-token summarisation tasks dominate
GROUPS_DEFAULT="longbench_qmsum|longbench_multi_news|longbench_vcsum,longbench_triviaqa|longbench_qasper,longbench_trec,longbench_multifieldqa_en"
SUITE=longbench MODEL="${MODEL:-$LLAMA31}" CHECKPOINT="${CHECKPOINT:-}" DECODE_MODE="${DECODE_MODE:-reference}" \
  GPUS="${GPUS:-0,1,2,3}" TASK_GROUPS="${TASK_GROUPS:-$GROUPS_DEFAULT}" OUT="$OUT" PYTHON_BIN="$PY" \
  bash scripts/eval_sharded.sh
$PY -m evals.compare "$OUT" --paper "${PAPER:-llama31_8b_instruct}" --markdown "$OUT/compare.md"
