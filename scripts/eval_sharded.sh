#!/usr/bin/env bash
# Run LongBench or RULER through evals.run with one task shard per GPU, then summarize.
#
#   SUITE=longbench MODEL=/path/to/Llama-3.1-8B-Instruct CHECKPOINT=ckpt.pt GPUS=2,3,4 \
#   TASKS=star-paper-7 OUT=results/evals/my_run bash scripts/eval_sharded.sh
#   SUITE=ruler MODEL=... GPUS=2,3,4,5,6,7 OUT=results/evals/ruler_baseline bash scripts/eval_sharded.sh
#
# Environment (defaults in brackets):
#   SUITE        longbench | ruler                            [longbench]
#   MODEL        base model directory (required)
#   CHECKPOINT   low-rank checkpoint; empty -> uncompressed baseline
#   DECODE_MODE  reference | triton | torch | sdpa            [reference]
#   GPUS         comma list of GPU ids                        [0,1,2]
#   TASKS        preset name or comma list of tasks           [star-paper-7 | paper-9]
#   TASK_GROUPS  explicit per-GPU groups "a,b|c|d,e" (overrides round-robin)
#   OUT          output directory (required)
#   BATCH        lm-eval batch size                           [4]
#   MAX_LEN      max context length                           [31500]
#   SEQLEN       RULER context length(s), space separated     [4096]
#   LIMIT        per-task example limit (smoke tests)         []
#   PYTHON_BIN   interpreter                                   [.venv/bin/python]
#   SUMMARY      preset used for the final average            [$TASKS if a preset, else all]
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO"

PYTHON_BIN=${PYTHON_BIN:-.venv/bin/python}
SUITE=${SUITE:-longbench}
MODEL=${MODEL:?set MODEL}
OUT=${OUT:?set OUT}
CHECKPOINT=${CHECKPOINT:-}
DECODE_MODE=${DECODE_MODE:-reference}
GPUS=${GPUS:-0,1,2}
BATCH=${BATCH:-4}
MAX_LEN=${MAX_LEN:-31500}
SEQLEN=${SEQLEN:-4096}
LIMIT=${LIMIT:-}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-1}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export TOKENIZERS_PARALLELISM=false

case "$SUITE" in
  longbench) TASKS=${TASKS:-star-paper-7}; list_cmd=(-m evals.harness --list-tasks "$TASKS"); preset_cmd=(-m evals.harness --presets); preset_prefix="" ;;
  ruler)     TASKS=${TASKS:-paper-9};      list_cmd=(-m evals.ruler --list-tasks "$TASKS");   preset_cmd=(-m evals.ruler --presets);   preset_prefix="ruler-"
             "$PYTHON_BIN" -m evals.sources paul_graham_essays squad punkt_tab ;;   # fetch once here, not in every shard
  *) echo "SUITE must be longbench or ruler" >&2; exit 1 ;;
esac

mkdir -p "$OUT"
IFS=',' read -r -a gpu_arr <<< "$GPUS"

if [[ -n "${TASK_GROUPS:-}" ]]; then
  IFS='|' read -r -a groups <<< "$TASK_GROUPS"
else
  all=$("$PYTHON_BIN" "${list_cmd[@]}" | tail -1)
  IFS=',' read -r -a task_arr <<< "$all"
  groups=()
  for ((i = 0; i < ${#gpu_arr[@]}; i++)); do groups+=(""); done
  for ((i = 0; i < ${#task_arr[@]}; i++)); do
    g=$((i % ${#gpu_arr[@]}))
    groups[$g]="${groups[$g]:+${groups[$g]},}${task_arr[$i]}"
  done
fi
if [[ ${#groups[@]} -gt ${#gpu_arr[@]} ]]; then
  echo "TASK_GROUPS has ${#groups[@]} groups but GPUS has ${#gpu_arr[@]} entries" >&2; exit 1
fi

ckpt_args=()
if [[ -n "$CHECKPOINT" ]]; then ckpt_args=(--checkpoint "$CHECKPOINT" --decode-mode "$DECODE_MODE"); else ckpt_args=(--baseline); fi
limit_args=()
if [[ -n "$LIMIT" ]]; then limit_args=(--limit "$LIMIT"); fi

pids=()
for ((i = 0; i < ${#groups[@]}; i++)); do
  gpu=${gpu_arr[$i]}
  [[ -z "${groups[$i]}" ]] && continue
  echo "GPU $gpu: ${groups[$i]}"
  if [[ "$SUITE" == longbench ]]; then
    suite_args=(--longbench --longbench-tasks "${groups[$i]}")
  else
    # shellcheck disable=SC2206
    suite_args=(--ruler --ruler-tasks "${groups[$i]}" --ruler-seqlen $SEQLEN)
  fi
  CUDA_VISIBLE_DEVICES=$gpu "$PYTHON_BIN" -m evals.run \
    --model "$MODEL" "${ckpt_args[@]}" "${suite_args[@]}" \
    --long-batch-size "$BATCH" --max-length "$MAX_LEN" "${limit_args[@]}" \
    --output "$OUT/shard_gpu$gpu.json" > "$OUT/shard_gpu$gpu.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do wait "$pid" || status=$?; done
if [[ $status -ne 0 ]]; then
  echo "at least one shard failed; see $OUT/*.log" >&2
  exit "$status"
fi

SUMMARY=${SUMMARY:-$TASKS}
if "$PYTHON_BIN" "${preset_cmd[@]}" | awk '{print $1}' | grep -qx "$SUMMARY"; then
  "$PYTHON_BIN" -m evals.summarize "$OUT" --preset "$preset_prefix$SUMMARY" --json "$OUT/summary.json"
else
  metric_args=(); [[ "$SUITE" == ruler ]] && metric_args=(--metric "${SEQLEN%% *}")
  "$PYTHON_BIN" -m evals.summarize "$OUT" --tasks "$SUMMARY" "${metric_args[@]}" --json "$OUT/summary.json"
fi
