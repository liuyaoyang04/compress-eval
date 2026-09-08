#!/usr/bin/env bash
# STAR-KV Table 1 protocol: WikiText-2 / C4 perplexity @4096 + the six zero-shot tasks,
# Llama-2-7B and Llama-3.1-8B-Instruct, one model per GPU, then a comparison with the paper.
#
#   GPUS=2,3 bash scripts/table1_ppl_zeroshot.sh
#   CHECKPOINT=ckpt.pt GPUS=2,3,4 bash scripts/table1_ppl_zeroshot.sh   # adds Llama-3.1 + checkpoint (reference path)
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
OUT=${OUT:-$OUT_ROOT/table1}
IFS=, read -r -a GPUS <<< "${GPUS:-0,1}"
mkdir -p "$OUT"
$PY -m evals.sources c4 wikitext2 piqa winogrande arc_easy arc_challenge openbookqa hellaswag >/dev/null
EVAL="--ppl --ppl-datasets wikitext2,c4 --tasks zero-shot --batch-size ${BATCH:-32}"   # --ppl-seqlen defaults to 4096

run() {  # run <gpu> <name> <evals.run args...>
  local gpu=$1 name=$2; shift 2
  CUDA_VISIBLE_DEVICES=$gpu nohup $PY -m evals.run "$@" --output "$OUT/$name.json" > "$OUT/$name.log" 2>&1 &
  echo "[$name] gpu=$gpu pid=$! log=$OUT/$name.log"
}
run "${GPUS[0]}" llama2_7b_baseline           --model "$LLAMA2"  --baseline $EVAL
run "${GPUS[1]}" llama31_8b_instruct_baseline --model "$LLAMA31" --baseline $EVAL
if [ -n "${CHECKPOINT:-}" ]; then
  run "${GPUS[2]:?CHECKPOINT needs a third GPU in GPUS}" llama31_8b_instruct_ckpt \
      --model "$LLAMA31" --checkpoint "$CHECKPOINT" --decode-mode reference $EVAL
fi
wait
$PY -m evals.compare "$OUT/llama2_7b_baseline.json" --paper llama2_7b --markdown "$OUT/compare_llama2_7b.md"
$PY -m evals.compare "$OUT/llama31_8b_instruct_baseline.json" --paper llama31_8b_instruct --markdown "$OUT/compare_llama31.md"
[ -n "${CHECKPOINT:-}" ] && $PY -m evals.compare "$OUT/llama31_8b_instruct_ckpt.json" --paper starkv60_llama31 --markdown "$OUT/compare_ckpt.md"
echo "done: $OUT"
