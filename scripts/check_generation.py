"""Smoke test on a real model: the same left-padded batch through every decode mode.

    CUDA_VISIBLE_DEVICES=2 .venv/bin/python scripts/check_generation.py \
        --model MODEL_DIR --checkpoint ckpt.pt --max-new-tokens 48

Two comparisons against the reference path (stock attention + low-rank
projections, full KV cache):

* free-running greedy generation: token agreement and first divergent step
  (after a flip the sequences differ, so this is a coarse signal);
* teacher-forced decode along the reference's own tokens: per-step logit
  deviation and top-1 agreement, which isolates the numerical difference of
  the decode path from the chaos of greedy decoding.
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lrkv.model import install_lowrank_attention, load_compressed_model, set_decode_mode  # noqa: E402

PROMPTS = [
    "The three laws of thermodynamics are",
    "In 1969, Neil Armstrong became the first person to walk on the Moon. The mission, Apollo 11, was launched from",
    "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
]


def _pad_id(model):
    eos = model.config.eos_token_id
    return eos if isinstance(eos, int) else eos[0]


@torch.no_grad()
def generate(model, enc, n):
    out = model.generate(**enc, max_new_tokens=n, min_new_tokens=n, do_sample=False, output_logits=True,
                         return_dict_in_generate=True, pad_token_id=_pad_id(model))
    return out.sequences[:, enc.input_ids.shape[1]:], torch.stack(out.logits, dim=1).float()


@torch.no_grad()
def teacher_forced(model, enc, ref_tok):
    """Per-step logits when the reference tokens are fed one at a time through the cache.

    position_ids follow the attention mask exactly as generate() does, so
    left-padded rows are positioned identically in both comparisons.
    """
    mask = enc.attention_mask
    pos = mask.long().cumsum(-1) - 1
    pos.masked_fill_(mask == 0, 1)
    out = model(input_ids=enc.input_ids, attention_mask=mask, position_ids=pos, use_cache=True)
    cache = out.past_key_values
    logits = [out.logits[:, -1]]
    last_pos = pos[:, -1:]
    for t in range(ref_tok.shape[1] - 1):
        mask = torch.cat([mask, torch.ones_like(mask[:, :1])], dim=1)
        last_pos = last_pos + 1
        o = model(input_ids=ref_tok[:, t:t + 1], attention_mask=mask, position_ids=last_pos,
                  past_key_values=cache, use_cache=True)
        cache = o.past_key_values
        logits.append(o.logits[:, -1])
    return torch.stack(logits, dim=1).float()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--max-new-tokens", type=int, default=48)
    p.add_argument("--modes", default="triton,torch,sdpa")
    p.add_argument("--dtype", default="bf16")
    args = p.parse_args()

    model, tok, info = load_compressed_model(args.model, args.checkpoint, dtype=args.dtype, decode_mode="reference")
    tok.padding_side = "left"
    enc = tok(PROMPTS, return_tensors="pt", padding=True).to(next(model.parameters()).device)
    print(f"batch {tuple(enc.input_ids.shape)}, pads per row: {(enc.attention_mask == 0).sum(1).tolist()}")

    ref_tok, ref_logits = generate(model, enc, args.max_new_tokens)
    scale = ref_logits.abs().max().item()
    print("\n[reference] " + " | ".join(repr(tok.decode(t, skip_special_tokens=True)[:60]) for t in ref_tok))
    tf_ref = teacher_forced(model, enc, ref_tok)
    print(f"[reference] teacher-forced loop vs generate: max|logit diff| {(tf_ref - ref_logits).abs().max():.3f} "
          f"(sanity check of the loop; logit scale {scale:.1f})")

    modes = [m for m in args.modes.split(",") if m]
    install_lowrank_attention(model, decode_mode=modes[0])
    for mode in modes:
        set_decode_mode(model, mode)
        t, _ = generate(model, enc, args.max_new_tokens)
        agree = (t == ref_tok).float().mean(1)
        first_div = [(int(r.nonzero()[0]) if r.any() else -1) for r in (t != ref_tok)]
        tf = teacher_forced(model, enc, ref_tok)
        dev = (tf - ref_logits).abs().amax(dim=(1, 2))
        top1 = (tf.argmax(-1) == ref_logits.argmax(-1)).float().mean(1)
        print(f"\n[{mode}] free-running token agreement per row {[f'{a:.2f}' for a in agree.tolist()]}, "
              f"first divergent step {first_div}")
        print(f"[{mode}] teacher-forced: max|logit diff| per row {[f'{d:.2f}' for d in dev.tolist()]} "
              f"(scale {scale:.1f}), top-1 agreement per row {[f'{a:.2f}' for a in top1.tolist()]}")
        print("  " + " | ".join(repr(tok.decode(x, skip_special_tokens=True)[:60]) for x in t))


if __name__ == "__main__":
    main()
