"""Perplexity on WikiText-2 and C4 (pinned validation shard): the STAR-KV Table 1 datasets.

Ported from STAR-KV ``eval.py`` unchanged in protocol: the test text is
concatenated, cut into non-overlapping ``seqlen`` windows, and scored with
``use_cache=False`` (prefill only, so the decode path is not exercised).

The window is ``PPL_SEQLEN`` = 4096 tokens: the STAR-KV paper does not state
it, but only 4096 reproduces its Table 1 baselines (Llama-2-7B 5.12 / 7.04,
Llama-3-8B-Instruct 7.74 / 12.61); upstream ``eval.py`` defaults to 1024 and
2048, which give 6.10 / 5.47 on Llama-2-7B.

C4 is the pinned ``en/c4-validation.00000-of-00008.json.gz`` shard of
``allenai/c4`` (revision in ``evals.sources``), read from the hash-checked local
copy that ``evals.sources`` fetches once; ``LRKV_C4_FILE`` overrides the path.
"""

from __future__ import annotations

import os
from typing import Dict, Optional

import torch
import torch.nn as nn
from tqdm import tqdm

from .sources import ensure as ensure_source

PPL_SEQLEN = 4096  # STAR-KV Table 1 protocol (see module docstring)
C4_LOCAL_ENV = "LRKV_C4_FILE"  # explicit path to the C4 shard; default is evals.sources' copy


def get_ppl_dataset(name: str, tokenizer, seqlen: int) -> torch.Tensor:
    """Token ids [1, N] of the evaluation text for ``name`` in {wikitext2, c4}."""
    from datasets import load_dataset
    name = name.strip().lower()
    if name == "wikitext2":
        data = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
        return tokenizer("\n\n".join(data["text"]), return_tensors="pt").input_ids
    if name == "c4":
        path = os.environ.get(C4_LOCAL_ENV) or ensure_source("c4")
        data = load_dataset("json", data_files={"validation": path}, split="validation")
        ids = tokenizer(" ".join(data[:1100]["text"]), return_tensors="pt").input_ids
        return ids[:, : 256 * seqlen]
    raise ValueError(f"unknown perplexity dataset {name!r} (wikitext2, c4)")


@torch.no_grad()
def evaluate_ppl(model, tokenizer, datasets, seqlen: int = PPL_SEQLEN, limit: Optional[int] = None,
                 device=None, verbose: bool = True) -> Dict[str, dict]:
    """{dataset: {"loss", "ppl", "nsamples", "seqlen"}} over non-overlapping windows."""
    model.eval()
    if device is None:
        device = next(model.parameters()).device
    if isinstance(datasets, str):
        datasets = [d for d in datasets.split(",") if d.strip()]
    results = {}
    for name in datasets:
        enc = get_ppl_dataset(name, tokenizer, seqlen)
        nsamples = enc.numel() // seqlen
        if limit is not None:
            nsamples = min(nsamples, limit)
        nlls = []
        it = range(nsamples)
        if verbose:
            it = tqdm(it, desc=f"PPL [{name} @ {seqlen}]")
        for i in it:
            batch = enc[:, i * seqlen:(i + 1) * seqlen].to(device)
            logits = model(input_ids=batch, use_cache=False).logits
            loss = nn.CrossEntropyLoss()(logits[:, :-1, :].reshape(-1, logits.size(-1)).float(),
                                         batch[:, 1:].reshape(-1))
            nlls.append(loss.float() * seqlen)
        avg = torch.stack(nlls).sum() / (len(nlls) * seqlen)
        ppl = torch.exp(avg).item()
        results[name] = {"loss": avg.item(), "ppl": ppl, "nsamples": nsamples, "seqlen": seqlen}
        if verbose:
            print(f"  {name:10s} seqlen={seqlen}  n={nsamples}  loss={avg.item():.4f}  ppl={ppl:.3f}")
    return results
