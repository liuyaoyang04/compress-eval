import copy
import os

import pytest
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STARKV_DIR = os.path.join(REPO, "baselines", "STAR-KV")

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")


def tiny_config(rope: str = "default", num_layers: int = 3):
    """A Llama config small enough for CPU tests but with head_dim=128 (GQA 4/2)."""
    from transformers import LlamaConfig
    kwargs = dict(
        vocab_size=512, hidden_size=512, intermediate_size=768, num_hidden_layers=num_layers,
        num_attention_heads=4, num_key_value_heads=2, head_dim=128, max_position_embeddings=4096,
        rope_theta=10000.0, attention_bias=False, tie_word_embeddings=False, use_cache=True,
    )
    if rope == "llama3":
        kwargs["rope_theta"] = 500000.0
        kwargs["rope_scaling"] = {"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0,
                                  "high_freq_factor": 4.0, "original_max_position_embeddings": 1024}
    return LlamaConfig(**kwargs)


def tiny_model(seed: int = 0, rope: str = "default", dtype=torch.float32, device="cpu", num_layers: int = 3,
               attn_implementation: str = "sdpa"):
    """Random tiny Llama with sharpened attention logits (q/k scaled) so softmax is not near-uniform."""
    from transformers import LlamaForCausalLM
    torch.manual_seed(seed)
    cfg = tiny_config(rope, num_layers)
    cfg._attn_implementation = attn_implementation
    model = LlamaForCausalLM(cfg)
    with torch.no_grad():
        for block in model.model.layers:
            block.self_attn.q_proj.weight.mul_(6.0)
            block.self_attn.k_proj.weight.mul_(6.0)
    model.generation_config.pad_token_id = 0
    return model.to(device=device, dtype=dtype).eval()


def clone(model):
    return copy.deepcopy(model)
