"""Download the study models into <repo>/models/ (one-time; needs a HuggingFace token with Llama access).

    HF_TOKEN=hf_... python scripts/fetch_models.py                 # both models, through LRKV_PROXY
    HF_TOKEN=hf_... python scripts/fetch_models.py Llama-2-7b-hf   # one of them
    python scripts/fetch_models.py --check                          # what is present

Models already on disk elsewhere can simply be symlinked: ln -s /path/to/Llama-3.1-8B-Instruct models/
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evals import sources  # noqa: E402

MODELS = {
    "Llama-2-7b-hf": "meta-llama/Llama-2-7b-hf",
    "Llama-3.1-8B-Instruct": "meta-llama/Llama-3.1-8B-Instruct",
}
IGNORE = ["*.bin", "*.pth", "original/*", "*.gguf"]  # safetensors only


def models_dir() -> str:
    return os.environ.get("LRKV_MODELS_DIR") or os.path.join(sources.REPO, "models")


def present(name: str) -> bool:
    d = os.path.join(models_dir(), name)
    return os.path.isfile(os.path.join(d, "config.json")) and os.path.isfile(os.path.join(d, "tokenizer.json"))


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("names", nargs="*", help=f"subset of {', '.join(MODELS)} (default: all)")
    p.add_argument("--check", action="store_true")
    args = p.parse_args(argv)
    names = args.names or list(MODELS)
    if args.check:
        for n in names:
            print(f"{n:24s} {'ok' if present(n) else 'MISSING'}  {os.path.join(models_dir(), n)}")
        return
    from huggingface_hub import snapshot_download
    for n in names:
        if present(n):
            print(f"{n}: present")
            continue
        with sources._online():
            print(f"{n}: downloading {MODELS[n]} -> {os.path.join(models_dir(), n)}")
            snapshot_download(MODELS[n], local_dir=os.path.join(models_dir(), n), ignore_patterns=IGNORE,
                              token=os.environ.get("HF_TOKEN"))
        print(f"{n}: {'ok' if present(n) else 'INCOMPLETE'}")


if __name__ == "__main__":
    main()
