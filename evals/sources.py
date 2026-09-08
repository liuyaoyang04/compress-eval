"""Pinned evaluation data: fetch once, then evaluate offline on any machine.

Two kinds of sources, both stored under ``LRKV_DATA_DIR`` (default ``<repo>/data``,
gitignored):

* ``FILES`` (``data/eval/``): raw files fetched by URL and verified by SHA256: the C4
  perplexity shard, the RULER essays / SQuAD / NLTK punkt_tab.
* ``DATASETS`` (``data/hf_datasets/``): the HuggingFace datasets lm-eval loads by id
  (zero-shot tasks, WikiText-2, LongBench), cached at a pinned repository revision
  in a repo-local ``HF_DATASETS_CACHE`` so every machine scores the same rows; a
  content hash over all rows is recorded for ``--verify``.

    python -m evals.sources             # fetch whatever is missing (optional proxy: LRKV_PROXY)
    python -m evals.sources --check     # presence + file hashes, no network
    python -m evals.sources --verify    # also re-hash every dataset row (a few minutes)

``configure_offline()`` (called first thing by ``evals.run``) points HF at the
repo-local cache and forbids network access, so an evaluation never depends on
``~/.cache`` or on the Hub. Downloads use ``LRKV_PROXY`` if set (an HTTP proxy
URL) and fall back to the mirror in ``HF_MIRROR`` (default ``https://hf-mirror.com``).
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR_ENV = "LRKV_DATA_DIR"
PROXY_ENV = "LRKV_PROXY"
DEFAULT_PROXY = ""  # direct connection; set LRKV_PROXY=http://host:port to download through a proxy
MIRROR_ENV = "HF_MIRROR"
DEFAULT_MIRROR = "https://hf-mirror.com"

C4_REVISION = "607bd4c8450a42878aa9ddc051a65a055450ef87"
C4_FILE = "en/c4-validation.00000-of-00008.json.gz"
ESSAYS_REVISION = "792d672b77d1a67720f7e9820d646358fc18792e"
SQUAD_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json"
_PUNKT = "nltk/nltk_data/gh-pages/packages/tokenizers/punkt_tab.zip"


# ----------------------------------------------------------------------------- files

@dataclass(frozen=True)
class Source:
    name: str
    filename: str
    urls: Tuple[str, ...]  # tried in order
    sha256: str
    note: str = ""


def _hf_file(repo: str, revision: str, path: str) -> Tuple[str, str]:
    return (f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{path}",
            f"{DEFAULT_MIRROR}/datasets/{repo}/resolve/{revision}/{path}")


FILES: Dict[str, Source] = {s.name: s for s in [
    Source("c4", "c4-validation.00000-of-00008.json.gz", _hf_file("allenai/c4", C4_REVISION, C4_FILE),
           "1f25b6af12da84115301d4ee93ea5246c8fea5bb4a2008472794d95b917cc97f",
           "C4 validation shard scored for perplexity (STAR-KV eval.py protocol)"),
    Source("paul_graham_essays", "paul_graham_essays.parquet",
           _hf_file("baber/paul_graham_essays", ESSAYS_REVISION, "essays.parquet"),
           "825448033be010042ffd6fc77c32bbcab61f2ef63196f0fc4bfbf5b388a1660e",
           "RULER essay haystack: the HF dataset baber/paul_graham_essays that lm-eval 0.4.12 reads"),
    Source("squad", "squad-dev-v2.0.json", (SQUAD_URL,),
           "80a5225e94905956a6446d296ca1093975c4d3b3260f1d6c8f68bc2ab77182d8",
           "SQuAD v2 dev, source of RULER ruler_qa_squad"),
    Source("punkt_tab", "punkt_tab.zip",
           (f"https://raw.githubusercontent.com/{_PUNKT}",
            "https://cdn.jsdelivr.net/gh/nltk/nltk_data@gh-pages/packages/tokenizers/punkt_tab.zip",
            f"https://ghfast.top/https://raw.githubusercontent.com/{_PUNKT}"),
           "e57f64187974277726a3417ca6f181ec5403676c717672eef6a748a7b20e0106",
           "NLTK sentence tokenizer that RULER uses to split the essay haystack"),
]}
SOURCES = FILES  # backwards-compatible alias


# -------------------------------------------------------------------------- datasets

@dataclass(frozen=True)
class HFDataset:
    name: str
    path: str                 # HF dataset id, as lm-eval's task YAML names it
    config: Optional[str]
    revision: str             # git commit of the dataset repository
    content_sha256: str = ""  # sha256 over every row of every split (see dataset_content_sha256); "" = not pinned yet
    note: str = ""


_LB_REV = "2e9ade51ebf45d98942056c0716234f9d5d257d5"
DATASETS: Dict[str, HFDataset] = {d.name: d for d in [
    HFDataset("piqa", "baber/piqa", None, "142f6d7367fd9877f0fb3b5734ea6a545f54cdd1", "18db0ab82b13e05ac0de2072fc3ab137c8ac6b788e2c17b8084c9f85cb365acc", "zero-shot"),
    HFDataset("winogrande", "allenai/winogrande", "winogrande_xl", "01e74176c63542e6b0bcb004dcdea22d94fb67b5", "766aba2c0798fc9e3a3ad78c18299ffe0e20638995dff4b13715503eed23d08b", "zero-shot"),
    HFDataset("arc_easy", "allenai/ai2_arc", "ARC-Easy", "210d026faf9955653af8916fad021475a3f00453", "5a8bc05a345b29d2630b299d668f7a6c5902424a26108675f27678749b6d26eb", "zero-shot"),
    HFDataset("arc_challenge", "allenai/ai2_arc", "ARC-Challenge", "210d026faf9955653af8916fad021475a3f00453", "83e8666c7401df211b38e4e74ec68ba541dc49c237a20c06cc58b4b8d00fde87", "zero-shot"),
    HFDataset("openbookqa", "allenai/openbookqa", "main", "388097ea7776314e93a529163e0fea805b8a6454", "b59daa2250bba3b5e700ef120f82e42435781a5228cd4cf275d253dbc32a069c", "zero-shot"),
    HFDataset("hellaswag", "Rowan/hellaswag", None, "218ec52e09a7e7462a5400043bb9a69a41d06b76", "45ff0dccec761368b347ef4e306d2ad1f9af1ae5167454f7afabf5d62c0d63c7", "zero-shot"),
    HFDataset("wikitext2", "Salesforce/wikitext", "wikitext-2-raw-v1", "b08601e04326c79dfdd32d625aee71d232d685c3", "5cfbd5f59171abceb2e9b2d4aff4e0087c859a1e3277aa2c7e4bd96eac346a81", "perplexity"),
] + [HFDataset(f"longbench_{t}", "Xnhyacinth/LongBench", t, _LB_REV, h, "LongBench (STAR-KV Table 2)") for t, h in [
    ("qasper", "3c824ad99738220bca91c88c9f7422e6ce6100def4234a1889d23efe5e88bb81"),
    ("qmsum", "dc28fbce883af82a4872f255a4124d63ffc984188336807e73a8c3fd15bb0183"),
    ("triviaqa", "3510a0aa9bdac695dae12642ee5427d02c93a9bb92c138cb3ece31852b5b716b"),
    ("multifieldqa_en", "a309909b2909d178796b58570e2c58e86c31219b90f4609380753e436bd01050"),
    ("trec", "0fe3b9349880a8576e5ae3984eddfd4419c0f485c14c0eced9b5fde918593b49"),
    ("multi_news", "c75cbb9d7a0126906768879cd41e3a1732aa5364d240594d61b65815eb6ac109"),
    ("vcsum", "6c94019e84dc2e00ad05da3f0c9e3ef53215e7edd83ca01c3402034408f0db06")]]}

# What each evaluation suite needs (names in FILES / DATASETS).
REQUIRED: Dict[str, List[str]] = {
    "ppl": ["wikitext2", "c4"],
    "zero-shot": ["piqa", "winogrande", "arc_easy", "arc_challenge", "openbookqa", "hellaswag"],
    "longbench": [n for n in DATASETS if n.startswith("longbench_")],
    "ruler": ["paul_graham_essays", "squad", "punkt_tab"],
}


# ------------------------------------------------------------------------ locations

def data_dir() -> str:
    return os.environ.get(DATA_DIR_ENV) or os.path.join(REPO, "data")


def files_dir() -> str:
    return os.path.join(data_dir(), "eval")


def hf_cache_dir() -> str:
    return os.path.join(data_dir(), "hf_datasets")


def proxies() -> Optional[Dict[str, str]]:
    p = os.environ.get(PROXY_ENV, DEFAULT_PROXY)
    return {"http": p, "https": p} if p else None


def configure_offline() -> None:
    """Use the repo-local dataset cache and forbid network access. Call before importing datasets / lm_eval."""
    os.environ.setdefault("HF_DATASETS_CACHE", hf_cache_dir())
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("DATASETS_VERBOSITY", "error")  # silence "using the latest cached version" on every load


@contextlib.contextmanager
def _online(endpoint: Optional[str] = None):
    """Temporarily allow network access through the proxy (and an alternative HF endpoint)."""
    keys = ["HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "HF_ENDPOINT", "HTTPS_PROXY", "HTTP_PROXY", "https_proxy", "http_proxy"]
    saved = {k: os.environ.get(k) for k in keys}
    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ["HF_DATASETS_OFFLINE"] = "0"
    p = os.environ.get(PROXY_ENV, DEFAULT_PROXY)
    for k in ["HTTPS_PROXY", "HTTP_PROXY", "https_proxy", "http_proxy"]:
        if p:
            os.environ[k] = p
        else:
            os.environ.pop(k, None)
    if endpoint:
        os.environ["HF_ENDPOINT"] = endpoint
    else:
        os.environ.pop("HF_ENDPOINT", None)
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# ----------------------------------------------------------------------------- files

def sha256_of(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def path_of(name: str) -> str:
    return os.path.join(files_dir(), FILES[name].filename)


def is_present(name: str) -> bool:
    p = path_of(name)
    return os.path.isfile(p) and sha256_of(p) == FILES[name].sha256


def _download(url: str, dst: str) -> None:
    import requests
    tmp = dst + ".part"
    with requests.get(url, stream=True, timeout=(30, 600), proxies=proxies()) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for block in r.iter_content(1 << 20):
                f.write(block)
    os.replace(tmp, dst)


def ensure(name: str, download: bool = True, verbose: bool = True) -> str:
    """Local path of file ``name``; download (hash-verified) only if it is missing or corrupt."""
    src = FILES[name]
    dst = path_of(name)
    if os.path.isfile(dst):
        if sha256_of(dst) == src.sha256:
            return dst
        if verbose:
            print(f"[sources] {dst} fails its SHA256 check; fetching again", file=sys.stderr)
    if not download:
        raise FileNotFoundError(f"{name}: {dst} is missing or corrupt; run `python -m evals.sources {name}`")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    errors = []
    for url in src.urls:
        try:
            if verbose:
                print(f"[sources] downloading {name} from {url}", file=sys.stderr)
            _download(url, dst)
        except Exception as e:  # noqa: BLE001 - try the next mirror
            errors.append(f"{url}: {type(e).__name__}: {e}")
            continue
        got = sha256_of(dst)
        if got == src.sha256:
            return dst
        os.remove(dst)
        errors.append(f"{url}: sha256 {got[:16]}... != pinned {src.sha256[:16]}...")
    raise RuntimeError(f"could not fetch {name} (proxy {os.environ.get(PROXY_ENV, DEFAULT_PROXY)!r}):\n  "
                       + "\n  ".join(errors))


# -------------------------------------------------------------------------- datasets

def _camelcase_to_snakecase(name: str) -> str:
    """Same rule as ``datasets.naming.camelcase_to_snakecase`` (LongBench -> long_bench), without importing datasets."""
    import re
    name = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
    name = re.sub(r"([a-z\d])([A-Z])", r"\1_\2", name)
    return name.replace("-", "_").lower()


def dataset_cache_path(name: str) -> str:
    """Directory the ``datasets`` library uses for this dataset at the pinned revision (no datasets import:
    importing it freezes the offline flags)."""
    d = DATASETS[name]
    builder = _camelcase_to_snakecase(d.path.split("/")[-1])
    namespace = d.path.split("/")[0]
    return os.path.join(hf_cache_dir(), f"{namespace}___{builder}", d.config or "default", "0.0.0", d.revision)


def dataset_present(name: str) -> bool:
    return os.path.isfile(os.path.join(dataset_cache_path(name), "dataset_info.json"))


def load_pinned_dataset(name: str):
    """The cached DatasetDict for ``name`` (all splits), offline."""
    from datasets import load_dataset
    d = DATASETS[name]
    if not dataset_present(name):
        raise FileNotFoundError(f"{name}: not in {hf_cache_dir()}; run `python -m evals.sources {name}`")
    return load_dataset(d.path, d.config, cache_dir=hf_cache_dir())  # offline: resolves to the cached (pinned) copy


def dataset_content_sha256(name: str) -> str:
    """sha256 over every row of every split (JSON with sorted keys), independent of the arrow layout."""
    ds = load_pinned_dataset(name)
    h = hashlib.sha256()
    for split in sorted(ds):
        h.update(f"[{split}:{len(ds[split])}]".encode())
        for row in ds[split]:
            h.update(json.dumps(row, sort_keys=True, ensure_ascii=False, default=str).encode())
            h.update(b"\n")
    return h.hexdigest()


def ensure_dataset(name: str, download: bool = True, verbose: bool = True) -> str:
    """Cache directory of dataset ``name`` at its pinned revision; fetched through the proxy if missing."""
    if dataset_present(name):
        return dataset_cache_path(name)
    if not download:
        raise FileNotFoundError(f"{name}: not in {hf_cache_dir()}; run `python -m evals.sources {name}`")
    d = DATASETS[name]
    os.makedirs(hf_cache_dir(), exist_ok=True)
    errors = []
    for endpoint in (None, os.environ.get(MIRROR_ENV, DEFAULT_MIRROR)):
        try:
            if verbose:
                print(f"[sources] fetching dataset {d.path} ({d.config}) @ {d.revision[:10]} via {endpoint or 'huggingface.co'}",
                      file=sys.stderr)
            with _online(endpoint):
                from datasets import load_dataset
                load_dataset(d.path, d.config, cache_dir=hf_cache_dir(), revision=d.revision)
            if dataset_present(name):
                return dataset_cache_path(name)
            errors.append(f"{endpoint or 'huggingface.co'}: fetched but {dataset_cache_path(name)} is not there")
        except Exception as e:  # noqa: BLE001
            errors.append(f"{endpoint or 'huggingface.co'}: {type(e).__name__}: {str(e)[:200]}")
    raise RuntimeError(f"could not fetch dataset {name}:\n  " + "\n  ".join(errors))


# ------------------------------------------------------------------------------ all

def require(suite: str, download_files: bool = True) -> None:
    """Raise with a clear message if anything ``suite`` needs is missing (files may be fetched on the spot)."""
    missing = []
    for n in REQUIRED[suite]:
        if n in FILES:
            try:
                ensure(n, download=download_files)
            except (FileNotFoundError, RuntimeError) as e:
                missing.append(str(e))
        elif not dataset_present(n):
            missing.append(f"dataset {n} ({DATASETS[n].path} {DATASETS[n].config or ''} @ {DATASETS[n].revision[:10]})")
    if missing:
        raise SystemExit(f"[sources] {suite}: missing data, run `python -m evals.sources` first:\n  " + "\n  ".join(missing))


def status(verify: bool = False) -> List[Tuple[str, str, str]]:
    rows = []
    for n, f in FILES.items():
        p = path_of(n)
        ok = os.path.isfile(p) and sha256_of(p) == f.sha256
        rows.append((n, "ok" if ok else "MISSING", p if ok else f"{p} (expected sha256 {f.sha256[:12]}...)"))
    for n, d in DATASETS.items():
        if not dataset_present(n):
            rows.append((n, "MISSING", f"{d.path} {d.config or ''} @ {d.revision[:10]}"))
            continue
        state = "ok"
        if verify:
            got = dataset_content_sha256(n)
            state = "ok" if (not d.content_sha256 or got == d.content_sha256) else f"CONTENT MISMATCH {got[:12]}"
            if not d.content_sha256:
                state = f"ok (unpinned content sha256 {got})"
        rows.append((n, state, dataset_cache_path(n)))
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description="Fetch or verify the pinned evaluation data.")
    p.add_argument("names", nargs="*", help=f"subset of {', '.join(list(FILES) + list(DATASETS))} (default: all)")
    p.add_argument("--check", action="store_true", help="verify presence and file hashes only, never download")
    p.add_argument("--verify", action="store_true", help="--check plus a content hash over every dataset row")
    p.add_argument("--data-dir", default=None, help=f"override {DATA_DIR_ENV} (default <repo>/data)")
    args = p.parse_args(argv)
    if args.data_dir:
        os.environ[DATA_DIR_ENV] = args.data_dir
    configure_offline()
    names = args.names or list(FILES) + list(DATASETS)
    failed = 0
    if not (args.check or args.verify):
        for n in names:
            try:
                ensure(n) if n in FILES else ensure_dataset(n)
            except (FileNotFoundError, RuntimeError) as e:
                failed += 1
                print(f"{n:24s} FAILED  {e}")
    wanted = set(names)
    for n, state, where in status(verify=args.verify):
        if n in wanted:
            print(f"{n:24s} {state:8s} {where}")
            failed += state.startswith(("MISSING", "CONTENT"))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
