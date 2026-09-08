"""Environment fingerprint stored in every result JSON (versions, GPU, code and data state)."""

from __future__ import annotations

import os
import platform
import socket
import subprocess
import time
from typing import Dict

from . import sources


def git_commit() -> str:
    try:
        out = subprocess.run(["git", "-C", sources.REPO, "rev-parse", "--short=12", "HEAD"], capture_output=True, text=True, timeout=10)
        if out.returncode != 0:
            return "no-git"
        dirty = subprocess.run(["git", "-C", sources.REPO, "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, timeout=10).stdout.strip()
        return out.stdout.strip() + ("-dirty" if dirty else "")
    except (OSError, subprocess.SubprocessError):
        return "no-git"


def environment_fingerprint() -> Dict[str, object]:
    import torch
    versions = {"python": platform.python_version(), "torch": torch.__version__}
    for mod in ("transformers", "lm_eval", "datasets", "triton", "accelerate"):
        try:
            versions[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001
            versions[mod] = None
    gpus = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else []
    data = {n: sources.is_present(n) for n in sources.FILES}
    data.update({n: sources.dataset_present(n) for n in sources.DATASETS})
    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "host": socket.gethostname(),
        "code": git_commit(), "versions": versions, "gpus": gpus,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "data_dir": sources.data_dir(), "data_present": data,
    }
