"""Low-rank KV checkpoint format, loading, and legacy STAR-KV conversion.

Format
------
A checkpoint is a plain ``state_dict`` (``torch.save``). For every compressed
attention layer ``i`` (prefix ``model.layers.{i}.self_attn.``) it holds:

    k_proj.VS.weight    [sum_h r_h, hidden]        per-KV-head latent projections, heads back to back
    k_proj.U.weight     [Hkv * head_dim, sum_h r_h] block-diagonal per head
    k_proj.head_ranks   int32 [Hkv]                r_h for every KV head
    v_proj.VS.weight    [r_v, hidden]              joint latent projection for the layer
    v_proj.U.weight     [Hkv * head_dim, r_v]

Everything else (q_proj, o_proj, MLP, norms, embeddings, lm_head) is optional:
a key that is present overrides the base model's tensor, a key that is absent
keeps it. A layer may compress only K or only V; the other projection stays a
dense ``nn.Linear`` (and is wrapped at full rank when the latent-cache
attention is installed). A layer with neither key stays uncompressed.

These are exactly the tensors STAR-KV's ``fuse_and_prune`` writes, so upstream
fused checkpoints load unchanged. Legacy STAR-KV checkpoints (``U`` / ``Sigma``
/ ``V`` with soft thresholds, the format of ``trained_weights.pt``) are
converted in memory by :func:`convert_legacy_starkv`, which reproduces
``export_kproj_for_triton`` / ``export_vproj_for_triton`` at the state-dict
level: ``keep = diag > alpha``, ``VS = V[keep] * soft_threshold(diag[keep])``,
``U = U[:, keep]``, per head for K and jointly for V.

CLI
---
    python -m lrkv.checkpoint inspect  ckpt.pt [--model MODEL_DIR]
    python -m lrkv.checkpoint convert  --input legacy.pt --output fused.pt [--factors-only]
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn

from .modules import HeadwiseLowRankLinear, LowRankLinear

LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.self_attn\.(k_proj|v_proj)\.(.+)$")
K_FACTOR_KEYS = ("VS.weight", "U.weight", "head_ranks")
V_FACTOR_KEYS = ("VS.weight", "U.weight")


def layer_prefix(i: int) -> str:
    return f"model.layers.{i}.self_attn."


# ---------------------------------------------------------------------------
# Inspection
# ---------------------------------------------------------------------------

def is_lowrank_state_dict(sd: Dict[str, torch.Tensor]) -> bool:
    return any(k.endswith("k_proj.VS.weight") or k.endswith("v_proj.VS.weight") for k in sd)


def is_legacy_starkv_state_dict(sd: Dict[str, torch.Tensor]) -> bool:
    return any("soft_thres_layer.alpha" in k for k in sd)


def compressed_layers(sd: Dict[str, torch.Tensor]) -> Dict[int, Dict[str, bool]]:
    """{layer_index: {"k": bool, "v": bool}} for every layer carrying low-rank factors."""
    out: Dict[int, Dict[str, bool]] = {}
    for k in sd:
        m = LAYER_RE.match(k)
        if m is None or m.group(3) != "VS.weight":
            continue
        i = int(m.group(1))
        out.setdefault(i, {"k": False, "v": False})[m.group(2)[0]] = True
    return dict(sorted(out.items()))


def infer_skip_layers(sd: Dict[str, torch.Tensor], num_layers: int) -> Tuple[int, ...]:
    """Layers that carry no low-rank factor at all. The checkpoint is the authority."""
    return tuple(i for i in range(num_layers) if i not in compressed_layers(sd))


def layer_ranks(sd: Dict[str, torch.Tensor], i: int) -> Tuple[Optional[list], Optional[int]]:
    """(k head ranks or None, v rank or None) for layer i of a low-rank state dict."""
    pre = layer_prefix(i)
    k = sd[pre + "k_proj.head_ranks"].tolist() if pre + "k_proj.head_ranks" in sd else None
    v = int(sd[pre + "v_proj.VS.weight"].shape[0]) if pre + "v_proj.VS.weight" in sd else None
    return k, v


def factor_keys(sd: Dict[str, torch.Tensor]) -> Iterable[str]:
    for k in sd:
        m = LAYER_RE.match(k)
        if m is not None and m.group(3) in (K_FACTOR_KEYS if m.group(2) == "k_proj" else V_FACTOR_KEYS):
            yield k


def factors_only(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Keep only the low-rank factor tensors (drops any fine-tuned dense weights)."""
    return {k: sd[k] for k in factor_keys(sd)}


# ---------------------------------------------------------------------------
# Legacy STAR-KV (U / Sigma / V + soft threshold) -> canonical
# ---------------------------------------------------------------------------

def soft_threshold(x: torch.Tensor, alpha: float, s: float, c: float) -> torch.Tensor:
    """STAR-KV ``soft_thres_func.forward``: ``coef * tanh(s * (x - alpha))``, coef = x if x > alpha else -c."""
    t = torch.tanh(s * (x - alpha))
    coef = torch.where(x > alpha, x, torch.full_like(x, -c))
    return coef * t


@dataclass
class LegacyLayerReport:
    layer: int
    k_ranks: Optional[list] = None
    k_dropped_zero_rows: int = 0
    k_offblock_energy: float = 0.0     # ||U outside its block-diagonal||^2 / ||U||^2 over kept columns
    v_rank: Optional[int] = None
    v_dropped_zero_rows: int = 0
    v_flattened: bool = False          # alpha == 0 and diag all ones: rank info was erased upstream


@dataclass
class LegacyReport:
    layers: list = field(default_factory=list)

    def max_offblock_energy(self) -> float:
        return max((r.k_offblock_energy for r in self.layers), default=0.0)

    def summary(self) -> str:
        lines = []
        for r in self.layers:
            k = f"K ranks={r.k_ranks} (max {max(r.k_ranks)})" if r.k_ranks else "K dense"
            v = f"V rank={r.v_rank}" if r.v_rank is not None else "V dense"
            extra = []
            if r.k_dropped_zero_rows or r.v_dropped_zero_rows:
                extra.append(f"dropped zero rows k={r.k_dropped_zero_rows} v={r.v_dropped_zero_rows}")
            if r.k_offblock_energy > 1e-6:
                extra.append(f"U off-block energy {r.k_offblock_energy:.2e}")
            if r.v_flattened:
                extra.append("V Sigma flattened (no rank info)")
            lines.append(f"  layer {r.layer:2d}: {k}  {v}" + (f"  [{'; '.join(extra)}]" if extra else ""))
        return "\n".join(lines)


def _legacy_effective(sd, pre, name):
    w = sd[pre + name + ".weight"].float()
    mask = sd.get(pre + name + ".mask")
    return w * mask.float() if mask is not None else w


def _threshold_params(sd, pre):
    return (float(sd[pre + "soft_thres_layer.alpha"].float()),
            float(sd[pre + "soft_thres_layer.s"].float()),
            float(sd[pre + "soft_thres_layer.c"].float()))


def _keep_rows(VS: torch.Tensor):
    """Indices of rows of VS that are not identically zero; a zero latent row contributes nothing."""
    nz = (VS != 0).any(dim=1).nonzero(as_tuple=False).view(-1)
    return nz


@torch.no_grad()
def convert_legacy_starkv(sd: Dict[str, torch.Tensor], keep_dtype: bool = True,
                          drop_zero_rows: bool = True) -> Tuple[Dict[str, torch.Tensor], LegacyReport]:
    """Convert a STAR-KV ``U/Sigma/V`` checkpoint to the canonical ``U/VS`` layout.

    The rule is the one STAR-KV's exporters use: a direction survives iff
    ``diag > alpha``; its VS row is ``V[row] * soft_threshold(diag)`` and its U
    column is copied. Rows whose VS is identically zero are dropped as well
    (exact, they contribute nothing); this also recovers the rank of the older
    "fused" checkpoints whose Sigma was reset to all-ones with alpha 0.

    K is sliced per head. If the training-time U carried cross-head entries
    (older STAR-KV without the block mask), they are dropped exactly as
    ``export_kproj_for_triton`` drops them, and their relative energy is
    reported so the discrepancy with the training-time reference is visible.
    """
    out: Dict[str, torch.Tensor] = {}
    report = LegacyReport()
    layers = sorted({int(LAYER_RE.match(k).group(1)) for k in sd if LAYER_RE.match(k)})
    consumed = set()

    for i in layers:
        pre = layer_prefix(i)
        rep = LegacyLayerReport(layer=i)
        kp, vp = pre + "k_proj.", pre + "v_proj."

        # ---- K: per head ------------------------------------------------
        if kp + "U.weight" in sd and kp + "Sigma_blocks.0.diag" in sd:
            U = _legacy_effective(sd, kp, "U")
            V = _legacy_effective(sd, kp, "V")
            dtype = sd[kp + "U.weight"].dtype if keep_dtype else torch.float32
            H = 0
            while kp + f"Sigma_blocks.{H}.diag" in sd:
                H += 1
            out_f = U.shape[0]
            assert out_f % H == 0, f"layer {i}: k_proj out={out_f} not divisible by {H} heads"
            Hd = out_f // H
            VS_blocks, U_blocks, col = [], [], 0
            offblock_num = offblock_den = 0.0
            for h in range(H):
                bp = kp + f"Sigma_blocks.{h}."
                diag = sd[bp + "diag"].float()
                alpha, s, c = _threshold_params(sd, bp)
                keep = (diag > alpha).nonzero(as_tuple=False).view(-1)
                s_eff = soft_threshold(diag, alpha, s, c)
                cols = col + keep
                VS_h = V[cols] * s_eff[keep, None]
                if drop_zero_rows and keep.numel():
                    nz = _keep_rows(VS_h)
                    rep.k_dropped_zero_rows += int(keep.numel() - nz.numel())
                    keep, cols, VS_h = keep[nz], cols[nz], VS_h[nz]
                if keep.numel() == 0:               # never emit a rank-0 head
                    cols = torch.tensor([col], dtype=torch.long)
                    VS_h = torch.zeros(1, V.shape[1])
                U_cols = U[:, cols]                  # [out_f, r_h], all rows
                U_h = U_cols[h * Hd:(h + 1) * Hd]
                offblock_den += float((U_cols ** 2).sum())
                offblock_num += float((U_cols ** 2).sum() - (U_h ** 2).sum())
                VS_blocks.append(VS_h)
                U_blocks.append(U_h)
                col += diag.numel()
            ranks = [int(v.shape[0]) for v in VS_blocks]
            out[kp + "VS.weight"] = torch.cat(VS_blocks, 0).to(dtype)
            U_new = torch.zeros(out_f, sum(ranks))
            off = 0
            for h, (u, r) in enumerate(zip(U_blocks, ranks)):
                U_new[h * Hd:(h + 1) * Hd, off:off + r] = u
                off += r
            out[kp + "U.weight"] = U_new.to(dtype)
            out[kp + "head_ranks"] = torch.tensor(ranks, dtype=torch.int32)
            rep.k_ranks = ranks
            rep.k_offblock_energy = offblock_num / offblock_den if offblock_den > 0 else 0.0
            consumed.update(k for k in sd if k.startswith(kp))
        elif kp + "Sigma.diag" in sd:
            raise NotImplementedError(
                f"layer {i}: k_proj was decomposed jointly (no per-head Sigma blocks); the latent-cache "
                f"attention needs per-KV-head K factors")

        # ---- V: joint -----------------------------------------------------
        if vp + "U.weight" in sd and vp + "Sigma.diag" in sd:
            U = _legacy_effective(sd, vp, "U")
            V = _legacy_effective(sd, vp, "V")
            dtype = sd[vp + "U.weight"].dtype if keep_dtype else torch.float32
            diag = sd[vp + "Sigma.diag"].float()
            alpha, s, c = _threshold_params(sd, vp + "Sigma.")
            rep.v_flattened = bool(alpha == 0.0 and torch.all(diag == 1.0))
            keep = (diag > alpha).nonzero(as_tuple=False).view(-1)
            s_eff = soft_threshold(diag, alpha, s, c)
            VS = V[keep] * s_eff[keep, None]
            if drop_zero_rows and keep.numel():
                nz = _keep_rows(VS)
                rep.v_dropped_zero_rows = int(keep.numel() - nz.numel())
                keep, VS = keep[nz], VS[nz]
            if keep.numel() == 0:
                keep = torch.zeros(1, dtype=torch.long)
                VS = torch.zeros(1, V.shape[1])
            out[vp + "VS.weight"] = VS.to(dtype)
            out[vp + "U.weight"] = U[:, keep].to(dtype)
            rep.v_rank = int(VS.shape[0])
            consumed.update(k for k in sd if k.startswith(vp))

        if rep.k_ranks is not None or rep.v_rank is not None:
            report.layers.append(rep)

    for k, v in sd.items():
        if k not in consumed:
            out[k] = v
    return out, report


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_state_dict_file(path: str, convert_legacy: bool = True, mmap: bool = True):
    """torch.load a checkpoint; legacy STAR-KV checkpoints are converted. Returns (sd, report|None)."""
    sd = torch.load(path, map_location="cpu", mmap=mmap, weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]
    report = None
    if is_legacy_starkv_state_dict(sd):
        if not convert_legacy:
            raise ValueError(f"{path} is a legacy STAR-KV U/Sigma/V checkpoint; pass convert_legacy=True")
        sd, report = convert_legacy_starkv(sd)
    elif not is_lowrank_state_dict(sd):
        raise ValueError(f"{path} carries no low-rank factors (no *.k_proj.VS.weight / *.v_proj.VS.weight keys)")
    return sd, report


def head_dim_of(config) -> int:
    return int(getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads)


@torch.no_grad()
def install_lowrank_modules(model: nn.Module, sd: Dict[str, torch.Tensor]) -> Dict[int, Dict[str, bool]]:
    """Replace k_proj / v_proj with low-rank modules shaped from the checkpoint's own tensors.

    Pruned factors have per-layer (and per-head) widths, so the modules must
    be built from the checkpoint before ``load_state_dict`` runs. Returns the
    {layer: {"k","v"}} map of what was installed.
    """
    layers = compressed_layers(sd)
    for i, which in layers.items():
        attn = model.model.layers[i].self_attn
        pre = layer_prefix(i)
        if which["k"]:
            VS, U = sd[pre + "k_proj.VS.weight"], sd[pre + "k_proj.U.weight"]
            ranks = sd[pre + "k_proj.head_ranks"].tolist()
            ref = attn.k_proj.weight
            assert U.shape[0] == ref.shape[0] and VS.shape[1] == ref.shape[1], (
                f"layer {i}: k_proj factors {tuple(U.shape)} x {tuple(VS.shape)} do not match the base "
                f"projection {tuple(ref.shape)}")
            attn.k_proj = HeadwiseLowRankLinear(VS.shape[1], U.shape[0], ranks, U.shape[0] // len(ranks),
                                                device=ref.device, dtype=ref.dtype)
        if which["v"]:
            VS, U = sd[pre + "v_proj.VS.weight"], sd[pre + "v_proj.U.weight"]
            ref = attn.v_proj.weight
            assert U.shape[0] == ref.shape[0] and VS.shape[1] == ref.shape[1], (
                f"layer {i}: v_proj factors {tuple(U.shape)} x {tuple(VS.shape)} do not match the base "
                f"projection {tuple(ref.shape)}")
            attn.v_proj = LowRankLinear(VS.shape[1], U.shape[0], VS.shape[0], device=ref.device, dtype=ref.dtype)
    return layers


@dataclass
class LoadInfo:
    path: Optional[str]
    compressed: Dict[int, Dict[str, bool]]
    skip_layers: Tuple[int, ...]
    converted_legacy: bool
    legacy_report: Optional[LegacyReport]
    overridden_dense_keys: int          # non-factor tensors the checkpoint overrode
    unexpected_keys: list


def load_lowrank_checkpoint(model: nn.Module, checkpoint, convert_legacy: bool = True) -> LoadInfo:
    """Install low-rank modules from ``checkpoint`` (path or state dict) and load every tensor it holds.

    Raises if any factor tensor fails to load; with ``strict=False`` an
    unloaded factor would silently stay at its random init.
    """
    path = None
    report = None
    if isinstance(checkpoint, str):
        path = checkpoint
        sd, report = load_state_dict_file(checkpoint, convert_legacy=convert_legacy)
    else:
        sd = checkpoint
        if is_legacy_starkv_state_dict(sd):
            sd, report = convert_legacy_starkv(sd)
    compressed = install_lowrank_modules(model, sd)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    factor_set = set(factor_keys(sd))
    critical = [k for k in missing if k in factor_set]
    if critical:
        raise RuntimeError(f"{len(critical)} factor tensors failed to load, e.g. {critical[:5]}")
    if unexpected:
        print(f"  warning: {len(unexpected)} checkpoint tensors have no target in the model, e.g. {unexpected[:5]}")
    num_layers = len(model.model.layers)
    return LoadInfo(
        path=path,
        compressed=compressed,
        skip_layers=infer_skip_layers(sd, num_layers),
        converted_legacy=report is not None,
        legacy_report=report,
        overridden_dense_keys=len(sd) - len(factor_set),
        unexpected_keys=list(unexpected),
    )


def save_state_dict(sd: Dict[str, torch.Tensor], path: str, only_factors: bool = False) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    sd = factors_only(sd) if only_factors else sd
    torch.save({k: v.contiguous() for k, v in sd.items()}, path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _describe(sd, model_dir: Optional[str] = None) -> dict:
    layers = compressed_layers(sd)
    n_factor = len(list(factor_keys(sd)))
    info = {
        "num_tensors": len(sd),
        "num_factor_tensors": n_factor,
        "num_other_tensors": len(sd) - n_factor,
        "compressed_layers": {str(i): w for i, w in layers.items()},
        "ranks": {},
    }
    for i in layers:
        k, v = layer_ranks(sd, i)
        info["ranks"][str(i)] = {"k_head_ranks": k, "v_rank": v}
    if model_dir:
        with open(os.path.join(model_dir, "config.json")) as f:
            cfg = json.load(f)
        info["skip_layers"] = list(infer_skip_layers(sd, cfg["num_hidden_layers"]))
    return info


def main():
    p = argparse.ArgumentParser(description="Inspect or convert low-rank KV checkpoints.")
    sub = p.add_subparsers(dest="cmd", required=True)
    pi = sub.add_parser("inspect", help="Print the compressed layers and ranks of a checkpoint")
    pi.add_argument("checkpoint")
    pi.add_argument("--model", default=None, help="Model directory (config.json) to list skipped layers")
    pi.add_argument("--json", default=None, help="Write the description to this JSON file")
    pc = sub.add_parser("convert", help="Convert a legacy STAR-KV U/Sigma/V checkpoint to the canonical layout")
    pc.add_argument("--input", required=True)
    pc.add_argument("--output", required=True)
    pc.add_argument("--factors-only", action="store_true",
                    help="Keep only the low-rank factors. STAR-KV fine-tunes every weight during KD, so "
                         "dropping the rest changes the model; use only for factor-only methods.")
    args = p.parse_args()

    if args.cmd == "inspect":
        sd, report = load_state_dict_file(args.checkpoint)
        if report is not None:
            print("legacy STAR-KV checkpoint, converted in memory:")
            print(report.summary())
        info = _describe(sd, args.model)
        print(json.dumps(info, indent=2))
        if args.json:
            with open(args.json, "w") as f:
                json.dump(info, f, indent=2)
    else:
        raw = torch.load(args.input, map_location="cpu", mmap=True, weights_only=False)
        if not is_legacy_starkv_state_dict(raw):
            raise SystemExit(f"{args.input} is not a legacy STAR-KV checkpoint")
        sd, report = convert_legacy_starkv(raw)
        print(report.summary())
        print(f"max U off-block energy over layers: {report.max_offblock_energy():.3e}")
        save_state_dict(sd, args.output, only_factors=args.factors_only)
        kept = len(factors_only(sd)) if args.factors_only else len(sd)
        print(f"wrote {args.output} ({kept} tensors{', factors only' if args.factors_only else ''})")


if __name__ == "__main__":
    main()
