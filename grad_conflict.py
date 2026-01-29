
from __future__ import annotations
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


def _select_params(model: nn.Module, scope: str) -> List[Tuple[str, torch.Tensor]]:
    scope = (scope or "router").lower()
    items: List[Tuple[str, torch.Tensor]] = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if scope == "all":
            items.append((n, p))
        elif scope == "encoder":
            if n.startswith("encoder.") or ".encoder." in n:
                items.append((n, p))
        else:  # router
            # Heuristic matches this repo's MoE FFN: gate, mod_bias, router noise/temp don't have params.
            if (".gate." in n) or (".mod_bias" in n) or ("moe" in n.lower() and "gate" in n.lower()):
                items.append((n, p))
    return items


def _flatten_grads(grads: List[Optional[torch.Tensor]]) -> torch.Tensor:
    vecs = []
    for g in grads:
        if g is None:
            continue
        vecs.append(g.detach().reshape(-1).float())
    if not vecs:
        return torch.zeros(1)
    return torch.cat(vecs, dim=0)


def _cos(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> float:
    a = a.float()
    b = b.float()
    na = a.norm() + eps
    nb = b.norm() + eps
    return float((a @ b) / (na * nb))


def _infer_pad_id(src_tokens: torch.Tensor) -> int:
    # Heuristic: pad dominates frequency.
    flat = src_tokens.reshape(-1)
    vals, counts = torch.unique(flat, return_counts=True)
    pad = int(vals[counts.argmax()].item())
    return pad


def _mask_src_by_mod(src: torch.Tensor, mod_ids: torch.Tensor, keep_mod: int, pad_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
    # src: (B,T,F) where src[:,:,0] are token ids. keep only tokens with mod==keep_mod.
    B, T, F = src.size()
    src2 = src.clone()
    keep = (mod_ids == keep_mod)
    # token ids
    tok = src2[:, :, 0]
    tok = tok.masked_fill(~keep, pad_id)
    src2[:, :, 0] = tok
    # lengths: count non-pad
    lens = (tok != pad_id).sum(dim=1).clamp_min(1)
    return src2, lens


def maybe_log_grad_conflict(step: int, split: str, model: nn.Module, loss_compute, batch: Dict[str, torch.Tensor]) -> None:
    if os.environ.get("GRAD_CONFLICT_LOG", "0") != "1":
        return
    every = int(os.environ.get("GRAD_CONFLICT_EVERY", "2000"))
    if every <= 0 or (int(step) % every != 0):
        return

    out_dir = Path(os.environ.get("GRAD_CONFLICT_DIR", "./moe_logs"))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "grad_conflict.jsonl"
    scope = os.environ.get("GRAD_CONFLICT_SCOPE", "router")

    # We require mod ids in src feature dim 1.
    if "src" not in batch or "tgt" not in batch:
        return
    src = batch["src"]
    tgt = batch["tgt"]
    src_len = batch.get("srclen", None)
    if src.dim() < 3 or src.size(-1) < 2:
        # no modality channel
        return

    src_tok = src[:, :, 0]
    mod_ids = src[:, :, 1].long()

    # Determine modalities to probe
    uniq = torch.unique(mod_ids)
    uniq = uniq.detach().cpu().tolist()
    # Drop likely padding modality if present (heuristic: modality id at pad positions)
    pad_id = _infer_pad_id(src_tok.detach())
    pad_mask = (src_tok == pad_id)
    if pad_mask.any():
        pad_mod = int(torch.mode(mod_ids[pad_mask].reshape(-1))[0].item())
        uniq = [m for m in uniq if m != pad_mod]
    # Cap count
    max_mod = int(os.environ.get("GRAD_CONFLICT_MAX_MODALITIES", "8"))
    uniq = uniq[:max_mod]
    if len(uniq) < 2:
        return

    params = _select_params(model, scope)
    if not params:
        return
    p_tensors = [p for _, p in params]

    grads_by_mod: Dict[int, torch.Tensor] = {}
    losses: Dict[int, float] = {}
    device = src.device

    # compute per-modality gradients
    for m in uniq:
        try:
            src_m, src_len_m = _mask_src_by_mod(src, mod_ids, int(m), pad_id=pad_id)
            # forward
            model.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=getattr(loss_compute, "use_amp", False)):
                model_out, attns = model(src_m, tgt, src_len_m, with_align=False)
                loss, _stats = loss_compute(batch, model_out, attns)
            if loss is None:
                continue
            loss_scalar = loss.mean()
            g = torch.autograd.grad(loss_scalar, p_tensors, retain_graph=False, create_graph=False, allow_unused=True)
            gv = _flatten_grads(list(g)).to("cpu")
            grads_by_mod[int(m)] = gv
            losses[int(m)] = float(loss_scalar.detach().cpu().item())
        except Exception:
            continue

    mods = sorted(grads_by_mod.keys())
    if len(mods) < 2:
        return

    cos_pairs = {}
    neg = 0
    total = 0
    cos_list = []
    for i in range(len(mods)):
        for j in range(i + 1, len(mods)):
            a = grads_by_mod[mods[i]]
            b = grads_by_mod[mods[j]]
            c = _cos(a, b)
            cos_pairs[f"{mods[i]}-{mods[j]}"] = c
            cos_list.append(c)
            total += 1
            if c < 0:
                neg += 1

    rec = {
        "ts": time.time(),
        "step": int(step),
        "split": str(split),
        "scope": str(scope),
        "pad_token_id": int(pad_id),
        "modalities": mods,
        "loss_by_mod": losses,
        "conflict_rate": float(neg / max(1, total)),
        "mean_cos": float(sum(cos_list) / max(1, len(cos_list))),
        "min_cos": float(min(cos_list)) if cos_list else 0.0,
        "pairs": cos_pairs,
        "n_params": int(sum(p.numel() for p in p_tensors)),
    }
    with out_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
