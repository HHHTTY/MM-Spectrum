
import math
import os
from typing import List, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from . import aux_registry  # local relative import


_ACTS = {"relu": F.relu, "gelu": F.gelu, "silu": F.silu}


def _cv2(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Coefficient of variation squared."""
    m = x.mean()
    return ((x - m).pow(2).mean() / (m * m + eps))


def _balance_aux_loss(
    probs: torch.Tensor,
    hard_counts: torch.Tensor,
    alpha: float = 0.02,
) -> torch.Tensor:

    S, E = probs.size()
    imp = probs.sum(dim=0)  # importance (differentiable)
    load = hard_counts.detach().to(probs.dtype)  # hard load (no grad)

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(imp, op=dist.ReduceOp.SUM)
        dist.all_reduce(load, op=dist.ReduceOp.SUM)

    imp_share = imp / (imp.sum() + 1e-8)
    load_share = load / (load.sum() + 1e-8)
    loss_cv = _cv2(imp_share) + _cv2(load_share)
    return probs.new_tensor(float(alpha)) * loss_cv


class _FFN(nn.Module):
    """Standard Transformer FFN expert: D -> d_ff -> D."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = "relu",
        add_bias: bool = True,
    ):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=add_bias)
        self.w2 = nn.Linear(d_ff, d_model, bias=add_bias)
        self.dropout = nn.Dropout(dropout)
        self.act = _ACTS.get(str(activation).lower(), F.relu)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.dropout(self.act(self.w1(x))))

    def update_dropout(self, p: float):
        self.dropout.p = p


class MoEPositionwiseFFN(nn.Module):
    """Encoder-side MoE FFN.

    Notes:
      - This module does NOT include LayerNorm / residual. The caller (encoder
        layer) must handle that, consistent with how onmt_local4 integrates it.

    Args:
      d_model: model dim D
      d_ff: default expert width (used when expert_dffs is None)
      num_experts: number of experts E
      k: top-k routing per token
      dropout: dropout in expert FFNs
      activation: relu/gelu/silu
      add_bias: whether experts have bias
      router_temp: temperature for logits (1.0 = no scaling)
      router_noise_std: training-time Gaussian noise std added to logits
      aux_alpha: coefficient for load-balance aux loss
      expert_dffs: optional list[int] of length E specifying per-expert d_ff
      comp_beta: coefficient for compute-aware regularization
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_experts: int,
        k: int = 2,
        dropout: float = 0.1,
        activation: str = "relu",
        add_bias: bool = True,
        router_temp: float = 1.0,
        router_noise_std: float = 0.0,
        aux_alpha: float = 0.02,
        expert_dffs: Optional[List[int]] = None,
        comp_beta: float = 0.0,
        # NEW: explicit modality-aware routing: logits = W h + b_modality
        modality_aware: bool = False,
        num_modalities: int = 0,
    ):
        super().__init__()
        assert num_experts >= 1 and 1 <= k <= num_experts
        self.num_experts = int(num_experts)
        self.k = int(k)

        self.gate = nn.Linear(d_model, num_experts, bias=False)

        # ---- Explicit Modality-aware routing (router bias by modality id) ----
        # logits = W h + b_modality, where b_modality is a learned vector in R^E.
        self.modality_aware = bool(modality_aware)
        self.num_modalities = int(num_modalities)
        if self.modality_aware:
            if self.num_modalities <= 0:
                raise ValueError(
                    "num_modalities must be > 0 when modality_aware is enabled"
                )
            self.mod_bias = nn.Embedding(self.num_modalities, self.num_experts)
            nn.init.zeros_(self.mod_bias.weight)
            self._warned_missing_mod_ids = False

        self.router_temp = float(router_temp)
        self.router_noise_std = float(router_noise_std)
        self.aux_alpha = float(aux_alpha)
        self.comp_beta = float(comp_beta)

        if self.router_temp <= 0:
            raise ValueError("router_temp must be > 0")

        # ---- Heterogeneous expert widths ----
        if expert_dffs is None:
            expert_dffs = [int(d_ff)] * self.num_experts
        if len(expert_dffs) != self.num_experts:
            raise ValueError(
                f"expert_dffs must have length num_experts={self.num_experts}, "
                f"got {len(expert_dffs)}"
            )
        self.expert_dffs = [int(x) for x in expert_dffs]

        self.experts = nn.ModuleList(
            [
                _FFN(
                    d_model,
                    dff,
                    dropout=dropout,
                    activation=activation,
                    add_bias=add_bias,
                )
                for dff in self.expert_dffs
            ]
        )
        self.dropout = nn.Dropout(dropout)

        # ---- Compute costs (normalized) for compute-aware regularization ----
        # Approx FLOPs ~ 2 * D * d_ff for 2-layer FFN (ignoring bias/act).
        costs = torch.tensor(
            [2.0 * float(d_model) * float(dff) for dff in self.expert_dffs],
            dtype=torch.float,
        )
        costs = costs / costs.mean().clamp_min(1e-8)
        self.register_buffer("expert_costs", costs)

    def forward(self, x: torch.Tensor, mod_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward.

        Args:
            x: (B, T, D)
        Returns:
            y: (B, T, D)
        """
        B, T, D = x.size()
        S = B * T
        xf = x.reshape(S, D)

        logits = self.gate(xf)  # (S, E)

        # Add modality bias if provided: logits += b_modality
        if getattr(self, "modality_aware", False):
            if mod_ids is None:
                # Don't spam logs; warn once if enabled but mod_ids not provided.
                if not getattr(self, "_warned_missing_mod_ids", False) and os.environ.get(
                    "MOE_WARN_MISSING_MOD", "1"
                ) == "1":
                    print(
                        "[MoE][WARN] modality_aware routing enabled but mod_ids is None; "
                        "skipping b_modality."
                    )
                    self._warned_missing_mod_ids = True
            else:
                if mod_ids.dim() == 2:
                    mod_flat = mod_ids.reshape(-1)
                else:
                    mod_flat = mod_ids
                mod_flat = mod_flat.to(dtype=torch.long, device=xf.device)
                logits = logits + self.mod_bias(mod_flat)

        # Training-time router noise to reduce early collapse.
        if self.training and self.router_noise_std > 0.0:
            logits = logits + torch.randn_like(logits) * self.router_noise_std

        # Temperature scaling (also allowed at inference).
        if self.router_temp != 1.0:
            logits = logits / self.router_temp

        # Full probabilities over all experts (for importance + compute reg).
        probs = F.softmax(logits, dim=-1)  # (S, E)

        # Top-k routing.
        topv, topi = torch.topk(logits, k=self.k, dim=-1)  # (S, k)
        gates = F.softmax(topv, dim=-1)  # (S, k)

        # ---- Aux losses (only when training with gradients) ----
        if self.training and torch.is_grad_enabled():
            # Hard counts for load-balance.
            hard_counts = torch.bincount(topi.reshape(-1), minlength=self.num_experts)
            alpha = float(getattr(self, "aux_alpha", 0.02))
            if alpha != 0.0:
                aux_registry.push(_balance_aux_loss(probs, hard_counts, alpha=alpha))

            # Compute-aware regularization: expected cost under router probs.
            beta = float(getattr(self, "comp_beta", 0.0))
            if beta != 0.0:
                # mean over tokens of sum_e p(t,e) * cost_e
                comp = (probs * self.expert_costs.unsqueeze(0)).sum(dim=-1).mean()
                aux_registry.push(probs.new_tensor(beta) * comp)

        # ---- Diagnostics (optional) ----
        with torch.no_grad():
            top1 = topi[:, 0]  # (S,)
            cnt = torch.bincount(top1, minlength=self.num_experts).float()
            frac = cnt / max(S, 1)

            p = frac.clamp_min(1e-12)
            H = -(p * p.log()).sum() / (
                math.log(self.num_experts) if self.num_experts > 1 else 1.0
            )
            imbalance = (p.max() / p.mean()).item() if p.mean() > 0 else float("inf")

            mean_gate_per_rank = []
            for r in range(self.k):
                sum_w = torch.zeros(self.num_experts, device=x.device, dtype=x.dtype)
                sum_w.scatter_add_(0, topi[:, r], gates[:, r])
                cnt_r = (
                    F.one_hot(topi[:, r], num_classes=self.num_experts)
                    .float()
                    .sum(dim=0)
                    .clamp_min(1.0)
                )
                mean_gate_per_rank.append((sum_w / cnt_r).detach().cpu())

            
            # --- Additional interpretability statistics (detached) ---
            # raw entropy / effective experts
            p_raw = frac.clamp_min(1e-12)
            H_raw = -(p_raw * p_raw.log()).sum()
            neff = float(torch.exp(H_raw).detach().cpu())
            # cv / kl to uniform / max share
            mean_p = p_raw.mean()
            std_p = p_raw.std(unbiased=False)
            cv = float((std_p / (mean_p + 1e-12)).detach().cpu())
            logE = math.log(self.num_experts) if self.num_experts > 1 else 1.0
            kl_uniform = float((p_raw * (p_raw.log() + logE)).sum().detach().cpu())
            max_share = float(p_raw.max().detach().cpu())

            # Top-k soft mass over experts: sum_r gate_r for the chosen expert at rank r.
            mass = torch.zeros(self.num_experts, device=xf.device, dtype=xf.dtype)
            for r in range(self.k):
                mass.scatter_add_(0, topi[:, r], gates[:, r])
            mass = (mass / max(S, 1)).detach().cpu()
            coverage_topk = int((mass > 0).sum().item())

            # Expected compute cost under full probs vs top-k routed gates.
            expected_cost_soft = float((probs * self.expert_costs.unsqueeze(0)).sum(dim=-1).mean().detach().cpu())
            topk_cost = self.expert_costs[topi]  # (S,k)
            expected_cost_topk = float((gates * topk_cost).sum(dim=-1).mean().detach().cpu())

            # Optional token-level trace (sampled): useful for case studies / stage-wise routing.
            trace = None
            if os.environ.get("MOE_LOG_TRACE", "0") == "1":
                max_n = int(os.environ.get("MOE_TRACE_MAX", "256"))
                n = min(int(S), max_n)
                # uniform sample without replacement
                if n > 0:
                    idx = torch.randperm(int(S), device=xf.device)[:n]
                    trace = {
                        "idx": idx.detach().cpu().tolist(),
                        "topi": topi[idx].detach().cpu().tolist(),
                        "gates": gates[idx].detach().cpu().tolist(),
                    }
                    if mod_ids is not None and mod_ids.numel() == S:
                        trace["mod"] = mod_ids.reshape(-1)[idx].detach().cpu().tolist()

            # Modality-grouped stats (requires mod_ids of shape (B,T) or (S,))
            mod_stats = None
            if mod_ids is not None:
                mod_flat = mod_ids.reshape(-1).to(dtype=torch.long, device=xf.device)
                # exclude padding modality id if possible
                uniq_mod = torch.unique(mod_flat)
                mod_top1_count = {}
                mod_top1_frac = {}
                mod_topk_mass = {}
                mod_expected_cost_soft = {}
                mod_expected_cost_topk = {}
                mod_mean_gate_per_rank = {}
                for m_id in uniq_mod.detach().cpu().tolist():
                    m_id = int(m_id)
                    mask = (mod_flat == m_id)
                    n_m = int(mask.sum().item())
                    if n_m <= 0:
                        continue
                    top1_m = top1[mask]
                    cnt_m = torch.bincount(top1_m, minlength=self.num_experts).float()
                    frac_m = (cnt_m / max(n_m, 1)).detach().cpu()
                    mod_top1_count[str(m_id)] = cnt_m.detach().cpu().tolist()
                    mod_top1_frac[str(m_id)] = frac_m.tolist()

                    mass_m = torch.zeros(self.num_experts, device=xf.device, dtype=xf.dtype)
                    for r in range(self.k):
                        mass_m.scatter_add_(0, topi[mask, r], gates[mask, r])
                    mass_m = (mass_m / max(n_m, 1)).detach().cpu()
                    mod_topk_mass[str(m_id)] = mass_m.tolist()

                    probs_m = probs[mask]
                    mod_expected_cost_soft[str(m_id)] = float((probs_m * self.expert_costs.unsqueeze(0)).sum(dim=-1).mean().detach().cpu())
                    topk_cost_m = self.expert_costs[topi[mask]]
                    mod_expected_cost_topk[str(m_id)] = float((gates[mask] * topk_cost_m).sum(dim=-1).mean().detach().cpu())

                    # rank-wise mean gate per expert (conditional on selection at that rank)
                    per_rank = []
                    for r in range(self.k):
                        sum_wm = torch.zeros(self.num_experts, device=xf.device, dtype=xf.dtype)
                        sum_wm.scatter_add_(0, topi[mask, r], gates[mask, r])
                        cnt_rm = (
                            F.one_hot(topi[mask, r], num_classes=self.num_experts)
                            .float()
                            .sum(dim=0)
                            .clamp_min(1.0)
                        )
                        per_rank.append((sum_wm / cnt_rm).detach().cpu().tolist())
                    mod_mean_gate_per_rank[str(m_id)] = per_rank

                mod_stats = {
                    "mod_top1_count": mod_top1_count,
                    "mod_top1_frac": mod_top1_frac,
                    "mod_topk_mass": mod_topk_mass,
                    "mod_expected_cost_soft": mod_expected_cost_soft,
                    "mod_expected_cost_topk": mod_expected_cost_topk,
                    "mod_mean_gate_per_rank": mod_mean_gate_per_rank,
                }

            self.last_stats = {
                "tokens": int(S),
                "num_experts": int(self.num_experts),
                "k": int(self.k),
                "top1_count": cnt.detach().cpu().tolist(),
                "top1_frac": frac.detach().cpu().tolist(),
                "topk_mass": mass.tolist(),
                "coverage_topk": coverage_topk,

                "entropy_norm": float(H.detach().cpu()),
                "entropy_raw": float(H_raw.detach().cpu()),
                "neff": neff,
                "imbalance": float(imbalance),
                "cv": cv,
                "kl_uniform": kl_uniform,
                "max_share": max_share,

                "mean_gate_per_rank": [t.tolist() for t in mean_gate_per_rank],
                "expected_cost_soft": expected_cost_soft,
                "expected_cost_topk": expected_cost_topk,

                "expert_dffs": list(self.expert_dffs),
                "expert_costs": self.expert_costs.detach().cpu().tolist(),
                "router_temp": float(getattr(self, "router_temp", 1.0)),
                "router_noise_std": float(getattr(self, "router_noise_std", 0.0)),
                "aux_alpha": float(getattr(self, "aux_alpha", 0.0)),
                "comp_beta": float(getattr(self, "comp_beta", 0.0)),
                "modality_aware": bool(getattr(self, "modality_aware", False)),
                "num_modalities": int(getattr(self, "num_modalities", 0)),
            }
            if trace is not None:
                self.last_stats["trace"] = trace
            if mod_stats is not None:
                self.last_stats.update(mod_stats)


            self._steps = getattr(self, "_steps", 0) + 1
            if os.environ.get("MOE_DEBUG", "0") == "1":
                every = int(os.environ.get("MOE_DEBUG_EVERY", "200"))
                if self.training and self._steps % every == 0:
                    msg = (
                        f"[MoE] step={self._steps} E={self.num_experts} k={self.k} "
                        f"H={self.last_stats['entropy_norm']:.3f} "
                        f"imbalance={self.last_stats['imbalance']:.2f} "
                        f"top1={self.last_stats['top1_count'].tolist()} "
                        f"alpha={float(getattr(self,'aux_alpha',0.0)):.4f} "
                        f"beta={float(getattr(self,'comp_beta',0.0)):.4f} "
                        f"sigma={float(getattr(self,'router_noise_std',0.0)):.4f}"
                    )
                    print(msg)

        # ---- Expert aggregation (reference implementation; correct but not fastest) ----
        y = torch.zeros(S, D, dtype=x.dtype, device=x.device)
        for r in range(self.k):
            e_idx = topi[:, r]  # (S,)
            w = gates[:, r].unsqueeze(1)  # (S, 1)
            for e in range(self.num_experts):
                take = (e_idx == e).nonzero(as_tuple=False).flatten()
                if take.numel() == 0:
                    continue
                ye = self.experts[e](xf.index_select(0, take)) * w.index_select(0, take)
                y.index_add_(0, take, ye)

        return y.reshape(B, T, D)

    # ---- setters (used by Trainer scheduling) ----
    def set_aux_alpha(self, alpha: float):
        self.aux_alpha = float(alpha)

    def set_router_noise(self, sigma: float):
        self.router_noise_std = float(sigma)

    def set_comp_beta(self, beta: float):
        self.comp_beta = float(beta)

    def update_dropout(self, p: float):
        for e in self.experts:
            e.update_dropout(p)
        self.dropout.p = p
