"""Implementation of "Attention is All You Need" (Transformer encoder).

This onmt_local4 variant includes an encoder-side MoE FFN option:
  - When enc_num_experts > 0, replace dense FFN with MoEPositionwiseFFN.
  - LayerNorm/residual for the MoE branch are handled in the encoder layer
    to match onmt PositionwiseFeedForward behaviour.

NEW in this patch:
  - Heterogeneous expert widths via enc_expert_dffs
  - Compute-aware regularization via enc_comp_beta
"""

import os
from typing import List, Optional

import torch
import torch.nn as nn

from onmt.encoders.encoder import EncoderBase
from onmt.modules import MultiHeadedAttention
from onmt.modules.moe_ffn import MoEPositionwiseFFN
from onmt.modules.position_ffn import ActivationFunction, PositionwiseFeedForward
from onmt.utils.misc import sequence_mask


def _parse_int_list(maybe_list) -> Optional[List[int]]:
    """Parse YAML/CLI value into list[int].

    Supports:
      - None
      - Python list/tuple (already parsed)
      - YAML string: "[2048, 4096, 8192]"
      - CSV string: "2048,4096,8192"
    """
    if maybe_list is None:
        return None
    if isinstance(maybe_list, (list, tuple)):
        return [int(x) for x in maybe_list]
    if isinstance(maybe_list, str):
        s = maybe_list.strip()
        if not s:
            return None
        # Try YAML first for bracket/list syntax.
        if s.startswith("["):
            try:
                import yaml  # type: ignore

                obj = yaml.safe_load(s)
                if isinstance(obj, (list, tuple)):
                    return [int(x) for x in obj]
            except Exception:
                # fall back to CSV parse below
                pass
        # CSV fallback
        return [int(x.strip()) for x in s.split(",") if x.strip()]
    raise TypeError(f"Unsupported type for list[int] parsing: {type(maybe_list)}")


class TransformerEncoderLayer(nn.Module):
    """A single layer of the transformer encoder."""

    def __init__(
        self,
        d_model: int,
        heads: int,
        d_ff: int,
        dropout: float,
        attention_dropout: float,
        max_relative_positions: int = 0,
        pos_ffn_activation_fn: ActivationFunction = ActivationFunction.relu,
        add_qkvbias: bool = False,
        # ---- Encoder MoE params ----
        enc_num_experts: int = 0,
        enc_num_experts_per_tok: int = 2,
        add_ffnbias: bool = True,
        enc_router_temp: float = 1.0,
        enc_router_noise_std: float = 0.0,
        enc_aux_loss_alpha: float = 0.02,
        # NEW: heterogeneous experts + compute-aware reg
        enc_expert_dffs: Optional[List[int]] = None,
        enc_comp_beta: float = 0.0,
        # NEW: explicit modality-aware routing
        enc_modality_aware_routing: bool = False,
        enc_num_modalities: int = 0,
    ):
        super().__init__()

        self.self_attn = MultiHeadedAttention(
            heads,
            d_model,
            dropout=attention_dropout,
            max_relative_positions=max_relative_positions,
            attn_type="self",
            add_qkvbias=add_qkvbias,
        )

        # Choose MoE-FFN or dense FFN.
        if enc_num_experts > 0:
            # Map ONMT activation enum -> string expected by MoE FFN.
            try:
                s = str(pos_ffn_activation_fn).lower()
                if s.endswith("gelu"):
                    act = "gelu"
                elif s.endswith("silu") or s.endswith("swish"):
                    act = "silu"
                else:
                    act = "relu"
            except Exception:
                act = "relu"

            self.feed_forward = MoEPositionwiseFFN(
                d_model=d_model,
                d_ff=d_ff,
                num_experts=enc_num_experts,
                k=enc_num_experts_per_tok,
                dropout=dropout,
                activation=act,
                add_bias=add_ffnbias,
                router_temp=enc_router_temp,
                router_noise_std=enc_router_noise_std,
                aux_alpha=enc_aux_loss_alpha,
                expert_dffs=enc_expert_dffs,
                comp_beta=enc_comp_beta,
                modality_aware=enc_modality_aware_routing,
                num_modalities=int(enc_num_modalities),
            )

            # For MoE branch we need explicit LN + residual here because
            # PositionwiseFeedForward in ONMT includes its own LN/residual.
            self.ffn_layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        else:
            self.feed_forward = PositionwiseFeedForward(
                d_model, d_ff, dropout, pos_ffn_activation_fn
            )

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

        if os.environ.get("ONMT_PRINT_FF", "0") == "1":
            print(
                f"[ENC-LAYER] feed_forward={type(self.feed_forward).__name__} "
                f"num_experts={getattr(self.feed_forward,'num_experts',0)} "
                f"expert_dffs={getattr(self.feed_forward,'expert_dffs',None)}"
            )

    def forward(self, layer_in, mask, mod_ids: Optional[torch.Tensor] = None):
        input_norm = self.layer_norm(layer_in)
        context, _ = self.self_attn(input_norm, input_norm, input_norm, mask=mask)
        x = layer_in + self.dropout(context)

        if isinstance(self.feed_forward, MoEPositionwiseFFN):
            ff_in = self.ffn_layer_norm(x)
            ff_out = self.feed_forward(ff_in, mod_ids=mod_ids)
            layer_out = x + self.dropout(ff_out)
        else:
            layer_out = self.feed_forward(x)
        return layer_out

    def update_dropout(self, dropout, attention_dropout):
        self.self_attn.update_dropout(attention_dropout)
        self.feed_forward.update_dropout(dropout)
        self.dropout.p = dropout


class TransformerEncoder(EncoderBase):
    """The Transformer encoder."""

    def __init__(
        self,
        num_layers,
        d_model,
        heads,
        d_ff,
        dropout,
        attention_dropout,
        embeddings,
        max_relative_positions,
        pos_ffn_activation_fn=ActivationFunction.relu,
        add_qkvbias=False,
        # ---- Encoder MoE params ----
        enc_num_experts=0,
        enc_num_experts_per_tok=2,
        add_ffnbias=True,
        enc_router_temp=1.0,
        enc_router_noise_std=0.0,
        enc_aux_loss_alpha=0.02,
        # NEW: heterogeneous experts + compute-aware reg
        enc_expert_dffs: Optional[List[int]] = None,
        enc_comp_beta: float = 0.0,
        # NEW: explicit modality-aware routing
        enc_modality_aware_routing: bool = False,
        enc_num_modalities: int = 0,
    ):
        super().__init__()

        self.embeddings = embeddings

        self.transformer = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    d_model,
                    heads,
                    d_ff,
                    dropout,
                    attention_dropout,
                    max_relative_positions=max_relative_positions,
                    pos_ffn_activation_fn=pos_ffn_activation_fn,
                    add_qkvbias=add_qkvbias,
                    enc_num_experts=enc_num_experts,
                    enc_num_experts_per_tok=enc_num_experts_per_tok,
                    add_ffnbias=add_ffnbias,
                    enc_router_temp=enc_router_temp,
                    enc_router_noise_std=enc_router_noise_std,
                    enc_aux_loss_alpha=enc_aux_loss_alpha,
                    enc_expert_dffs=enc_expert_dffs,
                    enc_comp_beta=enc_comp_beta,
                    enc_modality_aware_routing=enc_modality_aware_routing,
                    enc_num_modalities=enc_num_modalities,
                )
                for _ in range(num_layers)
            ]
        )
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

        if os.environ.get("ONMT_PRINT_MOE_ALPHA", "0") == "1":
            for i, layer in enumerate(self.transformer):
                ff = layer.feed_forward
                if isinstance(ff, MoEPositionwiseFFN):
                    print(
                        f"[MoE] layer={i} alpha={getattr(ff,'aux_alpha',None)} "
                        f"beta={getattr(ff,'comp_beta',None)} dffs={getattr(ff,'expert_dffs',None)}"
                    )

    @classmethod
    def from_opt(cls, opt, embeddings):
        """Alternate constructor."""
        enc_expert_dffs = _parse_int_list(getattr(opt, "enc_expert_dffs", None))
        enc_comp_beta = float(getattr(opt, "enc_comp_beta", 0.0))
        enc_modality_aware_routing = bool(getattr(opt, "enc_modality_aware_routing", False))
        enc_num_modalities = int(getattr(opt, "enc_num_modalities", 0))

        return cls(
            opt.enc_layers,
            opt.enc_hid_size,
            opt.heads,
            opt.transformer_ff,
            opt.dropout[0] if type(opt.dropout) is list else opt.dropout,
            opt.attention_dropout[0]
            if type(opt.attention_dropout) is list
            else opt.attention_dropout,
            embeddings,
            opt.max_relative_positions,
            pos_ffn_activation_fn=opt.pos_ffn_activation_fn,
            add_qkvbias=opt.add_qkvbias,
            enc_num_experts=getattr(opt, "enc_num_experts", 0),
            enc_num_experts_per_tok=getattr(opt, "enc_num_experts_per_tok", 2),
            add_ffnbias=getattr(opt, "add_ffnbias", True),
            enc_router_temp=getattr(opt, "enc_router_temp", 1.0),
            enc_router_noise_std=getattr(opt, "enc_router_noise_std", 0.0),
            enc_aux_loss_alpha=getattr(opt, "enc_aux_loss_alpha", 0.02),
            enc_expert_dffs=enc_expert_dffs,
            enc_comp_beta=enc_comp_beta,
            enc_modality_aware_routing=enc_modality_aware_routing,
            enc_num_modalities=enc_num_modalities,
        )

    def forward(self, src, src_len=None):
        # src is LongTensor (batch, len, nfeat). If nfeat>1, the 2nd column
        # corresponds to the first source feature (e.g., modality id).
        mod_ids = None
        if isinstance(src, torch.Tensor) and src.dim() == 3 and src.size(2) > 1:
            mod_ids = src[:, :, 1].contiguous()
        enc_out = self.embeddings(src)
        mask = ~sequence_mask(src_len).unsqueeze(1)
        mask = mask.unsqueeze(1)
        mask = mask.expand(-1, -1, mask.size(3), -1)
        for layer in self.transformer:
            enc_out = layer(enc_out, mask, mod_ids=mod_ids)
        enc_out = self.layer_norm(enc_out)
        return enc_out, None, src_len

    def update_dropout(self, dropout, attention_dropout):
        self.embeddings.update_dropout(dropout)
        for layer in self.transformer:
            layer.update_dropout(dropout, attention_dropout)
