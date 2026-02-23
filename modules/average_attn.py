# -*- coding: utf-8 -*-
"""Average Attention module."""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional
from onmt.modules.position_ffn import PositionwiseFeedForward
from onmt.modules.position_ffn import ActivationFunction


def cumulative_average_mask(batch_size: int, t_len: int,
                            device: Optional[torch.device] = None) -> Tensor:


    triangle = torch.tril(torch.ones(t_len, t_len,
                          dtype=torch.float, device=device))
    weights = torch.ones(1, t_len, dtype=torch.float, device=device) \
        / torch.arange(1, t_len + 1, dtype=torch.float, device=device)
    mask = triangle * weights.transpose(0, 1)

    return mask.unsqueeze(0).expand(batch_size, t_len, t_len)


def cumulative_average(layer_in: Tensor, layer_cache: tuple,
                       mask=None, step=None) -> Tensor:

    if layer_cache[0]:
        average_attention = (layer_in + step *
                             layer_cache[1]['prev_g']) / (step + 1)
        layer_cache[1]['prev_g'] = average_attention
        return average_attention
    else:
        return torch.matmul(mask.to(layer_in.dtype), layer_in)


class AverageAttention(nn.Module):

    def __init__(self, model_dim, dropout=0.1, aan_useffn=False,
                 pos_ffn_activation_fn=ActivationFunction.relu):
        self.model_dim = model_dim
        self.aan_useffn = aan_useffn
        super(AverageAttention, self).__init__()
        if aan_useffn:
            self.average_layer = PositionwiseFeedForward(model_dim, model_dim,
                                                         dropout,
                                                         pos_ffn_activation_fn
                                                         )
        self.gating_layer = nn.Linear(model_dim * 2, model_dim * 2)
        self.layer_cache = False, {'prev_g': torch.tensor([])}

    # @torch.jit.script
    def forward(self, layer_in, mask=None, step=None):

        batch_size = layer_in.size(0)
        t_len = layer_in.size(1)
        mask = cumulative_average_mask(batch_size, t_len, layer_in.device)\
            if not self.layer_cache[0] else None
        average_out = cumulative_average(
          layer_in, self.layer_cache, mask, step)
        if self.aan_useffn:
            average_out = self.average_layer(average_out)
        gating_out = self.gating_layer(torch.cat((layer_in,
                                                  average_out), -1))
        input_gate, forget_gate = torch.chunk(gating_out, 2, dim=2)
        gating_out = torch.sigmoid(input_gate) * layer_in + \
            torch.sigmoid(forget_gate) * average_out

        return gating_out, average_out
