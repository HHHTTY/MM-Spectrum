# onmt_local/onmt/modules/aux_registry.py
import torch

# 训练一步期间，MoE 各层把 aux loss 丢进来；trainer 再一次性取走
_AUX_LOSSES = []

def push(loss: torch.Tensor):
    if loss is not None:
        _AUX_LOSSES.append(loss)

def pop_sum(device=None):
    if not _AUX_LOSSES:
        return None
    s = torch.stack([x.to(device) for x in _AUX_LOSSES]).sum()
    _AUX_LOSSES.clear()
    return s
