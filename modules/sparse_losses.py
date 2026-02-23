import torch
import torch.nn as nn
from torch.autograd import Function
from torch.cuda.amp import custom_fwd, custom_bwd
from onmt.modules.sparse_activations import _threshold_and_support


class SparsemaxLossFunction(Function):

    @staticmethod
    @custom_fwd
    def forward(ctx, input, target):

        input_batch, classes = input.size()

        z_k = input.gather(1, target.unsqueeze(1)).squeeze()
        tau_z, support_size = _threshold_and_support(input, dim=1)
        support = input > tau_z
        x = torch.where(
            support, input**2 - tau_z**2,
            torch.tensor(0.0, device=input.device)
        ).sum(dim=1)
        ctx.save_for_backward(input, target, tau_z)

        return torch.clamp(x / 2 - z_k + 0.5, min=0.0)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        input, target, tau_z = ctx.saved_tensors
        sparsemax_out = torch.clamp(input - tau_z, min=0)
        delta = torch.zeros_like(sparsemax_out)
        delta.scatter_(1, target.unsqueeze(1), 1)
        return sparsemax_out - delta, None


sparsemax_loss = SparsemaxLossFunction.apply


class SparsemaxLoss(nn.Module):

    def __init__(self, weight=None, ignore_index=-100,
                 reduction='elementwise_mean'):
        assert reduction in ['elementwise_mean', 'sum', 'none']
        self.reduction = reduction
        self.weight = weight
        self.ignore_index = ignore_index
        super(SparsemaxLoss, self).__init__()

    def forward(self, input, target):
        loss = sparsemax_loss(input, target)
        if self.ignore_index >= 0:
            ignored_positions = target == self.ignore_index
            size = float((target.size(0) - ignored_positions.sum()).item())
            loss.masked_fill_(ignored_positions, 0.0)
        else:
            size = float(target.size(0))
        if self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'elementwise_mean':
            loss = loss.sum() / size
        return loss
