import torch
import torch.nn as nn


class Elementwise(nn.ModuleList):


    def __init__(self, merge=None, *args):
        assert merge in [None, 'first', 'concat', 'sum', 'mlp']
        self.merge = merge
        super(Elementwise, self).__init__(*args)

    def forward(self, emb):
        emb_ = [feat.squeeze(2) for feat in emb.split(1, dim=2)]
        assert len(self) == len(emb_)
        emb_out = [f(x) for f, x in zip(self, emb_)]
        if self.merge == 'first':
            return emb_out[0]
        elif self.merge == 'concat' or self.merge == 'mlp':
            return torch.cat(emb_out, 2)
        elif self.merge == 'sum':
            return sum(emb_out)
        else:
            return emb_out


class Cast(nn.Module):

    def __init__(self, dtype):
        super(Cast, self).__init__()
        self._dtype = dtype

    def forward(self, x):
        return x.to(self._dtype)
