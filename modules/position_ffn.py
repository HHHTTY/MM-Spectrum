import torch.nn as nn
import torch.nn.functional as F


class ActivationFunction(object):
    relu = "relu"
    gelu = "gelu"


ACTIVATION_FUNCTIONS = {
    ActivationFunction.relu: F.relu,
    ActivationFunction.gelu: F.gelu,
}


class PositionwiseFeedForward(nn.Module):


    def __init__(self, d_model, d_ff, dropout=0.1,
                 activation_fn=ActivationFunction.relu):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout_1 = nn.Dropout(dropout)
        self.activation = ACTIVATION_FUNCTIONS[activation_fn]
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):


        inter = self.dropout_1(self.activation(self.w_1(self.layer_norm(x))))
        output = self.dropout_2(self.w_2(inter))
        return output + x

    def update_dropout(self, dropout):
        self.dropout_1.p = dropout
        self.dropout_2.p = dropout
