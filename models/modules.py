import torch
import torch.nn as nn
import numpy as np

def get_activation(s_act):
    if s_act == "relu":
        return nn.ReLU(inplace=True)
    elif s_act == "sigmoid":
        return nn.Sigmoid()
    elif s_act == "softplus":
        return nn.Softplus()
    elif s_act == "linear":
        return None
    elif s_act == "tanh":
        return nn.Tanh()
    elif s_act == "leakyrelu":
        return nn.LeakyReLU(0.2, inplace=True)
    elif s_act == "softmax":
        return nn.Softmax(dim=1)
    elif s_act == "selu":
        return nn.SELU()
    elif s_act == "elu":
        return nn.ELU()
    elif s_act["type"] == 'superquadric_constraints':
        return superquadric_constraints(**s_act)
        
    else:
        raise ValueError(f"Unexpected activation: {s_act}")

class FC_vec(nn.Module):
    def __init__(
        self,
        in_chan=2,
        out_chan=1,
        l_hidden=None,
        activation=None,
        out_activation=None,
    ):
        super(FC_vec, self).__init__()

        self.in_chan = in_chan
        self.out_chan = out_chan
        l_neurons = l_hidden + [out_chan]
        activation = activation + [out_activation]

        l_layer = []
        prev_dim = in_chan
        for [n_hidden, act] in (zip(l_neurons, activation)):
            l_layer.append(nn.Linear(prev_dim, n_hidden))
            act_fn = get_activation(act)
            if act_fn is not None:
                l_layer.append(act_fn)
            prev_dim = n_hidden

        self.net = nn.Sequential(*l_layer)

    def forward(self, x):
        return self.net(x)

class superquadric_constraints(nn.Module):
    def __init__(
        self,
        size_weight=1.0,
        size_bias=0.001,
        shape_weight=1.0,
        shape_bias=0.19,
        **kwargs
    ):
        super(superquadric_constraints, self).__init__()
        self.size_weight = size_weight
        self.size_bias = size_bias
        self.shape_weight = shape_weight
        self.shape_bias = shape_bias
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):        
        # size
        x[:, -5:-2] = self.size_weight * self.sigmoid(x[:, -5:-2]) + self.size_bias
        # shape
        x[:, -2:] = self.shape_weight * self.sigmoid(x[:, -2:]) + self.shape_bias
        return x