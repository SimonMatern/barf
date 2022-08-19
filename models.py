import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import pytorch_lightning as pl
from kornia.utils import create_meshgrid
from torch.utils.data import DataLoader, TensorDataset
from einops import rearrange
from torchvision.io import read_image
from torchvision import transforms
import torchvision
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image
from collections import OrderedDict
from kornia.augmentation import RandomAffine, CenterCrop
import cv2 as cv



class Homography(nn.Module):
    def __init__(self, w=None):
        super().__init__()
        if w is None:
            self.H = torch.nn.Parameter(torch.zeros(8))
        else:
            self.H = torch.nn.Parameter(w)
        #self.register_parameter(name='H', param=torch.nn.Parameter(torch.randn(8)))

    def sl3_to_SL3(self,h):
        # homography: directly expand matrix exponential
        # https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.61.6151&rep=rep1&type=pdf
        h1,h2,h3,h4,h5,h6,h7,h8 = h.chunk(8,dim=-1)
        A = torch.stack([torch.cat([h5,h3,h1],dim=-1),
                         torch.cat([h4,-h5-h6,h2],dim=-1),
                         torch.cat([h7,h8,h6],dim=-1)],dim=-2)
        H = A.matrix_exp()
        return H
    
    def get_H(self):
        return self.sl3_to_SL3(self.H)


    def forward(self, x, inv=False):
        H = self.sl3_to_SL3(self.H)
        if inv:
            H = torch.linalg.inv(H)

        return H@x


class NeuralRenderer(nn.Module):
    def __init__(self, pos_enc = False, L = 10):
        super().__init__()
        self.pos_enc = pos_enc
        self.L = L

        if pos_enc:
            input_size = 2*2*L
        else:
            input_size = 2

        self.input_layer = nn.Linear(input_size,256)
        self.hidden = nn.Sequential(*[nn.Linear(256,256), nn.ReLU()]*6)
        self.output_layer = nn.Linear(256,3)

    def positional_encoding(self, input, L): # [B,...,N]
        shape = input.shape
        freq = 2**torch.arange(L,dtype=torch.float32).to(input.device)*np.pi # [L]
        spectrum = input[...,None]*freq # [B,...,N,L]
        sin,cos = spectrum.sin(),spectrum.cos() # [B,...,N,L]
        input_enc = torch.stack([sin,cos],dim=-2) # [B,...,N,2,L]
        input_enc = input_enc.view(*shape[:-1],-1) # [B,...,2NL]
        # coarse-to-fine: smoothly mask positional encoding for BARF
        return input_enc

    def forward(self, x):
        if self.pos_enc:
            x = self.positional_encoding(x, self.L)

        x = F.relu(self.input_layer(x))
        x = self.hidden(x)
        x = self.output_layer(x)
        return x


class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.

    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a
    # hyperparameter.

    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)

    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                             1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                             np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

    def forward_with_intermediate(self, input):
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate


class Siren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False,
                 first_omega_0=30, hidden_omega_0=30.):
        super().__init__()

        self.net = []
        self.net.append(SineLayer(in_features, hidden_features,
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features,
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)

            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                              np.sqrt(6 / hidden_features) / hidden_omega_0)

            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features,
                                      is_first=False, omega_0=hidden_omega_0))

        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        #coords = coords.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        output = self.net(coords)
        return output, coords

    def forward_with_activations(self, coords, retain_grad=False):
        '''Returns not only model output, but also intermediate activations.
        Only used for visualizing activations later!'''
        activations = OrderedDict()

        activation_count = 0
        x = coords.clone().detach().requires_grad_(True)
        activations['input'] = x
        for i, layer in enumerate(self.net):
            if isinstance(layer, SineLayer):
                x, intermed = layer.forward_with_intermediate(x)

                if retain_grad:
                    x.retain_grad()
                    intermed.retain_grad()

                activations['_'.join((str(layer.__class__), "%d" % activation_count))] = intermed
                activation_count += 1
            else:
                x = layer(x)

                if retain_grad:
                    x.retain_grad()

            activations['_'.join((str(layer.__class__), "%d" % activation_count))] = x
            activation_count += 1

        return activations