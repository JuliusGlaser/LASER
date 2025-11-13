"""
This module defines the autoencoder (AE) models.

References:


Author:
    Soundarya Soundarresan <soundarya.soundarresan@fau.de>
    Zhengguo Tan <zhengguo.tan@gmail.com>
    Julius Glaser <julius-glaser@gmx.de>
"""

from re import S
import torch

import torch.nn as nn

from torch.nn import functional as F

from typing import List

def _get_activ_fct(activ_fct:str):
    if activ_fct == 'ReLU':
        res_fct = nn.ReLU(True)
    elif activ_fct == 'SELU':
        res_fct = nn.SELU(True)
    elif activ_fct == 'Tanh':
        res_fct = nn.Tanh()
    else:
        print(f'Activation function {activ_fct} is not defined')
        quit()
    return res_fct

class CustomLinearEnc(nn.Module):
    """
    Definition of the first encoder layer for AE, where b0 entries of the acquisition are made non learnable 
    """
    def __init__(self, in_features, out_features, b0_mask, device, fixed_value=0):
        super(CustomLinearEnc, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        non_learnable_indices = [index for index, value in enumerate(b0_mask) if not value]
        self.neuron_indices = non_learnable_indices
        self.linear = nn.Linear(in_features, out_features)
        self.make_non_learnable()

    def make_non_learnable(self):
        with torch.no_grad():
            for index in self.neuron_indices:
                self.linear.weight[:,index]= 0  

    def forward(self, x):
        self.make_non_learnable()
        return torch.nn.functional.linear(x, self.linear.weight, self.linear.bias)

class CustomLinearDec(nn.Module):
    """
    Definition of the last decoder layer for AE, where b0 entries of the acquisition are made non learnable and the output is set to 1. 
    """
    def __init__(self, in_features, out_features, b0_mask, device, fixed_value=40, reco: bool =False):
        super(CustomLinearDec, self).__init__()
        self.b0_mask = b0_mask
        self.reco = reco
        self.in_features = in_features
        self.out_features = out_features
        non_learnable_indices = [index for index, value in enumerate(b0_mask) if not value]
        self.neuron_indices = non_learnable_indices
        self.fixed_value = fixed_value
        self.linear = nn.Linear(in_features, out_features)
        self.make_non_learnable()
        
        # # Create a mask to zero out the connections to the specific neuron
        # self.mask = torch.ones(out_features, in_features).to(device)
    
    def make_non_learnable(self):
        with torch.no_grad():
            for index in self.neuron_indices:
                self.linear.weight[index,:] = 0

    def forward(self, x):
        if not self.reco:
            self.make_non_learnable()
        output = torch.nn.functional.linear(x, self.linear.weight, self.linear.bias)
        if not self.reco:
            # Set the output of the specific neuron to the fixed value
            for index in self.neuron_indices:
                output[:,index] = self.fixed_value
        return output
    
class NormalizingLayer(nn.Module):
    #TODO: Layer to run before Encoder, which normalizes the real part of the complex data (b0 diff value to one) and removes the phase
    pass

class ScalingLayer(nn.Module):
    #TODO: Layer to run after Decoder, which scales the diffusion signal according to b0 and adds the phase
    
    pass

class DAE(nn.Module):
    """
    Denoising AutoEncoder
    """
    def __init__(self,
                 b0_mask,
                 input_features: int = 126,
                 latent_features: int = 11,
                 depth: int = 4,
                 activ_fct_str='Tanh',
                 encoder_features: List[int] = None,
                 device = 'cpu',
                 reco = False,
                 decoder_out_features: tuple[int, int, int] = (22, 33, 71)):

        super(DAE, self).__init__()

        encoder_module = []
        decoder_module = []

        activ_fct = _get_activ_fct(activ_fct_str)

        # ---------- ENCODER ----------
        encoder_module = []

        encoder_features = torch.linspace(start=input_features,
                                          end=latent_features,
                                          steps=depth+1).type(torch.int64)

        for d in range(depth):
            in_f  = encoder_features[d]
            out_f = encoder_features[d+1]

            if d == 0 and b0_mask is not None:
                encoder_module.append(CustomLinearEnc(in_f, out_f, b0_mask, device))
            else:
                encoder_module.append(nn.Linear(in_f, out_f))

            encoder_module.append(activ_fct)

        self.encoder_seq = nn.Sequential(*encoder_module)

        # ---------- DECODER BRANCHES (3 heads: 22, 33, 71) ----------
        self.decoders = nn.ModuleList()
        self.decoder_out_features = decoder_out_features

        for final_out in decoder_out_features:
            layers = []

            # latent_features -> ... -> final_out
            branch_features = torch.linspace(start=latent_features,
                                             end=final_out,
                                             steps=depth+1).type(torch.int64)

            for d in range(depth):
                in_f  = branch_features[d]
                out_f = branch_features[d+1]

                layers.append(nn.Linear(in_f, out_f))

                # activation after all but last layer
                if d < depth - 1:
                    layers.append(activ_fct)

            # note: NO Sigmoid here; we keep heads linear at the end
            self.decoders.append(nn.Sequential(*layers))

        # ---------- FINAL CUSTOM LAYER ----------
        total_branch_out = sum(decoder_out_features)  # 22 + 33 + 71 = 126

        # This custom layer consumes [22 | 33 | 71] concatenated and reconstructs input_features
        self.final_dec = CustomLinearDec(
            total_branch_out,
            input_features,
            b0_mask,
            device,
            reco=reco
        )
        self.final_activation = nn.Sigmoid()

    def encode(self, x):
        return self.encoder_seq(x)

    def decode_branches(self, z):
        """
        Runs the 3 separate decoder branches.
        Returns a list: [out_22, out_33, out_71]
        """
        return [dec(z) for dec in self.decoders]

    def decode(self, z):
        """
        Uses the 3 branches, concatenates their outputs, and applies the final custom layer.
        Returns:
          x_recon: final reconstruction (size input_features)
          branch_outs: list [out_22, out_33, out_71]
        """
        branch_outs = self.decode_branches(z)        # list of tensors
        concat = torch.cat(branch_outs, dim=-1)      # shape: (batch, 22+33+71)

        x_recon = self.final_dec(concat)
        x_recon = self.final_activation(x_recon)     # keep Sigmoid as in original

        return x_recon, branch_outs

    def forward(self, x):
        z = self.encode(x)
        x_recon, _ = self.decode(z)
        # x_recon: reconstructed full output (81)
        # branch_outs: [22, 33, 71] parts if you want to use them in the loss
        return x_recon

class VAE(nn.Module):
    """
    Variational AutoEncoder
    """
    def __init__(self,
                 b0_mask = None,
                 input_features=81,
                 latent_features=15,
                 depth=4,
                 activ_fct_str='Tanh',
                 device='cpu',
                 reco = False):

        super(VAE, self).__init__()
        encoder_module = []
        decoder_module = []

        activ_fct = _get_activ_fct(activ_fct_str)

        encoder_features = torch.linspace(start=input_features,
                                          end=latent_features,
                                          steps=depth+1).type(torch.int64)
        decoder_features = torch.flip(encoder_features, dims=(0, ))

        # encoder
        for d in range(depth - 1):
            if d == 0 and b0_mask is not None:
                encoder_module.append(CustomLinearEnc(encoder_features[d], encoder_features[d+1], b0_mask, device))
            else:
                encoder_module.append(nn.Linear(encoder_features[d], encoder_features[d+1]))
            encoder_module.append(activ_fct)

        self.encoder_seq = nn.Sequential(*encoder_module)

        # latent layer
        self.fc1 = nn.Linear(encoder_features[depth-1], encoder_features[depth])
        self.fc2 = nn.Linear(encoder_features[depth-1], encoder_features[depth])

        # decoder
        for d in range(depth): 
            if d < (depth - 1):
                decoder_module.append(nn.Linear(decoder_features[d], decoder_features[d+1]))
                decoder_module.append(activ_fct)
            else:
                if b0_mask is not None:
                    decoder_module.append(CustomLinearDec(decoder_features[d], decoder_features[d+1], b0_mask, device, reco=reco))
                else:
                    decoder_module.append(nn.Linear(decoder_features[d], decoder_features[d+1]))
                decoder_module.append(nn.Sigmoid())

        self.decoder_seq = nn.Sequential(*decoder_module)

    def encode(self, x):
        features = self.encoder_seq(x)
        return self.fc1(features), self.fc2(features)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder_seq(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar



def loss_function_mse(recon, orig):
    loss = F.mse_loss(recon, orig)

    if loss.isnan():
        return torch.tensor([1E-6])
    else:
        return loss
    
def loss_function_l1(recon, orig):
    loss = F.l1_loss(recon, orig)

    if loss.isnan():
        return torch.tensor([1E-6])
    else:
        return loss


def loss_function_kld( mu=None, logvar=None):
    """
    Reconstruction + KL divergence losses summed over all elements and batch

    Reference:
        * Kingma DP, Welling M.
          Auto-encoding Variational Bayes. ICLR (2014).
    Split up KLD and reconstruction loss
    Assume pdfs to be Gaussian to use analytical formula
    """

    if mu is None:
        mu = torch.tensor([0])

    if logvar is None:
        logvar = torch.tensor([0])

    # std = torch.exp(0.5 * logvar)

    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return KLD



class denoising_model(nn.Module):
    def __init__(self):
        super(denoising_model,self).__init__()
        self.seq1=nn.Sequential(
                        nn.Linear(81,60),
                        nn.ReLU(True),
                        nn.Linear(60,45),
                        nn.ReLU(True),
                        nn.Linear(45,20),
                        nn.ReLU(True),
                        nn.Linear(20,5),
                        nn.ReLU(True)
                        )

        self.seq2=nn.Sequential(
                        nn.Linear(5,20),
                        nn.ReLU(True),
                        nn.Linear(20,45),
                        nn.ReLU(True),
                        nn.Linear(45,60),
                        nn.ReLU(True),
                        nn.Linear(60,81),
                        nn.Sigmoid()
                        )

    def encoder(self,x):
        ou=self.seq1(x)
        return ou

    def decoder(self,x):
        ou = self.seq2(x)
        return ou

    def forward(self,x):
        ou=self.encoder(x)
        x=self.decoder(ou)
        return x