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
                 input_features: int = 81,
                 latent_features: int = 15,
                 depth: int = 4,
                 activ_fct_str='Tanh',
                 encoder_features: List[int] = None,
                 device = 'cpu',
                 reco = False):

        super(DAE, self).__init__()

        encoder_module = []
        decoder_module = []

        activ_fct = _get_activ_fct(activ_fct_str)

        # if encoder_features is None:

        #     encoder_features = torch.linspace(start=input_features, end=latent_features, steps=depth+1).type(torch.int64)

        # else:

        #     encoder_features = torch.tensor(encoder_features)

        # #     assert(depth == len(encoder_features))


        # decoder_features = torch.flip(encoder_features, dims=(0, ))

        encoder_features = torch.linspace(start=input_features,
                                          end=latent_features,
                                          steps=depth+1).type(torch.int64)
        decoder_features = torch.flip(encoder_features, dims=(0, ))


        for d in range(depth):
            if d == 0 and b0_mask is not None:
                encoder_module.append(CustomLinearEnc(encoder_features[d], encoder_features[d+1], b0_mask, device))
            else:
                encoder_module.append(nn.Linear(encoder_features[d], encoder_features[d+1]))
            encoder_module.append(activ_fct)
            
            if d < (depth - 1):
                decoder_module.append(nn.Linear(decoder_features[d], decoder_features[d+1]))
                decoder_module.append(activ_fct)
            else:
                decoder_module.append(CustomLinearDec(decoder_features[d], decoder_features[d+1], b0_mask, device, reco=reco))
                decoder_module.append(nn.Sigmoid())

        self.encoder_seq = nn.Sequential(*encoder_module)
        self.decoder_seq = nn.Sequential(*decoder_module)

    def encode(self, x):
        return self.encoder_seq(x)

    def decode(self, x):
        return self.decoder_seq(x)

    def forward(self, x):
        latent = self.encode(x)
        output = self.decode(latent)

        return output

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


class Shell_DAE_joint_lat(nn.Module):
    """
    Denoising AutoEncoder
    """
    def __init__(self,
                 b0_mask,
                 input_features: int = 81,
                 latent_features: int = 15,
                 depth: int = 4,
                 activ_fct_str='Tanh',
                 encoder_features: List[int] = None,
                 device = 'cpu',
                 reco = False):

        super(Shell_DAE_joint_lat, self).__init__()

        shell1 = 22
        shell2 = 33
        shell3 = 71

        encoder_module1 = []
        encoder_module2 = []
        encoder_module3 = []

        decoder_module1 = []
        decoder_module2 = []
        decoder_module3 = []

        b0_mask1 = b0_mask[0:22]
        b0_mask2 = b0_mask[22:55]
        b0_mask3 = b0_mask[55::]

        activ_fct = _get_activ_fct(activ_fct_str)

        # if encoder_features is None:

        #     encoder_features = torch.linspace(start=input_features, end=latent_features, steps=depth+1).type(torch.int64)

        # else:

        #     encoder_features = torch.tensor(encoder_features)

        # #     assert(depth == len(encoder_features))


        # decoder_features = torch.flip(encoder_features, dims=(0, ))

        encoder_features1 = torch.linspace(start=shell1,
                                          end=latent_features,
                                          steps=depth+1).type(torch.int64)
        decoder_features1 = torch.flip(encoder_features1, dims=(0, ))


        for d in range(depth):
            if d == 0 and b0_mask1 is not None:
                encoder_module1.append(CustomLinearEnc(encoder_features1[d], encoder_features1[d+1], b0_mask1, device))
            else:
                encoder_module1.append(nn.Linear(encoder_features1[d], encoder_features1[d+1]))
            encoder_module1.append(activ_fct)
            
            if d < (depth - 1):
                decoder_module1.append(nn.Linear(decoder_features1[d], decoder_features1[d+1]))
                decoder_module1.append(activ_fct)
            else:
                decoder_module1.append(CustomLinearDec(decoder_features1[d], decoder_features1[d+1], b0_mask1, device, reco=reco))
                decoder_module1.append(nn.Sigmoid())

        self.encoder_seq1 = nn.Sequential(*encoder_module1)
        self.decoder_seq1 = nn.Sequential(*decoder_module1)

        encoder_features2 = torch.linspace(start=shell2,
                                          end=latent_features,
                                          steps=depth+1).type(torch.int64)
        decoder_features2 = torch.flip(encoder_features2, dims=(0, ))


        for d in range(depth):
            if d == 0 and b0_mask2 is not None:
                encoder_module2.append(CustomLinearEnc(encoder_features2[d], encoder_features2[d+1], b0_mask2, device))
            else:
                encoder_module2.append(nn.Linear(encoder_features2[d], encoder_features2[d+1]))
            encoder_module2.append(activ_fct)
            
            if d < (depth - 1):
                decoder_module2.append(nn.Linear(decoder_features2[d], decoder_features2[d+1]))
                decoder_module2.append(activ_fct)
            else:
                decoder_module2.append(CustomLinearDec(decoder_features2[d], decoder_features2[d+1], b0_mask2, device, reco=reco))
                decoder_module2.append(nn.Sigmoid())

        self.encoder_seq2 = nn.Sequential(*encoder_module2)
        self.decoder_seq2 = nn.Sequential(*decoder_module2)

        encoder_features3 = torch.linspace(start=shell3,
                                          end=latent_features,
                                          steps=depth+1).type(torch.int64)
        decoder_features3 = torch.flip(encoder_features3, dims=(0, ))


        for d in range(depth):
            if d == 0 and b0_mask3 is not None:
                encoder_module3.append(CustomLinearEnc(encoder_features3[d], encoder_features3[d+1], b0_mask3, device))
            else:
                encoder_module3.append(nn.Linear(encoder_features3[d], encoder_features3[d+1]))
            encoder_module3.append(activ_fct)
            
            if d < (depth - 1):
                decoder_module3.append(nn.Linear(decoder_features3[d], decoder_features3[d+1]))
                decoder_module3.append(activ_fct)
            else:
                decoder_module3.append(CustomLinearDec(decoder_features3[d], decoder_features3[d+1], b0_mask3, device, reco=reco))
                decoder_module3.append(nn.Sigmoid())

        self.encoder_seq3 = nn.Sequential(*encoder_module3)
        self.decoder_seq3 = nn.Sequential(*decoder_module3)

        self.lat_fused = nn.Sequential(
            nn.Linear(latent_features*3, latent_features*3),
            activ_fct,
            nn.Linear(latent_features*3, latent_features*3),
        )

    def encode(self, x):
        x1 = x[:, 0:22]
        x2 = x[:, 22:55]
        x3 = x[:, 55::]
        z1 = self.encoder_seq1(x1)  # [B,10]
        z2 = self.encoder_seq2(x2)  # [B,10]
        z3 = self.encoder_seq3(x3)  # [B,10]
        return torch.cat([z1, z2, z3], dim=-1)  # [B,30]

    def decode(self, z_fused):
        z1_f = z_fused[:, 0:10]
        z2_f = z_fused[:, 10:20]
        z3_f = z_fused[:, 20:30]

        x1_hat = self.decoder_seq1(z1_f)
        x2_hat = self.decoder_seq2(z2_f)
        x3_hat = self.decoder_seq3(z3_f)

        return torch.cat([x1_hat, x2_hat, x3_hat], dim=-1)

    def forward(self, x):
        z_fused = self.encode(x) 
        z_fused = self.lat_fused(z_fused)               
        x_hat = self.decode(z_fused) 
        

        return x_hat


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