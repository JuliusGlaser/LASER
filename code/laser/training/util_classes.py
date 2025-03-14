"""
This module implements utility classes for the training script
Authors:

    Julius Glaser <julius-glaser@gmx.de>
"""

from __future__ import division
import h5py
import os
import sys
import yaml
import torch
import numpy as np
import torch.utils.data as data
from yaml import Loader
from laser.training.models.nn import autoencoder as ae

class Network_parameters:
    def __init__(self, config, acquisition_dir: str):
        """
        Network_parameters, storing important parameters and options for the creation of training data and the network to be trained.
        Explanation for parameters is saved in the prototype.yaml

        Args:
            config (yaml config): config file, storing all information here used, 
            acquisition_dir (str): path to directory where config is saved and data will be stored,
        """
        
        self.model = config['model']
        self.diff_model = config['diffusion_model']
        self.mask_usage = config['mask_usage']
        self.device = config['device']
        self.latent = config['latent']
        self.noise_type = config['noise_type']
        self.depth = config['depth']
        self.activ_fct = config['activation_fct']
        self.epochs = config['epochs']
        self.batch_size_train = eval(config['batch_size_train'])
        self.batch_size_test = eval(config['batch_size_test'])
        self.sphere_samples = config['sphere_samples']

        self.noise_range = config['noise_range']
        self.learning_rate = config['learning_rate']
        self.optimizer = config['optimizer']
        self.loss_function = config['loss_function']
        self.device = config['device']
        self.test_epoch_step = config['test_epoch_step']
        self.dvs_file_path = config['dvs_file_path']

        self.acquisition_dir = acquisition_dir
        self.tested_once = False
        self.config = config

    def createModel(self, b0_mask: np.array, inputFeatures: int, device:str)-> torch.nn.Module:
        """
        method to create neural network model.

        Args:
            b0_mask (np.array): mask to set b0 entries of sequence to have no impact in model training, 
            inputFeatures (str): path to directory where config is saved and data will be stored,
        
        returns:
            torch.nn.Module: neural network, inherited from nn.Module and defined in ae
        """
        ae_dict = {'DAE':ae.DAE, 
                   'VAE':ae.VAE}
        return ae_dict[self.model](b0_mask, input_features=inputFeatures, latent_features=self.latent, device=device,depth=self.depth, activ_fct_str=self.activ_fct).to(device)

    def selectLoss(self)->torch.nn.modules.loss:
        """
        method to define Loss function for NN training.

        Args:
        
        returns:
            torch.nn.modules.loss: torch implemented loss function
        """
        if self.loss_function == 'MSE':
            return torch.nn.MSELoss()
        elif self.loss_function == 'L1':
            return torch.nn.L1Loss()
        elif self.loss_function == 'Huber':
            return torch.nn.HuberLoss()

    def selectOptimizer(self, model: torch.nn.Module) -> torch.optim.Optimizer:
        """
        method to define Loss function for NN training.

        Args:
            model (torch.nn.Module): network to be optimized        
        returns:
            torch.optim.Optimizer: optimizer for model
        """

        if self.optimizer == 'SGD':
            return torch.optim.SGD(model.parameters(), self.learning_rate, weight_decay=1E-5)
        elif self.optimizer == 'Adam':
            return torch.optim.Adam(model.parameters(), self.learning_rate)

    def selectDevice(self):
        """
        method to select device for training.
        Args:      
        returns:
            torch.device: torch device to perform training on
        """
        return torch.device(self.device)

    def __str__(self) -> str:
        output_str = ''
        for entry in self.config:
            if (entry.startswith('kld')) and self.model == 'DAE':
                pass
            else:
                output_str += f'> used {entry}: {self.config[entry]}\n'
        return output_str


class Losses_class:
    def __init__(self, model):
        """
        Losses_class, storing Losses during training and testing.

        Args:
            model (str): string used to define model to train, 
        """
        if model == 'DAE' or model == 'DAE_from_VAE':
            self.train = []
            self.test = []
            self.mse = []

            self.D = []
            self.D_test = []

            self.D_recon_over_epochs = []
            self.D_noisy_over_epochs = []
            self.D_clean_over_epochs = []

            self.mse_train = 1.0

        elif model == 'VAE':
            self.train = []
            self.test = []
            self.mse = []

            self.kld = []
            self.testKld = []
            self.recon = []
            self.testRecon = []

            self.D = []
            self.D_test = []

            self.standards = []
            self.means = []

            self.D_recon_over_epochs = []
            self.D_noisy_over_epochs = []
            self.D_clean_over_epochs = []
            self.mse_train = 1.0

    def create_loss_file(self, network_parameters: Network_parameters):
        """
        loss file creation, storing all losses of training and testing

        Args:
            network_parameters (Network_parameters): Network parameters storing the config and resulting setup for the training, 
        """
        epoch = network_parameters.epochs
        latent = network_parameters.latent
        model = network_parameters.model
        device = network_parameters.device

        lossFile = h5py.File(network_parameters.acquisition_dir + '/valid_' + model + '_Latent' + str(latent).zfill(2) + 'EpochTrain' + str(epoch) + 'Loss.h5', 'w')

        lossFile.create_dataset('testLoss' + str(epoch), data=np.array(self.test))
        lossFile.create_dataset('trainLoss' + str(epoch), data=np.array(self.train))
        lossFile.create_dataset('mseLoss' + str(epoch), data=np.array(self.mse))

        lossFile.create_dataset('D_recon' + str(epoch), data=np.array(self.D_recon_over_epochs))
        lossFile.create_dataset('D_noisy' + str(epoch), data=np.array(self.D_noisy_over_epochs))
        lossFile.create_dataset('D_clean' + str(epoch), data=np.array(self.D_clean_over_epochs))

        if model == 'VAE':
            lossFile.create_dataset('kldLoss' + str(epoch), data=np.array(self.kld))
            lossFile.create_dataset('reconLoss' + str(epoch), data=np.array(self.recon))
            lossFile.create_dataset('testReconLoss' + str(epoch), data=np.array(self.testRecon))
            lossFile.create_dataset('testKldLoss' + str(epoch), data=np.array(self.testKld))
            lossFile.create_dataset('means' + str(epoch), data=np.array(self.means))
            lossFile.create_dataset('standards' + str(epoch), data=np.array(self.standards))

        lossFile.close()

    def create_mse_loss_txt_file(self, network_parameters: Network_parameters, model):
        """
        create mse loss file, storing average MSE losses of testing checkpoints and the model architecture as string 

        Args:
            network_parameters (Network_parameters): Network parameters storing the config and resulting setup for the training, 
        """
        #Create txt config file:
        completeName = os.path.join(network_parameters.acquisition_dir, "mseLoss.txt")

        txtFile = open(completeName, "w")

        txtFile.write("\n\n\n Results in Testing:\n\n")
        for line in range(len(self.mse)):
            txtFile.write("Loss with MSE after Epoch {} : {:12.7f}\n".format(network_parameters.test_epoch_step*line, self.mse[line]))
        txtFile.write('\n\n')
        txtFile.write(model.__str__())
        txtFile.close()