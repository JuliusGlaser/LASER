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

class Network_parameters:
    def __init__(self, config, acquisition_dir):
        self.model = config['model']
        self.diff_model = config['diffusion_model']
        self.device = config['device']
        self.latent = config['latent']
        self.noise_type = config['noise_type']
        self.depth = config['depth']
        self.activ_fct = config['activation_fct']
        self.epochs = config['epochs']
        self.batch_size_train = eval(config['batch_size_train'])
        self.batch_size_test = eval(config['batch_size_test'])


        self.noise_range = config['noise_range']
        self.N_diff = config['directions']
        self.learning_rate = config['learning_rate']
        self.D_loss_weight = config['D_loss_weight']
        if self.model == 'VAE':
            self.kld_start_epoch = config['kld_start_epoch']        
            self.kld_restart = config['kld_restart']
            self.kld_max_weight = config['kld_max_weight']
            self.kld_weight_increase = config['kld_weight_increase']

        self.optimizer = config['optimizer']
        self.loss_function = config['loss_function']
        self.device = config['device']
        self.test_epoch_step = config['test_epoch_step']

        self.acquisition_dir = acquisition_dir
        self.kld_weight = 0
        self.tested_once = False
        self.config = config

    def createModel(self, b0_mask, inputFeatures, device):
        ae_dict = {'DAE':ae.DAE, 
                   'VAE':ae.VAE}
        return ae_dict[self.model](b0_mask, input_features=inputFeatures, latent_features=self.latent, device=device,depth=self.depth, activ_fct_str=self.activ_fct).to(device)

    def selectLoss(self):
        if self.loss_function == 'MSE':
            return torch.nn.MSELoss()
        elif self.loss_function == 'L1':
            return torch.nn.L1Loss()
        elif self.loss_function == 'Huber':
            return torch.nn.HuberLoss()

    def selectOptimizer(self, model):
        if self.optimizer == 'SGD':
            return torch.optim.SGD(model.parameters(), self.learning_rate, weight_decay=1E-5)
        elif self.optimizer == 'Adam':
            return torch.optim.Adam(model.parameters(), self.learning_rate)

    def selectDevice(self):
        return torch.device(self.device)

    def update_kld_weight(self, training_epoch):
            interval = self.kld_max_weight/self.kld_weight_increase + 5
            if training_epoch >= self.kld_start_epoch:
                if self.kld_weight < self.kld_max_weight:
                    self.kld_weight += self.kld_weight_increase
                elif self.kld_restart:
                    if (training_epoch-self.kld_start_epoch)%interval == 0:
                        self.kld_weight = 0

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

        elif model == 'Diff_Dec_VAE':
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
        #Create txt config file:
        completeName = os.path.join(network_parameters.acquisition_dir, "mseLoss.txt")

        txtFile = open(completeName, "w")

        txtFile.write("\n\n\n Results in Testing:\n\n")
        for line in range(len(self.mse)):
            txtFile.write("Loss with MSE after Epoch {} : {:12.7f}\n".format(network_parameters.test_epoch_step*line, self.mse[line]))
        txtFile.write('\n\n')
        txtFile.write(model.__str__())
        txtFile.close()