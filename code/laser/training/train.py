"""
This module implements the VAE training for DW-MRI

Authors:

    Julius Glaser <julius.glaser@fau.de>
    Zhengguo Tan <zhengguo.tan@gmail.com>
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
import copy

# from torch.utils.data import DataLoader
from latrec.training.sim import dataset
from latrec.training.sim import dwi
from latrec.training.sim import epi
from latrec.training.models.nn import autoencoder as ae
import latrec.training.linsub



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

        if model == 'VAE' or model == 'Diff_Dec_VAE':
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

def train(network_parameters, loader_train, optim, model, device, loss_function, epoch, Losses, gamma_x, scalingMatrix):
    model.train()
    scalingMatrix = torch.tensor(scalingMatrix, device=device)
    HALF_LOG_TWO_PI = 0.91893
    train_loss = 0.0
    kld_loss = 0.0
    recon_loss = 0.0
    train_loss = 0.0
    mse_train = 0.0

    for batch_idx, (noisy_t, clean_t, D_clean, noise) in enumerate(loader_train):

        noisy_t = noisy_t.type(torch.FloatTensor).to(device)
        clean_t = clean_t.type(torch.FloatTensor).to(device)
        optim.zero_grad()

        if network_parameters.model == 'DAE' or network_parameters.model == 'DAE_from_VAE':
            recon_t = model(noisy_t)
            loss = loss_function(recon_t, clean_t)
        elif network_parameters.model == 'VAE' or network_parameters.model == 'Diff_Dec_VAE':
            recon_t, mu, logvar = model(noisy_t)
            k = (2*network_parameters.N_diff/network_parameters.latent)**2

            recon_t_scaled = recon_t*scalingMatrix
            clean_t_scaled = clean_t*scalingMatrix

            # recon_t_scaled = recon_t
            # clean_t_scaled = clean_t

            mse = loss_function(recon_t_scaled, clean_t_scaled)
            loggamma_x = np.log(gamma_x)
            gen_loss = torch.sum(torch.square((recon_t_scaled - clean_t_scaled)/gamma_x)/2.0 + loggamma_x + HALF_LOG_TWO_PI)/ network_parameters.batch_size_train
            KLD = ae.loss_function_kld(mu, logvar)/network_parameters.batch_size_train
            loss = gen_loss + k*KLD/network_parameters.batch_size_train
            kld_loss += (k*KLD).item()
            recon_loss += mse.item()
            mse_train += mse.item()

        loss.backward()
        optim.step()
        train_loss += loss.item()
        if batch_idx % 1000 == 0:
            print('Train Epoch: {:4d} [{:10d}/{:10d} ({:3.0f}%)]\tLoss: {:12.6f}'.format(
                epoch, batch_idx * len(noisy_t), len(loader_train.dataset),
                100. * batch_idx / len(loader_train),
                loss.item() / len(noisy_t)))
            
    Losses.train.append(train_loss / len(loader_train.dataset)) 
    #Losses.D.append(D_loss / len(loader_train.dataset))

    if network_parameters.model == 'VAE' or network_parameters.model == 'Diff_Dec_VAE':
        Losses.kld.append(kld_loss / len(loader_train.dataset))
        Losses.recon.append(recon_loss / len(loader_train.dataset))
        Losses.mse_train = (mse_train/len(loader_train.dataset))

    print('====> Epoch: {} Average loss Training: {:12.6f}'.format(epoch, Losses.train[-1]))

    return Losses

def test(network_parameters, loader_test, optimizer, model, device, loss_function, epoch, h5pyFile, Losses):
    model.eval()
    kld_loss = 0.0
    recon_loss = 0.0
    mse_loss = 0.0
    test_loss = 0.0

    test_clean = []
    test_noisy = []
    test_noise_amount = []
    test_recon = []
    D_clean_values = []

    with torch.no_grad():
        for batch_idx, (noisy_t, clean_t, D_clean, noise) in enumerate(loader_test):

            noisy_t = noisy_t.type(torch.FloatTensor).to(device)
            clean_t = clean_t.type(torch.FloatTensor).to(device)

            if network_parameters.model == 'DAE' or network_parameters.model == 'DAE_from_VAE':
                recon_t = model(noisy_t)
                loss = loss_function(recon_t, clean_t)
            elif network_parameters.model == 'VAE' or network_parameters.model == 'Diff_Dec_VAE':
                recon_t, mu, logvar = model(noisy_t)


                loss_lossf = loss_function(recon_t, clean_t)
                KLD = ae.loss_function_kld(mu, logvar)

                kld_loss += (network_parameters.kld_weight*KLD).item()
                recon_loss += loss_lossf.item()
                loss = loss_lossf*32 + network_parameters.kld_weight*KLD


            loss_mse = ae.loss_function_mse(recon_t, clean_t)
            mse_loss += loss_mse.item()
            test_loss += loss.item()

    Losses.test.append(test_loss / len(loader_test.dataset))
    Losses.mse.append(mse_loss / len(loader_test.dataset))

    if network_parameters.model == 'VAE' or network_parameters.model == 'Diff_Dec_VAE':
        Losses.testKld.append(kld_loss / len(loader_test.dataset))
        Losses.testRecon.append(recon_loss / len(loader_test.dataset))
        mu = mu.cpu().detach().numpy()
        Losses.means.append(mu)
        std = torch.exp(0.5 * logvar)
        std = std.cpu().detach().numpy()
        Losses.standards.append(std)

    print('====> Epoch: {} Average loss Testing: {:12.7f}'.format(epoch, Losses.test[-1]))
    print('====> Epoch: {} Average mse_loss Testing: {:12.7f}'.format(epoch, Losses.mse[-1]))

    #save trained model
    out_str=''
    if epoch == network_parameters.epochs:
        out_str = '/train_' + network_parameters.model + '_Latent' + str(network_parameters.latent).zfill(2) + 'final.pt'
        out_str2 = '/train_' + network_parameters.model + '_Latent' + str(network_parameters.latent).zfill(2) + 'final_trainable.tar'
        torch.save({'epoch':epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict':  optimizer.state_dict(),
                    'Losses': Losses,
                    'NetworkParameters': network_parameters},
                    network_parameters.acquisition_dir + out_str2)
    else:
        out_str = '/train_' + network_parameters.model + '_Latent' + str(network_parameters.latent).zfill(2) + '_epoch' + str(epoch).zfill(3) + '.pt'
    torch.save(model.state_dict(), network_parameters.acquisition_dir + out_str)
    
    return Losses

def setup(ACQ_DIR: str):

    stream = open(ACQ_DIR + '/config.yaml', 'r')
    config = yaml.load(stream, Loader)

    NetworkParameters = Network_parameters(config, ACQ_DIR)

    print(NetworkParameters)

    if NetworkParameters.N_diff == 126:
        f = h5py.File(r'../raw-data/data-126-dir/1.0mm_126-dir_R3x3_dvs.h5', 'r')
    elif NetworkParameters.N_diff == 32:
        f = h5py.File(ACQ_DIR + '/diff_encoding.h5', 'r')
    elif NetworkParameters.N_diff == 21:
        f = h5py.File('/home/hpc/iwbi/iwbi019h/DeepSubspaceMRI/deepsubspacemri/data_files/4shot_data/1.0mm_21-dir_R1x3_dvs.h5', 'r')

    bvals = f['bvals'][:]
    bvecs = f['bvecs'][:]
    scalingMatrix = np.ones(bvals.shape)
    # scalingMatrix = scalingMatrix * bvals/1000
    f.close()

    #b0 values don't have to be learned
    b0_threshold = 50
    b0_mask = bvals > b0_threshold

    if NetworkParameters.diff_model == 'DTI':
        # if bvals.ndim == 1:
            # bvals = bvals[:, np.newaxis]
        x_clean, original_D = dwi.model_DTI_new(bvals, bvecs, b0_threshold, 18, None)  #bvals = b, bvecs = g, returns y_pick
        original_D = original_D.T
        x_clean = x_clean.T
    elif NetworkParameters.diff_model == 'BAS':
        x_clean, original_D = dwi.model_BAS(bvals, bvecs, b0_threshold)  #bvals = b, bvecs = g, returns y_pick
        original_D = original_D.T
        x_clean = x_clean.T

    bvals = bvals[:, np.newaxis]

    B = epi.get_B(bvals, bvecs)

    return NetworkParameters, x_clean, original_D, B, b0_mask, scalingMatrix

def create_noised_dataset(train_set, test_set, NetworkParameters):

    train_set_full = copy.deepcopy(train_set)
    test_set_full = copy.deepcopy(test_set)
    
    for id in range(1, NetworkParameters.noise_range, 1):
        train_set_copy = copy.deepcopy(train_set)
        test_set_copy = copy.deepcopy(test_set)

        sd = 0.01 + id * 0.03

        for index in range(len(train_set_copy)):
            train_set_copy[index][0][:] = dataset.add_noise(train_set_copy[index][1], sd, NetworkParameters.noise_type)
            train_set_copy[index][3][:] = id

        for index in range(len(test_set_copy)):
            test_set_copy[index][0][:] = dataset.add_noise(test_set_copy[index][1], sd, NetworkParameters.noise_type)
            test_set_copy[index][3][:] = id

        train_set_full = data.ConcatDataset([train_set_full, train_set_copy])  
        test_set_full =  data.ConcatDataset([test_set_full, test_set_copy]) 
    print('train set size wo noise', len(train_set))
    print('test set size wo noise', len(test_set))
    print('train set size with noise', len(train_set)*NetworkParameters.noise_range)
    print('test set size with noise', len(test_set)*NetworkParameters.noise_range)
    return train_set_full, test_set_full

def create_data_loader(x_clean, original_D, NetworkParameters):
    tooLarge = False

    noise_amount = np.zeros(shape=(1, x_clean.shape[1]))
    noise_amount_full = np.transpose(noise_amount)

    x_clean = np.transpose(x_clean)
    x_no_noise_yet = copy.deepcopy(x_clean)
    original_D = np.transpose(original_D)

    print('shape x_clean = ', x_clean.shape)
    print('shape x_no_noise_yet = ', x_no_noise_yet.shape)
    print('shape original_D = ', original_D.shape)
    print('shape noise_amount_full = ', noise_amount_full.shape)
    q_dataset = dataset.qSpaceDataset(x_clean, x_no_noise_yet, original_D, noise_amount_full)

    q_datalen = len(q_dataset)

    train_siz = int(q_datalen * 0.8)
    test_siz = q_datalen - train_siz

    train_set, test_set = data.random_split(q_dataset, [train_siz, test_siz]) #get_D with test_set    

    train_set_noised, test_set_noised = create_noised_dataset(train_set, test_set, NetworkParameters)    

    loader_train = data.DataLoader(train_set_noised, batch_size=NetworkParameters.batch_size_train, shuffle=True)

    loader_test = data.DataLoader(test_set_noised, batch_size=NetworkParameters.batch_size_test, shuffle=False)

    return loader_train, loader_test


def main():

    torch.manual_seed(0)

    given_dir = sys.argv[1] if len(sys.argv) > 1 else print('No directory handed over!')

    print(f'> directory: {given_dir}\n')
    ACQ_DIR = given_dir

    NetworkParameters, x_clean, original_D, B, b0_mask, scalingMatrix = setup(ACQ_DIR)
    device = NetworkParameters.selectDevice()
    model = NetworkParameters.createModel(None, NetworkParameters.N_diff, device)
    loader_train, loader_test = create_data_loader(x_clean, original_D, NetworkParameters)

    
    
    loss_function = NetworkParameters.selectLoss()
    optimizer = NetworkParameters.selectOptimizer(model)

    f = h5py.File(ACQ_DIR + '/valid_' + NetworkParameters.model + '_Latent' + str(NetworkParameters.latent).zfill(2) + 'EpochTrain' + str(NetworkParameters.epochs) + '.h5', 'w')

    Losses = Losses_class(NetworkParameters.model)
    # %% training
    for epoch in range(1, NetworkParameters.epochs+1, 1):
        gamma_x = np.sqrt(Losses.mse_train)
        Losses = train(NetworkParameters, loader_train, optimizer, model, device, loss_function, epoch, Losses, gamma_x, scalingMatrix)
        

        if epoch == NetworkParameters.epochs or epoch % NetworkParameters.test_epoch_step == 0:
            # testing
            Losses = test(NetworkParameters, loader_test, optimizer, model, device, loss_function, epoch, f, Losses)
        if NetworkParameters.model == 'VAE':
            NetworkParameters.update_kld_weight(epoch)

    Losses.create_loss_file(NetworkParameters)
    Losses.create_mse_loss_txt_file(NetworkParameters, model)

    f.close()

if __name__ == "__main__":
    main()