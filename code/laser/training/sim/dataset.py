"""
This script creates q-space dataset

Author:
    Zhengguo Tan <zhengguo.tan@gmail.com>
    Soundarya Soundarresan <soundarya.soundarresan@fau.de>
"""

import torch

import numpy as np


def add_noise(x_clean, scale, noiseType = 'gaussian'):

    if noiseType== 'gaussian':
        x_noisy = x_clean + np.random.normal(loc = 0,
                                            scale = scale,
                                            size=x_clean.shape)
    elif noiseType== 'rician':
        noise1 =np.random.normal(0, scale, size=x_clean.shape)
        noise2 = np.random.normal(0, scale, size=x_clean.shape)
        x_noisy =  np.sqrt((x_clean + noise1) ** 2 + noise2 ** 2)

    x_noisy[x_noisy < 0.] = 0.
    x_noisy[x_noisy > 1.] = 1.

    return x_noisy

def _to_tensor(input):

    if isinstance(input, np.ndarray) and (not torch.is_tensor(input)):
        return torch.from_numpy(input)
    else:
        return input


class MrDictDataset(torch.utils.data.Dataset):

    def __init__(self, x_noisy, x_clean, transform=None):

        self.x_noisy = _to_tensor(x_noisy)
        self.x_clean = _to_tensor(x_clean)

        print('> MrDictDataset source shape: ', x_noisy.shape)
        print('> MrDictDataset target shape: ', x_noisy.shape)

        # transforms.ToTensor() scales images!!!
        # if transform is None:
        #     transform = transforms.Compose([transforms.ToTensor()])

        self.transform = transform

    def __len__(self):

        assert (len(self.x_noisy) == len(self.x_clean))
        return len(self.x_noisy)

    def __getitem__(self, idx):

        x_noisy = self.x_noisy[idx]
        x_clean = self.x_clean[idx]

        if self.transform is not None:
            x_noisy = self.transform(x_noisy)
            x_clean = self.transform(x_clean)

        return (x_noisy, x_clean)


class qSpaceDataset(torch.utils.data.Dataset):

    def __init__(self, x_noisy, x_clean, original_D_value, noise_amount, transform=None):

        self.x_noisy = x_noisy
        self.x_clean = x_clean
        self.original_D_value = original_D_value
        self.noise_amount = noise_amount

        print('> qSpaceDataset x_noisy shape: ', x_noisy.shape)

        self.N_atom = x_clean.shape[0]
        self.N_diff = x_clean.shape[1]
        

        # transforms.ToTensor() scales images!!!
        # if transform is None:
        #     transform = transforms.Compose([transforms.ToTensor()])

        self.transform = transform

    def __len__(self):

        assert (len(self.x_noisy) == len(self.x_clean) == len(self.noise_amount == len(self.original_D_value)))
        return len(self.x_noisy)

    def __getitem__(self, idx):

        x_noisy = self.x_noisy[idx]
        x_clean = self.x_clean[idx]
        original_D_value = self.original_D_value[idx]
        noise_amount = self.noise_amount[idx]

        if self.transform is not None:
            x_noisy = self.transform(x_noisy)
            x_clean = self.transform(x_clean)

        return (x_noisy, x_clean, original_D_value, noise_amount)
