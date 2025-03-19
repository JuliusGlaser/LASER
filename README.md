# **⚠️ This repository is under construction! ⚠️**
# A deep nonlinear subspace modeling and reconstruction for diffusion-weighted imaging using variational auto-encoder

## Introduction

This repository implements:

    - creation of diffusion signal dictionaries using the diffusion tensor model as well as the ball-and-stick model

    - training scheme for variational auto-encoder (VAE) using the diffusion signal dictionaries

    - reconstruction using the decoder of the trained VAE

    - plotting of all figures used in the corresponding paper

## Installation (using 'conda')

1. create a new conda environment named ('laser', you can use other names as you like) for python 3.9 (or higher):

    ```bash
    conda create -n laser python=3.9
    ```

2. activate the environment:

    ```bash
    conda activate laser
    ```

3. download and install `laser`:

    ```bash
    git clone https://github.com/JuliusGlaser/LASER.git
    ```
    and then ```cd``` to the LASER/code directory,
    ```bash
    pip install -e .
    ```

    This step will install all necessary dependencies for computation of all scripts on ```cpu```

4. (optional) ```cuda``` support:
    
    For CUDA support install the recommendet cuda version for your GPU and follow the steps under: https://pytorch.org/get-started/locally/

    Install the package using ```pip``` in your environment

    Also install cupy for the sigpy CUDA support:

    ```bash
    pip install cupy
    ```

## Features

The repo implements the following reconstruction:

    1. Multiplexed sensitivity-encoding diffusion-weighted imaging (MUSE, DOI: [https://doi.org/10.1007/s00261-022-03710-2](https://doi.org/10.1007/s00261-022-03710-2))

