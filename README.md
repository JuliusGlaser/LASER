# **⚠️ This repository is under construction! ⚠️**
# A deep nonlinear subspace modeling and reconstruction for diffusion-weighted imaging using variational auto-encoder

## Introduction

This repository implements:

- creation of diffusion signal dictionaries 
    using the diffusion tensor model as well as the ball-and-stick model

- training scheme for variational auto-encoder (VAE) 
    using the diffusion signal dictionaries

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

The repo implements the following reconstructions:

1. Multiplexed sensitivity-encoding diffusion-weighted imaging  
   (MUSE, DOI: [https://doi.org/10.1007/s00261-022-03710-2](https://doi.org/10.1007/s00261-022-03710-2))
2. LAtent Space dEcoded Reconstruction  
   (LASER, the proposed method of this publication)  
3. VAE-regularized reconstruction   
   (compare to qModel, DOI: [https://doi.org/10.1002/mrm.28756](https://doi.org/10.1002/mrm.28756))


Besides that it implements dictionary simulation for:

1. Diffusion tensor model (DOI: [https://doi.org/10.1016/S0006-3495(94)80775-1](https://doi.org/10.1016/S0006-3495(94)80775-1))

1. Ball-and-stick model (DOI: [https://doi.org/10.1016/j.neuroimage.2006.09.018](https://doi.org/10.1016/j.neuroimage.2006.09.018))

And auto-encoder models:
1. Denoising auto-encoder (DAE, DOI: [10.1126/science.1127647](10.1126/science.1127647))

2. Variational auto-encoder (VAE, DOI: [https://doi.org/10.48550/arXiv.1312.6114](https://doi.org/10.48550/arXiv.1312.6114))

## Data

The data used for the experiments has the following characteristics and needs to be downloaded before running reconstructions:

| Spatial Resolution (mm<sup>3</sup>) | Diffusion Mode | Acceleration (in-plane x slice) | Shots | Link |
|---|---|---|---|---|
| 1.00 | MDDW 3-shell (20, 30, 64 directions) with b-value of 1000, 2000, 3000 s/mm<sup>2</sup> | 3 x 3 | 2 | [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13171692.svg)](https://doi.org/10.5281/zenodo.13171692)|

## Examples

### Recreating publication results (for linux)

We assume being in the LASER directory and conda environment `laser` is activated and  
we only use CPU for training and reconstruction.
1. load the data
    First we load examplatory data, besides figure 5 we don't need to reconstruct all data,  
    so we will only load one k-space slice, coil sensitivities and the dvs file.

    ```bash
    cd data/raw/

    python load.py --file 1.0mm_126-dir_R3x3_kdat_slice_022.h5

    python load.py --file 1.0mm_126-dir_R3x3_coils.h5

    python load.py --file 1.0mm_126-dir_R3x3_dvs.h5
    ```

2. (optional) Run the VAE training for DT and BAS model
    As we provide checkpoints of the trained files, you don't need to rerun the training  
    (with the original settings it can be quite time consuming).  
    We assume here, you want to train the mdoel with the same parameters, so we use the   
    given configs in the directories and run the training with that.

    ```bash
    cd ../../code/laser/training/

    python train.py trained_data/VAE_BAS/

    python train.py trained_data/VAE_DTI/
    ```

    After successfull training you should see multiple files created in the corresponding directories.

3.  Run the reconstruction
    We assume you want to run MUSE reconstruction and LASER.  
    LASER needs the reconstructed shot_phases, so the shot_phase reconstruction  
    is run as well in the standard config.
    You don't need to adapt the config for this example.

    ```bash
    cd ../reconstruction/

    python reconstruction.py
    ```

    After succesfull reconstruction, the reconstructed slices should be available in  
    LASER/data/MUSE and LASER/data/LASER_BAS

    You can also run the reconstruction using the DT trained VAE, therefore you need to  
    adapt the config such that only LASER reconstruction is set to true (shot_reconstruction  
    is also not necessary anymore after first run) and the model directory now is the  
    DT model directory and just run 
    ```bash

    python reconstruction.py

    ```
4.  (optional) Run the LLR reconstruction
    To run the LLR reconstruction you'll need a complete different repository:
    [https://github.com/ZhengguoTan/NAViEPI](https://github.com/ZhengguoTan/NAViEPI)
    Follow the instructions there for the data here.

    Alternatively, the LLR reconstructed slice is available in zenodo.

5.  Fit the data
    We need to calculate do the fitting to get the fractional anisotropy (FA)  
    and colored fractional anisotropy (cFA) for figures 4 and 5.  
    For that, run:
    ```bash
    cd ../utility/
    python combine_and_fit.py
    ```
    This will calculate FA and cFA for the MUSE data. For the LASER data  
    please adapt the config options `reco_data_path` and `name` accordingly  
    and run the `combine_and_fit.py` script again.
    

6.  Run the plotting scripts of figures 3, 4, and 5

    6.1 Plot figure 3 by
    ```bash

    cd ../../../figures/fig3 

    python plot.py

    ```

    6.2 Plot figure 4 by
    ```bash

    cd ../fig4 

    python get_images.py

    ```

    6.3 Plot figure 5
    As we have only the one transversal slice reconstructed we can only  
    plot figure 5 b), to do so run:

    ```bash

    cd ../fig5/odf_plotting/ 

    python reconst_sfm.py

    ```

You have recreated all experiments of the publication, congratulations!

### Further examples

Explore the directories as you please and find further explanations.

## References

If you find the open-source codes/data useful, please cite:

```bibtex
@Article{glaser_2025_laser,
    Title     = {{A deep nonlinear subspace modeling and reconstruction for diffusion-weighted imaging using variational auto-encoder}},
    Author    = {Glaser, Julius and Tan, Zhengguo and Hofmann, Annika and Laun, Frederik B and Knoll, Florian},
    Journal   = {},
    Year      = {},
    Volume    = {},
    Pages     = {},
    doi       = {}
}
```
