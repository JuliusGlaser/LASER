## Build a new conda environment for LASER

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
    and
    ```bash
    pip install -r requirements.txt
    ```

    This step will install all necessary dependencies for computation of all scripts on ```cpu```

4. (necessary only for bootstrapping analysis and LPCA denoising) MRtrix3 installation

    Install MRtrix3 according to the installation guide given under:

    https://www.mrtrix.org/download/


5. (optional) ```cuda``` support:
    
    For CUDA support install the recommendet cuda version for your GPU and follow the steps under: https://pytorch.org/get-started/locally/

    Install the package using ```pip``` in your environment

    Also install cupy for the sigpy CUDA support:

    ```bash
    pip install cupy
    ```