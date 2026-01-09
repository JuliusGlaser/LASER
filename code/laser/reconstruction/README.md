## Reconstruction
The script `reconstruction.py` implements following reconstructions:
- MUSE
- LASER
- AE regularized MUSE
- MUSE + AE denoising

The LASER reconstruction is implemented with the following forward operator:
<p align="center">
  <img alt="Light" src="../../../figures/fig2/fwd.png" width="100%">
</p>


Modelling the minimization problem:

$$
||y-PΘFSΔ(b_0^{**}D(x))|| - λ||x||_{TV}
$$

<details>
  <summary>Click to get meaning of variables</summary>
With meaning as:

* $y$ is the k-space data
* $P$ is the undersampling pattern
* $Θ$ is the SMS operator
* $F$ is the forward Fourier transform
* $S$ is the multiplication with coil sensetivities
* $Δ$ is the multiplication with the shot-to-shot phases
* $b_0^{**}$ is the optimized, reconstructed b0 image
* $D$ is the voxel-wise decoding of x
* $x$ are the reconstructed latent images
* $λ$ is the regularization weight
* $||x||_{TV}$ is the TV-norm of the latent iamges

For $b_0^{**}$ the calculation is first:

<p align="center">
 <img src="https://latex.codecogs.com/svg.image?\mathbf{b_0}^*=\underset{\mathbf{b_0}}{\operatorname{argmin}}\left\|y(b=0)-A\mathbf{b_0}\right\|_2^2)" />
 </p>


with
* **b**<sub>0</sub> is a vector containing all 12 $b_0$ images of the acquisition
* $A$ being the MUSE forward operator $PΘFSΔ$

And further calculation of $b_0^{**}$ because $b_0^{*}$ shows artifacts in regions of high phase:

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?\displaystyle b_0^{**}=%5Cbegin%7Bcases%7D%5Ctext%7Bavg(%7D%5Cmathbf%7Bb_0%7D^*%5Ctext%7B)%7D%26%5Ctext%7Bif%20ang(%7D%7Bb_0%7D^*%5Ctext%7B)%7D%5Cgeq%2050%5Ctext%7B,%7D%5C%5C%20%7Bb_0%7D^*%26%5Ctext%7Botherwise.%7D%5Cend%7Bcases%7D" />
</p>

</details>

To run the reconstructions, you need to adapt the config files  
according to your use-case.  
The shot reconstruction only needs to be run once and only for the LASER  
reconstruction.  
For the denoising using the DAE, the data has to be reconstructed with PI first.

### Usage
Use `--config` to select a YAML file and run

```powershell
python reconstruction.py --config config_HR.yaml
```

```powershell
python reconstruction.py --config config_LR.yaml --part='1' --file_name_suffix='_us3'
python reconstruction.py --config config_LR.yaml --part='2' --file_name_suffix='_us3'
```
