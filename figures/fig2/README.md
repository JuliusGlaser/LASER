# **⚠️ This repository is under construction! ⚠️**
# In this folder the figure for the forward operator is shown

## Forward operator of LASER
<p align="center">
  <img alt="Light" src="fwd.png" width="100%">
</p>
The reconstruction minimization Problem is described by 

$$
||k-PΘFSΔ(|b_0^{*}|D_{Ψ^*}(\beta)exp(i\phi^*))|| - λ||\beta||_{TV}
$$

With meaning as:

* $k$ is the k-space data
* $P$ is the undersampling pattern
* $Θ$ is the SMS operator
* $F$ is the forward Fourier transform
* $S$ is the multiplication with coil sensetivities
* $Δ$ is the multiplication with the shot-to-shot phases
* $b_0^{*}$ is the reconstructed b0 image (optimizable)
* $D$ is the voxel-wise decoding of $\beta$
* $\beta$ are the reconstructed latent images (otimizable)
* $\phi^*$ are the reconstructed diffusion encoding phases
* $λ$ is the regularization weight
* $||x||_{TV}$ is the TV-norm of the latent iamges

Reconstruction is done by initializing $\beta$ with zeros and minimizing the difference between the sampled k-space data $k$ and $\beta$ after applying the forward operator.
The operations of decoding and multiplication with $b_0^{*}$ and $\phi^*$ are applied in a voxel-wise fashion along contrast dimension, while the remaining operations are applied in a image-wise manner.

[//]: <> (Examples for the reconstruction can be found in the `examples.ipynb`)