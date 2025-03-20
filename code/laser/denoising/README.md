## Denoising directory
`denoising_comp.py` implements a comparison between denoising raw MUSE data using:
* linear subspace models (based on singular-value decomposition)
* deep-learning non-linear subspace models (based on auto-encoder)

Both model-types first compress the data to a lower dimensional representation,  
and then decompress it to the original representation, suppressing a certain  
amount of noise.

The linear subspace models first "learn" a threshold for the important  
singular-values from a signal dictionary. This threshold defines  
which singular-values are kept after SVD of the data leading to a  
compression of the data so the subspace is only built by the  
most important singular-values from, which the high-dimensional  
signal is decompressed again.

The VAE actually learns the compression of the data by minimizing a loss-function  
during training between clean and noisy data. It is trained with a large signal  
dictionary and and learns denoising by compression with non-linear activation  
functions, which is benefitial, because of the highly non-linear nature  
of the diffusion data. 

To run `denoising_comp.py` adapt the config for your needs, naming the VAEs  
trained with DT and BAS model (if none it will be skipped), the path to the  
reconstructed MUSE data, the data shape and the number of samples for the   
linear subspace model "training" dictionary, to determine the SV-threshold.  
(The script will always run the linear subspace twice, once with DT and another  
time with the BAS diffusion model.)

Then just run:
```bash
python denoising_comp.py
```