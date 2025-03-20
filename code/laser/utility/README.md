## Utility directory

`combine_and_fit.py` implements:
- combining all reconstructed slices of the dataset into one file
- combining the latent slices of LASER reconstructions into one file
- fitting the data to receive fractional anisotropy and  
colored fractional anisotropy fits

To use the `combine_and_fit.py` script you need to adapt the config.  
Just look into the config to see the options.  
You can't run combination if not all slices are reconstructed.  

`convert_to_nifti.py` implements:
- conversion from h5py to nifti file format

This is useful if you want to use MITK or other tools to look at data  
and/or post-process your data. You need to adapt the script according  
to your data. 

