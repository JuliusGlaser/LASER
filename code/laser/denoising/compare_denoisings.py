f = h5py.File('/home/hpc/mfqb/mfqb102h/LASER/data/ref/PI_CombShots.h5', 'r')
ref = f['DWI'][:]
f.close()
print(ref.shape)
ref = np.divide(ref, ref[0, ...],
                        out=np.zeros_like(ref),
                        where=ref!=0)

f = h5py.File('/home/woody/mfqb/mfqb102h/meas_MID00201_FID00639_ep2d_diff_1_seg_3x1_126/reco_us_factor4_slice_015.h5', 'r')
print(list(f))
ref_us = f['DWI'][:]
ref_us.shape
f.close()
ref_us = np.divide(ref_us, ref_us[0, ...],
                        out=np.zeros_like(ref_us),
                        where=ref_us!=0)

f = h5py.File('/home/hpc/mfqb/mfqb102h/LASER/code/laser/denoising/denoised_comparison_reco_us_f4_slice_015.h5', 'r')
BAS_SVD = f['BAS_SVD'][:].T
BAS_SVD = np.divide(BAS_SVD, BAS_SVD[0, ...],
                        out=np.zeros_like(ref_us),
                        where=BAS_SVD!=0)
BAS_VAE = f['BAS_VAE'][:].T
BAS_VAE = np.divide(BAS_VAE, BAS_VAE[0, ...],
                        out=np.zeros_like(ref_us),
                        where=BAS_VAE!=0)
f.close()

N_diff = 126
diff_slider = widgets.IntSlider(min=0, max= N_diff-1, step=1)
slice_slider = widgets.IntSlider(min=0, max= N_z-1, step=1)

@widgets.interact(n_slice = slice_slider, n_diff = diff_slider)
def interact_plots(n_diff):
    fig, axs = plt.subplots(3, 2, figsize=(8, 8))  # Increase figure size
    vmax = np.max(abs(ref[n_diff, 31, :, :]))
    vmin = 0
    axs[0,0].imshow(abs(ref[n_diff, 31, :, :]), cmap='gray', vmin=0, vmax=1)
    axs[0,0].set_title('Ref')
    axs[0,0].axis('off')
    axs[0,1].imshow(abs(ref_us[n_diff, 0, :, :]), cmap='gray', vmin=0, vmax=1)
    axs[0,1].set_title('US')
    axs[0,1].axis('off')
    axs[1,0].imshow(abs(BAS_SVD[n_diff, 0, :, :]), cmap='gray', vmin=0, vmax=1)
    axs[1,0].set_title('SVD denoised')
    axs[1,0].axis('off')
    axs[1,1].imshow(abs(BAS_VAE[n_diff, 0, :, :]), cmap='gray', vmin=0, vmax=1)
    axs[1,1].set_title('VAE denoised')
    axs[1,1].axis('off')
    axs[2,0].imshow(abs(ref[n_diff, 31, :, :]) - abs(BAS_SVD[n_diff, 0, :, :]), cmap='gray', vmin=0, vmax=1)
    axs[2,0].set_title('diff SVD denoised ref')
    axs[2,0].axis('off')
    axs[2,1].imshow(abs(ref[n_diff, 31, :, :])-abs(BAS_VAE[n_diff, 0, :, :]), cmap='gray', vmin=0, vmax=1)
    axs[2,1].set_title('diff VAE denoised ref')
    axs[2,1].axis('off')
    