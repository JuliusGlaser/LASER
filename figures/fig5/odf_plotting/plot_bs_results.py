import matplotlib.pyplot as plt
import numpy as np
import h5py
from scipy.stats import gaussian_kde
import ipywidgets as widgets
import os
import nibabel as nib

def angle_between_vectors_arr(a, b, vec_dim=-1):
    dots = np.sum(a* b, axis=vec_dim)
    norms = np.linalg.norm(a, axis=vec_dim) * np.linalg.norm(b, axis=vec_dim)
    cos_theta = np.clip(dots / norms, -1.0, 1.0)
    theta_rad = np.arccos(cos_theta)        # in radians
    theta_deg = np.degrees(theta_rad)
    theta_deg[theta_deg > 90] = 180 - theta_deg[theta_deg > 90]  # angle between directions, not vectors
    return theta_deg

def angle_between_vectors(a, b):
    a, b = a.flatten(), b.flatten()
    cos_theta = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    cos_theta = np.clip(cos_theta, -1.0, 1.0)  # numerical safety
    return np.arccos(cos_theta) * 180 / np.pi  # in degrees

def compute_analysis_metrics(vec1_comb, org_vec1):
    n_bs = vec1_comb.shape[-2]
    # Compute average outer product
    M = np.einsum('...i,...j->...ij', vec1_comb, vec1_comb)  # shape (..., n, 3, 3)
    M = M.mean(axis=-3)  # average over the "n" dimension

    eigvals = np.zeros(M.shape[0:-1])
    eigvecs = np.zeros(M.shape)

    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            for k in range(M.shape[2]):
                if np.any(np.isnan(M[i,j,k,:,:])) or np.any(np.isinf(M[i,j,k,:,:])):
                    continue
                eval, evec = np.linalg.eig(M[i,j,k,:,:])
                idx = np.argsort(eval)[::-1]   # descending order
                eigvals[i,j,k,:] = eval[idx]
                eigvecs[i,j,k,:,:] = evec[:, idx]

    kappa = (1-np.sqrt((eigvals[...,1] + eigvals[...,2]) / (2*eigvals[...,0])))
    kappa[np.isnan(kappa)] = 0

    # Mean orientation is the first eigenvector of the dyadic tensor
    mean_orientations = eigvecs[...,:,0]  # shape (..., 3)
    # calculate angular bias: angle between mean orientation and ground truth
    bias_angle = angle_between_vectors_arr(mean_orientations, org_vec1)
    # calculate angular precision: angle between mean orientation and all bootstrap results and sort that
    precision_angle = angle_between_vectors_arr(mean_orientations[..., np.newaxis,:], vec1_comb)
    sorted_angle = np.sort(precision_angle, axis=-1)
    # take 95th percentile as measure of precision
    confidence_interval = 0.95
    idx_95 = int(np.round(confidence_interval * n_bs)) - 1
    print('95 confidence interval in reality index: ', idx_95)
    angle_95 = sorted_angle[..., idx_95]
    return kappa, bias_angle, angle_95

def plot_unit_sphere_samples(points):
    # Extract x, y, and z coordinates for plotting
    x_points, y_points, z_points = zip(*points)

    # Create a 3D plot for the unit sphere
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the sphere surface
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 50)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color='lightblue', alpha=0.3)

    # Plot each point on the unit sphere
    ax.scatter(x_points, y_points, z_points, color='red', s=50, label='Samples')

    # Label axes and set the aspect ratio to be equal
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_box_aspect([1,1,1])

    # Show the plot
    plt.legend()
    plt.show()


def run_statistics_on_angles_masked(dir, mask, slice_idx=[0]):
    # only angle one is relevant for analysis

    f = h5py.File(dir + r'bootstrap_analysis_PI.h5','r')
    vec1_comb_PI = f['vec1_comb'][:]
    org_vec1 = f['org_vec_2'][:]
    vec2_comb_PI = f['vec2_comb'][:]
    org_vec2 = f['org_vec_2'][:]
    angles_PI = f['angles'][:]
    f.close()

    vec1_comb_PI = vec1_comb_PI[:,:,slice_idx,:,:]*mask[..., slice_idx,None, None]
    org_vec1 = org_vec1[:,:,slice_idx,:]*mask[..., slice_idx, None]
    vec2_comb_PI = vec2_comb_PI[:,:,slice_idx,:,:]*mask[..., slice_idx,None, None]
    org_vec2 = org_vec2[:,:,slice_idx,:]*mask[..., slice_idx, None]
    angles_PI_mask = np.where(mask[..., slice_idx,None, None], angles_PI[:,:,slice_idx,:,:], np.nan)

    angle_1_PI_mask = angles_PI_mask[...,0].flatten()
    angle_1_PI_filtered_mask = angle_1_PI_mask[~np.isnan(angle_1_PI_mask) & ~np.isinf(angle_1_PI_mask)]*180/np.pi
    angle_2_PI_mask = angles_PI_mask[...,1].flatten()
    angle_2_PI_filtered_mask = angle_2_PI_mask[~np.isnan(angle_2_PI_mask) & ~np.isinf(angle_2_PI_mask)]*180/np.pi

    f = h5py.File(dir + r'bootstrap_analysis_MPPCA.h5','r')
    vec1_comb_MPPCA = f['vec1_comb'][:]
    vec2_comb_MPPCA = f['vec2_comb'][:]
    angles_MPPCA = f['angles'][:]
    f.close()

    vec1_comb_MPPCA = vec1_comb_MPPCA[:,:,slice_idx,:,:]*mask[..., slice_idx,None, None]
    vec2_comb_MPPCA = vec2_comb_MPPCA[:,:,slice_idx,:,:]*mask[..., slice_idx,None, None]
    angles_MPPCA_mask = np.where(mask[..., slice_idx,None, None], angles_MPPCA[:,:,slice_idx,:,:], np.nan)

    angle_1_MPPCA_mask = angles_MPPCA_mask[...,0].flatten()
    angle_1_MPPCA_filtered_mask = angle_1_MPPCA_mask[~np.isnan(angle_1_MPPCA_mask) & ~np.isinf(angle_1_MPPCA_mask)]*180/np.pi
    angle_2_MPPCA_mask = angles_MPPCA_mask[...,1].flatten()
    angle_2_MPPCA_filtered_mask = angle_2_MPPCA_mask[~np.isnan(angle_2_MPPCA_mask) & ~np.isinf(angle_2_MPPCA_mask)]*180/np.pi

    f = h5py.File(dir + r'bootstrap_analysis_LLR.h5','r')
    vec1_comb_LLR = f['vec1_comb'][:]
    vec2_comb_LLR = f['vec2_comb'][:]
    angles_LLR = f['angles'][:]
    f.close()

    vec1_comb_LLR = vec1_comb_LLR[:,:,slice_idx,:,:]*mask[..., slice_idx,None, None]
    vec2_comb_LLR = vec2_comb_LLR[:,:,slice_idx,:,:]*mask[..., slice_idx,None, None]
    angles_LLR_mask = np.where(mask[..., slice_idx,None, None], angles_LLR[:,:,slice_idx,:,:], np.nan)

    angle_1_LLR_mask = angles_LLR_mask[...,0].flatten()
    angle_1_LLR_filtered_mask = angle_1_LLR_mask[~np.isnan(angle_1_LLR_mask) & ~np.isinf(angle_1_LLR_mask)]*180/np.pi
    angle_2_LLR_mask = angles_LLR_mask[...,1].flatten()
    angle_2_LLR_filtered_mask = angle_2_LLR_mask[~np.isnan(angle_2_LLR_mask) & ~np.isinf(angle_2_LLR_mask)]*180/np.pi

    f = h5py.File(dir + r'bootstrap_analysis_DTI.h5','r')
    vec1_comb_DTI = f['vec1_comb'][:]
    vec2_comb_DTI = f['vec2_comb'][:]
    angles_DTI = f['angles'][:]
    f.close()

    vec1_comb_DTI = vec1_comb_DTI[:,:,slice_idx,:,:]*mask[..., slice_idx,None, None]
    vec2_comb_DTI = vec2_comb_DTI[:,:,slice_idx,:,:]*mask[..., slice_idx,None, None]
    angles_DTI_mask = np.where(mask[..., slice_idx,None, None], angles_DTI[:,:,slice_idx,:,:], np.nan)

    angle_1_DTI_mask = angles_DTI_mask[...,0].flatten()
    angle_1_DTI_filtered_mask = angle_1_DTI_mask[~np.isnan(angle_1_DTI_mask) & ~np.isinf(angle_1_DTI_mask)]*180/np.pi
    angle_2_DTI_mask = angles_DTI_mask[...,1].flatten()
    angle_2_DTI_filtered_mask = angle_2_DTI_mask[~np.isnan(angle_2_DTI_mask) & ~np.isinf(angle_2_DTI_mask)]*180/np.pi

    f = h5py.File(dir + r'bootstrap_analysis_BAS.h5','r')
    vec1_comb_BAS = f['vec1_comb'][:]
    vec2_comb_BAS = f['vec2_comb'][:]
    angles_BAS = f['angles'][:]
    f.close()

    vec1_comb_BAS = vec1_comb_BAS[:,:,slice_idx,:,:]*mask[..., slice_idx,None, None]
    vec2_comb_BAS = vec2_comb_BAS[:,:,slice_idx,:,:]*mask[..., slice_idx,None, None]
    angles_BAS_mask = np.where(mask[..., slice_idx,None, None], angles_BAS[:,:,slice_idx,:,:], np.nan)

    angle_1_BAS_mask = angles_BAS_mask[...,0].flatten()
    angle_1_BAS_filtered_mask = angle_1_BAS_mask[~np.isnan(angle_1_BAS_mask) & ~np.isinf(angle_1_BAS_mask)]*180/np.pi
    angle_2_BAS_mask = angles_BAS_mask[...,1].flatten()
    angle_2_BAS_filtered_mask = angle_2_BAS_mask[~np.isnan(angle_2_BAS_mask) & ~np.isinf(angle_2_BAS_mask)]*180/np.pi

    # data1 = [angle_1_PI_filtered_mask]
    # data2 = [angle_2_PI_filtered_mask]
    data1 = [angle_1_PI_filtered_mask, angle_1_MPPCA_filtered_mask, angle_1_LLR_filtered_mask, angle_1_DTI_filtered_mask, angle_1_BAS_filtered_mask]
    data2 = [angle_2_PI_filtered_mask, angle_2_MPPCA_filtered_mask, angle_2_LLR_filtered_mask, angle_2_DTI_filtered_mask, angle_2_BAS_filtered_mask]
    return data1, data2

def run_statistics_on_angles_masked_single_data(dir, mask, vec1_comb, vec2_comb, angles, org_vec1, org_vec2, slice_idx=[0]):
    # only angle one is relevant for analysis

    vec1_comb = vec1_comb[:,:,slice_idx,:,:]*mask[..., slice_idx,None, None]
    org_vec1 = org_vec1[:,:,slice_idx,:]*mask[..., slice_idx, None]
    vec2_comb = vec2_comb[:,:,slice_idx,:,:]*mask[..., slice_idx,None, None]
    org_vec2 = org_vec2[:,:,slice_idx,:]*mask[..., slice_idx, None]
    angles_mask = np.where(mask[..., slice_idx,None, None], angles[:,:,slice_idx,:,:], np.nan)

    angle_1_mask = angles_mask[...,0].flatten()
    angle_1_filtered_mask = angle_1_mask[~np.isnan(angle_1_mask) & ~np.isinf(angle_1_mask)]*180/np.pi
    angle_2_mask = angles_mask[...,1].flatten()
    angle_2_filtered_mask = angle_2_mask[~np.isnan(angle_2_mask) & ~np.isinf(angle_2_mask)]*180/np.pi

    return angle_1_filtered_mask, angle_2_filtered_mask

def plot_violins(data, dataNames, save=False, save_path=None, onlyOne=False):
    num_rows = len(data)
    if onlyOne:
        plot_colors = ["#FF0000"]
    else:
        plot_colors = ["#FF0000", "#750404","#0000FF", "#00FF00", "#B66700"]
    # Create 2x2 subplot grid
    fig, axes = plt.subplots(num_rows, 1, figsize=(12, 5))
    if num_rows > 1:
        axes = axes.flatten()
    if num_rows == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        parts = ax.violinplot(data[i], showmeans=False, showmedians=True, showextrema=False)

        for l, pc in enumerate(parts['bodies']):
            pc.set_facecolor(plot_colors[l])
            pc.set_edgecolor('black')
            pc.set_alpha(1)

        
        if onlyOne:
            ax.set_xticks([1])
            ax.set_xticklabels(["datalabel"])
        else:
            ax.set_xticks([1, 2, 3, 4, 5])
            ax.set_xticklabels(["PI", "MPPCA", "LLR", "DTI", "BAS"])
        if i > 1:
            hline = 0
            ax.axhline(hline, color='red', linestyle='--', linewidth=2, label=f"GT value")
        ax.set_title(dataNames[i])
        ax.set_ylabel("Values")

    plt.tight_layout()
    if save:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()

def plot_boxplots(data, dataNames, save=False, save_path=None, onlyOne=False):
    num_cols = len(data)
    fig, axes = plt.subplots(1, num_cols, figsize=(12, 5))
    if num_cols > 1:
        axes = axes.flatten()
    if num_cols == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        if onlyOne:
            ax.boxplot(data[i], tick_labels=["datalabel"])
        else:
            ax.boxplot(data[i], tick_labels=["PI", "MPPCA", "LLR", "DTI", "BAS"])
        if i > 1:
            hline = 0
            ax.axhline(hline, color='red', linestyle='--', linewidth=2, label=f"GT value")
        ax.set_title(dataNames[i])
        ax.set_ylabel("Values")

    plt.tight_layout()
    if save:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()

def print_stats(name, data, onlyOne=False):
    
    for l in range(len(data)):
        print(f'Median, mean, std of angle {l+1} in {name}:')
        if onlyOne:
            methods = ['dataLabel']
        else:
            methods = ['PI', 'MPPCA', 'LLR', 'DTI', 'BAS']
        for i, method in enumerate(methods):
        # for i, method in enumerate(['PI']):
            median = np.median(data[l][i])
            mean = np.mean(data[l][i])
            std = np.std(data[l][i])
            q1, median, q3 = np.percentile(data[l][i], [25, 50, 75])
            max = np.max(data[l][i])
            min = np.min(data[l][i])
            print(f"{method}:\t Median = {median:.2f}, \t Mean = {mean:.2f},\t Std = {std:.2f}")
            print(f"{method}:\t q1 = {q1:.2f}, \t q3 = {q3:.2f},\t max = {max:.2f},\t min = {min:.2f}")
        print('\n\n')

def precompute_violin(data, n_points=200):
    """
    Compute violin plot components: KDE + quartiles + whiskers.
    """
    data = np.asarray(data)
    kde = gaussian_kde(data)
    x_grid = np.linspace(data.min(), data.max(), n_points)
    density = kde(x_grid)
    q1, median, q3 = np.percentile(data, [25, 50, 75])
    iqr = q3 - q1
    whiskers = [max(data.min(), q1 - 1.5 * iqr), min(data.max(), q3 + 1.5 * iqr)]
    
    return {
        "x": x_grid.tolist(),
        "density": density.tolist(),
        "q1": float(q1),
        "median": float(median),
        "q3": float(q3),
        "whiskers": [float(whiskers[0]), float(whiskers[1])]
    }

def precompute_box(data):
    """
    Compute boxplot components: min, quartiles, median, max.
    """
    data = np.asarray(data)
    q1, median, q3 = np.percentile(data, [25, 50, 75])
    return {
        "min": float(data.min()),
        "q1": float(q1),
        "median": float(median),
        "q3": float(q3),
        "max": float(data.max())
    }

def save_precomputed_npz(violin_list, box_list, filename="precomputed.npz"):
    np.savez_compressed(filename,
                        violins=np.array(violin_list, dtype=object),
                        boxes=np.array(box_list, dtype=object))


# Load mask
index_f = nib.load(r'C:\msys64\home\glaserjs\bootstrap_analysis_1_fod\0\fixel_masks_all_15\index.nii')
index_data = index_f.get_fdata()
mask_1_plus_fiber_all = (index_data[:,:,0:25,0] > 0).astype(bool)
mask_2_fiber_all = (index_data[:,:,0:25,0] > 1).astype(bool)

dir = r'W:\radiologie\mrt-probanden\AG_Laun\Julius_Glaser\Revision_bipolar\fod\BS_analysis_slice_all_1000' + os.sep
save_dir = r'W:\radiologie\mrt-probanden\AG_Laun\Julius_Glaser\Revision_bipolar\fod\BS_analysis_slice_all_1000/results' + os.sep
# Load data

# # LLR
# f = h5py.File(dir + r'bootstrap_analysis_LLR.h5','r')
# print(f.keys())
# angles_PI = f['angles'][:]
# f.close()
# angle_1_PI = np.where(mask_1_plus_fiber_all[...,None], angles_PI[...,0], np.nan).flatten()

# angle_1_PI_filtered = angle_1_PI[~np.isnan(angle_1_PI) & ~np.isinf(angle_1_PI)]*180/np.pi
# angle_2_PI = np.where(mask_2_fiber_all[...,None], angles_PI[...,1], np.nan).flatten()
# angle_2_PI_filtered = angle_2_PI[~np.isnan(angle_2_PI) & ~np.isinf(angle_2_PI)]*180/np.pi

# # MPPCA
# f = h5py.File(dir + r'bootstrap_analysis_MPPCA.h5','r')
# print(f.keys())
# angles_MPPCA = f['angles'][:]
# f.close()
# angle_1_MPPCA = np.where(mask_1_plus_fiber_all[...,None], angles_MPPCA[...,0], np.nan).flatten()
# angle_1_MPPCA_filtered = angle_1_MPPCA[~np.isnan(angle_1_MPPCA) & ~np.isinf(angle_1_MPPCA)]*180/np.pi
# angle_2_MPPCA = np.where(mask_2_fiber_all[...,None], angles_MPPCA[...,1], np.nan).flatten()
# angle_2_MPPCA_filtered = angle_2_MPPCA[~np.isnan(angle_2_MPPCA) & ~np.isinf(angle_2_MPPCA)]*180/np.pi

# # LLR
# f = h5py.File(dir + r'bootstrap_analysis_LLR.h5','r')
# print(f.keys())
# angles_LLR = f['angles'][:]
# f.close()
# angle_1_LLR = np.where(mask_1_plus_fiber_all[...,None], angles_LLR[...,0], np.nan).flatten()
# angle_1_LLR_filtered = angle_1_LLR[~np.isnan(angle_1_LLR) & ~np.isinf(angle_1_LLR)]*180/np.pi
# angle_2_LLR = np.where(mask_2_fiber_all[...,None], angles_LLR[...,1], np.nan).flatten()
# angle_2_LLR_filtered = angle_2_LLR[~np.isnan(angle_2_LLR) & ~np.isinf(angle_2_LLR)]*180/np.pi

# # DTI
# f = h5py.File(dir + r'bootstrap_analysis_DTI.h5','r')
# print(f.keys())
# angles_DTI = f['angles'][:]
# f.close()
# angle_1_DTI = np.where(mask_1_plus_fiber_all[...,None], angles_DTI[...,0], np.nan).flatten()
# angle_1_DTI_filtered = angle_1_DTI[~np.isnan(angle_1_DTI) & ~np.isinf(angle_1_DTI)]*180/np.pi
# angle_2_DTI = np.where(mask_2_fiber_all[...,None], angles_DTI[...,1], np.nan).flatten()
# angle_2_DTI_filtered = angle_2_DTI[~np.isnan(angle_2_DTI) & ~np.isinf(angle_2_DTI)]*180/np.pi

# # BAS
# f = h5py.File(dir + r'bootstrap_analysis_BAS.h5','r')
# print(f.keys())
# angles_BAS = f['angles'][:]
# f.close()
# angle_1_BAS = np.where(mask_1_plus_fiber_all[...,None], angles_BAS[...,0], np.nan).flatten()
# angle_1_BAS_filtered = angle_1_BAS[~np.isnan(angle_1_BAS) & ~np.isinf(angle_1_BAS)]*180/np.pi
# angle_2_BAS = np.where(mask_2_fiber_all[...,None], angles_BAS[...,1], np.nan).flatten()
# angle_2_BAS_filtered = angle_2_BAS[~np.isnan(angle_2_BAS) & ~np.isinf(angle_2_BAS)]*180/np.pi

# print('Number of primary fits')
# print('angle_1_PI_filtered shape: ',angle_1_PI_filtered.shape)
# print('angle_1_MPPCA_filtered shape: ',angle_1_MPPCA_filtered.shape)
# print('angle_1_LLR_filtered shape: ',angle_1_LLR_filtered.shape)
# print('angle_1_BAS_filtered shape: ',angle_1_BAS_filtered.shape)
# print('angle_1_DTI_filtered shape: ',angle_1_DTI_filtered.shape)

# print('Number of secondary fits')
# print('angle_2_PI_filtered shape: ',angle_2_PI_filtered.shape)
# print('angle_2_MPPCA_filtered shape: ',angle_2_MPPCA_filtered.shape)
# print('angle_2_LLR_filtered shape: ',angle_2_LLR_filtered.shape)
# print('angle_2_BAS_filtered shape: ',angle_2_BAS_filtered.shape)
# print('angle_2_DTI_filtered shape: ',angle_2_DTI_filtered.shape)

# # Create Violin plots

# angles1 = [angle_1_PI_filtered , angle_1_MPPCA_filtered, angle_1_LLR_filtered , angle_1_DTI_filtered , angle_1_BAS_filtered ]  # flatten to 1D
# angles2 = [angle_2_PI_filtered , angle_2_MPPCA_filtered, angle_2_LLR_filtered , angle_2_DTI_filtered , angle_2_BAS_filtered ]  # flatten to 1D

# data = [angles1, angles2]
# dataNames = ["Angle 1", "Angle 2"]
# plot_colors = ["#FF0000", "#750404","#0000FF", "#00FF00", "#B66700"]
# # Create 2x2 subplot grid
# fig, axes = plt.subplots(1, 2, figsize=(12, 5))
# axes = axes.flatten()

# for i, ax in enumerate(axes):
#     parts = ax.violinplot(data[i], showmeans=False, showmedians=True, showextrema=False)

#     for l, pc in enumerate(parts['bodies']):
#         pc.set_facecolor(plot_colors[l])
#         pc.set_edgecolor('black')
#         pc.set_alpha(1)

#     # quartile1, medians, quartile3 = np.percentile(data[i], [25, 50, 75], axis=1)
#     # whiskers = np.array([
#     #     adjacent_values(sorted_array, q1, q3)
#     #     for sorted_array, q1, q3 in zip(data[i], quartile1, quartile3)])
#     # whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]
#     # inds = np.arange(1, len(medians)+1)
#     # ax.scatter(inds, medians, marker='o', color='white', s=30, zorder=3)
#     # ax.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)
#     # ax.vlines(inds, whiskers_min, whiskers_max, color='k', linestyle='-', lw=1)
#     # ax.axhline(7)
#     ax.set_xticks([1, 2, 3, 4, 5])
#     ax.set_xticklabels(["PI", "MPPCA", "LLR", "DTI", "BAS"])
#     if i > 1:
#         hline = 0
#         ax.axhline(hline, color='red', linestyle='--', linewidth=2, label=f"GT value")
#     ax.set_title(dataNames[i])
#     ax.set_ylabel("Values")

# plt.tight_layout()
# plt.savefig(save_dir + 'angle_violins_all.pdf', dpi=300)


# # Create 2x2 subplot grid
# fig, axes = plt.subplots(1, 2, figsize=(12, 5))
# axes = axes.flatten()

# for i, ax in enumerate(axes):
#     ax.boxplot(data[i], tick_labels=["PI", "MPPCA", "LLR", "DTI", "BAS"])
#     if i > 1:
#         hline = 0
#         ax.axhline(hline, color='red', linestyle='--', linewidth=2, label=f"GT value")
#     ax.set_title(dataNames[i])
#     ax.set_ylabel("Values")

# plt.tight_layout()
# plt.savefig(save_dir + 'angle_boxes_all.pdf', dpi=300)

# # Compute new mask

mask_1_plus_fiber_float_all = mask_1_plus_fiber_all.copy().astype(float)
mask_2_fiber_float_all = mask_2_fiber_all.copy().astype(float)

mask_1_plus_fiber_float_all[~mask_1_plus_fiber_all] = np.nan
mask_2_fiber_float_all[~mask_2_fiber_all] = np.nan

# # Calculate bias

# f = h5py.File(dir + r'bootstrap_analysis_PI.h5','r')
# print(f.keys())
# vec1_comb_PI = f['vec1_comb'][:]
# org_vec1 = f['org_vec_1'][:]
# vec2_comb_PI = f['vec2_comb'][:]
# org_vec2 = f['org_vec_2'][:]
# f.close()
# kappa_1_PI, bias_angle_1_PI, angle_95_1_PI = compute_analysis_metrics(vec1_comb_PI, org_vec1)
# kappa_2_PI, bias_angle_2_PI, angle_95_2_PI = compute_analysis_metrics(vec2_comb_PI, org_vec2)

# f = h5py.File(save_dir + r'PI_bias_angle_all_slices.h5','w')
# f.create_dataset('Bias_1_PI', data=bias_angle_1_PI*mask_1_plus_fiber_float_all)
# f.create_dataset('Bias_2_PI', data=bias_angle_2_PI*mask_2_fiber_float_all)
# f.create_dataset('Angle_95_1_PI', data=angle_95_1_PI*mask_1_plus_fiber_float_all)
# f.create_dataset('Angle_95_2_PI', data=angle_95_2_PI*mask_2_fiber_float_all)
# f.close()

# f = h5py.File(dir + r'bootstrap_analysis_MPPCA.h5','r')
# vec1_comb_MPPCA = f['vec1_comb'][:]
# vec2_comb_MPPCA = f['vec2_comb'][:]
# f.close()
# kappa_1_MPPCA, bias_angle_1_MPPCA, angle_95_1_MPPCA = compute_analysis_metrics(vec1_comb_MPPCA, org_vec1)
# kappa_2_MPPCA, bias_angle_2_MPPCA, angle_95_2_MPPCA = compute_analysis_metrics(vec2_comb_MPPCA, org_vec2)

# f = h5py.File(save_dir + r'MPPCA_bias_angle_all_slices.h5','w')
# f.create_dataset('Bias_1_MPPCA', data=bias_angle_1_MPPCA*mask_1_plus_fiber_float_all)
# f.create_dataset('Bias_2_MPPCA', data=bias_angle_2_MPPCA*mask_2_fiber_float_all)
# f.create_dataset('Angle_95_1_MPPCA', data=angle_95_1_MPPCA*mask_1_plus_fiber_float_all)
# f.create_dataset('Angle_95_2_MPPCA', data=angle_95_2_MPPCA*mask_2_fiber_float_all)
# f.close()

# f = h5py.File(dir + r'bootstrap_analysis_DTI.h5','r')
# vec1_comb_DTI = f['vec1_comb'][:]
# vec2_comb_DTI = f['vec2_comb'][:]
# org_vec1 = f['org_vec_1'][:]
# org_vec2 = f['org_vec_2'][:]
# f.close()
# kappa_1_DTI, bias_angle_1_DTI, angle_95_1_DTI = compute_analysis_metrics(vec1_comb_DTI, org_vec1)
# kappa_2_DTI, bias_angle_2_DTI, angle_95_2_DTI = compute_analysis_metrics(vec2_comb_DTI, org_vec2)

# f = h5py.File(save_dir + r'DTI_bias_angle_all_slices.h5','w')
# f.create_dataset('Bias_1_DTI', data=bias_angle_1_DTI*mask_1_plus_fiber_float_all)
# f.create_dataset('Bias_2_DTI', data=bias_angle_2_DTI*mask_2_fiber_float_all)
# f.create_dataset('Angle_95_1_DTI', data=angle_95_1_DTI*mask_1_plus_fiber_float_all)
# f.create_dataset('Angle_95_2_DTI', data=angle_95_2_DTI*mask_2_fiber_float_all)
# f.close()

# f = h5py.File(dir + r'bootstrap_analysis_BAS.h5','r')
# vec1_comb_BAS = f['vec1_comb'][:]
# vec2_comb_BAS = f['vec2_comb'][:]
# f.close()
# kappa_1_BAS, bias_angle_1_BAS, angle_95_1_BAS = compute_analysis_metrics(vec1_comb_BAS, org_vec1)
# kappa_2_BAS, bias_angle_2_BAS, angle_95_2_BAS = compute_analysis_metrics(vec2_comb_BAS, org_vec2)

# f = h5py.File(save_dir + r'BAS_bias_angle_all_slices.h5','w')
# f.create_dataset('Bias_1_BAS', data=bias_angle_1_BAS*mask_1_plus_fiber_float_all)
# f.create_dataset('Bias_2_BAS', data=bias_angle_2_BAS*mask_2_fiber_float_all)
# f.create_dataset('Angle_95_1_BAS', data=angle_95_1_BAS*mask_1_plus_fiber_float_all)
# f.create_dataset('Angle_95_2_BAS', data=angle_95_2_BAS*mask_2_fiber_float_all)
# f.close()

# f = h5py.File(dir + r'bootstrap_analysis_LLR.h5','r')
# vec1_comb_LLR = f['vec1_comb'][:]
# vec2_comb_LLR = f['vec2_comb'][:]
# org_vec1 = f['org_vec_1'][:]
# org_vec2 = f['org_vec_2'][:]
# f.close()
# kappa_1_LLR, bias_angle_1_LLR, angle_95_1_LLR = compute_analysis_metrics(vec1_comb_LLR, org_vec1)
# kappa_2_LLR, bias_angle_2_LLR, angle_95_2_LLR = compute_analysis_metrics(vec2_comb_LLR, org_vec2)

# f = h5py.File(save_dir + r'LLR_bias_angle_all_slices.h5','w')
# f.create_dataset('Bias_1_LLR', data=bias_angle_1_LLR*mask_1_plus_fiber_float_all)
# f.create_dataset('Bias_2_LLR', data=bias_angle_2_LLR*mask_2_fiber_float_all)
# f.create_dataset('Angle_95_1_LLR', data=angle_95_1_LLR*mask_1_plus_fiber_float_all)
# f.create_dataset('Angle_95_2_LLR', data=angle_95_2_LLR*mask_2_fiber_float_all)
# f.close()

# # Load analyzing masks
mask_dir = r'W:\radiologie\mrt-probanden\AG_Laun\Julius_Glaser\Revision_bipolar\fod\Analyzing_masks\All_slices' + os.sep

# mask_CC_f = nib.load(mask_dir + r'CorpusCallosum.nii')
# CC_all = mask_CC_f.get_fdata().astype(bool)

# crossing_section_f = nib.load(mask_dir + r'CrossingSection.nii')
# CS_all = crossing_section_f.get_fdata().astype(bool)                         #Crossing section front

# internal_capsule_f = nib.load(mask_dir + r'InternalCapsule.nii')
# IC_all = internal_capsule_f.get_fdata().astype(bool)                              #IC = Internal Capsule

# angles1_CC_all, angles2_CC_all = run_statistics_on_angles_masked(dir, CC_all, slice_idx=list(range(25)))
# mask_names = ['CorpusCallosum.nii','CrossingSection.nii', 'InternalCapsuleNew.nii']
mask_names = ['CorpusCallosum.nii']
data_to_process_list = ['PI', 'MPPCA', 'LLR', 'DTI', 'BAS']
plot_colors = ["#FF0000", "#750404","#0000FF", "#00FF00", "#B66700"]
# file_names = ['bootstrap_analysis_'+data_to_process+'.h5']

for mask in mask_names:
    mask_file = nib.load(mask_dir + mask)
    mask_data = mask_file.get_fdata().astype(bool)

    mask_name_str = mask.split('.nii')[0]
    fig1, ax1 = plt.subplots()
    ax1.set_xlim(0, len(data_to_process_list) + 1)
    ax1.set_xticklabels(data_to_process_list)
    ax1.set_ylim(0, 90)

    fig2, ax2 = plt.subplots()
    ax2.set_xlim(0, len(data_to_process_list) + 1)
    ax2.set_xticklabels(data_to_process_list)
    ax2.set_ylim(0, 90)

    for i, data_to_process in enumerate(data_to_process_list):
        data_file = 'bootstrap_analysis_'+data_to_process+'.h5'
        print('Processing file: ', data_file)
        f = h5py.File(dir + data_file,'r')
        vec1_comb = f['vec1_comb'][:]
        org_vec1 = f['org_vec_1'][:]
        vec2_comb = f['vec2_comb'][:]
        org_vec2 = f['org_vec_2'][:]
        angles = f['angles'][:]
        f.close()
        angle_1_filtered_mask, angle_2_filtered_mask = run_statistics_on_angles_masked_single_data(dir, mask_data, vec1_comb, vec2_comb, angles, org_vec1, org_vec2, slice_idx=list(range(25)))
        
        parts = ax1.violinplot(angle_1_filtered_mask, positions=[i+1], widths=0.5, points=300, showmeans=False, showmedians=True, showextrema=False)
        pc = parts['bodies'][0]
        pc.set_facecolor(plot_colors[i])
        pc.set_edgecolor('black')
        pc.set_alpha(1)
        for partname in ('cbars', 'cmins', 'cmaxes', 'cmeans', 'cmedians'):
            if partname in parts:
                vp = parts[partname]
                vp.set_color('black')
                vp.set_linewidth(1)
        

        parts = ax2.violinplot(angle_2_filtered_mask, positions=[i+1], widths=0.5, points=300, showmeans=False, showmedians=True, showextrema=False)
        pc = parts['bodies'][0]
        pc.set_facecolor(plot_colors[i])
        pc.set_edgecolor('black')
        pc.set_alpha(1)
        for partname in ('cbars', 'cmins', 'cmaxes', 'cmeans', 'cmedians'):
            if partname in parts:
                vp = parts[partname]
                vp.set_color('black')
                vp.set_linewidth(1)

        # violin_list = precompute_violin(data[i], n_points=200)

        # box_list = precompute_box(data[i])

        # save_precomputed_npz(violin_list, box_list, filename=save_dir + 'angle_violins_'+mask_name_str+'_all_' +data_to_process+'_'+dataNames[i]+'_precomputed.npz')
    fig1.savefig(save_dir+'angle1_violins_'+mask_name_str+'_all.pdf', dpi=300, bbox_inches="tight")
    fig2.savefig(save_dir+'angle2_violins_'+mask_name_str+'_all.pdf', dpi=300, bbox_inches="tight")
    # plot_violins(data, dataNames, save=True, save_path=save_dir+'angle_violins_'+mask_name_str+'_all_'+data_to_process+'.pdf', onlyOne=True)
    # plot_boxplots(data, dataNames, save=True, save_path=save_dir+'angle_boxes_'+mask_name_str+'_all_'+data_to_process+'.pdf', onlyOne=True)
    # print_stats(mask_name_str+ ' all mask', data, onlyOne=True)