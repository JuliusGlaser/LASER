import matplotlib.pyplot as plt
import numpy as np
import h5py
from pathlib import Path
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

def run_statistics_on_angles_masked_data(data1, data2, mask1, mask2=None):
    # only angle one is relevant for analysis
    if mask2 is None:
        mask2 = mask1

    out1 = []
    for d in data1:
        d_masked = (d*mask1).flatten()
        d_filtered_mask = d_masked[~np.isnan(d_masked) & ~np.isinf(d_masked)]
        out1.append(d_filtered_mask)

    out2 = []
    for d in data2:
        d_masked = (d*mask2).flatten()
        d_filtered_mask = d_masked[~np.isnan(d_masked) & ~np.isinf(d_masked)]
        out2.append(d_filtered_mask)

    
    return out1, out2

def plot_violins(data, dataNames, dataTitles, save_fig=False, save_dir="./", spacing=0.35, bias_or_prec="bias", region='', ylim_max=[90, 90]):
    """
    spacing: horizontal distance between adjacent violins (smaller => closer together).
             Try 0.25–0.40. Adjust widths below if you change spacing.
    """
    import os, numpy as np, matplotlib.pyplot as plt

    num_rows = len(data)
    plot_colors = ["#0050FF", "#FFD200","#FF00A2", "#00C400", "#FF6A00", "#000000", "#00FFFF", "#FFC0CB", "#808080"]

    # x positions packed tightly: start at 1.0 and step by `spacing`
    n = len(dataNames)
    positions = 1.0 + np.arange(n) * spacing
    widths = spacing * 0.9  # slightly less than spacing so neighboring violins nearly touch

    fig, axes = plt.subplots(num_rows, 1, figsize=(3, 3*num_rows), squeeze=False)
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        parts = ax.violinplot(
            data[i],
            positions=positions,
            widths=widths,
            showmeans=False,
            showmedians=True,
            showextrema=False
        )

        for l, pc in enumerate(parts['bodies']):
            pc.set_facecolor(plot_colors[l % len(plot_colors)])
            pc.set_edgecolor('black')
            pc.set_alpha(1)

        ax.set_xticks(positions)
        ax.set_xticklabels(dataNames, rotation=0)  # rotate if labels start colliding
        if i > 1:
            ax.axhline(0, color='red', linestyle='--', linewidth=2, label="GT value")
        ax.set_title(dataTitles[i])
        ax.set_ylabel("Values")

        # Optional: tighten x-limits so there isn't extra padding
        pad = spacing * 0.7
        ax.set_xlim(positions[0] - pad, positions[-1] + pad)
        ax.set_ylim(0, ylim_max[i])
        from matplotlib.ticker import MultipleLocator, FormatStrFormatter
        ax.yaxis.set_major_locator(MultipleLocator(10))        # ticks every 10
        ax.yaxis.set_major_formatter(FormatStrFormatter('%d')) # force integer formatting

        # --- Save each subplot (row) separately as a PDF ---
        if save_fig:
            os.makedirs(save_dir, exist_ok=True)
            subfig = plt.figure(figsize=(3, 3))
            sub_ax = subfig.add_subplot(111)
            parts_sub = sub_ax.violinplot(
                data[i],
                positions=positions,
                widths=widths,
                showmeans=False,
                showmedians=True,
                showextrema=False
            )
            for l, pc in enumerate(parts_sub['bodies']):
                pc.set_facecolor(plot_colors[l % len(plot_colors)])
                pc.set_edgecolor('black')
                pc.set_alpha(1)
            sub_ax.set_xticks(positions)
            sub_ax.set_xticklabels(dataNames)
            if i > 1:
                sub_ax.axhline(0, color='red', linestyle='--', linewidth=2)
            sub_ax.set_title(dataTitles[i])
            sub_ax.set_ylabel("Values")
            sub_ax.set_xlim(positions[0] - pad, positions[-1] + pad)
            sub_ax.set_ylim(0, ylim_max[i])
            from matplotlib.ticker import MultipleLocator, FormatStrFormatter
            sub_ax.yaxis.set_major_locator(MultipleLocator(10))        # ticks every 10
            sub_ax.yaxis.set_major_formatter(FormatStrFormatter('%d')) # force integer formatting

            filename = os.path.join(save_dir, f"violin_{region}_{bias_or_prec}_{dataTitles[i].replace(' ', '_')}.pdf")
            subfig.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close(subfig)

    plt.tight_layout()
    plt.show()

def plot_violin_grid(regions, dataNames, save_fig=False, save_dir="./", spacing=0.35, ylim_max_angle1=20, ylim_max_angle2=90):
    import os, numpy as np, matplotlib.pyplot as plt

    plot_colors = ["#0050FF", "#FFD200","#FF00A2", "#00C400", "#FF6A00", "#000000", "#00FFFF", "#FFC0CB", "#808080"]
    col_titles = ["Bias angle 1", "Precision angle 1", "Bias angle 2", "Precision angle 2"]

    n = len(dataNames)
    positions = 1.0 + np.arange(n) * spacing
    widths = spacing * 0.9

    fig, axes = plt.subplots(4, 4, figsize=(12, 10))
    for row_idx, (region, bias_a1, bias_a2, prec_a1, prec_a2) in enumerate(regions):
        row_data = [bias_a1, prec_a1, bias_a2, prec_a2]
        for col_idx in range(4):
            ax = axes[row_idx, col_idx]
            parts = ax.violinplot(
                row_data[col_idx],
                positions=positions,
                widths=widths,
                showmeans=False,
                showmedians=True,
                showextrema=False
            )
            for l, pc in enumerate(parts['bodies']):
                pc.set_facecolor(plot_colors[l % len(plot_colors)])
                pc.set_edgecolor('black')
                pc.set_alpha(1)

            ax.set_xticks(positions)
            if row_idx == len(regions) - 1:
                ax.set_xticklabels(dataNames, rotation=0)
            else:
                ax.set_xticklabels([])

            if col_idx < 2:
                ax.set_ylim(0, ylim_max_angle1)
            else:
                ax.set_ylim(0, ylim_max_angle2)

            pad = spacing * 0.7
            ax.set_xlim(positions[0] - pad, positions[-1] + pad)

            if row_idx == 0:
                ax.set_title(col_titles[col_idx])
            if col_idx == 0:
                ax.set_ylabel(region)

    plt.tight_layout()
    if save_fig:
        os.makedirs(save_dir, exist_ok=True)
        filename = os.path.join(save_dir, "violin_combined_4x4.pdf")
        fig.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

def plot_boxplots(data, dataNames, dataTitles, save_figs=False, save_dir="./", bias_or_prec="bias", region=''):
    num_cols = len(data)
    fig, axes = plt.subplots(1, num_cols, figsize=(12, 5))
    if num_cols > 1:
        axes = axes.flatten()
    if num_cols == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        # ax.boxplot(data[i], tick_labels=["PI"])
        ax.boxplot(data[i], tick_labels=dataNames)
        if i > 1:
            hline = 0
            ax.axhline(hline, color='red', linestyle='--', linewidth=2, label=f"GT value")
        ax.set_title(dataTitles[i])
        ax.set_ylabel("Values")

        # --- save each subplot (column) separately as a PDF ---
        if save_figs:
            os.makedirs(save_dir, exist_ok=True)
            subfig = plt.figure(figsize=(5, 5))
            sub_ax = subfig.add_subplot(111)

            # re-plot same data on its own figure for saving
            sub_ax.boxplot(data[i], tick_labels=dataNames)
            if i > 1:
                sub_ax.axhline(0, color='red', linestyle='--', linewidth=2, label=f"GT value")
            sub_ax.set_title(dataTitles[i])
            sub_ax.set_ylabel("Values")

            filename = os.path.join(save_dir, f"box_col_{i+1}_{region}_{bias_or_prec}_{dataTitles[i].replace(' ', '_')}.pdf")
            subfig.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close(subfig)
            print(f"Saved: {filename}")


    plt.tight_layout()
    plt.show()

def print_stats(name, dataNames, data):
    
    for l in range(len(data)):
        print(f'1st quartile, Median, 3rd quartile of angle {l+1} in {name}:')
        for i, method in enumerate(dataNames):
        # for i, method in enumerate(['PI']):
            quartile_1 = np.percentile(data[l][i], 25)
            quartile_3 = np.percentile(data[l][i], 75)
            median = np.median(data[l][i])
            mean = np.mean(data[l][i])
            std = np.std(data[l][i])
            precentile_90 = np.percentile(data[l][i], 90)
            print(f"{method}:\t\t 1st quartile = {quartile_1:.0f}, \t Median = {median:.0f},\t 3rd quartile = {quartile_3:.0f}")
        print('\n\n')

def main():

    bs_data_path = (
    Path(__file__).resolve()
    .parents[2]
    / 'data'
    / 'Paper_request_data'
    / 'bootstrap_data'
)

    file_path = (
                bs_data_path
                / 'bootstraps_1000_slice_all_PF'
                )
    
    f = h5py.File((file_path / 'bootstrap_analysis_PI.h5'), 'r')
    print(f.keys())
    vec1_comb_PI = f['vec1_comb'][:]
    org_vec1 = f['org_vec_1'][:]
    vec2_comb_PI = f['vec2_comb'][:]
    org_vec2 = f['org_vec_2'][:]
    f.close()
    kappa_1_PI, bias_angle_1_PI, angle_95_1_PI = compute_analysis_metrics(vec1_comb_PI, org_vec1)
    kappa_2_PI, bias_angle_2_PI, angle_95_2_PI = compute_analysis_metrics(vec2_comb_PI, org_vec2)

    f = h5py.File((file_path / 'bootstrap_analysis_PI+LPCA.h5'), 'r')
    vec1_comb_MPPCA = f['vec1_comb'][:]
    vec2_comb_MPPCA = f['vec2_comb'][:]
    f.close()
    kappa_1_MPPCA, bias_angle_1_MPPCA, angle_95_1_MPPCA = compute_analysis_metrics(vec1_comb_MPPCA, org_vec1)
    kappa_2_MPPCA, bias_angle_2_MPPCA, angle_95_2_MPPCA = compute_analysis_metrics(vec2_comb_MPPCA, org_vec2)

    f = h5py.File((file_path / 'bootstrap_analysis_LLR.h5'), 'r')
    vec1_comb_LLR = f['vec1_comb'][:]
    vec2_comb_LLR = f['vec2_comb'][:]
    f.close()
    kappa_1_LLR, bias_angle_1_LLR, angle_95_1_LLR = compute_analysis_metrics(vec1_comb_LLR, org_vec1)
    kappa_2_LLR, bias_angle_2_LLR, angle_95_2_LLR = compute_analysis_metrics(vec2_comb_LLR, org_vec2)

    f = h5py.File((file_path / 'bootstrap_analysis_Proposed_DT.h5'), 'r')
    vec1_comb_DTI = f['vec1_comb'][:]
    vec2_comb_DTI = f['vec2_comb'][:]
    f.close()
    kappa_1_DTI, bias_angle_1_DTI, angle_95_1_DTI = compute_analysis_metrics(vec1_comb_DTI, org_vec1)
    kappa_2_DTI, bias_angle_2_DTI, angle_95_2_DTI = compute_analysis_metrics(vec2_comb_DTI, org_vec2)

    f = h5py.File((file_path / 'bootstrap_analysis_Proposed_BAS.h5'), 'r')
    vec1_comb_BAS = f['vec1_comb'][:]
    vec2_comb_BAS = f['vec2_comb'][:]
    f.close()
    kappa_1_BAS, bias_angle_1_BAS, angle_95_1_BAS = compute_analysis_metrics(vec1_comb_BAS, org_vec1)
    kappa_2_BAS, bias_angle_2_BAS, angle_95_2_BAS = compute_analysis_metrics(vec2_comb_BAS, org_vec2)


    index_f = nib.load(bs_data_path / 'fixel_masks' / 'fixel_masks_all_new' / 'index.nii')
    index_data = index_f.get_fdata()
    mask_1_plus_fiber_all = (index_data[:,:,0:26,0] > 0).astype(bool)
    mask_2_fiber_all = (index_data[:,:,0:26,0] > 1).astype(bool)
    mask_2_fiber_all = mask_2_fiber_all*mask_1_plus_fiber_all

    mask_dir = bs_data_path / 'fixel_masks' / 'regions'

    mask_CC_f = nib.load(mask_dir / 'CorpusCallosum.nii')
    CC_all = mask_CC_f.get_fdata().astype(float)
    CC_all = CC_all[:,:,0:26]  # Corpus Callosum
    CC_all[CC_all==0] = np.nan

    crossing_section_f = nib.load(mask_dir / 'CrossingSection.nii')
    CS_all = crossing_section_f.get_fdata().astype(float)                         #Crossing section front
    CS_all = CS_all[:,:,0:26]
    CS_all[CS_all==0] = np.nan
    internal_capsule_f = nib.load(mask_dir / 'CorticoSpinal.nii')
    IC_all = internal_capsule_f.get_fdata().astype(float)                              #IC = Internal Capsule
    IC_all = IC_all[:,:,0:26]
    IC_all[IC_all==0] = np.nan

    mask_1_plus_fiber_float_all = mask_1_plus_fiber_all.copy().astype(float)
    mask_2_fiber_float_all = mask_2_fiber_all.copy().astype(float)

    mask_1_plus_fiber_float_all[~mask_1_plus_fiber_all] = np.nan
    mask_2_fiber_float_all[~mask_2_fiber_all] = np.nan


    
    save_dir = Path(__file__).resolve().parents[0]
    
    # All white matter voxels

    # bias
    bias_data_1 = [bias_angle_1_PI, bias_angle_1_MPPCA, bias_angle_1_LLR , bias_angle_1_DTI, bias_angle_1_BAS]
    bias_data_2 = [bias_angle_2_PI , bias_angle_2_MPPCA,bias_angle_2_LLR , bias_angle_2_DTI , bias_angle_2_BAS]
    angles1_all, angles2_all = run_statistics_on_angles_masked_data(bias_data_1, bias_data_2, mask_1_plus_fiber_float_all, mask2=mask_2_fiber_float_all)
    data = [angles1_all, angles2_all]
    dataNames = ['PI', 'PI +\nLPCA', 'LLR', 'Proposed\nDT', 'Proposed\nBAS']
    dataTitles = ["Angle 1", "Angle 2"]
    plot_violins(data, dataNames, dataTitles,save_dir=save_dir,save_fig=True, region='AllWhite', ylim_max=[20,90])
    # plot_boxplots(data, dataNames, dataTitles, save_figs=True)
    print_stats('All data mask', dataNames, data)

    # precision
    prec_data_1 = [angle_95_1_PI, angle_95_1_MPPCA, angle_95_1_LLR , angle_95_1_DTI, angle_95_1_BAS]
    prec_data_2 = [angle_95_2_PI, angle_95_2_MPPCA, angle_95_2_LLR , angle_95_2_DTI, angle_95_2_BAS]
    angles1_all, angles2_all = run_statistics_on_angles_masked_data(prec_data_1, prec_data_2, mask_1_plus_fiber_float_all, mask2=mask_2_fiber_float_all)
    data = [angles1_all, angles2_all]
    dataTitles = ["Angle 1", "Angle 2"]
    plot_violins(data, dataNames, dataTitles,save_dir=save_dir,save_fig=True, bias_or_prec='prec', region='AllWhite', ylim_max=[20,90])
    # plot_boxplots(data, dataNames, dataTitles, save_figs=True)
    print_stats('All data mask', dataNames, data)



    # Corpus Callosum

    # bias
    angles1_CC_all, angles2_CC_all = run_statistics_on_angles_masked_data(bias_data_1, bias_data_2, CC_all, mask2=mask_2_fiber_float_all*CC_all)
    data = [angles1_CC_all, angles2_CC_all]
    dataTitles = ["Angle 1", "Angle 2"]
    plot_violins(data, dataNames, dataTitles, save_dir=save_dir,save_fig=True, region='CC', ylim_max=[20,90])
    # plot_boxplots(data, dataNames, dataTitles, save_figs=True)
    print_stats('CC mask', dataNames, data)

    # precision
    prec1_CC_all, prec2_CC_all = run_statistics_on_angles_masked_data(prec_data_1, prec_data_2, CC_all,mask2=mask_2_fiber_float_all*CC_all)
    data = [prec1_CC_all, prec2_CC_all]
    dataTitles = ["Angle 1", "Angle 2"]
    plot_violins(data, dataNames, dataTitles, save_dir=save_dir,save_fig=True, ylim_max=[20,90], bias_or_prec='prec', region='CC')
    # plot_boxplots(data, dataNames, dataTitles, save_figs=True)
    print_stats('CC mask', dataNames, data)


    # Cortico Spinal Tract

    # bias
    angles1_IC_all, angles2_IC_all = run_statistics_on_angles_masked_data(bias_data_1, bias_data_2, IC_all, mask2=mask_2_fiber_float_all*IC_all)
    data = [angles1_IC_all, angles2_IC_all]
    dataTitles = ["Angle 1", "Angle 2"]
    plot_violins(data, dataNames, dataTitles, save_dir=save_dir,save_fig=True, ylim_max=[20,90], region='CST')
    # plot_boxplots(data, dataNames, dataTitles, save_figs=True)
    print_stats('IC mask', dataNames, data)

    # precision
    prec1_IC_all, prec2_IC_all = run_statistics_on_angles_masked_data(prec_data_1, prec_data_2, IC_all, mask2=mask_2_fiber_float_all*IC_all)
    data = [prec1_IC_all, prec2_IC_all]
    dataTitles = ["Angle 1", "Angle 2"]
    plot_violins(data, dataNames, dataTitles, save_dir=save_dir,save_fig=True, ylim_max=[20,90], bias_or_prec='prec', region='CST')
    # plot_boxplots(data, dataNames, dataTitles, save_figs=True)
    print_stats('IC mask', dataNames, data)


    # Crossing Section

    # bias
    angles1_CS_all, angles2_CS_all = run_statistics_on_angles_masked_data(bias_data_1, bias_data_2, CS_all, mask2=mask_2_fiber_float_all*CS_all)
    data = [angles1_CS_all, angles2_CS_all]
    dataTitles = ["Angle 1", "Angle 2"]
    plot_violins(data, dataNames, dataTitles, save_dir=save_dir,save_fig=True, ylim_max=[20,90], region='CS')
    # plot_boxplots(data, dataNames, dataTitles, save_figs=True)
    print_stats('CS mask', dataNames, data)

    # precision
    prec1_CS_all, prec2_CS_all = run_statistics_on_angles_masked_data(prec_data_1, prec_data_2, CS_all, mask2=mask_2_fiber_float_all*CS_all)
    data = [prec1_CS_all, prec2_CS_all]
    dataTitles = ["Angle 1", "Angle 2"]
    plot_violins(data, dataNames, dataTitles, save_dir=save_dir,save_fig=True, ylim_max=[20,90], region='CS', bias_or_prec='prec')
    # plot_boxplots(data, dataNames, dataTitles, save_figs=True)
    print_stats('CS mask', dataNames, data)

    regions = [
        ("AllWhite", angles1_all, angles2_all, angles1_all, angles2_all),
        ("CC", angles1_CC_all, angles2_CC_all, prec1_CC_all, prec2_CC_all),
        ("CST", angles1_IC_all, angles2_IC_all, prec1_IC_all, prec2_IC_all),
        ("CS", angles1_CS_all, angles2_CS_all, prec1_CS_all, prec2_CS_all),
    ]
    plot_violin_grid(regions, dataNames, save_fig=True, save_dir=save_dir)

if __name__ == "__main__":
    main()
