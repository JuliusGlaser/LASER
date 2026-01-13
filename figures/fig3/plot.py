import os
import h5py
import numpy as np
import sigpy as sp
import matplotlib.pyplot as plt
import ipywidgets as widgets

f= h5py.File( r'../../data/Paper_request_data/LR/GT_avg_PF.h5', 'r')
GT = f['DWI'][:]
f.close()

f= h5py.File(r'../../data/Paper_request_data/LR/recos/PI/PI_comb_us3_PF_1.h5', 'r')
PI = f['DWI'][:]
f.close()    

f= h5py.File(r'../../data/Paper_request_data/LR/recos/PI/denoised_results/Data_denoised.h5', 'r')
print(f.keys())
BAS_SVD = f['BAS_SVD_ss_11'][:].T
BAS_AE = f['BAS_AE'][:].T
DT_AE = f['DTI_AE'][:].T
f.close()


q_1000 = 2
q_2000 = 40
q_3000 = 57
q_values = [q_1000, q_2000, q_3000]
n_slice=14

x_slice = slice(10,105)
y_slice = slice(15,95)

rows = []
for q in q_values:
    print(q)
    key = 'GT'
    img = np.rot90(abs(GT[q, n_slice, x_slice, y_slice]),2)
    vmax = np.percentile(abs(GT[q, n_slice, ...]), 99)
    vmin = 0

    os.makedirs('q_' + str(q), exist_ok=True)
    fig, ax = plt.subplots(figsize=(10,10))
    ax.imshow(img, cmap='gray', vmin=vmin, vmax=vmax)
    ax.axis('off')
    fig.savefig('q_' + str(q) + os.sep + key + '.pdf', bbox_inches='tight', dpi=500)
    plt.close(fig)

    key = 'PI'
    img = np.rot90(abs(PI[q, n_slice, x_slice, y_slice]),2)
    fig, ax = plt.subplots(figsize=(10,10))
    ax.imshow(img, cmap='gray', vmin=vmin, vmax=vmax)
    ax.axis('off')
    fig.savefig('q_' + str(q) + os.sep + key + '.pdf', bbox_inches='tight', dpi=500)
    plt.close(fig)

    data = BAS_SVD
    key = 'BAS_SVD'
    img = np.rot90(abs(GT[q, n_slice, 10:105,15:95]),2) - np.rot90(abs(data[q, n_slice, 10:105,15:95]),2)
    fig, ax = plt.subplots(figsize=(10,10))
    ax.imshow(img, cmap='gray', vmin=-vmax, vmax=vmax)
    ax.axis('off')
    fig.savefig('q_' + str(q) + os.sep + key + '.pdf', bbox_inches='tight', pad_inches=0, dpi=500)
    plt.close(fig)

    rmse = np.sqrt(np.mean((abs(GT[q, n_slice, 10:105,15:95]) - abs(data[q, n_slice, 10:105,15:95])) ** 2))
    print('>> us_ref - ref')
    print('RMSE = ', rmse)

    data = DT_AE
    key = 'DT_AE'
    img = np.rot90(abs(GT[q, n_slice, 10:105,15:95]),2) - np.rot90(abs(data[q, n_slice, 10:105,15:95]),2)
    fig, ax = plt.subplots(figsize=(10,10))
    ax.imshow(img, cmap='gray', vmin=-vmax, vmax=vmax)
    ax.axis('off')
    fig.savefig('q_' + str(q) + os.sep + key + '.pdf', bbox_inches='tight', pad_inches=0, dpi=500)
    plt.close(fig)

    rmse = np.sqrt(np.mean((abs(GT[q, n_slice, 10:105,15:95]) - abs(data[q, n_slice, 10:105,15:95])) ** 2))
    print('>> us_ref - ref')
    print('RMSE = ', rmse)

    data = BAS_AE
    key = 'BAS_AE'
    img = np.rot90(abs(GT[q, n_slice, 10:105,15:95]),2) - np.rot90(abs(data[q, n_slice, 10:105,15:95]),2)
    fig, ax = plt.subplots(figsize=(10,10))
    ax.imshow(img, cmap='gray', vmin=-vmax, vmax=vmax)
    ax.axis('off')
    fig.savefig('q_' + str(q) + os.sep + key + '.pdf', bbox_inches='tight', pad_inches=0, dpi=500)
    plt.close(fig)

    rmse = np.sqrt(np.mean((abs(GT[q, n_slice, 10:105,15:95]) - abs(data[q, n_slice, 10:105,15:95])) ** 2))
    print('>> us_ref - ref')
    print('RMSE = ', rmse)

    rows.append(
        [
            np.rot90(abs(GT[q, n_slice, x_slice, y_slice]), 2),
            np.rot90(abs(PI[q, n_slice, x_slice, y_slice]), 2),
            np.rot90(abs(GT[q, n_slice, x_slice, y_slice]), 2)
            - np.rot90(abs(BAS_SVD[q, n_slice, x_slice, y_slice]), 2),
            np.rot90(abs(GT[q, n_slice, x_slice, y_slice]), 2)
            - np.rot90(abs(DT_AE[q, n_slice, x_slice, y_slice]), 2),
            np.rot90(abs(GT[q, n_slice, x_slice, y_slice]), 2)
            - np.rot90(abs(BAS_AE[q, n_slice, x_slice, y_slice]), 2),
        ]
    )

fig, axes = plt.subplots(3, 5, figsize=(15, 9))
for row_idx, q in enumerate(q_values):
    vmax = np.percentile(abs(GT[q, n_slice, ...]), 99)
    for col_idx in range(5):
        ax = axes[row_idx, col_idx]
        if col_idx < 2:
            ax.imshow(rows[row_idx][col_idx], cmap='gray', vmin=0, vmax=vmax)
        else:
            ax.imshow(rows[row_idx][col_idx], cmap='gray', vmin=-vmax, vmax=vmax)
        ax.axis('off')
fig.savefig('q_all_combined_3x5.pdf', bbox_inches='tight', dpi=500)
plt.close(fig)
