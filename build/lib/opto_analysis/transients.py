import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from ast import literal_eval
from scipy.ndimage import uniform_filter1d
import seaborn as sns
import os
import pickle
import scipy.stats as stats
import itertools
from matplotlib.ticker import PercentFormatter
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.metrics.pairwise import cosine_similarity
from os import listdir
from matplotlib.figure import figaspect
from itertools import combinations_with_replacement
from math import comb
from matplotlib.patches import Rectangle
from collections import Counter
from scipy.ndimage.filters import uniform_filter1d
import bisect
import logging
from scipy.signal import find_peaks
from opto_analysis.place_cell_opto import LoadAxon
# from place_cell_opto import load_py_var_mat


def rise_to_peak(data: LoadAxon, prominence_thresh=0.1, amplitude=0.12, interval=10, width_min=5, width_max=100):

    print(f'{data.axon_df.isna().any(axis=1).sum()} total frames missing in {data.name} {data.day}')

    mat = data.axon_df.iloc[:, :-5].to_numpy().transpose()
    result = np.zeros_like(mat)
    for i, row in enumerate(mat):
        peaks, properties = find_peaks(row, prominence=prominence_thresh, height=amplitude, distance=interval,
                                       width=[width_min, width_max])

        for n in range(len(peaks)):
            left_idx = properties['left_bases'][n]
            result[i, left_idx:peaks[n] + 1] = row[left_idx: peaks[n] + 1]

    t_mean = np.nanmean(result, axis=0)
    t_summary = pd.DataFrame(t_mean, columns=['mean amplitude'])
    t_summary['mouse'] = data.name
    t_summary['day'] = data.day

    vmax = np.median(np.max(result, axis=1))
    env_dict = dict(zip(data.axon_df['env'].unique(), data.axon_df.reset_index().groupby('env')['index'].idxmax().tolist()))
    print(env_dict)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True, layout='constrained')

    # First subplot (image)
    im = ax1.imshow(result, aspect='auto', cmap='gray_r', vmin=0.001, vmax=vmax)
    ax1.set_title(f'{data.name} {data.day} raster', pad=20)
    fig.colorbar(im, ax=ax1)

    # Second subplot (sum plot)
    ax2.plot(t_mean)
    ax2.set_title('mean amplitude')
    prev_ind = 0

    for condition, start_idx in env_dict.items():
        ax1.axvline(x=start_idx, color='red', linestyle='--', linewidth=2)
        ax1.text(start_idx, -1, condition, color='red', verticalalignment='bottom', horizontalalignment='right',
                 fontsize=12)
        ax2.axvline(x=start_idx, color='red', linestyle='--', linewidth=2)
        ax2.text(start_idx, 0, condition, color='red', verticalalignment='bottom', horizontalalignment='right',
                 fontsize=12)
        t_summary.loc[prev_ind: start_idx, 'env'] = condition
        prev_ind = start_idx

    # Save the figure
    plt.savefig(os.path.join(data.path, f'{data.name} {data.day} raster.png'))
    plt.show()

    return result, t_summary


def transient_summary_by_lap(data: LoadAxon):

    r2p, _ = rise_to_peak(data)

    beh_df = data.axon_df[['lap', 'env', 'mouse', 'day']].reset_index()

    data = []
    for row_idx in range(r2p.shape[0]):
        row = r2p[row_idx]

        start_col = None  # Track the start of a non-zero group
        group_sum = 0  # Track the sum of the current non-zero group

        # Iterate over each element in the row
        for col_idx in range(r2p.shape[1]):
            value = row[col_idx]

            if value != 0:  # If current value is non-zero
                if start_col is None:
                    start_col = col_idx  # Mark the start of the non-zero group
                group_sum += value  # Add value to the group sum
            else:
                if start_col is not None:
                    # End of a non-zero group, record it
                    data.append([row_idx, start_col, group_sum])
                    start_col = None  # Reset for the next group
                    group_sum = 0

        # If the row ends with a non-zero group, add it
        if start_col is not None:
            data.append([row_idx, start_col, group_sum])

    # Create a DataFrame from the reduced data
    df = pd.DataFrame(data, columns=['cell', 'frame', 'transient'])
    df = df.merge(beh_df, left_on='frame', right_on='index', how='left')
    df = df.drop(columns='index')
    df = df.sort_values(by=['cell', 'frame'])

    return df
