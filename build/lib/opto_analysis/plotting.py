import numpy as np
import seaborn as sns
import os
import matplotlib.pyplot as plt
from sklearn import preprocessing
from opto_analysis.place_cell_opto import LoadData
import pandas as pd
import scipy
from scipy import signal
from sklearn.metrics.pairwise import cosine_similarity
# from data_analysis.settings import defaults



class Subplots:
    def __init__(self, rows, cols, save_path=None):
        self.rows = rows
        self.cols = cols
        self.path = save_path
        if save_path:
            if not os.path.exists(save_path):
                print(f'Creating folder in path: {save_path}')
                os.mkdir(save_path)

    def _new_subplot(self):
        self.fig, self.axes = plt.subplots(self.rows, self.cols, sharey=True, sharex=True)
        self.current_row = 0
        self.current_col = 0

    def _next_subplot(self, title):
        if self.current_row == self.rows - 1 and self.current_col == self.cols - 1:
            self.show_save(title)
            self._new_subplot()
        elif self.current_col == self.cols - 1:
            self.current_col = 0
            self.current_row += 1
        else:
            self.current_col += 1
        return self.axes[self.current_row, self.current_col]

    def heatmap(self, data, title, red_condition):
        self._new_subplot()
        num_slices = data.shape[2]
        for i in range(num_slices):
            ax = self.axes[self.current_row, self.current_col]
            sns.heatmap(ax=ax, data=data[:, :, i], cbar=False, xticklabels=False, vmin=0, vmax=1)
            ax.set_title(title[i], color='red', fontweight='bold') if red_condition[i] else ax.set_title(title[i])
            self._next_subplot(title[i])
        self.show_save(title[i])

    def tuning(self, data, title):
        self._new_subplot()
        num_slices = data.shape[1]
        for i in range(num_slices):
            ax = self.axes[self.current_row, self.current_col]
            ax.plot(data[:, i])
            ax.set_title(title[i])
            ax.set(xlabel='location')
            ax.label_outer()
            self._next_subplot(title[i])
        self.show_save(title[i])

    def tuning_comparison(self, on, off, title):
        assert on.shape == off.shape, "data shape inconsistent"

        self._new_subplot()
        num_slices = on.shape[1]
        for i in range(num_slices):
            ax = self.axes[self.current_row, self.current_col]
            ax.plot(on[:, i], label='on')
            ax.plot(off[:, i], label='off')
            ax.set_title(title[i])
            ax.legend()
            self._next_subplot(title[i])
        self.show_save(title[i])

    def show_save(self, save_title):
        plt.tight_layout()
        plt.show()
        if self.path:
            self.fig.savefig(os.path.join(self.path, save_title))


def save_folder_handling(save_folder, path):
    if save_folder:
        folder_path = os.path.join(path, save_folder)
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)
        return folder_path
    return None


def cell_heatmap(data: LoadData, laps: np.ndarray, cells: np.ndarray, save_folder: str = None, red_cells: [int] = None, nrows: int=3, ncols: int=3):
    path = save_folder_handling(save_folder, data.path)
    title = [f'cell {cell}' for cell in cells]
    if red_cells:
        red_condition = [cell in red_cells for cell in cells]
    else:
        red_condition = [0] * len(cells)

    plts = Subplots(nrows, ncols, path)
    plts.heatmap(data.mean_activity[laps[0]:laps[-1]+1, :, cells], title, red_condition)


def lap_heatmap(data: LoadData, laps: np.ndarray, df: pd.DataFrame, save_folder: str = None, red_laps: [int] = None, nrows: int=3, ncols: int=3):
    path = save_folder_handling(save_folder, data.path)
    cells = df.sort_values(by=['COM'])['cell'].unique()
    title = [f'lap {lap}' for lap in laps]
    if red_laps:
        red_condition = [lap in red_laps for lap in laps]
    else:
        red_condition = [0] * len(laps)

    plts = Subplots(nrows, ncols, path)
    plts.heatmap(np.moveaxis(data.mean_activity[laps[0]: laps[-1]+1, :, cells],[0, 2], [-1, 0]), title, red_condition)

def cell_tuning(data: LoadData, laps: np.ndarray, cells: np.ndarray, save_folder: str = None, nrows: int=3, ncols: int=3):

    path = save_folder_handling(save_folder, data.path)
    tuning = np.nanmean(data.mean_activity[laps, :, cells[:, np.newaxis]], axis=1)
    title = [f'cell {c}' for c in cells]

    plts = Subplots(nrows, ncols, path)
    plts.tuning(tuning.transpose(), title)

    return tuning

def cell_opto_tuning(data: LoadData, on_laps: np.ndarray, off_laps: np.ndarray, cells: np.ndarray, save_folder: str = None, nrows: int=3, ncols: int=3):
    path = save_folder_handling(save_folder, data.path)
    on_tuning = np.nanmean(data.mean_activity[on_laps, :, cells[:, np.newaxis]], axis=1)
    off_tuning = np.nanmean(data.mean_activity[off_laps, :, cells[:, np.newaxis]], axis=1)
    title = [f'cell {c}' for c in cells]

    plts = Subplots(nrows, ncols, path)
    plts.tuning_comparison(on_tuning.transpose(), off_tuning.transpose(), title)
    return on_tuning, off_tuning


def emerge_cumcount(df: pd.DataFrame, emerge_col_name='emerge lap', groupby_var = ['mouse','day', 'opto'], row_var = 'day', max_lap=30):
    emerge = df[df[emerge_col_name] < max_lap]

    groupby_list = groupby_var + [emerge_col_name]

    emerge_grouped_by_mouse = emerge.groupby(groupby_list)['cell'].count().reset_index()

    all_lap = pd.DataFrame(np.arange(max_lap), columns=[emerge_col_name])
    all_lap_count = emerge_grouped_by_mouse.groupby(groupby_var).apply(
        lambda x: x.merge(all_lap, on=emerge_col_name, how='right')).reset_index(drop=True)
    all_lap_count['cell'] = all_lap_count['cell'].fillna(0)
    group_values = list(range(len(all_lap_count) // max_lap))
    all_lap_count['group'] = np.repeat(group_values, max_lap)
    all_lap_count = all_lap_count.groupby('group').apply(lambda x: x.fillna(method='bfill').fillna(method='ffill'))

    all_lap_count['cumcount'] = all_lap_count.groupby(groupby_var)['cell'].cumsum().astype('int')
    all_lap_count['total cells'] = all_lap_count.groupby(groupby_var)['cell'].transform('sum').astype('int')
    all_lap_count['cumcount percent'] = all_lap_count['cumcount'] / all_lap_count['total cells']


    #sns.relplot(data=all_lap_count, x=emerge_col_name, y="cumcount percent", row=row_var, hue="opto", kind="line")
    #plt.show()
    #sns.relplot(data=all_lap_count, x=emerge_col_name, y="cumcount percent", row=row_var, col="mouse", hue="opto", kind="line")

    return all_lap_count.reset_index(drop=True)


def population_vector_corr_mean(data: LoadData, laps, cells):

    cell_data = data.mean_activity[laps[0]:laps[-1]+1, :, cells]
    mean_rate = np.nanmean(cell_data, axis=0)
    corr = [scipy.stats.pearsonr(mean_rate.flatten(), cell_data[l, :, :].flatten())[0] for l in range(len(laps))]
    plt.plot(corr)
    plt.ylabel('pearson r')
    plt.xlabel('lap')

    return corr


def lap_by_lap_distance(data: LoadData, laps, cells):
    # check with corr reshape

    data = data.mean_activity[laps[0]:laps[-1]+1, :, cells]
    nlaps = len(laps)
    r_coef = np.zeros((nlaps, nlaps, len(cells)))
    cos_sim = np.zeros((nlaps, nlaps, len(cells)))

    for c in range(len(cells)):
        r_coef[:, :, c] = np.corrcoef(data[laps, :, c])
        cos_sim[:,:, c] = cosine_similarity(data[laps, :, c])

    fig, axs = plt.subplots(1, 2, sharey=True)
    axs = axs.ravel()

    colormap = sns.color_palette('Greys')
    sns.heatmap(ax=axs[0], data=np.nanmean(r_coef, axis=2), cmap = colormap, vmin=0, vmax=1, cbar=False)
    axs[0].set_title(f'lap by lap correlation')
    axs[0].set_aspect('equal')

    sns.heatmap(ax=axs[1], data=np.nanmean(cos_sim, axis=2), cmap = colormap, vmin=0, vmax=1, cbar=False)
    axs[1].set_title(f'lap by lap cosine similarity')
    axs[1].set_aspect('equal')

    plt.tight_layout()

    return r_coef, cos_sim


def backwards_shifting(data: LoadData, com_by_lap, laps, cells, min_corr=0.25, vmax_bin=5):

    nlaps = len(laps)
    npfs = len(cells)
    m = data.mean_activity[laps[0]: laps[-1]+1, :, cells]

    assert com_by_lap.shape == (npfs, nlaps), f"{com_by_lap.shape} does not match nlaps: {len(laps)} or nPFs: {len(cells)}"

    lag_mat = np.zeros((nlaps, nlaps, npfs))
    lag_mat[:] = np.nan
    emerge_lag = np.copy(lag_mat)
    xcorr_mat = np.zeros((nlaps, nlaps, npfs))
    xcorr_mat[:] = np.nan
    width = m.shape[1]

    # loop over individual PF (eg. one cell could have multiple PFs)
    for n in range(npfs):
        firing_laps = np.where(com_by_lap[n, :] > 0)[0]

        # calculate correlation between laps only once
        for l0 in range(len(firing_laps) - 1):
            for l1 in range(l0 + 1, len(firing_laps)):
                corr = signal.correlate(m[firing_laps[l0], :, n], m[firing_laps[l1], :, n])
                xcorr_mat[firing_laps[l0], firing_laps[l1], n] = np.max(corr)
                lag_mat[firing_laps[l0], firing_laps[l1], n] = np.argmax(corr) - width
                emerge_lag[firing_laps[l0] - firing_laps[0], firing_laps[l1] - firing_laps[0], n] = np.argmax(
                    corr) - width

    # remove invalid laps (xcorr < min_corr)
    x, y, z = np.where(xcorr_mat < min_corr)
    lag_mat[x, y, z] = np.nan
    emerge_lag[x, y, z] = np.nan

    # plot lag (compared to env switch) context-based
    fig, axs = plt.subplots(1, 2, sharey=True, sharex=True, figsize = (10, 10))
    axs = axs.ravel()

    sns.heatmap(ax=axs[0], data = np.nanmean(lag_mat, axis=2), cmap="PiYG", center=0, cbar=False, vmin=-vmax_bin, vmax=vmax_bin)
    axs[0].set_xlabel('lap')
    axs[0].set_xlabel('lap')
    axs[0].set_title(f'{data.name} {laps[0]} to {laps[-1]} xcorr')
    axs[0].set_aspect('equal')

    # plot lag (compared to first emerged) PF-based
    sns.heatmap(ax=axs[1], data=np.nanmean(emerge_lag, axis=2), cmap="PiYG", center=0, cbar=False, vmin=-vmax_bin, vmax=vmax_bin)
    axs[1].set_xlabel('lap after emerged')
    axs[1].set_xlabel('lap after emerged')
    axs[1].set_title(f'{data.name} {laps[0]} to {laps[-1]} xcorr emerged')
    axs[1].set_aspect('equal')

    return lag_mat, xcorr_mat, emerge_lag


def clean_axes(axes):
    for ax in np.asarray(axes).flat:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.margins(x=0.01, y=0.01)


def distplot(*args, **kwargs):
    for k, v in defaults.distplot_kwargs.items():
        if k not in kwargs:
            kwargs[k] = v
        elif isinstance(v, dict):
            kwargs[k] = v | kwargs[k]
    return sns.distplot(*args, **kwargs)
