import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy import nanmean
import scipy.io
from ast import literal_eval
from scipy.ndimage import uniform_filter1d
#from opto_analysis.plotting import cell_heatmap, cell_mean_over_laps
import seaborn as sns
import os.path
import pickle
import scipy.stats as stats
import itertools
from matplotlib.ticker import PercentFormatter
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.metrics.pairwise import cosine_similarity
from scipy.ndimage import uniform_filter1d
from os import listdir
#from opto_analysis.plotting import cell_heatmap
from matplotlib.figure import figaspect
from itertools import combinations_with_replacement
from math import comb
from matplotlib.patches import Rectangle
from collections import Counter
from scipy.ndimage.filters import uniform_filter1d
import bisect
import logging
from scipy.signal import find_peaks
from icasso import Icasso
from opto_analysis.place_cell_opto import load_py_var_mat, init_logger
import ruptures as rpt
import re

track_end = 0.6
track_start = 0.015


# icasso functions: bootstrap_fun & unmixing_fun
def bootstrap_fun(data, generator):
    return data[generator.choice(range(data.shape[0]), size=data.shape[0]), :]


def unmixing_fun(ica):
    return ica.components_

def roc(datA, datB, min_value = None, one_sided: bool = False,  fig: bool=False):
    """ """

    if min_value is not None:
        datA = datA[datA > min_value]
        datB = datB[datB > min_value]

    # Number of observations
    na = len(datA)
    nb = len(datB)
    nobs = na + nb

    # Sort class B
    b = np.sort(datB)

    ca = np.zeros(nb + 1)
    ib = np.zeros(nb + 1)

    # Calculate correct and incorrect assignments
    for i in range(nb):
        ca[i] = np.sum(datA >= b[i])
        ib[i] = nb - i

    # Final elements
    ca[-1] = 0
    ib[-1] = 0

    # Calculate hits and false alarms
    hits = ca / na
    fa = ib / nb

    # Flip arrays
    hits = np.flip(hits)
    fa = np.flip(fa)

    # Calculate AUC using trapezoidal integration
    intg = np.trapz(hits, fa)
    auc = np.abs(intg)

    # Adjust AUC if it's less than 0.5
    if (one_sided) and (auc < 0.5):
        auc = 1 - auc
        hits = (na - ca) / na
        fa = (nb - ib) / nb

    # Optionally plot the ROC curve (if fig is True)
    if fig:
        plt.plot(fa, hits, label=f'AUC = {auc:.2f}')
        plt.xlabel('False Alarm Rate')
        plt.ylabel('Hit Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.show()

    return auc


def find_integers_within_range(arr, range_size=10, count_threshold=5):
    n = len(arr)
    start_idx = 0
    result = []
    for end_idx in range(n):
        while arr[end_idx] - arr[start_idx] >= range_size:
            start_idx += 1
        if end_idx - start_idx + 1 >= count_threshold:
            result.extend(arr[start_idx:end_idx + 1])
    result_array = np.unique(result) if result else np.array([])
    return result_array


def find_pauses(ybinned, velocity, smooth_frames=20, velocity_thresh=0.08, continuous_frames=8, total_frames=10):
    """ return pause indices based on smoothed location and velocity """

    smoothed_y = uniform_filter1d(ybinned, size=smooth_frames)
    smoothed_v = uniform_filter1d(velocity, size=smooth_frames)
    track_ind = np.where((smoothed_y< track_end) & (smoothed_y>track_start))[0]
    track = smoothed_y[track_ind]
    track_v = smoothed_v[track_ind]

    indices = np.where(track_v < velocity_thresh)[0]
    pause_ind = find_integers_within_range(indices, total_frames, continuous_frames)
    print(f'{len(pause_ind)} frames pausing out of {len(track)} frames on the track. {np.round(len(pause_ind)*100/len(track), 2)} %')

    # mask = np.ones(len(track_v), dtype=bool)
    # mask[indices] = False
    # running_ind = np.where(mask)[0]

    return track_ind[pause_ind]

class LoadRaw:
    def __init__(self, mouse, envs, day):
        self.name = mouse
        self.day = day
        self.env = envs
        self.path = os.path.join('D:\\Opto\\Analysis', mouse, day)

        # load relevant files
        file = load_py_var_mat(self.path, 'align_cell_mean.mat', ['Fc3_DF'])
        self.mat = file['Fc3_DF']
        self.constants = {'nframes': self.mat.shape[0], 'ncells': self.mat.shape[1]}

        self.beh = self.beh_params()
        self.logger = init_logger(self.path, self.name, 'assembly')

        # keep only running frames & adjust indices after deleting frames
        # pause_ind = find_pauses(ybinned, velocity)
        # self.update_param_frames(pause_ind)

    def beh_params(self):

        beh_file = load_py_var_mat(self.path, 'reward-all-cond.mat')
        ybinned = beh_file['behavior']['ybinned'][0][0].transpose()
        velocity = beh_file['behavior']['velocity'][0][0].transpose()
        rewards = beh_file['behavior']['reward'][0][0].transpose()
        assert self.mat.shape[0] == len(ybinned), f'{self.name} behavior and imaging of different size'

        # load imaging params
        file = load_py_var_mat(self.path, 'align_cell_mean.mat', ['E', 'switch_frame', 'onFrames_ind', 'offFrames_ind'])
        E = file['E'].transpose()
        switch_frame = file['switch_frame'][0, :].astype('int') -1
        self.params = {'switch_frame': switch_frame}

        df = pd.DataFrame.from_dict(data={'y': ybinned[:, 0], 'velocity': velocity[:, 0], 'rewards': rewards[:, 0], 'lap': E[:, 0]})
        df['env'] = None
        df['opto'] = 'off'
        env_dict = dict(zip(self.env, np.split(np.arange(len(ybinned)), self.params['switch_frame'])))
        for env, frames in env_dict.items():
            df.loc[frames, 'env'] = env
        df['lap'] = df['lap'].replace(0, np.nan)
        df['lap'] = df['lap'].fillna(method='bfill')
        df['lap'] = df['lap'].fillna(value=df['lap'].max() + 1)
        df['lap'] = df['lap'].astype('int')
        df['rewards'] = df['rewards'].apply(lambda x: 1 if x > 7 else 0)

        if 'onFrames_ind' in file:
            opto_on_frame = file['onFrames_ind'][:, 0].astype('int') -1
            opto_off_frame = file['offFrames_ind'][:, 0].astype('int') - 1
            opto_length = opto_off_frame - opto_on_frame
            opto_on_frame = opto_on_frame[opto_length > 100]
            opto_off_frame = opto_off_frame[opto_length > 100]
            self.params.update({'opto_on': opto_on_frame, 'opto_off': opto_off_frame})
            for n in range(len(opto_on_frame)):
                df.loc[opto_on_frame[n]:opto_off_frame[n], 'opto'] = 'on'

        return df


    def update_param_frames(self, delete_frames: np.ndarray):

        self.logger.info(f'{self.name} deleting {delete_frames}')

        mask = np.ones(self.mat.shape[0], dtype=bool)
        mask[delete_frames] = False
        kept_ind = np.where(mask)[0]

        self.mat = self.mat[kept_ind, :]
        switch_frame = self.params['switch_frame']
        self.params['switch_frame'] = [bisect.bisect_left(kept_ind, x) for x in switch_frame]

        if 'opto_on' in self.params:
            opto_on_frame = self.params['opto_on']
            opto_off_frame = self.params['opto_off']
            self.params['opto_on'] = np.array([bisect.bisect_left(kept_ind, x) for x in opto_on_frame])
            self.params['opto_off'] = np.array([bisect.bisect_left(kept_ind, x) for x in opto_off_frame])

        self.constants =  {'nframes': self.mat.shape[0], 'ncells': self.mat.shape[1]}
        self.beh = self.beh.drop(index=delete_frames).reset_index(drop=True)

    def find_bad_frames(self, breakpoints, plot_range = None):

        z_data = stats.zscore(self.mat, axis=0, nan_policy='omit')
        if plot_range is not None:
            z_mean = np.nanmean(z_data[np.arange(plot_range[0], plot_range[1]), :], axis=1)
        else:
            z_mean = np.nanmean(z_data, axis=1)
        model = rpt.BottomUp(model='l2')
        algo_binseg = model.fit(z_mean)
        optimal_change_points = algo_binseg.predict(n_bkps=breakpoints)
        rpt.display(z_mean, optimal_change_points)
        if plot_range is not None:
            optimal_change_points = np.array(optimal_change_points) + plot_range[0]
        plt.show()

        z_deleted = np.delete(z_data, np.arange(optimal_change_points[0], optimal_change_points[1]), axis=0)
        plt.plot(np.mean(z_deleted, axis=1))
        plt.title(f'double check after deleting frames in {self.name}')
        #plt.axvline(x=optimal_change_points[0], color='red', linestyle='--', linewidth=2)
        #plt.axvline(x=optimal_change_points[1], color='red', linestyle='--', linewidth=2)
        if plot_range is not None:
            plt.xlim(plot_range)
        plt.show()

        return optimal_change_points[:-1]


def bin_z_mat(df, groupby_fun='mean', binwindow: int=5):

    nframes, ncells = df.shape
    ind = np.arange(nframes)
    bins = np.arange(0, nframes + binwindow, binwindow)
    binned = pd.cut(ind, bins, include_lowest=True)
    df['binned'] = binned
    if groupby_fun == 'mean':
        df_binned = df.groupby('binned').agg(np.nanmean)

        bin_mat = df_binned.iloc[0:-1, :ncells].to_numpy()

        active_neuron = np.where(np.sum(bin_mat, axis=0) > 0)[0]
        print(f'{len(active_neuron)} active cells out of {ncells} total cells')

        z = stats.zscore(bin_mat[:, active_neuron], axis=0, nan_policy='omit')
        # z = z[~np.isnan(z).any(axis=1), :]  # remove any nans in z_mat
        z[np.isnan(z)] = 0
        return z

    else:
        df_binned = df.groupby('binned').agg(groupby_fun)
        print('CAUTION: return a df instead of the z mat')
        return df_binned.reset_index()

def plot_raster(data: LoadRaw, z=None, z_thresh=0, z_max=2, binwindow=5):

    if z is None:
        z = bin_z_mat(pd.DataFrame(data.mat), binwindow=binwindow)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True, layout='constrained')
    im = ax1.imshow(z.transpose(), aspect='auto', cmap='gray_r', vmin=z_thresh, vmax=z_max)
    ax1.set_title(f'{data.name} raster', pad=20)
    fig.colorbar(im, ax=ax1)

    # Second subplot (sum plot)
    ax2.plot(np.nanmean(z.transpose(), axis=0))
    ax2.set_title('mean z score')
    prev_ind = 0

    for switch_frame in data.params['switch_frame']:
        ax1.axvline(x=int(switch_frame/binwindow), color='black', linestyle='--', linewidth=1.5)
        ax2.axvline(x=int(switch_frame/binwindow), color='black', linestyle='--', linewidth=1.5)

    if 'opto_on' in data.params:
        for n in range(len(data.params['opto_on'])):
            ax1.hlines(y=data.constants['ncells']+10, xmin=int(data.params['opto_on'][n]/binwindow),
                       xmax=int(data.params['opto_off'][n]/binwindow), color='red', linewidth=2)

    # Save the figure
    plt.savefig(os.path.join(data.path, f'{data.name} raster z score.png'))
    plt.show()

def n_assembly(z):
    """ Step 2: determine # of cell assemblies
    """

    cov_mat = np.cov(z.transpose())
    evals, _ = np.linalg.eigh(cov_mat)
    evals = np.sort(evals)[::-1]

    ncells = z.shape[1]
    nbins = z.shape[0]

    # find the boundary of the eigenvalues of the cov mat (Marcenko-Pastur)
    lambda_max = (1 + np.sqrt(ncells / nbins)) ** 2
    print(f'sig high thresh: {lambda_max}')
    lambda_min = (1 - np.sqrt(ncells / nbins)) ** 2
    print(f'sig low thresh: {lambda_min}')

    n_sig_assembly = sum(evals > lambda_max)
    print(f'Step 2: identified {n_sig_assembly} cell assemblies out of {ncells} total neurons')
    n_sig_neurons = sum(evals < lambda_min)
    print(f'# neurons: {n_sig_neurons}')

    return n_sig_assembly


def find_assembly(z, n_sig_assembly, niter = 25, clusterdist = 0.6, clusterthresh = 0.5, learningrate=0.025,
                  silhouette_thresh=0.6):
    """ Step 3: first reduce z_mat dim by n_sig_assembly, then fastICA on projected space,
    icasso to boost reliability of ica (bootstrap niter times and cluster result components) """

    ica_params = {}
    icasso = Icasso(FastICA, ica_params=ica_params, iterations=niter, bootstrap=True, vary_init=True)

    # PCA on z_mat
    pca = PCA(n_components=n_sig_assembly)
    reduced_mat = pca.fit_transform(z)
    pca_weights = pca.components_

    # icasso on ICA
    icasso.fit(data=reduced_mat, fit_params={}, bootstrap_fun=bootstrap_fun, unmixing_fun=unmixing_fun)
    distance = clusterdist
    W_, scores = icasso.get_centrotype_unmixing(distance=distance)
    n_consistent_assembly = np.sum(np.array(scores)>silhouette_thresh)

    # clustermore = 0
    # clusterless = 0
    # while W_.shape[0] > n_sig_assembly * (1 + clusterthresh):
    #     # overfit: Getting too many clusters! Increase within cluster distance to reduce # of clusters
    #     distance = distance + learningrate
    #     W_, scores = icasso.get_centrotype_unmixing(distance=distance)
    #     clusterless = clusterless +1
    #
    # while W_.shape[0] < n_sig_assembly:
    #     # underfit: reduce within cluster distance to make sure getting more clusters than n_sig_assembly
    #     distance = distance - learningrate
    #     W_, scores = icasso.get_centrotype_unmixing(distance=distance)
    #     clustermore = clustermore +1

    plt.plot(range(1, len(scores) + 1), scores)
    plt.axvline(x=n_sig_assembly, linestyle='--', color='black', label=f'{n_sig_assembly} significant assembly')
    plt.scatter(n_consistent_assembly+1, scores[n_consistent_assembly], marker='x', color='red',
                label=f'{n_consistent_assembly} passed {silhouette_thresh} silhouette threshold')
    plt.legend()
    plt.xlabel('Component')
    plt.ylabel('Silhouette score')
    plt.title(f'distance={distance}, {len(scores)} clusters')
    plt.show()

    # icasso.plot_mds(distance=distance)
    # plt.show()

    #print(f'unmixing distance {distance}, {n_sig_assembly} assemblies from {W_.shape[0]} clusters, '
    #      f'after {clusterless} times increasing distance & {clustermore} times reducing distance')

    ica_weights = W_[:np.min((n_sig_assembly, n_consistent_assembly)), :]
    weights = ica_weights @ pca_weights  # weights to project neurons into ICA space

    return weights, scores
    #return ica_weights, pca_weights, scores

def calculate_assembly_strength(weights, z):

    # Calculate coactivation strength
    ncells = np.shape(z)[1]
    nbins = np.shape(z)[0]
    n_sig_assembly = weights.shape[0]
    diag_ind = np.diag_indices(ncells)
    assemb_strength = np.zeros((n_sig_assembly, nbins))
    for n in range(n_sig_assembly):
        P = np.outer(weights[n, :], weights[n, :])
        P[diag_ind] = 0   # set diagonal to zeros to remove contribution from a single cell
        R = (z @ P) * (z)
        assemb_strength[n, :] = np.sum(R, axis=1)

    return assemb_strength

def find_cells_per_assembly(weights, z_thresh=2):

    z_weights = stats.zscore(weights, axis=1)
    n_assemb, n_sig_neuron = np.where((z_weights > z_thresh) | (z_weights < -z_thresh))
    max_assembly = max(n_assemb)
    assembly_cells = {n: n_sig_neuron[n_assemb == n] for n in range(max_assembly + 1)}

    return assembly_cells

def plot_assembly(z: np.ndarray, strength: np.ndarray, cells: np.ndarray, data: LoadRaw, z_thresh=0.01, z_max=2, binwindow=5):


    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True, layout='constrained')
    im = ax1.imshow(z[:, cells].transpose(), aspect='auto', cmap='gray_r', vmin=z_thresh, vmax=z_max)
    ax1.set_title(f'cells: {cells}')
    fig.colorbar(im, ax=ax1)

    # Second subplot (sum plot)
    ax2.plot(strength)
    ax2.set_title('assembly strength')
    prev_ind = 0

    for switch_frame in data.params['switch_frame']:
        ax1.axvline(x=int(switch_frame / binwindow), color='black', linestyle='--', linewidth=1.5)
        ax2.axvline(x=int(switch_frame / binwindow), color='black', linestyle='--', linewidth=1.5)

    if 'opto_on' in data.params:
        for n in range(len(data.params['opto_on'])):
            ax1.hlines(y=len(cells)+ 0.5, xmin=int(data.params['opto_on'][n] / binwindow),
                       xmax=int(data.params['opto_off'][n] / binwindow), color='red', linewidth=2)
            ax2.hlines(y=-1, xmin=int(data.params['opto_on'][n] / binwindow),
                       xmax=int(data.params['opto_off'][n] / binwindow), color='red', linewidth=2)


def match_opto_env(data: LoadRaw):

    diff_env_indices = data.params['switch_frame'][1::2]
    envs = [f'{env[:-5]}' for env in data.env[1::2]]
    mat = np.split(data.mat, diff_env_indices, axis=0)

    return dict(zip(envs, mat))


class AssemblyPipeline:

    def __init__(self, binwindow=5):

        self.binwindow = binwindow

    def visualize_all_envs(self, data: LoadRaw):

        z_all = bin_z_mat(data.mat, binwindow=self.binwindow)
        plot_raster(data=data, z=z_all, binwindow=self.binwindow)

        return z_all

    def find_assembly_weights(self, data: LoadRaw):

        env_mat = match_opto_env(data)
        strength_ = []

        for env, mat in env_mat.items():
            save_path = os.path.join(data.path, f'{data.name}_{env}_assembly_weights.npy')
            z = bin_z_mat(mat, binwindow=self.binwindow)
            n_sig_assembly = n_assembly(z)
            weights = find_assembly(z, n_sig_assembly)
            np.save(save_path, weights)
            assemb_strength = calculate_assembly_strength(weights, z)
            np.save(os.path.join(data.path, f'{data.name}_{env}_assembly_strength.npy'), assemb_strength)
            print(f'saved {data.name} {env} assembly weights and strength to {data.path}')
            df = pd.DataFrame(assemb_strength.transpose(), columns=[f'{n}' for n in range(assemb_strength.shape[0])])
            strength_df = pd.melt(df, value_vars=[f'{n}' for n in range(assemb_strength.shape[0])],
                                  var_name='assembly', value_name='strength')
            strength_df['assembly env'] = env
            strength_.append(strength_df)


        # set up df
        strength_df = pd.concat(strength_).reset_index()
        strength_df['mouse'] = data.name

        df = pd.DataFrame(np.arange(int(data.constants['nframes']/self.binwindow)), columns=['bin'])
        ind = [int(n / self.binwindow) for n in data.params['switch_frame']]
        prev_index = 0
        for i, index in enumerate(ind):
            df.loc[prev_index:index, 'env'] = data.env[i]
            prev_index = index + 1

        df.loc[prev_index:, 'env'] = data.env[-1]

        on_ind = [int(n / self.binwindow) for n in data.params['opto_on']]
        off_ind = [int(n / self.binwindow) for n in data.params['opto_off']]
        df['opto'] = 'off'

        for n in range(len(data.params['opto_on'])):
            df.loc[on_ind[n]:off_ind[n], 'opto'] = 'on'

        strength_df = pd.merge(left=strength_df, right=df, how='left', left_on='index', right_on='bin')
        strength_df.to_parquet(os.path.join(data.path, f'{data.name}_assembly_strength.parquet'))

        return strength_df

    def get_assembly_summary(self, data: LoadRaw):

        weight_files = AssemblyPipeline.find_files(data)
        strength_files = AssemblyPipeline.find_files(data, 'assembly_strength.npy')
        envs = [re.search(r'e[LR]\d{2,3}_(.*?)_assembly_weights.npy', filename).group(1) for filename in weight_files]

        for n in range(len(weight_files)):
            print(f'processing {data.name} {envs[n]}')
            env = envs[n]
            weights = np.load(os.path.join(data.path, weight_files[n]))
            assembly_weights = np.load(os.path.join(data.path, strength_files[n]))
            plt.plot(np.mean(assembly_weights, axis=0))
            plt.title(f'{data.name} mean assembly weights {env}')
            plt.show()
            cell_assembly[env] = find_cells_per_assembly(weights)


        assemb_strength = np.concatenate(weights_all, axis=0)
        strength_df = pd.DataFrame(assemb_strength.transpose(),
                                   columns=[f'{n}' for n in range(assemb_strength.shape[0])])
        strength_df['env'] = None
        strength_df['animal'] = data.name
        strength_df['opto'] = 'off'
        ind = [int(n / self.binwindow) for n in data.params['switch_frame']]
        prev_index = 0
        for i, index in enumerate(ind):
            strength_df.loc[prev_index:index, 'env'] = data.env[i]
            prev_index = index + 1

        strength_df.loc[prev_index:, 'env'] = data.env[-1]

        on_ind = [int(n / self.binwindow) for n in data.params['opto_on']]
        off_ind = [int(n / self.binwindow) for n in data.params['opto_off']]

        for n in range(len(data.params['opto_on'])):
            strength_df.loc[on_ind[n]:off_ind[n], 'opto'] = 'on'

        return strength_df, cell_assembly

    @staticmethod
    def find_files(data: LoadRaw, keywords='assembly_weights.npy'):
        """ find the matlab file under day_path directory"""

        onlyfiles = [f for f in os.listdir(data.path) if f.endswith(keywords)]

        return onlyfiles
