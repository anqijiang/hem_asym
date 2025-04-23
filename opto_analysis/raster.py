import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy import nanmean
import scipy.io
from ast import literal_eval
from scipy.ndimage import uniform_filter1d
# from opto_analysis.plotting import cell_heatmap, cell_mean_over_laps
from opto_analysis.place_cell_opto import *
import seaborn as sns
import os.path
import pickle
import scipy.stats as stats
import itertools
from matplotlib.ticker import PercentFormatter
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
import re


def find_elbow(y):
    """
    Given arrays x (time/index) and y (values), this function computes the elbow
    point defined as the point with maximum distance to the line joining the first
    and last points.
    """
    # Create endpoint coordinates
    x = np.arange(len(y))
    point1 = np.array([x[0], y[0]])
    point2 = np.array([x[-1], y[-1]])

    # Compute the line vector and its unit vector
    line_vec = point2 - point1
    line_vec_norm = line_vec / np.linalg.norm(line_vec)

    # For each point, compute the vector from point1
    vec_from_first = np.vstack((x - point1[0], y - point1[1])).T

    # Project each point onto the line (dot product)
    proj_lengths = np.dot(vec_from_first, line_vec_norm)
    proj_points = np.outer(proj_lengths, line_vec_norm) + point1

    # Calculate perpendicular distances from each point to the line
    distances = np.linalg.norm(vec_from_first - proj_points, axis=1)

    # The index with the maximum distance is our elbow
    elbow_index = np.argmax(distances)
    return elbow_index, distances

def load_py_var_mat(day_path, keywords):
    """ find the matlab file under day_path directory"""
    onlyfiles = [f for f in listdir(day_path) if os.path.isfile(os.path.join(day_path, f))]
    file_name = [f for f in onlyfiles if f.endswith(keywords)][0]
    print('loading file: ', file_name)
    file = scipy.io.loadmat(os.path.join(day_path, file_name))

    return file

def map_string(input_string):
    # map 'control_later/first_day(\d)' to 'control_day(\d)'
    result = re.sub(r'^(control)(.*?(first|later))?_day(\d)', r'\1_day\4', input_string)
    return result


# icasso functions: bootstrap_fun & unmixing_fun
def bootstrap_fun(data, generator):
    return data[generator.choice(range(data.shape[0]), size=data.shape[0]), :]


def unmixing_fun(ica):
    return ica.components_

def init_logger(path, name):
    log_file = os.path.join(path, f'{name}_raster.log')

    logger = logging.getLogger('PF_logger')
    logger.setLevel(logging.INFO)

    logger.propagate = False

    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter('%(asctime)s %(message)s', datefmt='%m/%d/%y %I:%M %p')

    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


class LoadAxonImagingData:
    def __init__(self, mouse, env, day):
        self.path = os.path.join('D:\\Axon\\Analysis', mouse, day)
        self.name = mouse
        self.env = env

        if os.path.exists(os.path.join(self.path, f'{self.name}_raster.pickle')):
            self.load()
        else:

            mat = np.genfromtxt(os.path.join(self.path, 'grouped_axons.csv'), delimiter=',', skip_header=1)
            self.mat = mat[:, 1:]
            self.constants = {'ncells': np.shape(self.mat)[1]}
            df = pd.read_parquet(os.path.join(self.path, f'{self.name}_{day}_axon.parquet'), engine='fastparquet')
            envs = df['env'].unique()
            self.running_mat = {}
            for e in envs:
                running_mat = df.loc[df['env'] == e].iloc[:, :-5].to_numpy()
                self.running_mat[e] = running_mat

            pattern = re.compile(r'.*-all-cond(\d?)\.mat$')
            onlyfiles = [f for f in os.listdir(self.path) if os.path.isfile(os.path.join(self.path, f))]
            beh_file_name = [f for f in onlyfiles if pattern.match(f)][0]
            print(f'found beh file: {beh_file_name} '
                  f'\namong files: {onlyfiles} ')
            axon_beh = scipy.io.loadmat(os.path.join(self.path, beh_file_name))

            # axon_beh = load_py_var_mat(self.path, 'cond.mat')

            self.params = {'switch_frame': axon_beh['end_frame'][:, 0],
                           'start_frame': axon_beh['start_frame'][:, 0],
                           'ybinned': axon_beh['behavior']['ybinned'][0, :][0][0, 1:], 'lap': axon_beh['E'][0, 1:]}

            switch_frame = np.array(self.params['switch_frame'])
            switch_frame = np.insert(switch_frame, 0, 0)
            start_frame = np.array(self.params['start_frame']) - switch_frame[:-1]
            self.params['env_start'] = dict(zip(self.env, start_frame))

            # setup logger
            self.logger = init_logger(self.path, f'{self.name}_new')

    def delete_frames(self, bad_frames):

        data = np.nanmean(self.mat, axis=1)
        plt.plot(data)
        plt.scatter(bad_frames, [[np.max(data)]*len(bad_frames)], color='r')
        plt.title('bad frames to delete')
        plt.show()

        cont = input('continue to delete frames? 1 to continue, any other key to exit \n')
        if int(cont) != 1:
            print('exit: keep original data')
            return None

        img_frames = len(data)
        run_ind = np.setdiff1d(np.arange(img_frames), bad_frames)
        self.logger.info(f'deleting frames: {bad_frames}')

        # update attributes
        self.mat = self.mat[run_ind, :]
        for k,v in self.params.items():
            self.params[k] = np.array([bisect.bisect_left(run_ind, x) for x in v])

        self.save_to_file()

    def delete_pause(self, pause_frame=40, dist_thresh=0.05, plt_frames=5000):
        """ align with behavior and remove frames when animal is not running

        :param pause_frame: 2x duration of frames not moving to be considered as pauses
        :param dist_thresh: distance smaller than this will be considered not moving
        :return: run ind,  opto on ind while running (if applicable)
        """

        self.logger.info(f'Step 0: Using only active period to find coactivity patterns. '
                    f'Not moving at least {dist_thresh} for {pause_frame} frames is considered as pausing')

        beh_mat = load_py_var_mat(self.path, 'cond.mat')
        ybinned = beh_mat['behavior']['ybinned'][0][0].transpose()
        velocity = beh_mat['behavior']['velocity'][0][0].transpose()
        beh_frames = ybinned.shape[0]
        img_frames = self.mat.shape[0]
        if beh_frames != img_frames:
            print(f'{self.name} behavior and imaging of different size: beh {beh_frames}, img {img_frames}')
            return

        # find ind for pauses
        trackend = 0.605
        trackstart = 0.015
        vr_ind = (ybinned > trackstart) & (ybinned < trackend)
        # ybinned_vr = ybinned[vr_ind]
        v_vr = velocity[vr_ind]
        v_thresh = np.quantile(v_vr, 0.05)
        v0_ind = np.where(velocity < v_thresh)[0]  # find indices for when velocity is lower than 5% of all velocity
        v0_dis = ybinned[np.minimum(v0_ind + pause_frame, beh_frames - 1)] \
                 - ybinned[np.maximum(v0_ind - pause_frame, 0)]  # double check with frames and distance
        pause_ind = v0_ind[v0_dis[:, 0] < dist_thresh]
        run_ind = np.setdiff1d(np.arange(img_frames), pause_ind)
        self.logger.info(f'{len(pause_ind)} frames pausing, {len(run_ind)} frames running, {beh_frames} frames in total')

        # plot to verify
        plt.scatter(np.arange(plt_frames), ybinned[pause_ind[:plt_frames]])
        plt.title(f'{self.name} Double check: behav trace to delete (pauses), first {plt_frames} frames')
        plt.show()

        plt.plot(ybinned[run_ind[:plt_frames]])
        plt.title(f'{self.name} Double check: behav trace after removing pauses, first {plt_frames} frames')
        plt.show()

        self.delete_frames(pause_ind)

    def load(self, var='raster'):
        """ load previously saved pickle file"""

        pickle_file = os.path.join(self.path, f'{self.name}_{var}.pickle')
        file = open(pickle_file, 'rb')
        print(f'Loading {self.name} {var} stored from cache: {pickle_file}')
        temp_dict = pickle.load(file)
        file.close()
        self.__dict__.update(temp_dict)

    def save_to_file(self, var='raster'):
        """ auto save class """

        data_path = os.path.join(self.path, f'{self.name}_{var}.pickle')
        print(f'Saving {self.name} {var} to file at {data_path}')

        with open(data_path, 'wb') as output_file:
            pickle.dump(self.__dict__, output_file, pickle.HIGHEST_PROTOCOL)


class LoadImagingData:
    def __init__(self, mouse, env, day):
        self.path = os.path.join('D:\\Opto\\Analysis', mouse, day)
        self.name = mouse
        self.params = {}
        self.env = env

        if os.path.exists(os.path.join(self.path, f'{self.name}_raster.pickle')):
            self.load()
        else:
            mat_file = load_py_var_mat(self.path, 'align_cell_mean.mat')
            self.mat = mat_file['Fc3_DF']
            self.constants = {'ncells': np.shape(self.mat)[1]}

            self.params['switch_frame'] = mat_file['switch_frame'][0, :].astype('int') -1

            # add opto_on_lap and opto_off_lap if exist
            if all(var in mat_file.keys() for var in ['opto_off_lap', 'opto_on_lap']):
                opto_off_frame = mat_file['offFrames_ind'][:, 0].astype('int') - 1
                opto_on_frame = np.fmax(mat_file['onFrames_ind'][:, 0].astype('int') - 1, 0)
                opto_length = opto_off_frame - opto_on_frame
                self.params['opto_on_frame'] = opto_on_frame[opto_length > 100]
                self.params['opto_off_frame'] = opto_off_frame[opto_length > 100]

            # setup logger
            self.logger = init_logger(self.path, f'{self.name}_new')
            self.opto_df = self.opto_summary()

    def delete_frames(self, bad_frames):

        data = np.nanmean(self.mat, axis=1)
        plt.plot(data)
        plt.scatter(bad_frames, [[np.max(data)]*len(bad_frames)], color='r')
        plt.title('bad frames to delete')
        plt.show()

        cont = input('continue to delete frames? 1 to continue, any other key to exit \n')
        if int(cont) != 1:
            print('exit: keep original data')
            return None

        img_frames = len(data)
        run_ind = np.setdiff1d(np.arange(img_frames), bad_frames)
        self.logger.info(f'deleting frames: {bad_frames}')

        # update attributes
        self.mat = self.mat[run_ind, :]
        for k,v in self.params.items():
            self.params[k] = np.array([bisect.bisect_left(run_ind, x) for x in v])
        self.opto_df = self.opto_summary()

        self.save_to_file()

    def delete_pause(self, pause_frame=40, dist_thresh=0.05, plt_frames=5000):
        """ align with behavior and remove frames when animal is not running

        :param pause_frame: 2x duration of frames not moving to be considered as pauses
        :param dist_thresh: distance smaller than this will be considered not moving
        :return: run ind,  opto on ind while running (if applicable)
        """

        self.logger.info(f'Step 0: Using only active period to find coactivity patterns. '
                    f'Not moving at least {dist_thresh} for {pause_frame} frames is considered as pausing')

        beh_mat = load_py_var_mat(self.path, 'cond.mat')
        ybinned = beh_mat['behavior']['ybinned'][0][0].transpose()
        velocity = beh_mat['behavior']['velocity'][0][0].transpose()
        beh_frames = ybinned.shape[0]
        img_frames = self.mat.shape[0]
        if beh_frames != img_frames:
            print(f'{self.name} behavior and imaging of different size: beh {beh_frames}, img {img_frames}')
            return

        # find ind for pauses
        trackend = 0.605
        trackstart = 0.015
        vr_ind = (ybinned > trackstart) & (ybinned < trackend)
        # ybinned_vr = ybinned[vr_ind]
        v_vr = velocity[vr_ind]
        v_thresh = np.quantile(v_vr, 0.05)
        v0_ind = np.where(velocity < v_thresh)[0]  # find indices for when velocity is lower than 5% of all velocity
        v0_dis = ybinned[np.minimum(v0_ind + pause_frame, beh_frames - 1)] \
                 - ybinned[np.maximum(v0_ind - pause_frame, 0)]  # double check with frames and distance
        pause_ind = v0_ind[v0_dis[:, 0] < dist_thresh]
        run_ind = np.setdiff1d(np.arange(img_frames), pause_ind)
        self.logger.info(f'{len(pause_ind)} frames pausing, {len(run_ind)} frames running, {beh_frames} frames in total')

        # plot to verify
        plt.scatter(np.arange(plt_frames), ybinned[pause_ind[:plt_frames]])
        plt.title(f'{self.name} Double check: behav trace to delete (pauses), first {plt_frames} frames')
        plt.show()

        plt.plot(ybinned[run_ind[:plt_frames]])
        plt.title(f'{self.name} Double check: behav trace after removing pauses, first {plt_frames} frames')
        plt.show()

        self.delete_frames(pause_ind)


    def opto_summary(self) -> pd.DataFrame:

        opto_summary = pd.DataFrame(columns = ['env', 'switch frame', 'opto on frame', 'opto off frame'])
        opto_summary['env'] = self.env
        opto_summary['switch frame'] = np.concatenate(([0], self.params['switch_frame']))
        opto_ind = [bisect.bisect_right(self.params['switch_frame'], x) for x in self.params['opto_on_frame']]
        opto_summary.loc[opto_ind, 'opto on frame'] = self.params['opto_on_frame']
        opto_summary.loc[opto_ind, 'opto off frame'] = self.params['opto_off_frame']

        return opto_summary

    def load(self, var='raster'):
        """ load previously saved pickle file"""

        pickle_file = os.path.join(self.path, f'{self.name}_{var}.pickle')
        file = open(pickle_file, 'rb')
        print(f'Loading {self.name} {var} stored from cache: {pickle_file}')
        temp_dict = pickle.load(file)
        file.close()
        self.__dict__.update(temp_dict)

    def save_to_file(self, var='raster'):
        """ auto save class """

        data_path = os.path.join(self.path, f'{self.name}_{var}.pickle')
        print(f'Saving {self.name} {var} to file at {data_path}')

        with open(data_path, 'wb') as output_file:
            pickle.dump(self.__dict__, output_file, pickle.HIGHEST_PROTOCOL)


def separate_env(data: LoadImagingData) -> {}:
    return dict(zip(data.env, np.split(data.mat, data.params['switch_frame'])))


def np_to_long_df(data: np.ndarray, start_ind: int = 0, var_name: str = 'cell', val_name: str = 'value') -> pd.DataFrame:
    # Get the shape of the array to create appropriate indices and columns
    n_rows, n_cols = data.shape

    # Create row and column indices for the long-format DataFrame
    row_index = np.repeat(np.arange(n_rows), n_cols) + start_ind

    # Reshape the data array to a 1D array for values in the long format
    values = data.flatten()

    # Create a DataFrame using the indices and values
    df = pd.DataFrame({var_name: row_index, val_name: values})

    return df


def fr_stats(fr_mat1: np.ndarray, fr_mat2: np.ndarray, alpha=0.01) -> {}:
    """
    :param fr_mat1: cell* nshuffle, control, opto off firing rate - opto on firing rate
    :param fr_mat2: same size as fr_mat1, opto
    """

    f, p = stats.ttest_ind(fr_mat1, fr_mat2, axis=1, equal_var=False)

    # sidak adjustment for family-wise error
    alpha_adj = 1 - (1 - alpha) ** (1 / len(p))

    no_effect = np.where(p > alpha_adj)[0]
    fr_mean1 = np.mean(fr_mat1, axis=1)
    fr_mean2 = np.mean(fr_mat2, axis=1)
    fr_diff = fr_mean1 - fr_mean2
    mat2bigger = np.where(fr_diff<0 & (~np.isin(fr_diff, no_effect)))[0]
    mat1bigger = np.where(fr_diff > 0 & (~np.isin(fr_diff, no_effect)))[0]
    effects = {'no_effect': no_effect, 'opto increased firing': mat2bigger, 'opto decreased firing': mat1bigger}

    return effects


class Raster:
    def __init__(self, fig_path, binsize=3, thresh=0):
        self.fig_path = fig_path
        self.bin = binsize
        self.zthresh = thresh

    def load_cache(self, path: str, savename: str):
        """ load previously saved pickle file"""

        file_path = os.path.join(path, f'bin{self.bin}_z{self.zthresh}_{savename}.pickle')
        if os.path.exists(file_path):
            file = open(file_path, 'rb')
            print(f'Loading stored {savename} from cache: {file_path}')
            temp_dict = pickle.load(file)
            file.close()
            return temp_dict
        else:
            print(f'No stored {savename}')
            return None

    def save_to_file(self, var, savename, path):
        """ save var as savename """

        data_path = os.path.join(path, f'bin{self.bin}_z{self.zthresh}_{savename}.pickle')
        print(f'Saving {savename} to file at {data_path}')

        with open(data_path, 'wb') as output_file:
            pickle.dump(var, output_file, pickle.HIGHEST_PROTOCOL)

    def bin_z_mat(self, data_file: LoadImagingData):

        path = data_file.path

        z_mat = self.load_cache(path, 'zmat')
        binned_firing = self.load_cache(path, 'binned_firing')
        #
        if z_mat is not None and binned_firing is not None:
            return z_mat, binned_firing

        z_mat = {}
        binned_firing = {}
        # data = separate_env(data_file)
        data = data_file.running_mat

        for k, mat in data.items():
            nframes = np.shape(mat)[0]
            ncells = np.shape(mat)[1]

            env_start = data_file.params['env_start'][k]

            df = pd.DataFrame(mat[env_start:, :])
            ind = np.arange(env_start, nframes)
            bins = np.arange(env_start, nframes + self.bin, self.bin)
            binned = pd.cut(ind, bins, include_lowest=True)
            df['bin'] = binned
            df_binned = df.groupby('bin').mean()
            bin_mat = df_binned.iloc[0:-1, :ncells].to_numpy()

            inactive_neuron = np.where(np.sum(bin_mat, axis=0) == 0)[0]
            #active_mat = bin_mat[:, np.nansum(bin_mat, axis=0) > 0]
            ncells = ncells - len(inactive_neuron)
            print(f'{ncells} active cells out of {ncells} total cells in {k}')

            #z = stats.zscore(active_mat[:, :], axis=0, nan_policy='omit')
            z = stats.zscore(bin_mat[:, :], axis=0, nan_policy='omit')
            #z = z[~np.isnan(z).any(axis=1), :]  # remove any nans in z_mat
            z[np.isnan(z)] = 0
            z_mat[k] = z
            firing = z>self.zthresh
            binned_firing[k] = firing.astype(int)

        self.save_to_file(z_mat, 'zmat', path)
        self.save_to_file(binned_firing, 'binned_firing', path)

        return z_mat, binned_firing

    def plot_raster(self, binned_firing: {},  title: str): #,opto_summary: pd.DataFrame, smooth=3):

        # assert len(binned_firing) == len(opto_summary), "z_mat and opto_summary length not matched"

        cell_counts = []
        # opto_summary_z = pd.DataFrame(index=binned_firing.keys(), columns=['opto on z', 'opto off z', 'total z'])

        for env, mat in binned_firing.items():
            print(env)
            fr_loc = np.argwhere(mat)
            nframes, ncells = np.shape(mat)

            # ax[1] cell count per frame
            cell_counts_local = pd.DataFrame(columns=['frame', 'active cell count', 'env', 'opto'])
            cell_counts_local['frame'] = np.arange(nframes)
            cell_counts_local['active cell count'] = np.sum(mat, axis=1)
            cell_counts_local['env'] = env
            cell_counts_local['opto'] = 'off'

            # ax[0] raster plot
            fig, ax = plt.subplots(2, 1, sharex=True, figsize = (2*nframes/ncells, 6))
            ax[0].scatter(fr_loc[:, 0], fr_loc[:, 1], linewidths=0.5, s=1)
            ax[0].set_xlim([-5, nframes+5])
            plt.xlabel('bins')
            ax[0].set_ylabel('cells')
            # fig.set_size_inches(3*nframes/ncells, 6)
            smoothed_frame = uniform_filter1d(cell_counts_local['active cell count'].tolist(), size=3)  # smooth cells_per_frame for plotting
            horz_line_y = np.max(smoothed_frame) + 5
            ax[1].set_xlim([-5, nframes + 5])
            ax[1].plot(smoothed_frame)
            ax[1].set_ylabel('# cells firing per frame')
            #ax[1].set_title(f'# cells firing per frame, smooth window {smooth}')

            # # add opto on period red box
            # if ~opto_summary.loc[opto_summary['env']==env].isna().any(axis=None):
            #     row = opto_summary.loc[opto_summary['env']==env]
            #     opto_on_z = ((row['opto on frame']- row['switch frame']) / self.bin).to_numpy()
            #     opto_off_z = ((row['opto off frame']- row['switch frame']) / self.bin).to_numpy()
            #     opto_summary_z.loc[env, :] = [int(np.floor(opto_on_z)), int(np.ceil(opto_off_z)), nframes]
            #     ax[0].add_patch(Rectangle((opto_on_z[0], 0), opto_off_z[0] - opto_on_z[0], ncells,
            #                               color="red", linestyle='dotted', fc='none'))
            #     ax[1].plot([opto_on_z[0], opto_off_z[0]], [horz_line_y, horz_line_y], color='red')  # add opto
            #     cell_counts_local.iloc[int(np.floor(opto_on_z)): int(np.ceil(opto_off_z)), 2] = 'on'
            # else:
            #     opto_summary_z.loc[env, 'total z'] = nframes

            cell_counts.append(cell_counts_local)

            titleName = f'{title} raster plot in {env}'
            plt.suptitle(titleName)
            #fig.set_dpi(80)
            plt.savefig(os.path.join(self.fig_path, f'{titleName}.svg'))
            plt.show()

            plt.scatter(fr_loc[:, 0], fr_loc[:, 1], linewidths=0.5, s=1)
            plt.xlim([-5, nframes + 5])
            plt.xlabel('bins')
            plt.ylabel('cells')

            # if ~opto_summary.loc[opto_summary['env'] == env].isna().any(axis=None):
            #     row = opto_summary.loc[opto_summary['env'] == env]
            #     opto_on_z = ((row['opto on frame'] - row['switch frame']) / self.bin).to_numpy()
            #     opto_off_z = ((row['opto off frame'] - row['switch frame']) / self.bin).to_numpy()
            #     opto_summary_z.loc[env, :] = [int(np.floor(opto_on_z)), int(np.ceil(opto_off_z)), nframes]
            #     plt.gca().add_patch(Rectangle((opto_on_z[0], 0), opto_off_z[0] - opto_on_z[0], ncells,
            #                               color="red", linestyle='dotted', fc='none'))
            #     cell_counts_local.iloc[int(np.floor(opto_on_z)): int(np.ceil(opto_off_z)), 2] = 'on'
            # else:
            #     opto_summary_z.loc[env, 'total z'] = nframes

            cell_counts.append(cell_counts_local)

            titleName = f'{title} in {env} raster only'
            #plt.suptitle(titleName)
            plt.savefig(os.path.join(self.fig_path, f'{titleName}.jpg'))
            #plt.savefig(os.path.join(self.fig_path, f'{titleName}.svg'))
            plt.show()

        cell_counts = pd.concat(cell_counts).reset_index(drop=True)

        return cell_counts #, Raster.add_control(opto_summary_z)

    @staticmethod
    def add_control(opto_summary_z):

        copy_data = opto_summary_z[opto_summary_z['opto on z'].notna()]
        copy_data.index = [f'control_{i}' for i in copy_data.index]
        opto_summary_z = pd.concat([opto_summary_z, copy_data])

        pattern = re.compile(r'^control_(.*)_day(\d)$')

        # Iterate over the DataFrame's index and update 'total' values
        for idx in opto_summary_z.index:
            match = pattern.match(idx)
            if match:
                num = int(match.group(2))
                opto_summary_z.at[idx, 'total z'] = opto_summary_z.at[f'control_day{num}', 'total z']

        return opto_summary_z.dropna()

    def bootstrap_fr(self, fr_mat: np.ndarray, start_frame, end_frame, nshuffle=500, nbins=1000) -> np.ndarray:

        fr_dist = np.zeros((fr_mat.shape[0], nshuffle))
        data = fr_mat[:, start_frame: end_frame]
        ind = np.arange(end_frame - start_frame)

        dur = nbins * (self.bin / 30.98 / 60)

        for n in range(nshuffle):
            rand_ind = np.random.choice(ind, nbins)
            fr_dist[:, n] = np.sum(data[:, rand_ind], axis=1) / dur

        return fr_dist

    def opto_all_cells(self, binned_firing: {}, opto_z: pd.DataFrame) -> {}:

        opto_z.reset_index(inplace=True)
        opto_z['env'] = opto_z['index'].apply(map_string)
        n_opto_env = len(opto_z)

        fr_dist_on = opto_z.apply(
            lambda row: self.bootstrap_fr(fr_mat=binned_firing[row['env']].transpose(), start_frame=row['opto on z'],
                                           end_frame=row['opto off z']), axis=1)
        fr_dist_after = opto_z.apply(
            lambda row: self.bootstrap_fr(fr_mat=binned_firing[row['env']].transpose(), start_frame=row['opto off z'],
                                           end_frame=row['total z']), axis=1)
        fr_diff = [fr_dist_after[n] - fr_dist_on[n] for n in range(n_opto_env)]
        effect_df = []

        for n in range(int(n_opto_env / 2)):
            env = opto_z.loc[n, 'env']
            effects = fr_stats(fr_diff[n], fr_diff[n + int(n_opto_env / 2)])
            effect_df_local = pd.DataFrame([(key, value) for key, values in effects.items() for value in values],
                                           columns=['opto effect', 'cell'])
            effect_df_local['env'] = env
            effect_df.append(effect_df_local)

        effect_df = pd.concat(effect_df).reset_index(drop=True)
        return effect_df

    def quantify_raster(self, binned_firing: {}, opto_summary_control: pd.DataFrame):

        f = self.bin / 30.98 / 60
        cell_opto_summary = []
        opto_later_conds = ['before opto', 'during opto', 'after opto']
        opto_first_conds = ['bug in code', 'during opto', 'after opto']

        for n in range(len(opto_summary_control)):
            env = opto_summary_control.iloc[n, 0]
            print(env)
            if env in binned_firing:
                v = binned_firing[env]
                treatment_group = 'opto'
            else:
                control_env = f'control_{env[-4:]}'
                treatment_group = 'control'
                v = binned_firing[control_env]
            mean_FR = (np.sum(v, axis=0)) / (np.shape(v)[0]*f)
            cut_offs = opto_summary_control.iloc[n, 1:-1].tolist()
            print(cut_offs)
            opto_conds = np.split(v, cut_offs)
            z_length = [len(m) for m in opto_conds]

            if z_length[0] > 0:
                # opto later conditions
                for cond in range(3):
                    FR = np.sum(opto_conds[cond], axis=0)/ (z_length[cond]*f)
                    local_df = pd.DataFrame(enumerate(FR), columns=['cell', 'FR'])
                    local_df['opto'] = opto_later_conds[cond]
                    local_df['env'] = env
                    local_df['baseline FR'] = mean_FR
                    local_df['treatment group'] = treatment_group
                    cell_opto_summary.append(local_df)
            else:
                for cond in range(1,3):
                    FR = np.sum(opto_conds[cond], axis=0)/ (z_length[cond]*f)
                    local_df = pd.DataFrame(enumerate(FR), columns=['cell', 'FR'])
                    local_df['opto'] = opto_first_conds[cond]
                    local_df['env'] = env
                    local_df['baseline FR'] = mean_FR
                    local_df['treatment group'] = treatment_group
                    cell_opto_summary.append(local_df)

        cell_opto_summary = pd.concat(cell_opto_summary)
        cell_opto_summary['norm FR'] = (cell_opto_summary['FR'] - cell_opto_summary['baseline FR']) / cell_opto_summary['baseline FR']

        return cell_opto_summary


    def opto_effect_firing_intensity(self, binned_firing: {}, opto_summary_control: pd.DataFrame):

        cell_opto_summary = []

        def add_column(row, before_thresh, after_thresh):
            if row['frame'] < before_thresh:
                opto = 'before'
            elif before_thresh <= row['frame'] < after_thresh:
                opto = 'on'
            else:
                opto = 'after'
            return opto

        def norm_count(row, before_frames, on_frames, after_frames):
            # print(row)
            if row['opto'] == 'before':
                val = row['frame'] / before_frames
            elif row['opto'] == 'on':
                val = row['frame'] / on_frames
            elif row['opto'] == 'after':
                val = row['frame'] / after_frames
            return val

        for n in range(len(opto_summary_control)):
            env = opto_summary_control.iloc[n, 0]
            if env in binned_firing:
                v = binned_firing[env]
            else:
                control_env = f'control_{env[-4:]}'
                print(control_env)
                v = binned_firing[control_env]
            frames, cells = np.where(v)
            cell_count = pd.DataFrame(np.array([frames, cells]).transpose(), columns=['frame', 'cell'])
            on_z, off_z, total_z = opto_summary_control.iloc[n, 1:4]
            before_dur = on_z * (self.bin / 30.98 / 60)
            on_dur = (off_z - on_z) * (self.bin / 30.98 / 60)
            after_dur = (total_z - off_z) * (self.bin / 30.98 / 60)
            cell_count['opto'] = cell_count.apply(add_column, axis=1, args=(on_z, off_z))
            cell_opto = cell_count.groupby(['cell', 'opto']).count().reset_index()
            cell_opto['firing intensity'] = cell_opto.apply(norm_count, axis=1, args=(before_dur, on_dur, after_dur))
            cell_opto['env'] = env
            cell_opto_summary.append(cell_opto)

            # sns.pointplot(data=cell_opto, x='opto', y='firing intensity',  =['before', 'on', 'after'])
            sns.histplot(data=cell_opto, x="firing intensity", hue="opto", element="step")
            plt.title(env)
            plt.show()

        cell_opto_summary = pd.concat(cell_opto_summary).reset_index(drop=True)
        #cell_summary = cell_opto_summary.pivot(index='cell', columns=['env', 'opto'], values='frame')
        #cell_summary = cell_summary.fillna(0)
        #cell_summary = cell_summary.reset_index().melt(id_vars=['cell'])
        total_fi = cell_opto_summary.groupby(['cell', 'env'])['frame'].sum().reset_index()
        total_fi = total_fi.merge(opto_summary_control, )

        cell_firing_env = cell_summary.groupby(['cell', 'env'])['value'].sum().reset_index().merge(
            opto_summary_control.reset_index()[['index', 'total z']], left_on='env', right_on='index', how='left')
        cell_firing_env['mean firing'] = cell_firing_env['value'].div(cell_firing_env['total z'], axis=0) / (self.bin / 30.98 / 60)
        cell_summary = cell_summary.merge(cell_firing_env[['cell', 'env', 'mean firing']], on=['cell', 'env'], how='left')
        cell_opto_summary = cell_opto_summary.merge(cell_summary[['cell', 'env', 'opto', 'mean firing']], on=['cell', 'env', 'opto'], how='right')
        cell_opto_summary = cell_opto_summary.fillna(0)
        cell_opto_summary['delta firing'] = (cell_opto_summary['firing intensity'] - cell_opto_summary['mean firing']) / \
                                            cell_opto_summary['mean firing']

        return cell_opto_summary

    @staticmethod
    def normed_firing_intensity(cell_summary: pd.DataFrame, opto_summary_z: pd.DataFrame):
        on_diff_summary = []
        before_diff_summary = []
        before_col_name = []
        for env in opto_summary_z.index:
            df = cell_summary[env]
            df_local = (df['on'] - df['after']) / df['after']
            on_diff_summary.append(df_local)
            normed = df.div(df['after'], axis=0)

            if 'before' in df.columns:
                df_local = (df['before'] - df['after']) / df['after']
                before_diff_summary.append(df_local)
                before_col_name.append(env)
                sns.pointplot(data=normed, order=['before', 'on', 'after'])
            else:
                sns.pointplot(data=normed, order=['on', 'after'])
            plt.title(env)
            plt.show()

        on_diff_summary = pd.concat(on_diff_summary, axis=1).reset_index(drop=True)
        on_diff_summary.columns = opto_summary_z.index.tolist()

        before_diff_summary = pd.concat(before_diff_summary, axis=1).reset_index(drop=True)
        before_diff_summary.columns = before_col_name

        return on_diff_summary, before_diff_summary

