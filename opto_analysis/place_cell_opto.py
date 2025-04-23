import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import seaborn as sns
import os.path
import pickle
import re
import scipy.stats as stats
import itertools
from matplotlib.ticker import PercentFormatter
from sklearn import preprocessing
from itertools import compress
from sklearn.metrics import r2_score
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import linregress
from fastparquet import write
import bisect
from scipy.stats import wilcoxon
from scipy.stats.stats import pearsonr
import logging
from scipy import signal
import scipy.io as sio
from scipy import stats
# from opto_analysis.plotting import *

def determine_ca3_value(mouse_str):
    if 'L' in mouse_str:
        return 'left'
    elif 'R' in mouse_str:
        return 'right'
    else:
        return np.nan  # Handle cases where neither 'L' nor 'R' is found


def load_py_var_mat(day_path, keywords, varname=None):
    """ find the matlab file under day_path directory"""
    file_name = [f for f in os.listdir(day_path) if f.endswith(keywords)][0]
    # print('loading file: ', file_name)
    if varname is None:
        file = scipy.io.loadmat(os.path.join(day_path, file_name))
        print(f'loading file {file_name}')
    else:
        file = scipy.io.loadmat(os.path.join(day_path, file_name), variable_names=varname)
        print(f'loading file {file_name}: variable {varname}')

    return file


def mean_bin_over_laps(merged_data: pd.DataFrame, nbin=40) -> np.ndarray:
    """average cell activity within each ybin over each lap for all cells
    :param merged_data: dataframe[cell activity; ybinned; lap]
    :param nbin: number of bins
    :return: laps * location on track (binned) * cell
    """
    nlaps = merged_data.iat[-1, merged_data.columns.get_loc("lap")]+1

    cell_locs = [i for i, col in enumerate(merged_data.columns) if col.isdigit()]
    ncells = cell_locs[-1] + 1

    # ybinned track start and track end values from lab MATLAB scripts
    binEdges = np.linspace(0.015, 0.605, nbin + 1)
    bins = pd.cut(merged_data.ybinned, binEdges)
    mean_activity = (merged_data.iloc[:, cell_locs].groupby([merged_data.lap, bins], observed=False).mean())
    mean_activity = mean_activity.values.reshape((nlaps, nbin, ncells))

    return mean_activity


def map_string(input_string):
    # map 'control_later/first_day(\d)' to 'control_day(\d)'
    result = re.sub(r'^(control)(.*?(first|later))?_day(\d)', r'\1_day\4', input_string)
    return result


def init_logger(path, name, logger_name='PF_params', get_logger='__name__'):
    log_file = os.path.join(path, f'{name}_{logger_name}.log')

    logger = logging.getLogger(get_logger)
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


class LoadData:
    def __init__(self, mouse, env, day, folder):
        self.path = os.path.join('D:\\', folder, 'Analysis', mouse, day)
        self.env = env
        self.day = day

    def delete_laps(self, laps):
        pass

    def save_to_file(self):
        """ autosave var with savename"""
        pass


class LoadAxon(LoadData):
    def __init__(self, mouse, env, day):
        super().__init__(mouse, env, day, folder='Axon')
        saved_file = os.path.join(self.path, f'{mouse}_data.pickle')

        if os.path.exists(saved_file):
            file = open(saved_file, 'rb')
            print(f'Loading {mouse} stored from cache: {saved_file}')
            temp_dict = pickle.load(file)
            file.close()
            self.__dict__.update(temp_dict)

        else:
            self.name = mouse
            self.params = {}

            df_file_name = os.path.join(self.path,  f'{mouse}_{day}_axon.parquet')
            axon_df = pd.read_parquet(df_file_name, engine='fastparquet')
            axon_df['lap'] = axon_df['lap'].astype('int')
            self.axon_df = axon_df.reset_index()
            self.mean_activity = mean_bin_over_laps(axon_df, nbin=40)
            self.constants = {'nbins': np.shape(self.mean_activity)[1], 'ncells': np.shape(self.mean_activity)[2]}

            # self.axon_df['env'] = self.axon_df['env'].replace(dict(enumerate(env)))
            self.params['switch_lap'] = self.axon_df.groupby('env')['lap'].max().tolist()
            self.params['env_laps'] = self.axon_df.groupby('env')['lap'].unique().to_dict()

            self.logger = init_logger(self.path, f'{self.name}_new')


class LoadOpto(LoadData):
    def __init__(self, mouse, env, day):
        super().__init__(mouse, env, day, folder='Opto')
        saved_file = os.path.join(self.path, f'{mouse}_data.pickle')

        if os.path.exists(saved_file):
            file = open(saved_file, 'rb')
            print(f'Loading {mouse} stored from cache: {saved_file}')
            temp_dict = pickle.load(file)
            file.close()
            self.__dict__.update(temp_dict)

        else:
            self.name = mouse
            self.params = {}

            mat_file = load_py_var_mat(self.path, 'align_cell_mean.mat')
            mean_activity = mat_file['cell_binMean'].transpose((1, 0, 2))
            mean_activity[np.isnan(mean_activity)] = 0
            self.mean_activity = mean_activity
            self.constants = {'nbins': np.shape(self.mean_activity)[1], 'ncells': np.shape(self.mean_activity)[2]}


            if 'env_switch_lap' in mat_file:
                self.params['switch_lap'] = mat_file['env_switch_lap'][:, 0].astype('int') - 1
            # add opto_on_lap and opto_off_lap if exist
            if all(var in mat_file.keys() for var in ['opto_off_lap', 'opto_on_lap']):
                opto_off_lap = mat_file['opto_off_lap'][:, 0].astype('int')
                opto_on_lap = np.fmax(mat_file['opto_on_lap'][:, 0].astype('int') - 1, 0)
                opto_length = opto_off_lap - opto_on_lap
                self.params['opto_on_lap'] = opto_on_lap[opto_length > 1]
                self.params['opto_off_lap'] = opto_off_lap[opto_length > 1]

            # setup logger
            self.logger = init_logger(self.path, f'{self.name}_new')
            #self.separate_laps()

    def delete_laps(self, laps):

        self.logger.info(f'{self.name} deleting laps {laps}')
        self.mean_activity = np.delete(self.mean_activity, laps, 0)

        # adjust for deleted laps
        self.logger.info(f"switch laps before deleting pause laps: {self.params['switch_lap']}")
        for n in range(len(self.params['switch_lap'])):
            self.params['switch_lap'][n] = self.params['switch_lap'][n] - np.sum(self.params['switch_lap'][n] > laps)
        self.logger.info(f"switch laps after deleting pause laps: {self.params['switch_lap']}")

        self.logger.info(f"opto on laps before deleting: {self.params['opto_on_lap']}")
        self.logger.info(f"opto off laps before deleting: {self.params['opto_off_lap']}")
        for n in range(len(self.params['opto_off_lap'])):
            self.params['opto_off_lap'][n] = self.params['opto_off_lap'][n] - np.sum(
                self.params['opto_off_lap'][n] > laps)
            self.params['opto_on_lap'][n] = self.params['opto_on_lap'][n] - np.sum(self.params['opto_on_lap'][n] > laps)
        self.logger.info(f"opto on laps after deleting: {self.params['opto_on_lap']}")
        self.logger.info(f"opto off laps after deleting: {self.params['opto_off_lap']}")
        self.separate_laps()

    def separate_laps(self):
        lap_arrays = np.split(np.arange(np.shape(self.mean_activity)[0]), self.params['switch_lap'])
        self.params['env_laps'] = dict(zip(self.env, lap_arrays))

        if 'opto_on_lap' in self.params:

            ind = [bisect.bisect_right(self.params['switch_lap'], x) for x in self.params['opto_on_lap']]
            opto_in_env = np.array(self.env)[ind]
            self.params['opto_on_env'] = {}
            self.params['opto_after_max_length'] = {}  # max # of laps after opto off in opto and control envs

            for n, env in enumerate(opto_in_env):
                opto_off_lap = self.params['opto_off_lap'][n]
                opto_on_lap = self.params['opto_on_lap'][n]
                assert opto_off_lap in self.params['env_laps'][env]
                opto_laps = np.arange(opto_on_lap, opto_off_lap)
                self.params['opto_on_env'][env] = opto_laps

                # setup corresponding control
                control_env_name = f'control_{env[-10:]}'
                control_env_laps = self.params['env_laps'][f'control_{env[-4:]}']
                control_env_start = opto_on_lap - self.params['env_laps'][env][0] + control_env_laps[0]
                control_env_last = np.min((control_env_start + len(opto_laps), control_env_laps[-1]))
                control_laps = np.arange(control_env_last - len(opto_laps), control_env_last)
                self.params['opto_on_env'][control_env_name] = control_laps

                # max # of laps within env after turning off opto
                opto_env_max_lap = self.params['env_laps'][env][-1] - opto_off_lap
                control_env_max_lap = control_env_laps[-1] - control_laps[-1]
                max_n_lap = np.min((opto_env_max_lap, control_env_max_lap))
                self.params['opto_after_max_length'][env] = max_n_lap
                self.params['opto_after_max_length'][control_env_name] = max_n_lap

    def save_to_file(self):
        """ autosave var with savename"""

        data_path = os.path.join(self.path, f'{self.name}_data.pickle')
        self.separate_laps()
        with open(data_path, 'wb') as output_file:
            pickle.dump(self.__dict__, output_file, pickle.HIGHEST_PROTOCOL)

        self.logger.info(f'{self.name} parameters saved')


class PlaceCell:
    def __init__(self):

        self.thresh = {'minDF': 0.1, 'min_lap': 4, 'active_lap': 4, 'total_lap': 6}

    def set_pf_thresh(self, new_thresh: {}):
        self.thresh |= new_thresh

    def get_pf_features(self, data: LoadData, laps, cell, PF_loc_left, PF_loc_right, min_lap=4):

        features = dict(
            zip(['COM', 'emerge lap', 'ratio', 'out field ratio', 'out in ratio', 'adjusted ratio', 'peak amp',
                 'precision', 'slope', 'p', 'r2'], [np.nan] * 11))
        temp_pf = data.mean_activity[laps, PF_loc_left:PF_loc_right+1, cell]  # potential PF
        features['cell'] = cell

        temp_thresh = temp_pf > 0 # self.thresh['minDF']
        # ratio: total laps is the # laps
        firing_lap = np.sum(temp_thresh, axis=1)
        firing_lap_ind = np.where(firing_lap > self.thresh['minWidth'])[0]
        features['ratio'] = np.round(len(firing_lap_ind) / len(laps), 2)

        out_field = np.setdiff1d(np.arange(data.constants['nbins']), np.arange(PF_loc_left, PF_loc_right))
        out_field_F = np.mean(data.mean_activity[laps[:, np.newaxis], out_field, cell])
        out_field_lap = np.sum((data.mean_activity[laps[:, np.newaxis], out_field, cell] > self.thresh['minDF']),
                               axis=1)
        features['out field ratio'] = np.round((np.sum(out_field_lap > 0)) / len(laps), 2)

        temp_com = np.array([np.nan] * len(laps))

        if len(firing_lap_ind) >= min_lap:
            in_field_F = np.mean(temp_pf)
            features['out in ratio'] = np.round(out_field_F / in_field_F, 2)
            peak_lap = np.max(temp_pf[firing_lap_ind, :], axis=1)
            features['peak amp'] = np.round(np.mean(peak_lap), 2)

            # calculate COM
            width = np.shape(temp_pf)[1]
            temp_w = temp_pf * np.arange(width)  # activity * bin
            np.seterr(invalid='ignore')  # suppress zero divide error msg
            temp_com = np.sum(temp_w, axis=1) / np.sum(temp_pf, axis=1)  # com each lap
            features['precision'] = np.round(1 / (np.nanstd(temp_com[firing_lap_ind])), 2)  # non-zero division
            COM = np.nansum(temp_com[firing_lap_ind] * peak_lap) / np.nansum(peak_lap)  # com weighed by peak activity per lap
            features['COM'] = np.round(COM + PF_loc_left, 2)

            # backwards shifting
            slope, _, r, p, _ = linregress(firing_lap_ind, temp_com[firing_lap_ind])
            features['slope'] = np.round(slope, 2)
            features['p'] = np.round(-np.log10(p), 2)
            features['r2'] = np.round(r ** 2, 2)

            # find emerge lap (first reliable firing lap)
            for n in range(len(firing_lap_ind) - self.thresh['active_lap']):
                if firing_lap_ind[n + self.thresh['active_lap'] - 1] <= firing_lap_ind[n] + self.thresh['total_lap'] - 1:
                    features['emerge lap'] = firing_lap_ind[n]
                    firing_lap_after_emerge = firing_lap_ind[firing_lap_ind >= features['emerge lap']]
                    features['adjusted ratio'] = np.round(len(firing_lap_after_emerge) / (len(laps) - features['emerge lap']), 2)
                    temp_com[:features['emerge lap']] = np.nan
                    break

        return features, temp_com+PF_loc_left

    @staticmethod
    def _shuffle(single_cell: np.ndarray, nshuffle, nbins):
        """ shuffle to determine PF significance. keep the ISI structure of the original cell

        """
        all_transient = single_cell.flat

        # find structures of ISI and put continuous gaps and transients into different groups.
        # this aims to keep the same structure of calcium dynamics during shuffling
        group_bounds = np.where(np.diff(all_transient > 0) != 0)[0]
        group_bounds = np.insert(group_bounds + 1, [0, len(group_bounds)], [0, len(all_transient)])
        ngroups = len(group_bounds) - 1

        shuffle_cell = np.empty((nshuffle, np.shape(single_cell)[0], nbins))

        for m in range(nshuffle):
            ind0 = 0
            randind = np.random.choice(ngroups, ngroups, replace=False)
            single_shuffle = shuffle_cell[m]
            # build shuffle with structure
            for group_num in randind:
                group_start = group_bounds[group_num]
                group_end = group_bounds[group_num + 1]
                size = group_end - group_start
                single_shuffle.flat[ind0:ind0 + size] = all_transient[group_start:group_end]
                ind0 += size

        return np.moveaxis(shuffle_cell, 0, -1)

    def check_PF(self, data:LoadData, laps):
        pass

    @staticmethod
    def backwards_shifting_corr(cell_activity, firing_laps, min_corr=0.25):

        nlaps = np.shape(cell_activity)[0]
        width = np.shape(cell_activity)[1]

        lag_mat = np.zeros((nlaps, nlaps))
        lag_mat[:] = np.nan
        emerge_lag = np.copy(lag_mat)
        xcorr_mat = np.zeros((nlaps, nlaps))
        xcorr_mat[:] = np.nan

        # calculate correlation between laps only once
        for l0 in range(len(firing_laps) - 1):
            for l1 in range(l0 + 1, len(firing_laps)):
                corr = signal.correlate(cell_activity[firing_laps[l0], :], cell_activity[firing_laps[l1], :])
                if np.nanmax(corr) < min_corr:
                    continue

                xcorr_mat[firing_laps[l0], firing_laps[l1]] = np.nanmax(corr)
                corr_com = np.sum(corr * np.arange(len(corr))) / np.sum(corr) - width
                # print(corr_com, np.nanargmax(corr)-width)
                lag_mat[firing_laps[l0], firing_laps[l1]] = corr_com
                emerge_lag[firing_laps[l0] - firing_laps[0], firing_laps[l1] - firing_laps[0]] = corr_com

        return lag_mat, emerge_lag, xcorr_mat

    @staticmethod
    def plot_mean_place_map(data: LoadData, laps: np.ndarray, df: pd.DataFrame, title_name: str):

        cell_order = df.sort_values(by=['COM'])['cell'].unique()
        mean_field = np.nanmean(data.mean_activity[laps, :, cell_order[:, np.newaxis]], axis=1)
        sns.heatmap(mean_field, xticklabels=5, yticklabels=20)
        plt.xlabel('location on track')
        plt.ylabel('cells')
        plt.title(f'{data.name} {title_name}')
        plt.savefig(os.path.join(data.path, f'{data.name} {title_name}'))
        plt.show()

        return mean_field


class PlaceCellPeak(PlaceCell):
    def __init__(self):
        super().__init__()

        self.set_pf_thresh({'minDF': 0.1, 'minWidth': 2, 'maxWidth': 17, 'nshuffle': 600, 'pval': 0.01,
                            'bndry_thresh': 0.4, 'minRatio': 0.3, 'min_laps': 4})

    def check_PF(self, data: LoadData, laps):

        data.logger.info(f'{data.name} in laps {laps[0]} to {laps[-1]} PF thresh: \n'
                         f"{self.thresh['nshuffle']} shuffles, pval={self.thresh['pval']}, "
                         f" PF boundary/peak={self.thresh['bndry_thresh']}, PF width from {self.thresh['minWidth']} "
                         f"to {self.thresh['maxWidth']}, min % laps firing={self.thresh['minRatio']}, "
                         f"min peak amplitude = {self.thresh['minDF']}")

        cell_mean = np.nanmean(data.mean_activity[laps, :, :], axis=0)
        cell_peak = np.nanmax(cell_mean, axis=0)
        cell_trough = np.nanmin(cell_mean, axis=0)
        cell_bndry = cell_peak * self.thresh['bndry_thresh']

        temp_cells = np.where((cell_peak > self.thresh['minDF']) & (self.thresh['minDF'] > cell_trough))[0]

        PF_id = 0
        df_row = []
        # lag_cell = []
        # emerge_lag_cell = []
        # xcorr_cell = []
        bw_com = []

        for cell in temp_cells:

            # has contiguous region that fired across laps
            region = np.where(cell_mean[:, cell] > cell_bndry[cell])[0]  # continuous regions
            if len(region) < self.thresh['minWidth']:
                data.logger.info(f'{data.name}: cell {cell} failed PF width: {region}')
                continue

            shuffle_thresh = 10  # set an arbitrary large int
            pfs = self._find_pf_bndry(region, self.thresh['minWidth'])  # n potential PFs

            # loop through all potential PFs within cell
            for n in pfs:
                PF_loc_left = n[0]
                PF_loc_right = n[-1]

                if len(n) > self.thresh['maxWidth']:
                    data.logger.info(
                        f'{data.name}: cell {cell} {PF_loc_left} to {PF_loc_right}: {len(n)} width too large')
                    continue

                pf_features, com = self.get_pf_features(data, laps, cell, PF_loc_left, PF_loc_right, self.thresh['min_lap'])

                if pf_features['adjusted ratio'] < self.thresh['minRatio'] or np.isnan(pf_features['adjusted ratio']):
                    data.logger.info(
                        f'{data.name}: cell {cell} {PF_loc_left} to {PF_loc_right}: {pf_features} failed thresh')
                    continue

                # shuffle only once per cell
                if shuffle_thresh == 10:  # and mean_peak < shuffle_thresh:
                    shuffle_cell = self._shuffle(data.mean_activity[laps, :, cell], self.thresh['nshuffle'], data.constants['nbins'])
                    shuffle_mean = np.nanmean(shuffle_cell, axis=0)
                    shuffle_peak = np.nanmax(shuffle_mean, axis=0)
                    shuffle_trough = np.nanmin(shuffle_mean, axis=0)
                    shuffle_thresh = np.quantile(shuffle_peak, 1 - self.thresh['pval'])  # update shuffle_thresh value
                    lower_thresh = np.quantile(shuffle_trough, self.thresh['pval'])

                # pass the shuffle
                if (cell_peak[cell] > shuffle_thresh) and (cell_trough[cell] <= lower_thresh):
                    data.logger.info(f'{data.name}: cell {cell} {PF_loc_left} to {PF_loc_right}: passed!')
                    # lag_mat, emerge_lag, xcorr_mat = self.backwards_shifting_corr(data.mean_activity[laps, :, cell], np.where(com>0)[0])

                    pf_features |= {'PF id': PF_id, 'left': PF_loc_left, 'right': PF_loc_right}
                    df_row.append(pf_features)
                    bw_com.append(com)
                    # lag_cell.append(lag_mat)
                    # emerge_lag_cell.append(emerge_lag)
                    # xcorr_cell.append(xcorr_mat)
                    PF_id = PF_id + 1
                else:
                    data.logger.info(f'{data.name}: cell {cell} {PF_loc_left} to {PF_loc_right}: '
                                     f'failed by shuffle sig {cell_peak[cell]} < {shuffle_thresh} or {cell_trough[cell]} > {lower_thresh}')

        data.logger.info(f'{data.name} in laps {laps[0]} to {laps[-1]}: {len(df_row)} place fields')

        df = pd.DataFrame(df_row)
        # bw_shift_corr = dict(zip(['lag', 'emerge lag', 'xcorr'], [np.array(var) for var in [lag_cell, emerge_lag_cell, xcorr_cell]]))

        # plot
        if len(df) > 0:
            self.plot_mean_place_map(data, laps, df, f'lap {laps[0]} to {laps[-1]}')
            return df[
                ['cell', 'PF id', 'left', 'right', 'COM', 'emerge lap', 'ratio', 'out field ratio', 'out in ratio',
                 'adjusted ratio', 'peak amp', 'precision', 'slope', 'p', 'r2']], bw_com
        else:
            print(f'warning: no PFs found')
            return [], []

    @staticmethod
    def _find_pf_bndry(ind, minwidth) -> [np.ndarray]:
        """ find continuous region longer than minwidth above zero

        :param ind: np array to check for continuous regions longer minwidth (indices where smooth_mean >0)
        :param minwidth: min width of place field, check if the continuous region is larger than min PF
        :return: np array with group identity of continuous region longer than minwidth above zero
        """
        diff = np.diff(ind)  # Calculate the differences between consecutive elements
        split_indices = np.where(diff > 1)[0]  # Find indices where the difference is greater than 1
        groups = np.split(ind, split_indices + 1)  # Split the array into subarrays
        temp_pf = [bndry for bndry in groups if len(bndry) > minwidth]
        return temp_pf




def opto_ttest(data, laps_on, laps_off, left_bndry, right_bndry, cell):

    on_data = data[laps_on, slice(left_bndry, right_bndry), cell].flatten()
    off_data = data[laps_off, slice(left_bndry, right_bndry), cell].flatten()

    s, p = stats.ttest_ind(on_data, off_data)

    on_firing = np.sum(on_data)
    off_firing = np.sum(off_data)

    if on_firing > off_firing:
        post_hoc = 'on'
    else:
        post_hoc = 'off'

    return p, post_hoc


def opto_tuning_place_cell(mouse: LoadData, PF_df: pd.DataFrame, alpha = 0.05):

    data = mouse.mean_activity
    mouse_df = PF_df.loc[PF_df['mouse'] == mouse.name]
    tuning_summary = []

    for env in mouse.params['opto_on_env']:
        print(env)
        on_laps = mouse.params['opto_on_env'][env]
        real_env = map_string(env)
        last_lap_env = mouse.params['env_laps'][real_env][-1]
        last_lap = np.min((last_lap_env, on_laps[-1]+1+len(on_laps)))   # make sure last lap of off_laps is in the same env
        off_laps = np.arange(on_laps[-1]+1, last_lap)

        df = mouse_df.loc[mouse_df['env'] == real_env]
        ncells = len(df)
        p = df.apply(lambda row: opto_ttest(data, on_laps, off_laps, row['left'], row['right'], row['cell']),
                     axis=1, result_type='expand')
        p.rename(columns={0: 'p', 1: 'more firing'}, inplace=True)
        adjusted_alpha = 1 - (1 - alpha) ** (1 / ncells)
        sig = p['p'] < adjusted_alpha
        p['sig'] = np.where(sig, 'Yes', 'No')
        p['opto_env'] = env

        env_df = pd.concat([df[['mouse', 'env', 'cell', 'PF id', 'left', 'right', 'COM', 'emerge lap', 'ratio',
                                'out field ratio', 'out in ratio', 'adjusted ratio', 'peak amp', 'precision']], p], axis=1)
        tuning_summary.append(env_df)

    return pd.concat(tuning_summary).reset_index(drop=True)


def opto_tuning_place_cell_ratio(df: pd.DataFrame):

    total = df.groupby('opto_env')['sig'].count().reset_index()
    total.rename(columns={'sig': 'total PFs'}, inplace=True)

    off = df.loc[(df['sig'] == 'Yes') & (df['more firing'] == 'off')].groupby('opto_env')['sig'].count().reset_index()
    off.rename(columns={'sig': 'more firing off'}, inplace=True)
    total = total.merge(off, how='left')

    on = df.loc[(df['sig'] == 'Yes') & (df['more firing'] == 'on')].groupby('opto_env')['sig'].count().reset_index()
    on.rename(columns={'sig': 'more firing on'}, inplace=True)
    total = total.merge(on, how='left')

    total['on %'] = total['more firing on'] / total['total PFs'] * 100
    total['off %'] = total['more firing off'] / total['total PFs'] * 100
    total.fillna(0, inplace=True)

    total_long = pd.melt(total, id_vars=['opto_env'], value_vars=['on %', 'off %'], var_name='opto effect', value_name='%')
    total_long['cond'] = total_long['opto_env'].apply(lambda row: row.split('_')[0])
    total_long['env'] = total_long['opto_env'].apply(lambda row: '_'.join(row.split('_')[1:3]))
    sns.catplot(data=total_long, x="opto effect", y="%", col="env", hue="cond", kind="point")
    plt.show()

    return total


class StatsPF:
    @staticmethod
    def opto_tuning(tuning_on, tuning_off):
        pass

    @staticmethod
    def place_field_feature(df1, df2, feature_col):
        pass


class Pipeline:
    def __init__(self, group: str):
        # mouse, env, day
        self.group = group
        self.path = os.path.join('D:\\Opto\\Analysis', group)
        if not os.path.exists(self.path):
            print(f'creating folder at path: {self.path}')
            os.mkdir(self.path)

        # initialize
        self.place_cell_handler = PlaceCellPeak()

    def find_PF(self, mice: [LoadData], max_lap=40):
        combined_df = []
        bw_com = []

        #PF_save_path = os.path.join(self.path, f'{self.group}_PFs.parquet')
        #if os.path.exists(PF_save_path):
        #    return pd.read_parquet(PF_save_path, engine='fastparquet')

        for mouse in mice:
            for env, laps in mouse.params['env_laps'].items():
                df, com = self.place_cell_handler.check_PF(mouse, laps)
                df['mouse'] = mouse.name
                df['day'] = env[-4:]
                df['opto'] = env[:-5]
                df['env'] = env
                df.to_csv(os.path.join(mouse.path, f'{mouse.name}_PFs_{env}.csv'), index=False)
                combined_df.append(df)
                bw_com.extend(com[:, :max_lap].tolist())

        combined_df = pd.concat(combined_df).reset_index(drop=True)
        cols = combined_df.columns.tolist()
        cols = cols[-4:] + cols[:-4]
        combined_df = combined_df[cols]
        combined_df = pd.concat([combined_df, pd.DataFrame(np.array(bw_com), columns=[f'{n}' for n in np.arange(max_lap)])], axis=1)

        combined_df.to_parquet(PF_save_path, compression='gzip')  # save to parquet

        return combined_df

    def reliability(self, mice: [LoadData], df: pd.DataFrame, first_laps = 40):

        reliab_summary = []

        for mouse in mice:
            mouse_df = df.loc[df['mouse']==mouse.name]
            for env, laps in mouse.params['env_laps'].items():
                cells = mouse_df.loc[mouse_df['env']==env]['cell'].unique()
                data = mouse.mean_activity[slice(laps[0],laps[0]+first_laps), :, cells]
                mean_tuning = np.mean(data, axis=0)
                reliab = np.zeros((len(cells), first_laps))
                for n in range(len(cells)):
                    reliab[n, :] = [scipy.stats.pearsonr(mean_tuning[:, n], data[l, :, n])[0] for l in range(first_laps)]
                local_reliab = pd.DataFrame(reliab, columns = [f'{n}' for n in range(first_laps)])
                local_reliab['env'] = env
                local_reliab['cell'] = cells
                local_reliab['mouse'] = mouse.name
                reliab_summary.append(local_reliab)

        reliab_summary = pd.concat(reliab_summary, axis=0).reset_index(drop=True)

        #reliab_summary_long = reliab_summary.melt()


        return reliab_summary

    @staticmethod
    def backwards_shifting_emerge(df: pd.DataFrame, anchor_lap = 15):
        max_lap = int(df.columns[-1])+1
        com_emerge, final_location = Pipeline.align_to_emerge_lap(df.iloc[:, -max_lap:].to_numpy(), anchor_lap)
        com_emerge = com_emerge - final_location
        bw_emerge = df.copy()
        bw_emerge.iloc[:, -max_lap:] = com_emerge
        bw_emerge = bw_emerge.loc[bw_emerge['emerge lap'] <= int(np.ceil(max_lap/2))]

        bw_emerge_long = pd.melt(bw_emerge, id_vars=['mouse', 'env'], value_vars=[f'{n}' for n in np.arange(max_lap)],
                                 var_name='lap', value_name='shift')
        bw_emerge_long.dropna(subset=['shift'], inplace=True)
        bw_emerge_long['day'] = bw_emerge_long['env'].apply(lambda x: x[-4:])
        bw_emerge_long['opto'] = bw_emerge_long['env'].apply(lambda x: x[:-5])

        sns.relplot(data=bw_emerge_long, x='lap', y='shift', hue='opto', col='day', kind='line')
        plt.show()

        sns.relplot(data=bw_emerge_long, x='lap', y='shift', hue='opto', col='day', row='mouse', kind='line')
        plt.show()

        return bw_emerge_long

    @staticmethod
    def align_to_emerge_lap(arr, stop_moving_lap = 15):
        # Find the indices of the first non-nan values in each row
        mask = ~np.isnan(arr)
        first_non_nan_indices = np.argmax(mask, axis=1)

        # Create an array to store the result
        result = np.empty_like(arr)
        length = arr.shape[1]
        final_location = np.zeros((arr.shape[0], 1))

        for i in range(arr.shape[0]):
            start_index = first_non_nan_indices[i]
            result[i, :length - start_index] = arr[i, start_index:]
            result[i, length - start_index::] = np.nan
            final_location[i] = np.nanmedian(result[i, stop_moving_lap:])

        return result, final_location

    def opto_pf_features(self, mice: [LoadData], df: pd.DataFrame, min_lap: int=5):

        PF_save_path = os.path.join(self.path, f'{self.group}_PFs_opto.parquet')
        #if os.path.exists(PF_save_path):
        #    return pd.read_parquet(PF_save_path, engine='fastparquet')

        opto = []

        for mouse in mice:
            for env, laps in mouse.params['opto_on_env'].items():

                condition = env.split('_')[0]   # eg. 'control' or 'opto'
                day = '_'.join(env.split('_')[-2:])     # eg. 'later_day1' or 'first_day2'

                raw_env = map_string(env)   # eg. 'opto_first_day1', 'control_first_day1' => 'opto_first_day1', 'control_day1'

                mouse_df = df.loc[(df['mouse'] == mouse.name) & (df['env']==raw_env)]
                mouse_df = mouse_df[['cell', 'left', 'right', 'emerge lap', 'PF id']].to_numpy()

                # setup same # of laps before and after as opto on laps
                after_length = mouse.params['opto_after_max_length'][env]
                if after_length > min_lap:
                    after_laps = np.arange(laps[-1]+1, laps[-1]+1+np.min((after_length, len(laps))))
                    #after_laps = np.arange(laps[-1]+1, mouse.params['env_laps'][raw_env][-1])
                else:
                    after_laps = None
                first_lap = laps[0]-len(laps)

                # print(mouse.name, env, after_laps)

                for row in mouse_df:
                    features, _ = self.place_cell_handler.get_pf_features(mouse, laps, row[0], row[1], row[2])
                    opto = add_features_list_of_dict(opto, features, {'mouse': mouse.name, 'stage': 'on', 'env': day,
                                                                           'condition': condition, 'id': row[4]})

                    if after_laps is not None:
                        #print(mouse.name, env, after_laps, row)
                        features, _ = self.place_cell_handler.get_pf_features(mouse, after_laps, row[0], row[1], row[2])
                        opto = add_features_list_of_dict(opto, features,
                                                              {'mouse': mouse.name, 'stage': 'after', 'env': day,
                                                               'condition': condition, 'id': row[4]})

                    begin_lap = np.max((row[3]+mouse.params['env_laps'][raw_env][0], first_lap))
                    if begin_lap < laps[0] - min_lap:
                        features, _ = self.place_cell_handler.get_pf_features(mouse, np.arange(begin_lap, laps[0]), row[0], row[1], row[2])
                        opto = add_features_list_of_dict(opto, features,
                                                              {'mouse': mouse.name, 'stage': 'before', 'env': day,
                                                               'condition': condition, 'id': row[4]})

        opto_df = pd.DataFrame(opto)
        cols = opto_df.columns.tolist()
        cols = cols[-5:] + cols[:-5]

        #opto_df[cols].to_parquet(PF_save_path, compression='gzip')  # save to parquet

        return opto_df[cols]

    @staticmethod
    def plot_normalized_precision(df: pd.DataFrame):

        feature_df = df.copy()
        feature_df.dropna(subset=['precision'], inplace=True)

        # Create a mask for rows where 'stage' is 'before'
        mask = (feature_df['env'] == 'later_day1') & (feature_df['stage'] == 'before')

        # Merge the DataFrame with itself on 'name', 'cell', and 'env' columns when 'stage' is 'before'
        feature_df = feature_df.merge(feature_df[mask][['mouse', 'id', 'env', 'condition', 'precision']],
                                      on=['mouse', 'id', 'env', 'condition'], suffixes=('', '_before'), how='left')

        # Calculate the normalized 'precision' by dividing the original 'precision' by the 'precision_before' values
        feature_df['precision_normalized'] = feature_df['precision'] / feature_df['precision_before']

        sns.pointplot(data=feature_df, x='stage', y='precision', order=['before', 'on', 'after'],
                      hue='condition', dodge=True)
        plt.show()

        sns.pointplot(data=feature_df, x='stage', y='precision_normalized', order=['before', 'on', 'after'],
                      hue='condition', dodge=True)
        plt.show()

        sns.catplot(data=feature_df, x='stage', y='precision_normalized', order=['before', 'on', 'after'],
                    hue='condition', dodge=True, row='mouse', kind='point')
        plt.show()

        return feature_df

    def same_cells_cross_days(self, df: pd.DataFrame, threshold=4):

        col_names = df.columns.tolist()
        bw_col = col_names.index('0')
        features_df = df.iloc[:, :bw_col].copy()

        # Merge the dataframes on 'cell' and 'mouse' columns
        merged_df = pd.merge(features_df.loc[features_df['day']=='day1'], features_df.loc[features_df['day']=='day2'],
                             on=['cell', 'mouse', 'opto'], suffixes=('_day1', ''), how='outer')

        # Calculate the absolute difference between 'COM' values
        merged_df['COM_diff'] = abs(merged_df['COM_day1'] - merged_df['COM'])

        # Filter rows based on the threshold
        result_df = merged_df[merged_df['COM_diff'] <= threshold]

        # Drop the 'COM_diff' column if not needed
        col_del = [string for string in result_df.columns.tolist() if '_day1' in string]
        col_del.append('COM_diff')
        same_df = result_df.drop(columns=col_del).reset_index(drop=True)

        diff_df = pd.concat([features_df.loc[features_df['day']=='day2'], same_df])
        diff_df.drop_duplicates(subset=['env', 'mouse', 'cell', 'PF id'], keep=False, inplace=True, ignore_index=True)

        same_df['day2_PF'] = 'same'
        diff_df['day2_PF'] = 'new'
        day2_df = pd.concat([same_df, diff_df]).reset_index(drop=True)

        day2_emerge = emerge_cumcount(day2_df, 'emerge lap', ['day', 'opto', 'day2_PF'], 'day2_PF')

        return day2_df, day2_emerge

    def cross_day_stability(self, mice: [LoadData], df: pd.DataFrame, max_lap = 40):

        # cell_count = df.groupby(['mouse', 'opto', 'cell'])['day'].nunique().reset_index()
        # same_cells = cell_count[cell_count['day'] > 1]
        # same_cells = same_cells.drop(columns='day').reset_index(drop=True)
        # same_df = same_cells.merge(df[['mouse', 'opto', 'day', 'env', 'cell', 'COM', 'PF id']], on=['mouse', 'opto', 'cell'], how='left')
        corr_df = []

        for mouse in mice:
            mouse_df = df.loc[df['mouse'] == mouse.name]
            for env in mouse_df['opto'].unique():
                mouse_env_df = mouse_df.loc[mouse_df['env']==f'{env}_day1'].drop_duplicates(subset=['cell']).sort_values(by=['COM'])[['mouse', 'opto', 'cell', 'COM', 'PF id']].reset_index()
                sorted_PF = self.place_cell_handler.plot_mean_place_map(mouse, mouse.params['env_laps'][f'{env}_day1'][:max_lap],
                                                                        mouse_env_df, f'{env} day1')
                sorted_PF_day2 = self.place_cell_handler.plot_mean_place_map(mouse, mouse.params['env_laps'][f'{env}_day2'][:max_lap],
                                                                        mouse_env_df, f'{env} day1 PFs in day2')

                df1 = pd.DataFrame(sorted_PF)
                df2 = pd.DataFrame(sorted_PF_day2)
                cell_corr = df1.corrwith(df2, axis=1)
                mouse_env_df['overday corr'] = cell_corr
                mouse_env_df = mouse_env_df.merge(mouse_df.loc[mouse_df['env']==f'{env}_day2'][['mouse', 'opto', 'cell', 'COM']].drop_duplicates(subset=['cell']), on=['mouse', 'opto', 'cell'], how='left', suffixes=('_day1', '_day2'))
                #print(mouse.name, env, mouse_env_df, len(sorted_PF))
                corr_df.append(mouse_env_df)

        return pd.concat(corr_df)


def add_features_list_of_dict(list_of_dict: [], original_dict: {}, new_dict: {}):
    original_dict |= new_dict
    list_of_dict.append(original_dict)
    return list_of_dict


def calculate_spatial_information(data):
    """
    Calculate spatial information for each cell.

    Parameters:
    data (np.array): 3D numpy array with dimensions (trial, location, cell)

    Returns:
    np.array: 1D array with spatial information for each cell
    """
    # Calculate mean firing rate for each cell in each location
    mean_firing_rate = np.nanmean(data, axis=0)  # Shape: (location, cell)

    # Calculate overall mean firing rate for each cell
    overall_mean_firing_rate = np.nanmean(mean_firing_rate, axis=0)  # Shape: (cell,)

    # Avoid division by zero
    overall_mean_firing_rate += np.finfo(float).eps
    mean_firing_rate += np.finfo(float).eps

    # Calculate occupancy probability for each location
    num_locations = mean_firing_rate.shape[0]
    occupancy_prob = 1 / num_locations  # Assuming uniform occupancy

    # Calculate spatial information for each cell
    normalized_firing_rate = mean_firing_rate / overall_mean_firing_rate  # Shape: (location, cell)
    log_term = np.log2(normalized_firing_rate)  # Shape: (location, cell)
    spatial_information = np.sum(normalized_firing_rate * log_term, axis=0) * occupancy_prob  # Shape: (cell,)

    return spatial_information

