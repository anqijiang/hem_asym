import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from ast import literal_eval
#from opto_analysis.plotting import cell_heatmap, cell_mean_over_laps, cell_mean_over_laps_opto_compare
import seaborn as sns
from os import listdir
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



def load_py_var_mat(day_path, keywords):
    """ find the matlab file under day_path directory"""
    onlyfiles = [f for f in listdir(day_path) if os.path.isfile(os.path.join(day_path, f))]
    file_name = [f for f in onlyfiles if f.endswith(keywords)][0]
    print('loading file: ', file_name)
    file = scipy.io.loadmat(os.path.join(day_path, file_name))

    return file

def init_logger(path, name):
    logging.basicConfig(filename=os.path.join(path, f'{name}_PF_params.log'),
                        format='%(asctime)s %(message)s', datefmt='%m/%d/%y %I:%M %p', filemode='a', force=True)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    return logger


class PF_analysis:
    """
    this class takes inputs from a folder that contains the MATLAB output of mean_activity
     (3d ndarray laps * location on track * cell), checks cache, and uses two methods
    to identify place fields, and organize them into corresponding pandas dataframes.
    """

    def __init__(self, mouse, env, day):

        self.mouse = mouse
        self.env = env
        self.day = day
        self.save_path = os.path.join('D:\\Opto\\Analysis', mouse, day)
        logger = init_logger(self.save_path, self.mouse)

        # check for cache in the directory first
        pickle_file = os.path.join(self.save_path, f'{self.mouse}_PF.pickle')
        if os.path.exists(pickle_file):
            print(f'Loading {self.mouse} from cache: {pickle_file}')
            self.load()
        else:
            print(f'no cache for {mouse}')

            # load experimental conditions from matlab file
            mat_file = load_py_var_mat(self.save_path, 'align_cell_mean.mat')
            mean_activity = mat_file['cell_binMean'].transpose((1, 0, 2))
            mean_activity[np.isnan(mean_activity)] = 0
            self.mean_activity = mean_activity
            self.raw = np.copy(self.mean_activity)

            # basics of mean_activity
            self.ncell = self.mean_activity.shape[2]
            self.nbins = self.mean_activity.shape[1]
            self.nlaps = self.mean_activity.shape[0]

            # load env switch lap
            if mat_file['env_switch_lap'].shape[1] == 0:  # if empty 'env_switch_lap'
                self.switch_lap = []
            else:
                self.switch_lap = mat_file['env_switch_lap'][:, 0].astype('int') -1
            self.switch_lap = np.concatenate(([0], self.switch_lap))  # add 0 as the first switch

            # match env with laps (eg. laps = {env: laps})
            assert len(self.env) == len(self.switch_lap), "# of env does not match with laps"
            if len(self.env) > 1:
                lap_arrays = np.split(np.arange(self.nlaps), self.switch_lap)[1::]
                self.laps = dict(zip(self.env, lap_arrays))
            else:
                self.laps = {self.env[0]: np.arange(self.nlaps)}

            # load additional experimental conditions if they exist
            if all(var in mat_file.keys() for var in ['opto_off_lap', 'opto_on_lap']):
                opto_off_lap = mat_file['opto_off_lap'][:, 0].astype('int') -1
                opto_on_lap = np.fmax(mat_file['opto_on_lap'][:, 0].astype('int') -1, 0)
                opto_length = opto_off_lap - opto_on_lap
                self.opto_length = opto_length[opto_length>1]

                # remove artifact
                self.opto_on_lap = opto_on_lap[opto_length>1]
                self.opto_off_lap = opto_off_lap[opto_length>1]

                # identify opto_on laps in each env
                ind = [bisect.bisect_right(self.switch_lap[1:], x) for x in self.opto_on_lap]  # identify env opto is on
                l = [np.arange(self.opto_on_lap[x], self.opto_off_lap[x]) for x in range(len(self.opto_on_lap))]
                opto_in_env = list(np.array(self.env)[ind])
                self.opto_env = dict(zip(opto_in_env, l))    # {env: all opto on laps in env}
                self.opto_compare = {}

            self.PFs = {}
            self.nPF = {}
            self.COM_shift = {}
            self.PF_summary_peak = pd.DataFrame()
            self.PF_summary_opto = pd.DataFrame()

            # check if mouse behavior is consistent between laps
            self.deleted_lap = []
            if 'pause_lap' in mat_file.keys():
                pause_lap = mat_file['pause_lap']
                if len(pause_lap) > 0:
                    beh_trace = mat_file['E']
                    pause_lap = pause_lap[0, :].astype('int') -1  # convert matlab indexing to python indexing
                    print(f'pause laps in {self.mouse}: {pause_lap}')
                    laps = input('type "T" to confirm deleting above laps, "F" to keep original, or input other laps to delete: \n')
                    if laps == 'T' or laps.upper() == 'T':
                        logger.info(f'{self.mouse} delete pause laps {pause_lap}')
                        self.delete_laps(pause_lap)

    def __repr__(self):
        return 'Data from {} on {} in env {}'.format(self.mouse, self.day, self.env)

    def delete_laps(self, laps):
        logger = init_logger(self.save_path, self.mouse)

        print(f'{self.mouse} deleting laps {laps}')
        self.deleted_lap.append(laps)
        self.mean_activity = np.delete(self.mean_activity, laps, 0)
        self.nlaps = self.mean_activity.shape[0]

        # find envs affected
        ind = [bisect.bisect_right(self.switch_lap[1::], l) for l in laps]
        env = np.array(self.env)
        envs_affected = list(env[(np.unique(ind))])
        logger.info(f'{self.mouse} laps in envs affected: {envs_affected}, may need re-analysis')
        print(f'{self.mouse} laps in envs affected: {envs_affected}, may need re-analysis')

        # adjust for deleted laps
        print('switch laps before deleting pause laps: ', self.switch_lap)
        for n in range(len(self.switch_lap)):
            self.switch_lap[n] = self.switch_lap[n] - np.sum(self.switch_lap[n] > laps)
        lap_arrays = np.split(np.arange(self.nlaps), self.switch_lap[1::])
        self.laps = dict(zip(self.env, lap_arrays))
        logger.info(f'{self.mouse} deleting laps {laps}')
        print('switch laps after deleting pause laps: ', self.switch_lap)

        print(f'{self.mouse} opto on/off laps before deleting: {self.opto_env}')
        for n in range(len(self.opto_on_lap)):
            self.opto_off_lap[n] = self.opto_off_lap[n] - np.sum(self.opto_off_lap[n] > laps)
            self.opto_on_lap[n] = self.opto_on_lap[n] - np.sum(self.opto_on_lap[n] > laps)
        ind = [bisect.bisect_right(self.switch_lap[1:], x) for x in self.opto_on_lap]  # identify env opto is on
        l = [np.arange(self.opto_on_lap[x], self.opto_off_lap[x]) for x in range(len(self.opto_on_lap))]
        opto_in_env = list(np.array(self.env)[ind])
        self.opto_env = dict(zip(opto_in_env, l))  # {env: all opto on laps in env}
        print(f'{self.mouse} opto on/off laps after deleting: {self.opto_env}')

    def delete_laps_env_rerun(self, env):
        logger = init_logger(self.save_path, self.mouse)

        if len(self.PF_summary_peak.loc[self.PF_summary_peak['env'] == env]) > 0:
            print(f'{self.mouse} delete and rerun PF analysis in {env} after deleting laps')
            logger.info(f'{self.mouse} delete and rerun PF analysis in {env}')
            self.PF_summary_peak.drop(self.PF_summary_peak[self.PF_summary_peak['env'] == env].index, inplace=True)
            self.check_PF_peak(env)
        #if len(self.PF_summary_opto) > 0:

    @staticmethod
    def _group_id(ind, minwidth):
        """ find continuous region longer than minwidth above zero

        :param ind: np array to check for continuous regions longer minwidth (indices where smooth_mean >0)
        :param minwidth: min width of place field, check if the continuous region is larger than min PF
        :return: np array with group identity of continuous region longer than minwidth above zero
        """
        minbins = int(np.ceil(minwidth / 5))
        indlist = np.insert(ind, 0, ind[0] - 1)
        jumpind = np.where(np.diff(indlist) > 1)[0]  # where ind is not continuous
        jumplist = np.insert(jumpind, [0, len(jumpind)], [0, len(ind)])

        group_bounds = []
        for n in range(len(jumpind) + 1):
            ind1 = jumplist[n + 1] - 1
            ind0 = jumplist[n]
            width = ind[ind1] - ind[ind0]
            if width >= minbins:
                group_bounds.append((ind[ind0], ind[ind1] + 1))

        return group_bounds

    def shuffle(self, single_cell, nshuffle):
        """ shuffle to determine PF significance. keep the ISI structure of the original cell

        """
        nlaps = np.shape(single_cell)[0]
        nbins = np.shape(single_cell)[1]
        all_transient = single_cell.flat

        # find structures of ISI and put continuous gaps and transients into different groups.
        # this aims to keep the same structure of calcium dynamics during shuffling
        group_bounds = np.where(np.diff(all_transient > 0) != 0)[0]
        group_bounds = np.insert(group_bounds + 1, [0, len(group_bounds)], [0, len(all_transient)])
        ngroups = len(group_bounds) - 1

        shuffle_cell = np.empty((nshuffle, nlaps, nbins))

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

    def check_PF_peak(self, PF_env, laps = None, nshuffle=600, pval=0.01, bndry_thresh = 0.4, maxWidth = 17, minWidth = 2,
                      minRatio = 0.3, minDF = 0.05, min_laps = 4, opto_check = 0):
        """ determine PF identity by peak method (Grijseels, 2021)

        :param bndry_thresh: min percentage of maximum activity to be considered as part of PF
        :param PF_env: check PF within which env, preferably use the same env as class structure inputs
        :param laps: laps to examine PFs in, default uses the laps associated with envs in dict(self.laps)
        :return: dataframe with only significant PF
        """

        if laps is None:
            laps = self.laps[PF_env]

        # check if already ran in this env or not
        if not self.PF_summary_peak.empty and opto_check == 0:
            exist_env = self.PF_summary_peak.env.unique()
            if sum(PF_env == exist_env) > 0:
                print(f"has already checked {self.mouse} PF in {PF_env}")
                self.remapping(self.PFs[PF_env], laps[0], laps[-1])
                return self.PF_summary_peak

        logger = init_logger(self.save_path, self.mouse)
        logger.info(f'{self.mouse} in {PF_env}: {nshuffle} shuffles, pval={pval}, bndry_thresh={bndry_thresh}, '
                     f'PF width {minWidth} to {maxWidth}, minRatio={minRatio}, minDF = {minDF}, opto_check = {opto_check}')

        cell_mean = np.nanmean(self.mean_activity[laps, :, :], axis=0)
        cell_peak = np.nanmax(cell_mean, axis=0)
        cell_bndry = cell_peak * bndry_thresh
        nlaps = len(laps)
        PF_features = [[]]
        bw_shift = [[]]
        PF_id = 0

        for cell in range(self.ncell):
            if np.max(cell_mean[:, cell]) < minDF:
                continue

            # zscore transform the heatmap (normalize lap by lap activity)
            zmat = stats.zscore(self.mean_activity[laps, :, cell], axis=1, nan_policy='omit')
            replace_num = np.nanmin(zmat)
            transmat = np.nan_to_num(zmat, nan=replace_num)
            zpeak = np.mean(transmat, axis=0)
            if np.max(zpeak) < 0:
                logger.info(f'{self.mouse} in {PF_env}: {cell} failed zscore {np.max(zpeak)}')
                continue

            # has contiguous region that fired across laps
            region = np.where(cell_mean[:, cell] > cell_bndry[cell])[0]  # continuous regions
            if len(region)<minWidth:
                logger.info(f'{self.mouse} in {PF_env}: {cell} failed PF width: {region}')
                continue

            shuffle_thresh = 10
            bndry = self._group_id(region, minWidth)  # n potential PFs
            for n in range(len(bndry)):
                PF_loc_left = bndry[n][0]
                PF_loc_right = bndry[n][1]
                temp_pf = self.mean_activity[laps, PF_loc_left:PF_loc_right, cell]  # potential PF
                temp_thresh = temp_pf > np.max((cell_bndry[cell], minDF))
                # ratio: total laps is the # laps
                firing_lap = np.sum(temp_thresh, axis=1)
                firing_lap_ind = np.where(firing_lap >= minWidth)[0]

                if PF_loc_right - PF_loc_left > maxWidth:
                    logger.info(f'{self.mouse} in {PF_env}: {cell} {PF_loc_left} to {PF_loc_right}: width too large')
                    continue
                if len(firing_lap_ind) < min_laps:  # fire at least min_laps times over all laps
                    logger.info(f'{self.mouse} in {PF_env}: {cell} {PF_loc_left} to {PF_loc_right}: only fired {len(firing_lap_ind)} laps')
                    continue

                emerge_lap = self._first_lap_PF(firing_lap_ind)
                if np.isnan(emerge_lap):   # fire consistently at least at some point
                    logger.info(f'{self.mouse} in {PF_env}: {cell} {PF_loc_left} to {PF_loc_right}: not fire consistently for emerge lap')
                    continue

                firing_lap_ind = firing_lap_ind[firing_lap_ind>=emerge_lap]
                ratio_temp = len(firing_lap_ind) / (nlaps - emerge_lap)
                if ratio_temp < minRatio:  # fire at least minRatio% laps after it's become a PF
                    logger.info(f'{self.mouse} in {PF_env}: {cell} {PF_loc_left} to {PF_loc_right}: '
                                f'not fire enough laps by ratio, {len(firing_lap_ind)} out of {nlaps}')
                    continue

                mean_peak = np.max(cell_mean[PF_loc_left:PF_loc_right, cell])
                # shuffle only once per cell
                if (shuffle_thresh == 10) & (mean_peak < shuffle_thresh):
                    # proceed to shuffle
                    # print(f'{self.mouse}: shuffle cell {cell}, {PF_loc_left} to {PF_loc_right}')
                    shuffle_cell = self.shuffle(self.mean_activity[laps, :, cell], nshuffle)
                    shuffle_mean = np.nanmean(shuffle_cell, axis=0)
                    shuffle_peak = np.nanmax(shuffle_mean, axis=0)
                    shuffle_thresh = np.quantile(shuffle_peak, 1 - pval)  # update shuffle_thresh value

                # pass the shuffle
                if mean_peak > shuffle_thresh:
                    in_field_F = np.mean(temp_pf)
                    out_field = np.setdiff1d(np.arange(self.nbins),
                                             np.arange(PF_loc_left, PF_loc_right))
                    out_field_F = np.mean(self.mean_activity[laps[:, np.newaxis], out_field, cell])
                    out_in_ratio = np.round(out_field_F / in_field_F, 2)
                    COM, width, precision, slope, r2, p, com_by_laps = self._PF_features(temp_pf, PF_loc_left, firing_lap_ind)
                    PF_features.append(
                        [PF_env, cell, PF_id, emerge_lap, COM, PF_loc_left, PF_loc_right,
                         np.round(ratio_temp, 2), precision, out_in_ratio, slope, r2, p])
                    bw_shift.append(com_by_laps)
                    PF_id = PF_id + 1
                else:
                    logger.info(f'{self.mouse} in {PF_env}: {cell} {PF_loc_left} to {PF_loc_right}: '
                                f'failed by shuffle sig {mean_peak} < {shuffle_thresh}')

        PF_array = np.array(PF_features[1:][:])
        if len(PF_array) == 0:
            print(f'{self.mouse} has no PFs in {PF_env}')
            logger.info(f'{self.mouse} has no PFs in {PF_env}')
            return pd.DataFrame()
        else:
            PF_summary_peak_local = pd.DataFrame(PF_array, columns=['env', 'cell', 'PF id', 'emerge lap', 'COM', 'left',
                                                                    'right', 'ratio', 'precision', 'out in ratio',
                                                                    'slope', 'r2','-log(p)'])
            PF_summary_peak_local = PF_summary_peak_local.astype(dict(zip(['ratio', 'precision', 'out in ratio', 'slope',
                                                                           'r2', '-log(p)', 'COM', 'cell','emerge lap',
                                                                           'PF id', 'left', 'right'], ['float']*12)))
            PF_summary_peak_local = PF_summary_peak_local.astype(dict(zip(['cell','emerge lap', 'PF id', 'left', 'right'], ["Int64"]*5)))

        if opto_check == 1:
            print(self.mouse, 'in', PF_env, 'total # of place cells:', str(len(np.unique(PF_array[:, 1]))))
            return PF_summary_peak_local
        else:
            bw_array = np.array(bw_shift[1:][:])
            self.COM_shift[PF_env] = bw_array
            self.nPF.update({PF_env: len(np.unique(PF_array[:, 1]))})
            print(self.mouse, 'in', PF_env, 'total # of place cells:', str(len(np.unique(PF_array[:, 1]))))

            self.PF_summary_peak = pd.concat([self.PF_summary_peak, PF_summary_peak_local], ignore_index=True)
            self.PF_summary_peak.reset_index(drop=True)
            self.sort_by_COM(PF_env)
            # self.save_to_file()
            return self.PF_summary_peak

    @staticmethod
    def _PF_features(temp_pf, PF_loc_left, firing_laps):
        """ spatial precision, calculated as the inverse of COM std over laps

        :param temp_pf: temp PF of a cell, lap * PF boundary * cell
        :param PF_loc_left: PF left boundary
        :param firing_laps: index of laps after emergence (already fires consistently)
        :param return: COM, width, precision, slope, r2, p, com_by_laps
        """

        width = np.shape(temp_pf)[1]
        laps = np.shape(temp_pf)[0]
        temp_w = temp_pf * np.arange(width)  # activity * bin

        np.seterr(invalid='ignore')  # suppress zero divide error msg
        temp_com = np.sum(temp_w, axis=1) / np.sum(temp_pf, axis=1)  # com each lap
        precision = 1/(np.nanstd(temp_com[firing_laps]))  # non-zero division
        peak_lap = np.max(temp_pf[firing_laps, :], axis=1)
        COM = np.nansum(temp_com[firing_laps] * peak_lap) / np.sum(peak_lap)   # com weighed by peak activity per lap

        # backwards shifting
        slope, _, r, p, _ = linregress(np.arange(laps)[firing_laps], temp_com[firing_laps])
        r2 = r**2

        return np.round(COM+PF_loc_left,2), width, np.round(precision,2), np.round(slope,2), np.round(r2,2), \
               np.round(-np.log10(p),2), list(temp_com+PF_loc_left)

    def plot_all_cells(self, env, save = 0, red_cells = None, laps = None, autoscale=1):

        if red_cells is None:
            red_cells = self.PFs[env]
        if laps is None:
            laps = self.laps[env]

        mean_activity = self.mean_activity[laps, :, :]
        cell_heatmap(mean_activity, self.save_path, place_cells=red_cells, foldername=f'heatmaps {env}', scale=autoscale, savecellfig=save)

    def _first_lap_PF(self, active_laps):
        """ find the first stable lap of identified PFs """

        first_lap = np.nan
        for n in range(len(active_laps) - 4):
            if active_laps[n+3] <= active_laps[n] + 5:
                first_lap = active_laps[n]
                return first_lap
        return first_lap

    def reliability(self, env, start_lap, last_lap, cells=None, min_lap = 10):
        """ calculate lap-by-lap reliability/stability of each cell firing in envs. also called PV corr
        :param env: use PFs and laps in this env
        :param start_lap: start lap to calculate mean PF to compare with each lap
        :param last_lap: last lap to calculate mean PF to compare with each lap
        :param cells: specify cells, otherwise default PFs in the env
        :param min_lap: each PF to be included should have at least min_lap # of laps to calculate mean PFs from

        """

        if cells is None:
            cells = np.sort(self.PFs[env])

        # only calculate reliability after emerged, step1 find emerge lap for each PF
        PF_env = self.PF_summary_peak.loc[(self.PF_summary_peak['cell'].isin(cells)) &
                                          (self.PF_summary_peak['env'] == env) &
                                          (self.PF_summary_peak['emerge lap'] <= last_lap-self.laps[env][0]-min_lap)]
        if PF_env['cell'].dtype != 'int':
            PF_env = PF_env.astype(dict(zip(['cell', 'emerge lap'], ['int']*2)))
        PF_df = PF_env.groupby(by='cell').min()
        emerge_lap = PF_df['emerge lap'].tolist()
        cells = PF_df.index.tolist()
        # assert all(np.array([*PF_df.index]) == np.sort(self.PFs[env])), 'check PF_summary_peak, PFs not matched'

        data = self.mean_activity[start_lap:last_lap,:,cells]
        nlaps = data.shape[0]
        ncells = len(cells)
        mean_PF = np.nanmean(data, axis=0)
        reliab = np.zeros((ncells, nlaps))

        for n in range(ncells):
            reliab[n,:] = [scipy.stats.pearsonr(mean_PF[:, n], data[l, :, n])[0] for l in range(nlaps)]
            emerge_ind = np.max((emerge_lap[n]-(start_lap-self.laps[env][0]), 0))    # if emerge lap < start_lap
            reliab[n, range(emerge_ind)] = np.nan

        mean_r = np.nanmean(reliab, axis=0)
        plt.plot(mean_r)

        # add opto on line
        if env in self.opto_env:
            rmax = np.max(mean_r)
            rsd = np.std(mean_r)
            opto_on = self.opto_env[env][0] - start_lap
            opto_off = self.opto_env[env][-1] - start_lap
            plt.plot([opto_on, opto_off], [rmax+rsd, rmax+rsd], color = 'red')
        plt.xlabel(f'laps in {env}')
        plt.ylabel('pearson r')
        plt.title(f'{self.mouse} in {env} lap by lap reliability, {nlaps} laps')
        plt.show()

        # visualize reliab for each cell over laps
        reliab_sort = np.argsort(emerge_lap)     # sort cells by emerge lap
        #colormap = sns.color_palette('Greys')
        sns.heatmap(reliab[reliab_sort, :])  #, cmap=colormap)
        if env in self.opto_env:
            plt.plot([opto_on, opto_on], [0, ncells], 'k')
            plt.plot([opto_off, opto_off], [0, ncells], 'k')
        plt.xlabel('laps')
        plt.ylabel(f'cells in {env}')
        plt.show()

        return reliab[reliab_sort, :]

    def lap_by_lap_map(self, env, PF = None):

        if PF is None:
            PF = self.PFs[env]
        laps = self.laps[env]
        opto_on = 100000
        opto_off = 100000
        if env in self.opto_env:
            opto_on = self.opto_env[env][0] - laps[0]
            opto_off = self.opto_env[env][-1] - laps[0]

        m = self.mean_activity[laps[:, np.newaxis], :, PF]
        # out_field_firing

        # cell_heatmap(map, self.save_path, scale=1)
        for cell in range(len(laps)):
            figcount = cell % 9

            if figcount == 0:
                fig, axs = plt.subplots(3, 3, sharex=True, sharey=True)
                axs = axs.ravel()

            celltitle = 'lap ' + str(cell)
            sns.heatmap(ax=axs[figcount], data=m[cell, :, :], cbar=False,
                        xticklabels=False, yticklabels=False, vmax=1)
            if np.isin(cell, [opto_on, opto_off]):
                axs[figcount].set_title(celltitle, color='red', fontweight='bold')
            else:
                axs[figcount].set_title(celltitle)

            if figcount == 8:
                for ax in axs.flat:
                    ax.set(xlabel='location', ylabel='cells')
                    ax.label_outer()
                plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
                plt.show()

        plt.show()
        # return out_field_firing

    def emerge_lap_hist(self, PF_env, start_lap = 0, cut_off_lap =30):
        PF_summary = self.PF_summary_peak
        laps = PF_summary.loc[PF_summary['env'] == PF_env]['emerge lap'] - start_lap
        cut_laps = laps[laps <=cut_off_lap]
        y, _, _ = plt.hist(cut_laps,  weights=np.ones(len(cut_laps)) / len(cut_laps), alpha=0.5)
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
        plt.xlabel('emerge lap')
        hist_title = f'{self.mouse} emerge lap in {PF_env}'
        plt.title(hist_title)

        if PF_env in self.opto_env:
            opto_on_lap = self.opto_env[PF_env][0] - self.laps[PF_env][0]
            opto_off_lap = self.opto_env[PF_env][-1] - self.laps[PF_env][0]
            # only plot opto off lap if opto first condition
            if opto_on_lap == self.laps[PF_env][0]:
                plt.plot([opto_off_lap, opto_off_lap], [0, y.max()], '--k')
            # plot both opto on and off laps if opto later condition
            else:
                plt.plot([opto_on_lap, opto_on_lap], [0, y.max()], '--r')
                plt.plot([opto_off_lap, opto_off_lap], [0, y.max()], '--k')

        plt.show()

        return laps

    def emerge_lap_cumhist(self, PF_env, start_lap=0, cut_off_lap = 30):
        PF_summary = self.PF_summary_peak
        laps = PF_summary.loc[PF_summary['env'] == PF_env]['emerge lap'] - start_lap
        cut_laps = laps[laps <=cut_off_lap]
        plt.hist(cut_laps, density=True, histtype='step', cumulative=True)
        #plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
        plt.xlabel('emerge lap')
        plt.title(PF_env)

        return laps

    def overday_PF_match(self, env_day1, env_day2, max_lap = 30, min_lap = 10):
        """ find PFs on day2 that are formed on day1 and compare them with newly formed PFs on day2"""

        day1_df = self.PF_summary_peak.loc[self.PF_summary_peak['env'] == env_day1]
        day2_df = self.PF_summary_peak.loc[self.PF_summary_peak['env'] == env_day2]

        df_days = day1_df.merge(day2_df, 'right', on='cell')
        df_days.drop(columns = ['env_x', 'PF id_x', 'slope_x', 'r2_x', '-log(p)_x', 'left_x', 'right_x',
                                'env_y', 'PF id_y', 'slope_y', 'r2_y', '-log(p)_y', 'left_y', 'right_y'], inplace=True)

        df_days = df_days.astype(dict(zip(['COM_x', 'ratio_x', 'precision_x','COM_y', 'ratio_y', 'precision_y', 'emerge lap_x'], ['float']*7)))
        df_days = df_days.astype(dict(zip(['cell', 'emerge lap_y'], ['int'] * 2)))

        new_cells = df_days['emerge lap_x'].isnull()
        same_cells = ~new_cells

        # emerge lap
        lap1 = df_days['emerge lap_x'].dropna().to_numpy()
        lap2a = df_days[same_cells]['emerge lap_y'].to_numpy()
        lap2b = df_days[new_cells]['emerge lap_y'].to_numpy()
        plt.hist(lap1[lap1<max_lap], density=True, histtype='step', cumulative=True)
        plt.hist(lap2a[lap2a<max_lap], density=True, histtype='step', cumulative=True)
        plt.hist(lap2b[lap2b<max_lap], density=True, histtype='step', cumulative=True)
        plt.legend([env_day1, f'{env_day2} same PF', f'{env_day2} new PF'], loc = 'lower right')
        plt.title(f'{self.mouse} emerge lap')
        emerge_lap_dict = dict(zip([env_day1, f'{env_day2} same PF', f'{env_day2} new PF'], [lap1, lap2a, lap2b]))
        plt.show()
        day2_same_cells = df_days.loc[same_cells, 'cell'].unique()
        day2_new_cells = df_days.loc[new_cells, 'cell'].unique()

        # reliability
        reliab_dict = {}
        if env_day2 in self.opto_env:
            start_lap = np.max((self.laps[env_day2][0], self.opto_env[env_day2][0]-min_lap))         # start lap to calculate mean PF
            last_lap = np.min((self.laps[env_day2][-1], start_lap+max_lap+min_lap))     # last lap to calculate mean PF
            print(f'{self.mouse} in {env_day2} same PF')
            reliab_2a = self.reliability(env_day2, start_lap, last_lap, cells=day2_same_cells)
            # padding for equal size matrix between animals
            reliab_dict[f'{env_day2} same PF'] = np.pad(reliab_2a, ((0,0), (0, max_lap+min_lap-(last_lap-start_lap))), constant_values=np.nan)
            print(f'{self.mouse} in {env_day2} new PF')
            reliab_2b = self.reliability(env_day2, start_lap, last_lap, cells=day2_new_cells)
            reliab_dict[f'{env_day2} new PF'] = np.pad(reliab_2b, ((0,0), (0, max_lap+min_lap-(last_lap-start_lap))), constant_values=np.nan)

        elif env_day2 == 'control_day2':
            # control_first_day2
            start_lap = self.laps[env_day2][0]
            last_lap = np.min((self.laps[env_day2][-1], start_lap+max_lap+min_lap))
            print(f'{self.mouse} control_first_day2 same PF:')
            reliab_dict['control_first_day2 same PF'] = self.reliability(env_day2, start_lap, last_lap, cells=day2_same_cells)
            print(f'{self.mouse} control_first_day2 new PF:')
            reliab_dict['control_first_day2 new PF'] = self.reliability(env_day2, start_lap, last_lap, cells=day2_new_cells)

            # control_later_day2
            if 'opto_later_day2' in self.opto_env:
                start_lap = np.max((self.laps['opto_later_day2'][0], self.opto_env['opto_later_day2'][0]-min_lap)) \
                            -self.laps['opto_later_day2'][0] + self.laps['control_day2'][0]
                last_lap = np.min((self.laps['control_day2'][-1], start_lap + max_lap + min_lap))  # last lap to calculate mean PF
                print(f'{self.mouse} control_later_day2 same PF:')
                reliab_2a = self.reliability(env_day2, start_lap, last_lap, cells=day2_same_cells)
                reliab_dict['control_later_day2 same PF'] = np.pad(reliab_2a, ((0,0), (0, max_lap+min_lap-(last_lap-start_lap))), constant_values=np.nan)
                print(f'{self.mouse} control_later_day2 new PF:')
                reliab_2b = self.reliability(env_day2, start_lap, last_lap,cells=day2_new_cells)
                reliab_dict['control_later_day2 new PF'] = np.pad(reliab_2b, ((0,0), (0, max_lap+min_lap-(last_lap-start_lap))), constant_values=np.nan)

        return df_days, emerge_lap_dict, reliab_dict

    def sort_by_COM(self, PF_env):
        """plot lap-averaged PF activity for all PFs sorted by location of COM and save PFs in the env

        :param PF_summary: PF_summary dataframe by method
        :return:
        """
        PF_summary = self.PF_summary_peak
        PF_summary['COM'] = pd.to_numeric(PF_summary['COM'])
        cell_order = PF_summary.loc[PF_summary['env'] == PF_env].sort_values(by=['COM'])['cell']
        cells = cell_order.to_numpy().astype(int)
        _, idx = np.unique(cells, return_index=True)
        cell_sorted = cells[np.sort(idx)]
        self.PFs.update({PF_env: cell_sorted})

        laps = self.laps[PF_env]
        lap_range = slice(laps[0], laps[-1]+1)
        sns.heatmap(np.nanmean(self.mean_activity[lap_range,:,cell_sorted],axis = 0).transpose(), xticklabels=5, yticklabels=10, vmin = 0, vmax = 1.1)
        plt.xlabel('location on track')
        plt.ylabel('cells')
        plt.title(f'{self.mouse} in {PF_env}')
        plt.show()
        plt.close()

        return self.PFs

    def remapping(self, cells, first_lap = 0, final_lap = None):
        """ plot the trial-averaged cell activity from first_lap to final_lap

        :param cells: output from sort_by_COM
        :param first_lap: first lap of an ENV
        :param final_lap: final lap of an ENV
        :return: trial-averaged cell activity
        """
        if first_lap == 0 and final_lap is None:
            PF_order = np.transpose(np.squeeze(self.mean_PF[:, [cells]]))
        else:
            if final_lap is None:
                final_lap = self.nlaps
            PF_order = np.transpose(np.squeeze(np.nanmean(self.mean_activity[slice(first_lap,final_lap),:,cells], axis = 0)))
        sns.heatmap(PF_order, xticklabels=5, yticklabels=10, vmin = 0, vmax = 1)
        plt.xlabel('location on track')
        plt.ylabel(f'cells in {self.mouse}')
        plt.show()

        return PF_order

    @staticmethod
    def remapping_corr(map1, map2):

        assert map1.shape == map2.shape, 'maps of different sizes'
        ncells = np.shape(map1)[0]
        corr = np.zeros((ncells, 1))
        for cell in range(ncells):
            corr[cell], _ = stats.pearsonr(map1[cell, :], map2[cell, :])
        print('mean corr is: ', np.nanmean(corr))
        plt.hist(corr, weights=np.ones(ncells)/ncells, alpha=0.5)
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
        plt.xlabel('pearson r')
        plt.ylabel('cell count')

        return corr

    def same_cells_over_days(self, env, same_thresh = 8):
        """ identify same place cells in the same environment on the second day"""

        env_day1 = f'{env}_day1'
        env_day2 = f'{env}_day2'

        PFs_day1 = self.PFs[env_day1]
        PFs_day2 = self.PFs[env_day2]
        overday_PFs = np.intersect1d(PFs_day1, PFs_day2)

        # check if COM is the same
        PF_summary = self.PF_summary_peak
        COM_day1 = PF_summary[(PF_summary['cell'].astype(int).isin(overday_PFs)) & (PF_summary['env'] == env_day1)]
        COM_day2 = PF_summary[(PF_summary['cell'].astype(int).isin(overday_PFs)) & (PF_summary['env'] == env_day2)]
        COM_merge = self._merge_opto_df(COM_day1, COM_day2, same_thresh)
        same_COM = COM_merge[['cell', 'COM_off', 'COM_on']]
        COM_diff = same_COM['COM_off'] - same_COM['COM_on']
        overday_COMs = same_COM[np.abs(COM_diff < same_thresh)]['cell'].astype(int).unique()
        # overday_COMs = overday_PFs  # identify overday COMs by cell identity instead COM (code above)
        print(f'{self.mouse} has {len(PFs_day1)} PFs in {env_day1}, {len(PFs_day2)} PFs in {env_day2}, '
              f'{len(overday_COMs)} PFs with same COMs in both days')

        # reorder PFs by COM
        sorted_overday_PFs = [x for _, x in sorted(zip(overday_COMs, PFs_day1))]
        day1_map = self.remapping(sorted_overday_PFs, self.laps[env_day1][0], self.laps[env_day1][-1])
        day2_map = self.remapping(sorted_overday_PFs, self.laps[env_day2][0], self.laps[env_day2][-1])
        map_corr = self.remapping_corr(day1_map, day2_map)
        plt.show()

        return overday_COMs, map_corr, sorted_overday_PFs

    def shift_xcorr(self, env, padding = 3):
        """ calculate lap-by-lap population vector cross correlation of place cells between laps in env. Eg.
        Priestly... Losonczy 2021 Fig 2.

        :param env: which env the cells and laps are from
        :param padding: enlarge the recognized PF by padding bins left and right

        """

        laps = self.laps[env]
        cells = self.PFs[env]
        m = self.mean_activity[slice(laps[0], laps[-1]+1), :, cells]
        df = self.PF_summary_peak.loc[self.PF_summary_peak['env']== env]
        lag_mat = np.zeros((len(laps), len(laps), len(df)))
        lag_mat[:] = np.nan
        emerge_lag = np.copy(lag_mat)
        xcorr_mat = np.zeros((len(laps), len(laps), len(df)))
        xcorr_mat[:] = np.nan

        # loop over individual PF (eg. one cell could have multiple PFs)
        for c in range(len(df)):
            cell = df.iloc[c]['cell']
            emerge_lap = df.iloc[c]['emerge lap']
            left = np.max((df.iloc[c]['left']-padding, 0))
            right = np.min((df.iloc[c]['right']+padding, self.nbins-1))
            temp_pf = self.mean_activity[slice(laps[0], laps[-1]+1), slice(left, right), cell]   # only compare in-field firing
            # only consider laps that fired (criteria: activity > 0.1, fire at least 2 bins)
            firing_laps = np.where(np.sum(temp_pf>0.1, axis=1)>1)[0]
            firing_laps = firing_laps[firing_laps >= emerge_lap]
            m = temp_pf[firing_laps, :]
            width = np.shape(temp_pf)[1]

            for l0 in range(len(firing_laps)-1):
                for l1 in range(l0+1, len(firing_laps)):
                    corr = signal.correlate(m[l0, :], m[l1, :])
                    xcorr_mat[firing_laps[l0], firing_laps[l1], c] = np.max(corr)
                    lag_mat[firing_laps[l0], firing_laps[l1], c] = np.argmax(corr) - width
                    emerge_lag[firing_laps[l0] - firing_laps[0], firing_laps[l1] - firing_laps[0], c] = np.argmax(corr) - width

        # plot lag (compared to env switch) context-based
        mean_lag = np.nanmean(lag_mat, axis=2)
        sns.heatmap(mean_lag,  cmap="PiYG", center = 0)
        plt.xlabel('lap')
        plt.ylabel('lap')
        plt.title(f'{self.mouse} place cells in {env} xcorr lag')
        if env in self.opto_env:
            plt.plot([0, len(laps)],
                     [self.opto_env[env][-1] - self.laps[env][0], self.opto_env[env][-1] - self.laps[env][0]],
                     'k', '--', alpha = 0.5)
            plt.plot([0, len(laps)],
                     [self.opto_env[env][0] - self.laps[env][0], self.opto_env[env][0] - self.laps[env][0]],
                     'k', '--', alpha = 0.5)
        plt.show()

        # plot lag (compared to first emerged) PF-based
        sns.heatmap(np.nanmean(emerge_lag, axis=2), cmap="PiYG", center=0)
        plt.xlabel('lap after emerged')
        plt.ylabel('lap after emerged')
        plt.title(f'{self.mouse} place cells in {env} xcorr lag after emerged')
        plt.show()

        return lag_mat, xcorr_mat, emerge_lag

    def shift_scatter(self, env):
        """ scatter plot of place cells backwards/forwards shifting in env"""

        df = self.PF_summary_peak
        slope = df.loc[df['env'] == env]['slope'].values.astype(float)
        r2 = df.loc[df['env'] == env]['r2'].values.astype(float)
        p = df.loc[df['env'] == env]['-log(p)'].values.astype(float)
        plt.scatter(slope, r2, c=p, cmap= 'gray_r', vmax=5)
        plt.colorbar()
        plt.xlabel('slope (bin/lap)')
        plt.ylabel('r2')
        plt.title(f'slope vs. r2 in {env}')

    def opto_check(self, env, min_lap = 10):
        """ check the effect of opto in env. Mostly takes care of the control condition
        :param min_lap: in opto later condition, use cells that have emerged at least min_lap # of laps before opto on
        Note: using all opto on laps and opto off laps for comparison. Much more opto off laps than opto on laps
        """

        opto_later_day = f'opto_later_{env[-4:]}'
        opto_first_day = f'opto_first_{env[-4:]}'
        PF_features = [[]]

        # if opto_later_day in self.opto_env:
        if (env in ['opto_later_day1', 'opto_later_day2']) & (env in self.opto_env):
            opto_later_start = self.opto_env[env][0] - self.laps[env][0]   # within env, opto on lap
            last_lap = np.min((self.opto_env[env][-1]+1+min_lap, self.laps[env][-1]+1)) # make sure last lap is in same env
            after_laps = np.arange(self.opto_env[env][-1]+1, last_lap)  # min_lap # of laps after opto on
            print(f'{self.mouse} {env} during laps: {self.opto_env[env]}')
            print(f'{self.mouse} {env} after laps: {after_laps}, env last lap: {self.laps[env][-1]}')

            # find PFs in opto later env and recalculate their PF features during opto on and opto off
            PF_df = self.PF_summary_peak.loc[(self.PF_summary_peak['env'] == env) &
                                             (self.PF_summary_peak['emerge lap'] <= opto_later_start - min_lap)]
            cells = PF_df['cell'].unique()

            for c in cells:
                cell_df = PF_df.loc[PF_df['cell']==c]

                # PF features before opto on
                before_laps = np.arange(self.laps[env][0]+cell_df['emerge lap'].min(), self.opto_env[env][0])

                for n in range(len(cell_df)):   # for cell that has multiple PFs, loop over each PF

                    ratio, out_ratio, out_in_ratio, COM, precision, amp, slope, r2, p = self._PF_features_opto(before_laps,c,
                                                                                      cell_df.iloc[n]['left'],
                                                                                      cell_df.iloc[n]['right'])
                    PF_features.append([self.mouse, env, 'before', c, cell_df.iloc[n]['PF id'], COM,
                                        ratio, out_ratio, precision, out_in_ratio, amp, slope, r2, p])

                # PF features during opto on
                    ratio, out_ratio, out_in_ratio, COM, precision, amp, slope, r2, p = self._PF_features_opto(self.opto_env[env], c,
                                                                        cell_df.iloc[n]['left'],
                                                                        cell_df.iloc[n]['right'])
                    PF_features.append([self.mouse, env, 'during', c, cell_df.iloc[n]['PF id'], COM,
                                        ratio, out_ratio, precision, out_in_ratio, amp, slope, r2, p])

                # PF features after opto on
                    ratio, out_ratio, out_in_ratio, COM, precision, amp, slope, r2, p = self._PF_features_opto(after_laps, c,
                                                                        cell_df.iloc[n]['left'],
                                                                        cell_df.iloc[n]['right'])
                    PF_features.append([self.mouse, env, 'after', c, cell_df.iloc[n]['PF id'], COM,
                                        ratio, out_ratio, precision, out_in_ratio, amp, slope, r2, p])

        # only use # laps after opto on to calculate PF features
        elif (env in ['opto_first_day1', 'opto_first_day2']) & (env in self.opto_env):
            opto_off = self.opto_env[env][-1] - self.laps[env][0]   # within env, opto off lap
            PF_df = self.PF_summary_peak.loc[(self.PF_summary_peak['env'] == env) &
                                             (self.PF_summary_peak['emerge lap'] <= opto_off + min_lap)]
            cells = PF_df['cell'].unique()
            first_lap = self.laps[env][0] + opto_off + min_lap
            if first_lap > self.laps[env][-1] - 5:
                print(f'{self.mouse} does not have enough laps in {env}')
                return self.PF_summary_opto
            last_lap = np.min((first_lap+min_lap+1, self.laps[env][-1]+1))
            after_laps = np.arange(first_lap, last_lap)

            print(f'{self.mouse} {env} after laps: {after_laps}, env last lap: {self.laps[env][-1]}')

            for c in cells:
                cell_df = PF_df.loc[PF_df['cell'] == c]
                # check min_lap # of laps after opto off
                for n in range(len(cell_df)):
                    ratio, out_ratio, out_in_ratio, COM, precision,amp,  slope, r2, p = self._PF_features_opto(after_laps, c,
                                                                    cell_df.iloc[n]['left'], cell_df.iloc[n]['right'])
                    PF_features.append([self.mouse, env, 'after', c, cell_df.iloc[n]['PF id'], COM,
                                        ratio, out_ratio, precision, out_in_ratio, amp, slope, r2, p])

        elif env in ['control_day1', 'control_day2']:

            if opto_first_day in self.opto_env:
                # matching laps comparable to opto first day1/day2
                opto_off = self.opto_env[opto_first_day][-1] - self.laps[opto_first_day][0]  # within env, opto off lap
                last_lap = self.laps[env][0] + opto_off + min_lap * 2
                if last_lap > self.laps[env][-1]:
                    print(f'{self.mouse} {env} has fewer laps than {opto_first_day}, use last {min_lap} laps instead. {last_lap} < {self.laps[env][-1]}')
                    last_lap = self.laps[env][-1]
                after_laps = np.arange(last_lap - min_lap, last_lap)

                env_label = f'control_{opto_first_day}'
                PF_df = self.PF_summary_peak.loc[(self.PF_summary_peak['env'] == env) &
                                                 (self.PF_summary_peak['emerge lap'] <= opto_off + min_lap)]
                cells = PF_df['cell'].unique()
                print(f'{self.mouse} {env_label} after laps: {after_laps}, env last lap: {self.laps[env][-1]}')

                for c in cells:
                    cell_df = PF_df.loc[PF_df['cell'] == c]

                    for n in range(len(cell_df)):
                        ratio, out_ratio, out_in_ratio, COM, precision, amp, slope, r2, p = self._PF_features_opto(
                            after_laps, c, cell_df.iloc[n]['left'], cell_df.iloc[n]['right'])
                        PF_features.append([self.mouse, env_label, 'after', c, cell_df.iloc[n]['PF id'], COM,
                                            ratio, out_ratio, precision, out_in_ratio, amp, slope, r2, p])

            if opto_later_day in self.opto_env:   # compare with opto_later same day
                # opto_length_later = len(self.opto_env[opto_later_day])   # opto on # laps
                # opto_later_after_length = self.laps[opto_later_day][-1] - self.opto_env[opto_later_day][-1]  # opto off after # laps

                opto_on_env = self.opto_env[opto_later_day][0] - self.laps[opto_later_day][0]
                opto_off_env = self.opto_env[opto_later_day][-1] - self.laps[opto_later_day][0]
                first_lap = self.laps[env][0]+opto_on_env
                if first_lap > self.laps[env][-1]-(opto_off_env-opto_on_env):
                    print(f'{self.mouse} {env} does not have enough laps for comparison with {opto_later_day}')
                    return self.PF_summary_opto
                opto_on_laps = np.arange(first_lap, self.laps[env][0]+opto_off_env)
                after_first_lap = opto_on_laps[-1]+1
                opto_after_length = np.min((self.laps[opto_later_day][-1] - (self.opto_env[opto_later_day][-1]+1), min_lap))
                last_lap = after_first_lap + opto_after_length
                if (after_first_lap > self.laps[env][-1]-opto_after_length) & (after_first_lap < self.laps[env][-1]-4):
                    print(after_first_lap, opto_after_length, last_lap)
                    last_lap = self.laps[env][-1]
                    after_length = last_lap - after_first_lap
                    print(f'{self.mouse} {env} has fewer laps than {opto_later_day}, use last {after_length} laps instead')
                elif after_first_lap > self.laps[env][-1]-4:
                    print(f'{self.mouse} {env} does not have enough laps for comparison with {opto_later_day}')
                    return self.PF_summary_opto
                after_laps = np.arange(after_first_lap, last_lap)

                # opto later comparison
                # opto_on_laps = np.arange(self.laps[env][-1] - opto_later_after_length - opto_length_later,
                #                          self.laps[env][-1] - opto_later_after_length)   # same # opto on laps in control
                # after_laps = np.arange(opto_on_laps[-1] + 1, self.laps[env][-1] + 1)   # same # opto off after laps in control
                # opto_on_start = opto_on_laps[0] - self.laps[env][0]   # first opto on lap in env
                env_label = f'control_{opto_later_day}'
                print(f'{self.mouse} {env_label} during laps: {opto_on_laps}')
                print(f'{self.mouse} {env_label} after laps: {after_laps}, env last lap: {self.laps[env][-1]}')

                # find PFs in opto later env and recalculate their PF features during opto on and opto off
                PF_df = self.PF_summary_peak.loc[(self.PF_summary_peak['env'] == env) &
                                                 (self.PF_summary_peak['emerge lap'] <= opto_on_env - min_lap)]
                cells = PF_df['cell'].unique()
                for c in cells:
                    cell_df = PF_df.loc[PF_df['cell']==c]
                    # PF features before opto on
                    before_laps = np.arange(self.laps[env][0]+cell_df['emerge lap'].min(), opto_on_laps[0])
                    # print(c, before_laps)
                    for n in range(len(cell_df)):
                        ratio, out_ratio, out_in_ratio, COM, precision, amp, slope, r2, p = self._PF_features_opto(before_laps, c,
                                                                                          cell_df.iloc[n]['left'],
                                                                                          cell_df.iloc[n]['right'])
                        PF_features.append([self.mouse, env_label, 'before', c, cell_df.iloc[n]['PF id'], COM,
                                            ratio, out_ratio, precision, out_in_ratio,amp,  slope, r2, p])

                    # PF features during opto on
                        ratio, out_ratio, out_in_ratio, COM, precision, amp, slope, r2, p = self._PF_features_opto(opto_on_laps, c,
                                                                            cell_df.iloc[n]['left'],
                                                                            cell_df.iloc[n]['right'])
                        PF_features.append([self.mouse, env_label, 'during', c, cell_df.iloc[n]['PF id'], COM,
                                            ratio, out_ratio, precision, out_in_ratio, amp, slope, r2, p])

                    # PF features after opto on
                        ratio, out_ratio, out_in_ratio, COM, precision, amp, slope, r2, p = self._PF_features_opto(after_laps, c,
                                                                            cell_df.iloc[n]['left'],
                                                                            cell_df.iloc[n]['right'])
                        PF_features.append([self.mouse, env_label, 'after', c, cell_df.iloc[n]['PF id'], COM,
                                            ratio,out_ratio, precision, out_in_ratio, amp, slope, r2, p])

        PF_array = np.array(PF_features[1:][:])

        if len(PF_array)>0:
            PF_summary_peak_local = pd.DataFrame(PF_array,
                                                 columns=['mouse','env', 'opto lap', 'cell', 'PF id', 'COM','ratio',
                                                          'out ratio','precision', 'out in ratio','amplitude',
                                                          'slope', 'r2', '-log(p)'])
            PF_summary_peak_local = PF_summary_peak_local.astype(
                dict(zip(['ratio', 'out ratio', 'precision', 'out in ratio', 'amplitude', 'slope',
                          'r2', '-log(p)', 'COM', 'cell', 'PF id'], ['float'] * 11)))
            PF_summary_peak_local = PF_summary_peak_local.astype(dict(zip(['cell', 'PF id'], ["Int64"] * 2)))
            self.PF_summary_opto = pd.concat([self.PF_summary_opto, PF_summary_peak_local], ignore_index=True)
            self.PF_summary_opto.reset_index(drop=True)
        return self.PF_summary_opto

    def plot_opto_feature(self, feature):

        df = self.PF_summary_opto

        for o in ['before', 'during']:
            df_o = df.loc[df['opto lap'] == o]

            fig, axs = plt.subplots(1, 2, sharey=True)
            axs = axs.ravel()

            sns.pointplot(ax=axs[0], x='env', y=feature, data=df_o, errorbar='sd',
                          order=['control_opto_later_day1', 'opto_later_day1'])
            axs[0].set_title(f'{self.mouse} {o} opto day1')
            #axs[0].get_legend().remove()

            sns.pointplot(ax=axs[1], x='env', y=feature, data=df_o, errorbar='sd',
                          order=['control_opto_later_day2', 'opto_later_day2'])
            axs[1].set_title(f'{self.mouse} {o} opto day2')
            # axs[1].get_legend().remove()
            axs[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))

        # after opto
        df_after = df.loc[df['opto lap'] == 'after']
        fig, axs = plt.subplots(1,2, sharey=True)
        axs = axs.ravel()

        sns.pointplot(ax = axs[0],  x='env', y=feature, data=df_after, errorbar='sd',
                      order=['control_opto_later_day1', 'opto_later_day1'])
        axs[0].set_title(f'{self.mouse} after opto first day1')
        #axs[0].get_legend().remove()

        sns.pointplot(ax=axs[1], x='env', y=feature, data=df_after, errorbar='sd',
                      order=['control_opto_later_day2', 'opto_later_day2'])
        axs[1].set_title(f'{self.mouse} after opto first day2')
        axs[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))

        fig, axs = plt.subplots(1,2, sharey=True)
        axs = axs.ravel()

        sns.pointplot(ax = axs[0],  x='env', y=feature, data=df_after, errorbar='sd',
                      order=['control_opto_first_day1', 'opto_first_day1'])
        axs[0].set_title(f'{self.mouse} after opto first day1')
        #axs[0].get_legend().remove()

        sns.pointplot(ax=axs[1], x='env', y=feature, data=df_after, errorbar='sd',
                      order=['control_opto_first_day2', 'opto_first_day2'])
        axs[1].set_title(f'{self.mouse} after opto first day2')
        axs[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))



    def _PF_features_opto(self, laps, cell,  PF_loc_left, PF_loc_right, minDF=0.1, minWidth=2, min_lap =3):

        temp_pf = self.mean_activity[laps, PF_loc_left:PF_loc_right, cell]  # potential PF
        temp_thresh = temp_pf > minDF
        # ratio: total laps is the # laps
        firing_lap = np.sum(temp_thresh, axis=1)
        firing_lap_ind = np.where(firing_lap >= minWidth)[0]
        ratio_temp = np.round(len(firing_lap_ind) /len(laps), 2)

        out_field = np.setdiff1d(np.arange(self.nbins), np.arange(PF_loc_left, PF_loc_right))
        out_field_F = np.mean(self.mean_activity[laps[:, np.newaxis], out_field, cell])
        out_field_lap = np.sum((self.mean_activity[laps[:, np.newaxis], out_field, cell] > minDF), axis=1)
        out_field_ratio = np.round((np.sum(out_field_lap>0)) / len(laps), 2)
        # print(cell, out_field_ratio)

        if len(firing_lap_ind) > min_lap:
            in_field_F = np.mean(temp_pf)
            out_in_ratio = np.round(out_field_F / in_field_F, 2)
            amp = np.mean(np.max(temp_pf[firing_lap_ind, :], axis=1))
            COM, width, precision, slope, r2, p, _  = self._PF_features(temp_pf, PF_loc_left, firing_lap_ind)
            return ratio_temp, out_field_ratio, out_in_ratio, COM, precision, np.round(amp,2), slope, r2, p

        else:
            return ratio_temp, out_field_ratio, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    def _opto_check(self, env, opto_on_laps, opto_off_laps, same_thresh, off_laps, save, savename, min_laps = 8):

        # check cache
        opto_pq = os.path.join(self.save_path, 'opto', f'{self.mouse}_{savename}_{off_laps}.parquet')
        opto_folder = os.path.join(self.save_path, 'opto')
        if os.path.exists(opto_pq):
            opto = pd.read_parquet(opto_pq, engine='fastparquet')
            print(f'already checked {self.mouse} in {env}')
            if opto is None or len(opto) ==0:
                print(f'{self.mouse} has no PFs in {env} opto on/off, no comparison can be made')
            else:
                return self.load('opto', f'corr_{savename}_{off_laps}'), opto, self.load('opto', f'opto_effect_{savename}_{off_laps}')

        # choose which laps to represent opto off laps
        opto_length = len(opto_on_laps)
        if off_laps == 'after':
            opto_off_laps = opto_off_laps[opto_off_laps > opto_on_laps[-1]]
            opto_off_laps = opto_off_laps[1:opto_length+1]
        elif off_laps == 'last':
            opto_off_laps = opto_off_laps[-opto_length-1:-1]
        elif type(off_laps) == int:
            opto_off_laps = opto_off_laps[off_laps:off_laps+opto_length]
        else:
            print(f'using all {len(opto_off_laps)} opto off laps as comparison to {opto_length} opto on laps')

        if len(opto_off_laps) < min_laps or opto_length < min_laps:
            print(f'{self.mouse} does not have enough laps in {env}, opto on: {opto_length}; opto off: {len(opto_off_laps)}')
            return None, None, None

        # place cell features dataframe
        opto_on = self.check_PF_peak(env, opto_on_laps, opto_check=1)
        if len(opto_on) == 0:
            opto = pd.DataFrame()
            opto.to_parquet(opto_pq, compression='gzip')  # save to parquet
            print(f'{self.mouse} in {env} opto on no PFs')
            return None, None, None
        opto_off = self.check_PF_peak(env, opto_off_laps, opto_check=1)
        if len(opto_off) == 0:
            opto = pd.DataFrame()
            opto.to_parquet(opto_pq, compression='gzip')  # save to parquet
            print(f'{self.mouse} in {env} opto off no PFs')
            return None, None, None
        opto = self._merge_opto_df(opto_off, opto_on, same_thresh)

        # calculating PFs found by entire env features in opto on vs. off laps
        last_opto_lap = opto_on_laps[-1] - self.laps[env][0]
        for cell in self.PFs[env]:
            df = self.PF_summary_peak.loc[(self.PF_summary_peak['cell'] == cell) & (self.PF_summary_peak['env'] == env)
                                          & (self.PF_summary_peak['emerge lap']< last_opto_lap)]
            pfs = len(df)
            for ind in range(pfs):
                PF_loc_left = df.iloc[ind]['left']
                PF_loc_right = df.iloc[ind]['right']
                temp_pf = self.mean_activity[opto_on_laps, PF_loc_left:PF_loc_right, cell]  # potential PF
                temp_thresh = temp_pf > 0.1        # minDF as 0.1
                # ratio: total laps is the # laps
                firing_lap = np.sum(temp_thresh, axis=1)
                firing_lap_ind = np.where(firing_lap >= 2)[0]      # minWidth as 2
                ratio = len(firing_lap_ind)/ len(opto_on_laps)

                in_field_F = np.mean(temp_pf)
                out_field = np.setdiff1d(np.arange(self.nbins),
                                         np.arange(PF_loc_left, PF_loc_right))
                out_field_F = np.mean(self.mean_activity[opto_on_laps[:, np.newaxis], out_field, cell])
                out_in_ratio = np.round(out_field_F / in_field_F, 2)

                COM, width, precision, slope, r2, p, com_by_laps = self._PF_features(temp_pf, PF_loc_left,
                                                                                     firing_lap_ind)

                PF_features.append(
                    [PF_env, cell, PF_id, emerge_lap, COM, PF_loc_left, PF_loc_right,
                     np.round(ratio, 2), precision, out_in_ratio, slope, r2, p])


        # place cell COM shift scatter plot
        opto['COM_diff'] = np.abs(opto['COM_off'] - opto['COM_on'])
        rows = opto[(~opto['COM_off'].isnull()) & (~opto['COM_on'].isnull())]
        shift_cells = rows[rows['COM_diff'] >= same_thresh]['cell'].astype(int).to_numpy()
        if len(shift_cells)>0:
            com_off = rows['COM_off']
            com_on = rows['COM_on']
            colors = rows['COM_diff']
            plt.scatter(com_off, com_on, c=colors, cmap='gist_heat')
            plt.axis('square')
            plt.xlabel('COM opto off')
            plt.ylabel('COM opto on')
            com_shift_title = f'{self.mouse} in {env} opto on vs off COM, {off_laps} off laps'
            plt.title(com_shift_title)
            if save == 1:
                if not os.path.exists(opto_folder):
                    print(f'creating folder at path: {opto_folder}')
                    os.mkdir(opto_folder)
                plt.savefig(os.path.join(opto_folder, com_shift_title))
            plt.show()
        self._opto_tuning(opto_on_laps, opto_off_laps, plot_tuning=1, save=0, cells = shift_cells)

        print(f'{self.mouse} # place fields thru structural change in {env}: {str(len(shift_cells))}')

        # opto effect pie chart by individual place fields
        on_only = opto.loc[(~opto['COM_on'].isnull()) & (opto['COM_off'].isnull()),'cell'].astype(int).to_numpy()
        off_only = opto.loc[(~opto['COM_off'].isnull()) & (opto['COM_on'].isnull()),'cell'].astype(int).to_numpy()
        same_COM = rows.loc[rows['COM_diff'] <= same_thresh, 'cell'].astype(int).to_numpy()
        field_count = np.array([len(on_only), len(off_only), len(same_COM), len(shift_cells)])
        pie_labels = ['on only', 'off only', 'both same COM', 'both shift COM']
        plt.pie(field_count, labels= pie_labels)
        pie_title = f'{self.mouse} in {env} opto effects on place fields, {off_laps} off laps'
        plt.title(pie_title)
        if save == 1:
            plt.savefig(os.path.join(opto_folder, pie_title))
        plt.show()
        opto_PF_effect = dict(zip(pie_labels, [on_only, off_only, same_COM, shift_cells]))

        # place cell map
        title_on = f'{self.mouse} in {env} opto on'
        plt.title(title_on)
        map1 = self.remapping(self.PFs[env], opto_on_laps[0], opto_on_laps[-1])
        title_off = f'{self.mouse} in {env} opto off, {off_laps} off laps'
        plt.title(title_off)
        map2 = self.remapping(self.PFs[env], opto_off_laps[0], opto_off_laps[-1])
        corr = self.remapping_corr(map1, map2)
        title_corr = f'{self.mouse} in {env} place cell firing in opto on vs off, {off_laps}'
        plt.title(title_corr)
        if save == 1:
            plt.savefig(os.path.join(opto_folder, title_corr))
        plt.show()

        # save variables
        opto['mouse'] = self.mouse
        opto.to_parquet(opto_pq, compression='gzip')  # save to parquet
        self.save_to_file(corr, subdir = 'opto', savename=f'corr_{savename}_{off_laps}')     # save to pickle
        self.save_to_file(opto_PF_effect, subdir = 'opto', savename = f'opto_effect_{savename}_{off_laps}')
        #self.save_to_file()

        return corr, opto, opto_PF_effect

    def _opto_tuning(self, opto_on_laps, opto_off_laps, plot_tuning, save, cells):
        """ plot place cell tuning in env in opto on and opto off"""

        mean_PF_off = np.zeros((len(cells), self.nbins))
        mean_PF_on = np.zeros((len(cells), self.nbins))
        ind = 0

        for c in cells:
            firing_laps_off = np.nansum(self.mean_activity[opto_off_laps[:, np.newaxis],:, c], axis=2)[:, 0] > 0
            firing_laps_on = np.nansum(self.mean_activity[opto_on_laps[:, np.newaxis], :, c], axis=2)[:, 0] > 0
            laps_off = opto_off_laps[firing_laps_off]
            laps_on = opto_on_laps[firing_laps_on]
            mean_PF_off[ind, :] = np.mean(self.mean_activity[laps_off,:, c], axis=0)
            mean_PF_on[ind, :] = np.mean(self.mean_activity[laps_on, :, c], axis=0)
            ind = ind+1

        if plot_tuning ==1:
            cell_mean_over_laps_opto_compare(mean_PF_off.transpose(), mean_PF_on.transpose(),
                                             cells, savecellfig=save, save_path=self.save_path)

        return mean_PF_off, mean_PF_on

    def opto_tuning_ratio(self, env):
        """ find which cells have reduced/enhanced tuning from opto """

        cells = self.PFs[env]
        opto_on_laps = self.opto_env[env]
        opto_off_laps = np.setdiff1d(self.laps[env], opto_on_laps)
        mean_PF_off, mean_PF_on = self._opto_tuning(opto_on_laps, opto_off_laps, 0, 0, cells)

        mean_diff = mean_PF_off - mean_PF_on
        p = np.zeros((len(cells), 1))
        for c in range(len(cells)):
            _, p[c] = wilcoxon(mean_diff[c, :])

        return p

    @staticmethod
    def _merge_opto_df(opto_off, opto_on, same_thresh):
        """ exact merge of two dataframes by cell and approximate merge by COM"""

        # format opto_off and opto_on
        opto_off.drop(columns = 'env', inplace=True)
        opto_on.drop(columns = 'env', inplace=True)
        col_name = opto_off.columns
        opto_off = opto_off.astype(dict(zip(col_name, [float] * len(col_name))))
        opto_on = opto_on.astype(dict(zip(col_name, [float] * len(col_name))))

        # first merge all and then examine approx merge by finding duplicates
        opto = opto_off.merge(opto_on, on='cell', how='outer', suffixes=('_off', '_on'))
        dup_ind = opto.duplicated(keep = False, subset=['cell'])
        unique_rows = opto[~dup_ind]
        mPFs = opto[dup_ind]['cell'].unique()  # cells with multiple place fields

        # approx merge based on COM
        asof_merge_off = pd.merge_asof(opto_off.loc[opto_off['cell'].isin(mPFs)].sort_values('COM').rename(columns={'COM':'COM_off'}),
            opto_on.loc[opto_on['cell'].isin(mPFs)].sort_values('COM').rename(columns={'COM':'COM_on'}),
            left_on='COM_off', right_on='COM_on', tolerance= same_thresh, direction='nearest', suffixes=('_off', '_on'))
        asof_merge_off.iloc[list(asof_merge_off['cell_on'] != asof_merge_off['cell_off']), 10:20] = np.nan
        asof_merge_off = asof_merge_off.drop(columns = ['cell_on'])\
            .rename(columns = {'cell_off':'cell'})

        asof_merge_on = pd.merge_asof(opto_on.loc[opto_on['cell'].isin(mPFs)].sort_values('COM').rename(columns={'COM':'COM_on'}),
            opto_off.loc[opto_off['cell'].isin(mPFs)].sort_values('COM').rename(columns={'COM':'COM_off'}),
            left_on='COM_on', right_on='COM_off', tolerance=same_thresh, direction='nearest', suffixes=('_on', '_off'))
        asof_merge_on.iloc[list(asof_merge_on['cell_on'] != asof_merge_on['cell_off']), 10:20] = np.nan
        asof_merge_on = asof_merge_on.drop(columns = ['cell_off'])\
            .rename(columns = {'cell_on':'cell'})

        # combine all the merges together
        opto_merge = pd.concat([unique_rows, asof_merge_off, asof_merge_on]).sort_values('cell').reset_index(drop=True)
        # remove duplicated PFs from concat
        opto_merge = opto_merge[~((opto_merge.duplicated(keep = False, subset = ['PF id_off'])) & (opto_merge['COM_on'].isnull()))
                                & ~((opto_merge.duplicated(keep = False, subset=['PF id_on'])) & (opto_merge['COM_off'].isnull()))]

        return opto_merge

    def lap_corr_cell(self, env = None, cells = None, laps = None):
        """ lap-by-lap correlation for each cell """

        if cells is None:
            cells = self.PF_summary_peak['cell'].unique().astype(int)
        if laps is None:
            laps = np.arange(self.nlaps)
        if env is not None:
            cells = self.PFs[env]
            laps = self.laps[env]

        nlaps = len(laps)
        r_coef = np.zeros((nlaps, nlaps, len(cells)))
        #cos_sim = np.zeros((nlaps, nlaps, len(cells)))
        cos_f = np.zeros((int(nlaps*(nlaps-1)/2), len(cells)))
        f_ind = np.triu_indices(n=nlaps, k=1)

        count = 0
        for c in cells:
            r_coef[:,:,count] = np.corrcoef(self.mean_activity[laps,:,c])
            mat = cosine_similarity(self.mean_activity[laps, :, c])
            #cos_sim[:,:,count] = mat
            cos_f[:, count] = mat[f_ind]
            colormap = sns.color_palette('Greys')
            sns.heatmap(mat, xticklabels=15, yticklabels=15, cmap=colormap)
            plt.xlabel('trials')
            plt.ylabel('trials')
            plt.title(f'cell {c} lap by lap correlation')
            plt.show()
            count = count +1

        return r_coef, cos_f

    def binarize_firing(self, cells, thresh = 0.2):
        """ binarize all activity by thresh of max activity by each cell
        """

        PF_mat = self.mean_activity[:,:,cells]
        peak_firing = np.max(PF_mat, axis = (0,1))
        thresh_firing = peak_firing * thresh
        binarize_mat = np.zeros_like(PF_mat)
        for i in range(len(peak_firing)):
            binarize_mat[:,:,i] = (PF_mat[:,:,i] > thresh_firing[i])*1

        return binarize_mat

    def pop_corr_mat(self, cells = None, binarize = 0):
        """ use binarized firing mat to calculate correlation matrix for each trial. a way to quantify and visualize remapping.

        :param binarize: binarize firing intensity or not (binarize (default) = 1, not binarize = 0)
        :param thresh: % of max activity for each cell to count as "firing"
        :return: coef for correlation matrix
        """
        if cells is None:
            cells = self.PF_summary_peak.cell.unique().astype(int)
            print(f'using all PFs of {self.mouse} in envs: {list(self.PFs.keys())}')

        if binarize == 1:
            mat = self.binarize_firing()
        else:
            mat = self.mean_activity[:,:,cells]
        flat_mat = np.reshape(mat, (np.shape(mat)[0], -1))
        r_coef = np.corrcoef(flat_mat)
        colormap = sns.color_palette('Greys')
        sns.heatmap(r_coef, xticklabels=50, yticklabels=50, cmap = colormap)
        plt.xlabel('trials')
        plt.ylabel('trials')
        if hasattr(self, 'opto_on_lap'):
            for n in range(len(self.opto_on_lap)):
                plt.plot([self.opto_off_lap[n], self.opto_off_lap[n]], [self.opto_off_lap[n], self.opto_on_lap[n]],
                         color='red', linestyle='dashed')
                plt.plot([self.opto_on_lap[n], self.opto_off_lap[n]], [self.opto_off_lap[n], self.opto_off_lap[n]],
                         color='red', linestyle='dashed')
                plt.plot([self.opto_off_lap[n], self.opto_on_lap[n]], [self.opto_on_lap[n], self.opto_on_lap[n]],
                         color='red', linestyle='dashed')
                plt.plot([self.opto_on_lap[n], self.opto_on_lap[n]], [self.opto_on_lap[n], self.opto_off_lap[n]],
                         color='red', linestyle='dashed')
        for s in range(1, len(self.switch_lap)):
            plt.plot([0, self.switch_lap[s]], [self.switch_lap[s],self.switch_lap[s]], color = 'yellow', linestyle = 'dashed')
            plt.plot([self.switch_lap[s], self.switch_lap[s]], [0,self.switch_lap[s]], color = 'yellow', linestyle = 'dashed')

        plt.title(f'{self.mouse} lap by lap corr')

        return r_coef

    def pop_cos_sim(self, env='all env', cells = None):
        if env == 'all env':
            if cells is None:
                cells = self.PF_summary_peak['cell'].unique().astype(int)
            mat_flat = np.reshape(self.mean_activity[:, :, cells], (self.nlaps, -1))
        else:
            mat_laps = self.laps[env]
            if cells is None:
                cells = self.PFs[env]
            mat_flat = np.reshape(self.mean_activity[mat_laps[:, np.newaxis], :, cells], (len(mat_laps), -1))

        cos_sim = cosine_similarity(mat_flat)
        colormap = sns.color_palette('Greys')
        sns.heatmap(cos_sim, xticklabels=10, yticklabels=10, cmap = colormap)
        axis_label = 'laps in '+ env
        plt.xlabel(axis_label)
        plt.ylabel(axis_label)
        plt.title(f'{self.mouse} lap-by-lap cosine similarity in {env}')

        return cos_sim

    def firing_dark_hist(self, cells, laps, xrange, thresh = 0.2):
        """ binarized cell firing during dark env histogram, normalized by laps for comparison """

        bi_mat = self.binarize_firing(cells, thresh)
        dark_mat = np.sum(bi_mat[laps, :,:], axis = (0,1))/len(laps)

        # remove silent neurons
        # spont_fr = dark_mat[dark_mat !=0]

        plt.hist(dark_mat, range = xrange, alpha = 0.5, weights=np.ones(len(dark_mat)) / len(dark_mat))
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1))

        return dark_mat

    def before_after_PFs(self, env):
        """ plot spontaneous firing before becoming place cells"""

        # check if cells are PFs, prerequisites
        assert not self.PF_summary_peak.empty, "run check_PF_peak first"
        df = self.PF_summary_peak.loc[self.PF_summary_peak['env'] == env]
        cells = df['cell'].tolist()
        laps = self.laps[env]
        if env in self.opto_env:
            opto_lap = self.opto_env[env][0]
        #assert np.isin(cells, all_PFs) == len(cells), "not all cells are place cells"

        # plot average place fields for each cell and their firing before
        lap_range = slice(laps[0], laps[-1] + 1)
        bi_activity = self.binarize_firing(cells, 0.15)[lap_range,:,:]
        activity = self.mean_activity[lap_range, :, cells]

        for i in range(len(cells)):
            figcount = i % 3

            if figcount == 0:
                fig, ax = plt.subplots(3, 3, sharex=True)
                #ax = ax.ravel()

            # plot heatmap of mean activity for each cell with marked opto switch lap and emerge lap
            cell = df.iloc[i]['cell']
            emerge_lap = df.iloc[i]['emerge lap']
            sns.heatmap(activity[:,:,i], ax=ax[figcount, 0], cbar=False, xticklabels=False, yticklabels=10)
            ax[figcount, 0].set_title(f'{str(cell)}: emerge {str(emerge_lap)}')
            ax[figcount, 0].plot([0,40], [emerge_lap, emerge_lap], 'w:')
            if 'opto_lap' in locals():
                ax[figcount, 0].plot([0,40], [opto_lap, opto_lap], 'o:')

            # double check with binarization
            sns.heatmap(bi_activity[:,:,i], ax=ax[figcount, 1], cbar=False, xticklabels=False, yticklabels=10)
            left = df.iloc[i]['left']
            right = df.iloc[i]['right']
            ax[figcount, 1].set_title(f'{str(left)} to {str(right)}')


            # # histogram of firing location before emerge vs. after emerge laps
            # cell_bi_activity = bi_activity[:,:,i]
            # loc = cell_bi_activity * np.arange(0,40)
            # before_laps = loc[:emerge_lap[0],:]
            # after_laps = loc[emerge_lap[0]:,:]
            # before = before_laps[before_laps>0]
            # after = after_laps[after_laps>0]
            # if len(before)>0:
            #     ax[figcount, 2].hist(before, range = (0,40), alpha = 0.5, weights = np.ones(len(before))/len(before))
            #     ax[figcount, 2].hist(after, range = (0,40), alpha = 0.5, weights = np.ones(len(after))/len(after))
            #     ax[figcount, 2].yaxis.set_major_formatter(PercentFormatter(1))
            #     ax[figcount, 2].legend(['before', 'after'])
            # else:
            #     ax[figcount, 2].hist(after, range=(0, 40), alpha=0.5, weights = np.ones(len(after))/len(after))
            #     ax[figcount, 2].yaxis.set_major_formatter(PercentFormatter(1))
            #
            if figcount == 2:
                plt.show()

    def Antoine_struct(self):

        envs = ['control_day1', 'control_day2']
        for env in envs:
            print(env)
            s = {'name': self.mouse}
            df = self.PF_summary_peak.loc[self.PF_summary_peak['env'] == env]
            onset_lap = df['emerge lap'].tolist()
            s['onset_lap'] = onset_lap
            s['COM'] = self.COM_shift[env]
            laps = self.laps[env]
            PF_activity = np.zeros((len(laps), self.nbins, len(onset_lap)))    # 3d mat: PFs * bins * laps
            PF_activity[:] = np.nan
            #print(PF_activity.shape)
            for n in range(len(df)):
                #print(n, df.iloc[n]['cell'])
                left = df.iloc[n]['left']
                right = df.iloc[n]['right']
                PF_activity[:, left:right, n] = self.mean_activity[laps, left:right, df.iloc[n]['cell']]
                s['COM'][n] = s['COM'][n]+left
            s['PFs'] = PF_activity
            save_path = os.path.join('D:\\Opto\\Analysis\\Antoine_struct', f'{self.mouse}_{env}.mat')
            sio.savemat(save_path, {'s': s})
        return s

    def save_to_file(self, val = None, subdir = None, savename = None):
        """ autosave var with savename"""

        logger = init_logger(self.save_path, self.mouse)
        logger.info(f'{self.mouse} progress saved')

        if val is None:
            val = self.__dict__
        if savename is None:
            savename = 'PF'
        if subdir is None:
            data_path = os.path.join(self.save_path, f'{self.mouse}_{savename}.pickle')
        else:
            data_path = os.path.join(self.save_path, subdir, f'{self.mouse}_{savename}.pickle')
        print(f'Saving {savename} to file at {data_path}')
        with open(data_path, 'wb') as output_file:
            pickle.dump(val, output_file, pickle.HIGHEST_PROTOCOL)

    def load(self, subdir = None, var_name = 'PF'):
        """ load previously saved pickle file"""

        if subdir is None:
            pickle_file = os.path.join(self.save_path, f'{self.mouse}_{var_name}.pickle')
        else:
            pickle_file = os.path.join(self.save_path, subdir, f'{self.mouse}_{var_name}.pickle')
        file = open(pickle_file, 'rb')
        temp_dict = pickle.load(file)
        file.close()
        if var_name == 'PF':
            self.__dict__.update(temp_dict)
        else:
            return temp_dict

    def spatial_info(self, cells):
        s = np.zeros((self.nbins, 1))
        fr = self.mean_PF[:, cells]

# # pass the shuffle
# if cell_peak[cell] > np.max((shuffle_sig[cell], minDF)):  #  and cell_low[cell] < np.quantile(shuffle_low, pval)
#
#     # check for possibility of multiple PFs:
#     sig_peak = np.where(cell_mean[:, cell] > shuffle_sig[cell])[0]   # check above sig shuffle region
#     region = np.where(cell_mean[:, cell] > cell_peak[cell] * bndry_thresh)[0]   # continuous regions
#     bndry = self._group_id(region, 3)   # n potential PFs
#     for n in range(len(bndry)):
#         PF_loc_left = bndry[n][0]
#         PF_loc_right = bndry[n][1]
#
#         # peak inside the PF boundary
#         if np.sum(np.isin(sig_peak, range(PF_loc_left, PF_loc_right))) <= 0:
#             pass
#         elif PF_loc_right - PF_loc_left > maxWidth:  # PF width has to be smaller than maxWidth
#             pass
#         else:
#             # min Fc3_DF to count as firing
#             temp_pf = self.mean_activity[laps, PF_loc_left:PF_loc_right, cell]
#             temp_thresh = temp_pf > cell_peak[cell]*bndry_thresh
#             # ratio: total laps is the # laps
#             firing_lap = np.sum(temp_thresh, axis=1)
#             firing_lap_ind = np.where(firing_lap > 0)[0]
#
#             if len(firing_lap_ind) < min_laps:   # fire at least min_laps times over all laps
#                 pass
#             else:
#                 first_firing_lap = self._first_lap_PF(firing_lap_ind)
#
#                 if np.isnan(first_firing_lap):  # fire at least 4 out of 6 laps consistently at some point
#                     pass
#                 else:
#                     ratio_temp = np.sum(firing_lap > 0)/(nlaps-first_firing_lap)
#                     in_field_F = np.mean(self.mean_activity[laps, PF_loc_left:PF_loc_right, cell])
#                     out_field = np.setdiff1d(np.arange(self.nbins), np.arange(PF_loc_left, PF_loc_right))
#                     out_field_F = np.mean(self.mean_activity[laps[:, np.newaxis], out_field, cell])
#                     out_in_ratio = out_field_F/ in_field_F
#                     COM, width, precision, slope, r2, p, com_by_laps = self._PF_features(temp_pf, PF_loc_left)
#                     PF_features.append([PF_env, cell, PF_id, first_firing_lap, COM, PF_loc_left, PF_loc_right,
#                                         np.round(ratio_temp,2), precision, out_in_ratio, slope, r2, p])
#                     bw_shift.append(com_by_laps)
#                     PF_id = PF_id+1