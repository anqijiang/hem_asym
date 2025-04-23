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


def load_py_var_mat(day_path, keywords):
    """ find the matlab file under day_path directory"""
    onlyfiles = [f for f in listdir(day_path) if os.path.isfile(os.path.join(day_path, f))]
    file_name = [f for f in onlyfiles if f.endswith(keywords)][0]
    print('loading file: ', file_name)
    file = scipy.io.loadmat(os.path.join(day_path, file_name))

    return file


# icasso functions: bootstrap_fun & unmixing_fun
def bootstrap_fun(data, generator):
    return data[generator.choice(range(data.shape[0]), size=data.shape[0]), :]


def unmixing_fun(ica):
    return ica.components_


class Cell_struct:
    """
    this class structure finds and takes 2d matrix (cell * frame) without behavioral correlates. The goal is to reveal
    intrinsic cell network structure. Refer to Lopes-dos-Santos 2013 for details
    """
    def __init__(self, mouse, env, day, binwindow, savetitle):

        self.mouse = mouse
        self.day = day
        self.env = env
        self.path = os.path.join('D:\\Opto\\Analysis', mouse, day)
        self.binwindow = binwindow
        binwindow_str = '_'.join(map(str, binwindow))
        self.savetitle = f'{savetitle}_{binwindow_str}'

        # loading raw data
        mat_file = load_py_var_mat(self.path, 'align_cell_mean.mat')
        mean_activity = mat_file['cell_binMean'].transpose((1, 0, 2))
        mean_activity[np.isnan(mean_activity)] = 0
        self.mean_activity = mean_activity
        self.nlaps = self.mean_activity.shape[0]
        self.raw = mat_file['Fc3_DF']
        self.mat = np.copy(self.raw)
        self.del_frames = []
        self.ncells = np.shape(self.mat)[1]

        self.switch_frame = mat_file['switch_frame'][0, :].astype('int') -1
        self.switch = np.copy(self.switch_frame)
        assert len(self.env) == len(self.switch_frame)+1, "# of env does not match with laps"
        switch_lap = mat_file['env_switch_lap'][:, 0].astype('int') -1
        self.switch_lap = np.concatenate(([0], switch_lap, [self.nlaps]))  # add 0 as the first switch
        laps = [np.arange(self.switch_lap[x], self.switch_lap[x + 1]) for x in range(len(self.switch_lap) - 1)]
        self.laps = dict(zip(self.env, laps))

        # check for cache in the directory for analyzed data
        pickle_assemb_bin = os.path.join(self.path, f'{self.mouse}_{self.savetitle}_assemb_bin.pickle')
        print(pickle_assemb_bin)
        if os.path.exists(pickle_assemb_bin):
            self.load(pickle_assemb_bin, 'assemb_bin')
        else:
            print(f'no cache for {self.mouse}')
            self.assemb_bin = {}
            keys = ['z_mat', 'inactive_neuron', 'cov_mat', 'n_sig_assembly', 'n_sig_neurons',
                    'assemb_strength', 'weights', 'assemb_neuron', 'frame_scale']
            for bin in binwindow:
                self.assemb_bin[bin] = {}
                for key in keys:
                    self.assemb_bin[bin][key] = {}

        # relate to experimental setup
        self.cond = {}
        if all(var in mat_file.keys() for var in ['onFrames_ind', 'offFrames_ind']):

            opto_pq = os.path.join(self.path, f'{self.mouse}_{self.savetitle}_opto_summary.parquet')
            if os.path.exists(opto_pq):
                print('loading opto_summary from cache')
                self.opto_summary = pd.read_parquet(opto_pq, engine='fastparquet')
                nconds_opto = self.opto_summary.shape[0]
                nconds_assemb_bin = len(self.assemb_bin[self.binwindow[0]]['n_sig_assembly'])
                if nconds_opto > nconds_assemb_bin:
                    self.control_env()
                elif nconds_opto < nconds_assemb_bin:
                    y = input(f'{self.mouse} length of opto_summary {nconds_opto} and assemb_bin {nconds_assemb_bin} not matched ')
                    display(self.opto_summary)
                    print(self.assemb_bin[self.binwindow[0]]['z_mat'].keys())
                sep_frame = self.opto_summary['switch frame'].dropna().astype('int').to_list()
                self.sep_frame = np.array(sep_frame[1:])

            else:
                opto_summary = pd.DataFrame(columns=['env', 'switch frame', 'opto on frame', 'opto off frame',
                                                     'opto on lap', 'opto off lap'])
                # opto_summary.loc[:, 'switch frame'] = np.concatenate(([0], self.switch_frame))
                opto_summary = opto_summary.astype({'switch frame': 'Int64', 'opto on frame': 'Int64',
                                                    'opto off frame': 'Int64'})
                self.opto_summary = opto_summary
                self.sep_frame = None

            opto_off_frame = mat_file['offFrames_ind'][:, 0].astype('int') -1  # matlab indexing to python indexing
            opto_on_frame = mat_file['onFrames_ind'][:, 0].astype('int') -1
            opto_length_frame = opto_off_frame - opto_on_frame
            self.opto_off_frame = opto_off_frame[opto_length_frame > 100]  # remove opto artifact
            self.opto_on_frame = opto_on_frame[opto_length_frame > 100]
            self.opto_on = np.copy(self.opto_on_frame)
            self.opto_off = np.copy(self.opto_off_frame)

            opto_on_lap = np.fmax(mat_file['opto_on_lap'][:, 0].astype('int') - 1, 0)
            opto_off_lap = mat_file['opto_off_lap'][:, 0].astype('int') - 1
            opto_length_lap = opto_off_lap - opto_on_lap
            self.opto_off_lap = opto_off_lap[opto_length_lap > 1]
            self.opto_on_lap = opto_on_lap[opto_length_lap > 1]

    def load(self, pickle_file, var):
        """ load previously saved pickle file"""

        file = open(pickle_file, 'rb')
        print(f'Loading {self.mouse} stored {var} from cache: {pickle_file}')
        temp_dict = pickle.load(file)
        file.close()
        self.__dict__.update({var:temp_dict})

    def save_to_file(self, var):
        """ auto save class """

        data_path = os.path.join(self.path, f'{self.mouse}_{self.savetitle}_{var}.pickle')
        print(f'Saving {self.mouse} {var} to file at {data_path}')

        with open(data_path, 'wb') as output_file:
            pickle.dump(self.__dict__[var], output_file, pickle.HIGHEST_PROTOCOL)

    def plot_raster(self, binwindow = None, thresh = 0, smooth=3, show = 1, save=1):

        if binwindow is None:
            binwindow = self.binwindow[0]

        if len(self.assemb_bin[binwindow]['z_mat']) < 1:
            self.bin_norm_mat(singleb=True)
        elif len(self.assemb_bin[binwindow]['z_mat']) != len(self.cond) and len(self.cond) > 0:
            self.bin_norm_mat(singleb=True)

        loc = {}
        cells_per_frame = {}
        opto_on_conds = self.opto_summary['opto on frame'].dropna().index.to_list()  # conds with opto on
        cell_counts = pd.DataFrame()
        conds_dict = {1:['control'], 2: ['during', 'after'], 3:['before', 'during', 'after']}

        for k in self.assemb_bin[binwindow]['z_mat']:
            env = self.opto_summary.loc[k, 'env']
            conds_env = self.opto_summary.loc[self.opto_summary['env']==env].index.to_list()
            nconds = len(conds_env)

            loc[k] = np.argwhere(self.assemb_bin[binwindow]['z_mat'][k] > thresh)
            ncells = np.max(loc[k][:,1])
            nframes = np.max(loc[k][:,0]) - np.min(loc[k][:,0]) + 1

            # ax[1] cell count per frame
            cells_per_frame[k] = np.zeros((nframes, 1))
            count = Counter(list(loc[k][:, 0]))    # {frame: # cells firing}

            cells_per_frame[k] = [*count.values()]
            cell_counts_local = pd.DataFrame(cells_per_frame[k], columns=['active cell count'])
            #print(conds_env)
            cell_counts_local['cond'] = conds_dict[nconds][k-conds_env[0]]
            cell_counts_local['env'] = env
            cell_counts = pd.concat([cell_counts, cell_counts_local])

            if show ==1:
                # ax[0] raster plot
                fig, ax = plt.subplots(2, 1, sharex=True, figsize = (2*nframes/ncells, 6))
                ax[0].scatter(loc[k][:, 0], loc[k][:, 1], linewidths=0.5, s=1)
                ax[0].set_xlim([np.min(loc[k][:,0]) -5, np.max(loc[k][:,0])+5])
                plt.xlabel('bins')
                ax[0].set_ylabel('cells')
                # fig.set_size_inches(3*nframes/ncells, 6)
                smoothed_frame = uniform_filter1d(list(cells_per_frame[k]), size=smooth)  # smooth cells_per_frame for plotting
                horz_line_y = np.max(smoothed_frame) + 5
                ax[1].set_xlim([np.min(loc[k][:, 0]) - 5, np.max(loc[k][:, 0]) + 5])
                ax[1].plot(smoothed_frame)
                ax[1].set_ylabel('# cells firing per frame')
                ax[1].set_title(f'# cells firing per frame, smooth window {smooth}')

                # add opto on period red box
                if k in opto_on_conds:
                    opto_on_z = (self.opto_summary.loc[k, 'opto on frame'] - self.opto_summary.loc[k, 'switch frame']) * \
                                self.assemb_bin[binwindow]['frame_scale'][k]
                    opto_off_z = (self.opto_summary.loc[k, 'opto off frame'] - self.opto_summary.loc[k, 'switch frame']) * \
                                 self.assemb_bin[binwindow]['frame_scale'][k]
                    ax[0].add_patch(Rectangle((opto_on_z, 0), opto_off_z - opto_on_z, ncells,
                                              color="red", linestyle='dotted', fc='none'))
                    ax[1].plot([opto_on_z, opto_off_z], [horz_line_y, horz_line_y], color='red')  # add opto

                titleName = f'{self.mouse} raster plot cond {k} {env}'
                plt.suptitle(titleName)
                fig.set_dpi(80)

            if save == 1:
                plt.savefig(os.path.join(self.path, titleName), bbox_inches='tight')
            plt.show()

        cell_counts.reset_index(drop=True, inplace=True)

        return loc, cells_per_frame, cell_counts

    def plot_opto_effect(self, loc = None, cells_per_frame = None):

        if (loc is None) | (cells_per_frame is None):
            # no conds created yet or no opto conds, conds separated by envs instead of opto
            if (len(self.cond) == 0) | (len(self.cond) == len(self.env)):
                print('separate conditions by opto')
                self.control_env()
            loc, cells_per_frame, cell_counts_df = self.plot_raster(show = 0, save=0)

        opto_on_conds = self.opto_summary['opto on frame'].dropna().index.to_list()  # conds with opto on
        legend_dict = {2: ['opto on', 'opto off'], 3:['before', 'opto on', 'after']}

        for k in opto_on_conds:
            env = self.opto_summary.loc[k, 'env']
            hist_conds = self.opto_summary.loc[self.opto_summary['env'] == env].index.to_list()
            control_env = f'control_{env[5:]}'
            control_conds = self.opto_summary.loc[self.opto_summary['env'] == control_env].index.to_list()

            fig, axs = plt.subplots(1,2, sharey=True, sharex=True)
            for n in hist_conds:
                axs[0].hist(cells_per_frame[n], alpha = 0.5, fill = False, density=True, histtype='step')
            #axs[0].yaxis.set_major_formatter(PercentFormatter(1))
            for m in control_conds:
                axs[1].hist(cells_per_frame[m], alpha = 0.5, fill = False, density=True, histtype='step')
            axs[0].set_title(f'{env}')
            axs[1].set_title(f'{control_env}')
            print(hist_conds)
            axs[0].legend(legend_dict[len(hist_conds)])
            axs[1].legend(legend_dict[len(hist_conds)])
            axs[0].set_ylabel('probability density')
            axs[1].set_xlabel('cells firing at any frame')
            plt.suptitle(f'{self.mouse}')
            plt.savefig(os.path.join(self.path, f'{self.mouse} raster count in {env}'))
            plt.show()

        for env in ['later_day1', 'later_day2', 'first_day1', 'first_day2']:
            exp_env = f'opto_{env}'
            control_env = f'control_{env}'
            exp_data = cell_counts_df.loc[cell_counts_df['env'] == exp_env]
            control_data = cell_counts_df.loc[cell_counts_df['env'] == control_env]
            if (len(exp_data)>0) & (len(control_data)>0):
                sns.pointplot(data=exp_data, x='cond', y='active cell count', alpha=0.5, color='red')
                sns.pointplot(data=control_data, x='cond', y='active cell count', alpha=0.5, color='k')
                plt.title(f'{self.mouse} in {exp_env} (red) and {control_env} (black)')
                plt.savefig(os.path.join(self.path, f'{self.mouse} raster count in {exp_env} and {control_env}'))
                plt.show()

        return cells_per_frame, cell_counts_df

    def bad_frames(self, bad_cond, bad_frame):
        """ delete bad frames from the raw data """

        m = np.mean(self.cond[bad_cond], axis=1)
        plt.plot(m)
        plt.plot([bad_frame[0], bad_frame[-1]], [np.max(m), np.max(m)], '-r')
        plt.title(f'cond {bad_cond} frames to delete')
        plt.show()

        if 'cond' in self.__dict__:
            # check before deleting
            cont = input('continue to delete frames? 1 to continue, any other key to exit \n')
            if int(cont) == 1:
                print(f'deleting frames {bad_frame[0]} to {bad_frame[-1]}, total {len(bad_frame)} frame, in cond {bad_cond}')
                logging.info(f'Step 0: {self.mouse} deleting bad frames {bad_frame[0]} to {bad_frame[-1]} in cond {bad_cond}')
                good_frame = np.delete(self.cond[bad_cond], bad_frame, axis=0)
                cond_frame = self.opto_summary.loc[bad_cond, 'switch frame']
                self.mat = np.delete(self.mat, bad_frame + cond_frame, axis=0)
                n = np.mean(good_frame, axis=1)
                plt.plot(n)
                plt.title(f'cond {bad_cond} remaining good frames')
                plt.show()
                self.cond[bad_cond] = good_frame
                self.del_frames.extend([*bad_frame+cond_frame])
                assert len(self.del_frames) + np.shape(self.mat)[0] == np.shape(self.raw)[0], '# frames deleted unmatched'

                # update sep_frames and opto_on/ off frames
                self.sep_frame = None
                cut_frame = bad_frame[-1] + cond_frame
                self.opto_on_frame = [new - len(bad_frame) if new > cut_frame else new for new in self.opto_on_frame]
                self.opto_off_frame = [new - len(bad_frame) if new > cut_frame else new for new in self.opto_off_frame]
                self.switch_frame = [new - len(bad_frame) if new > cut_frame else new for new in self.switch_frame]
                self._separate_cond()

                # update control envs based on the new cond that has deleted bad frames already
                if len(self.cond) > len(self.sep_frame) + 1:
                    [self.cond.pop(key) for key in np.arange(len(self.sep_frame)+1, len(self.cond))]
                    if self.opto_summary.shape[0] > len(self.sep_frame)+1:
                        self.opto_summary = self.opto_summary.drop(np.arange(len(self.sep_frame)+1, self.opto_summary.shape[0]))
                    self.control_env()

    def reset_bad_frames(self):
        """ if accidentally deleted wrong frames, load original data again"""

        print('resetting variables: mat, del_frames, switch_frame, opto_on_frame, opto_off_frame')
        self.mat = np.copy(self.raw)
        self.del_frames = []
        # relate to experimental setup
        self.cond = {}
        self.switch_frame = self.switch
        self.sep_frame = None
        self.opto_off_frame = self.opto_off
        self.opto_on_frame = self.opto_on
        self._separate_cond()

    def _separate_cond(self, sep_frame = None):
        """ separate original mat into different chunks by experimental conditions"""

        # check if self.opto_summary already loaded from cache
        if self.opto_summary.shape[0] > len(self.cond) and len(self.cond) >0:
            print('already ran _separate_cond before')
            return self.opto_summary

        # separate conditions by envs
        if sep_frame is None:
            self.sep_frame = self.switch_frame

        # separate conditions by opto on/off and envs
        elif sep_frame == 'opto':
            chunks = np.concatenate((self.switch_frame, self.opto_on_frame, self.opto_off_frame))
            sep_frame = np.sort(chunks)
            ind = np.concatenate(([True], (sep_frame[1:] - sep_frame[:-1]) > 10))  # remove duplicates
            sep_frame = sep_frame[ind]
            self.sep_frame = sep_frame[sep_frame>0]

        # overwrite self.cond if exists already
        if len(self.cond) > 0:
            k = [*self.cond.keys()]
            print(f'overwriting self.cond: {k}')
            for keys in k:
                self.cond.pop(keys, None)

        if len(self.sep_frame) > 1:
            frames_cond = np.split(self.mat, self.sep_frame)
            self.cond = dict(enumerate(frames_cond))
        else:
            self.cond = {'0': self.mat}

        # overwrite self.opto_summary if exists already
        if self.opto_summary.shape[0] >0:
            print('overwriting existing opto_summary')
            display(self.opto_summary)
            opto_summary = self.opto_summary.head(0).copy()
        else:
            opto_summary = self.opto_summary.copy()

        # fill in opto_summary conds
        nframes = np.shape(self.mat)[0]
        sep_conds = np.concatenate((self.sep_frame, [nframes]))
        switch_conds = np.concatenate((self.switch_frame, [nframes]))
        assert len(switch_conds) == len(self.env), "length of sep_frame does not match length of envs"

        # match conds with envs in self.opto_summary
        sep = 0
        for n in range(len(switch_conds)):
            while sep < len(sep_conds) and sep_conds[sep] <= switch_conds[n]:
                opto_summary.loc[sep, 'env'] = self.env[n]
                sep = sep+1

        self.opto_summary = opto_summary
        sep_frame0 = np.concatenate(([0], self.sep_frame))
        self.opto_summary['switch frame'] = sep_frame0
        opto_ind = [bisect.bisect_right(self.sep_frame, x) for x in self.opto_on_frame]
        self.opto_summary.loc[opto_ind, 'opto on frame'] = self.opto_on_frame
        self.opto_summary.loc[opto_ind, 'opto off frame'] = self.opto_off_frame
        self.opto_summary.loc[opto_ind, 'opto on lap'] = self.opto_on_lap
        self.opto_summary.loc[opto_ind, 'opto off lap'] = self.opto_off_lap
        display(self.opto_summary)

        logging.info(f'Step 0: separate file into different conds using {self.sep_frame} '
                     f'into {len(self.cond)} chunks')

        return self.cond

    def control_env(self):
        """ separate control env to compare opto effects on ensembles. make 2 copies of control to self.cond
        to compare with opto first and opto later conditions. """

        opto_control_env = ['opto_first_day1', 'opto_first_day2', 'opto_later_day1', 'opto_later_day2']
        # drop duplicated rows
        if len(self.opto_summary) > 22:
            self.opto_summary.dropna(subset=['frame scale'], inplace=True)
            opto_pq = os.path.join(self.path, f'{self.mouse}_{self.savetitle}_opto_summary.parquet')
            self.opto_summary.to_parquet(opto_pq, compression='gzip')  # save to parquet

        if 'cond' not in self.__dict__:
            print('loading from cache')
        elif any(np.isin(['control_first_day1', 'control_first_day2', 'control_later_day1', 'control_later_day2'], self.opto_summary.env)):
            print('already ran control_env before')
        elif len(self.cond) ==0 or len(self.cond) == len(self.env):
            print('running self._separate_cond')
            self._separate_cond('opto')
            self.control_env()
        elif len(self.cond) > len(self.sep_frame) +1:
            print('already copied control env')
        else:
            if 'control_day1' in self.env and 'control_day2' in self.env:
                day1_cond = self.opto_summary.loc[self.opto_summary['env']=='control_day1'].index.values
                control_day1 = self.cond[day1_cond[0]]
                day2_cond = self.opto_summary.loc[self.opto_summary['env']=='control_day2'].index.values
                control_day2 = self.cond[day2_cond[0]]
            else:
                return print('no control env in this file')

            # add copies of control and separate it like in opto_first and opto_later
            opto_env = np.intersect1d(self.env, opto_control_env)
            control_chunks = dict(zip(opto_control_env,[control_day1, control_day2, control_day1, control_day2]))
            concat_dict = {'env':[]}

            for e in opto_env:
                cond = self.opto_summary.loc[self.opto_summary['env'] == e]
                opto_on_frame = cond['opto on frame'].dropna().values
                opto_off_frame = cond['opto off frame'].dropna().values
                opto_length = opto_off_frame - opto_on_frame

                # if opto on in the environment
                if sum(opto_length) > 10:
                    opto_length = int(opto_length[0])
                    control_env_day = e[-5:]
                    switch = min(cond['switch frame'].values)
                    opto_start = opto_on_frame - switch   # first frame opto is on
                    opto_start = int(opto_start[0])

                    # if opto_first or opto_later
                    if opto_start > 10:    # opto later
                        nconds = len(self.cond)
                        self.cond[nconds] = control_chunks[e][:opto_start]   # before opto on
                        control_name = 'control_later' + control_env_day
                        concat_dict['env'].append(control_name)
                    elif (opto_start >-1) and (opto_start < 11):    # opto first
                        control_name = 'control_first' + control_env_day
                    else:
                        return print('check opto_summary: switch frame and opto on frame unmatched')

                    nconds = len(self.cond)
                    self.cond[nconds] = control_chunks[e][opto_start:opto_start+opto_length]  # opto on
                    concat_dict['env'].append(control_name)

                    if len(control_chunks[e][opto_start+opto_length:]) >100:   # make sure enough frames after opto on
                        self.cond[nconds+1] = control_chunks[e][opto_start+opto_length:]
                        concat_dict['env'].append(control_name)

            concat_df = pd.DataFrame(concat_dict)
            self.opto_summary = pd.concat([self.opto_summary, concat_df], ignore_index=True)
            opto_pq = os.path.join(self.path, f'{self.mouse}_{self.savetitle}_opto_summary.parquet')
            self.opto_summary.to_parquet(opto_pq, compression='gzip')  # save to parquet

        return self.opto_summary

    def construct_mat(self, thresh = 0.3, sep_frame = None):
        """ Step 1: binarize and z-transform original mat by time window and % of max firing by each cell.
        :param thresh: % of the max firing to be counted as firing
        :param binwindow: how many frames to be counted as one bin. each frame is roughly 32.27ms; 12 is roughly 400ms
        """

        # binarize firing based on max activity of each cell
        if len(self.cond) == 0:
            self._separate_cond(sep_frame)
        binarize_firing = {}

        for k in self.cond:
            mat = self.cond[k]
            peak_firing = np.max(mat, axis=0)
            thresh_firing = peak_firing * thresh
            binarize_firing[k] = (mat> thresh_firing)*1

        # bin frames based on binwindow
        logging.info(f'Step 1: binarize original mat using threshold {thresh}')
        self.bin_norm_mat(binarize_firing, sep_frame, 1, 10)

    def bin_norm_mat(self, dict_mat=None, sep_frame = None, binarize=0, plot_cells=10, singleb = False):
        """ Step 1 Alternative: bin by the mean value and z-transform the mat
        :param mat: 2d matrix (frame * cell) to be binned
        :param binarize: if mat is binarized or not
        :param plot_cells: how many cells to visualize to compare z_mat with mat
        """

        if dict_mat is None:
            if len(self.cond) == 0:
                self._separate_cond(sep_frame)
            dict_mat = self.cond

        for b in self.binwindow:

            print(f'Step 1: {self.mouse} z-transform by bin {b}, roughly {np.round(b*32.28, -1)} ms')

        # remove silent neuron
            for k in dict_mat:
                mat = dict_mat[k]
                nframes = np.shape(mat)[0]
                ncells = np.shape(mat)[1]

                df = pd.DataFrame(mat)
                ind = np.arange(nframes)
                bins = np.arange(0, nframes+b, b)
                binned = pd.cut(ind, bins, include_lowest=True)
                df['bin'] = binned
                if binarize == 0:
                    df_binned = df.groupby('bin').mean()
                elif binarize == 1:
                    df_binned = df.groupby('bin').max()
                bin_mat = df_binned.iloc[0:-1, :ncells].to_numpy()

                inactive_neuron = np.where(np.sum(bin_mat, axis=0) == 0)[0]
                active_mat = bin_mat[:, np.nansum(bin_mat, axis=0) > 0]
                ncells = ncells-len(inactive_neuron)
                if ncells != self.ncells:
                    print(f'Note: inactive neurons in cond {k}: {inactive_neuron}')
                    self.assemb_bin[b]['inactive_neuron'][k] = inactive_neuron
                    # self.mean_activity = self.mean_activity[:, :, np.sum(mat, axis=0) > 0]
                    self.assemb_bin[b]['cov_mat'][k] = np.zeros((ncells, ncells))

                z_mat = stats.zscore(active_mat[:, :], axis=0, nan_policy='omit')
                z_mat = z_mat[~np.isnan(z_mat).any(axis=1), :]     # remove any nans in z_mat
                self.assemb_bin[b]['z_mat'][k] = z_mat
                self.assemb_bin[b]['frame_scale'][k] = np.shape(z_mat)[0] / np.shape(mat)[0]

                # update variables associated with z_mat for later calculation and plotting
                self.opto_summary.loc[k, 'frame scale'] = z_mat.shape[0]/nframes

            if singleb:
                return None

        # compare z_mat with original mat
        self.check_bin(np.arange(plot_cells))

    def check_bin(self, cells):
        """ visualize original mat and constructed mat to ensure a valid representation after binarization
        """

        nbins = len(self.binwindow)
        for k in self.cond:
            fig, axs = plt.subplots(nbins+1, 1)
            axs = axs.ravel()

            axs_count = 0
            sns.heatmap(ax=axs[axs_count], data=self.cond[k][:,cells].transpose(), cbar=False, cmap=plt.cm.gray_r)
            for key, b in self.assemb_bin.items():
                axs_count = axs_count+1
                sns.heatmap(ax=axs[axs_count], data=b['z_mat'][k][:,cells].transpose(), cbar=False, cmap=plt.cm.gray_r)
                axs[axs_count].set_title(f'norm data bin {key}')
            axs[0].set_title('original data')
            fig.suptitle(f'cond {(k)}')
            #fig.tight_layout()
            plt.show()
            plt.close()

    def n_assembly(self):
        """ Step 2: determine # of cell assemblies
        """

        logging.basicConfig(filename=os.path.join(self.path, f'{self.savetitle}_params.log'), level=logging.DEBUG,
                            format='%(asctime)s %(message)s', datefmt='%m/%d/%y %I:%M %p', filemode='w')
        logging.getLogger('matplotlib.font_manager').disabled = True

        for key, b in self.assemb_bin.items():

            evals = {}

            # covariance mat
            for k in b['z_mat']:
                b['cov_mat'][k] = np.cov(b['z_mat'][k].transpose())
                evals[k], _ = np.linalg.eigh(b['cov_mat'][k])
                evals[k] = np.sort(evals[k])[::-1]

                ncells = b['z_mat'][k].shape[1]
                nbins = b['z_mat'][k].shape[0]

                # find the boundary of the eigenvalues of the cov mat (Marcenko-Pastur)
                lambda_max = (1 + np.sqrt(ncells / nbins)) ** 2
                print(f'Cond {k} sig high thresh: {lambda_max}')
                lambda_min = (1 - np.sqrt(ncells / nbins)) ** 2
                print(f'Cond {k} sig low thresh: {lambda_min}')

                b['n_sig_assembly'][k] = sum(evals[k] > lambda_max)
                temp = b['n_sig_assembly'][k]
                logging.info(f'Step 2: identified {temp} cell assemblies '
                             f'out of {ncells} total neurons in cond {k}')
                b['n_sig_neurons'][k] = sum(evals[k] < lambda_min)
                temp_neurons = b['n_sig_neurons'][k]
                print(f'# neurons: {temp_neurons}')

    def ICA(self, z_thresh=2, niter = 25, clusterdist = 0.65, clusterthresh = 0.5, seed = 42):
        """ Step 3: first reduce z_mat dim by n_sig_assembly, then fastICA on projected space,
        icasso to boost reliability of ica (bootstrap niter times and cluster result components) """

        logging.info(f'Step 3: icasso: repeat ICA {niter} times')
        logging.info(f'Step 3: cells with weight zscore > {z_thresh} are associated with the pattern')

        opto_on_conds = self.opto_summary['opto on frame'].dropna().index.to_list()  # conds with opto on

        ica_params = {}
        icasso = Icasso(FastICA, ica_params=ica_params, iterations=niter, bootstrap=True, vary_init=True)

        for key, b in self.assemb_bin.items():
            z_mat = b['z_mat']
            n_sig_assembly = b['n_sig_assembly']

            for k in z_mat:
                # PCA on z_mat
                pca = PCA(n_components=n_sig_assembly[k])
                reduced_mat = pca.fit_transform(z_mat[k])
                pca_weights = pca.components_

                # icasso on ICA
                icasso.fit(data=reduced_mat, fit_params={}, random_state=seed,
                           bootstrap_fun=bootstrap_fun, unmixing_fun=unmixing_fun)
                distance = clusterdist
                W_, scores = icasso.get_centrotype_unmixing(distance=distance)

                clustermore = 0
                clusterless = 0
                while W_.shape[0] > n_sig_assembly[k] * (1 + clusterthresh):
                    # overfit: Getting too many clusters! Increase within cluster distance to reduce # of clusters
                    distance = distance + 0.025
                    W_, scores = icasso.get_centrotype_unmixing(distance=distance)
                    clusterless = clusterless +1
                while W_.shape[0] < n_sig_assembly[k]:
                    # underfit: reduce within cluster distance to make sure getting more clusters than n_sig_assembly
                    distance = distance - 0.025
                    W_, scores = icasso.get_centrotype_unmixing(distance=distance)
                    clustermore = clustermore +1
                print(f'cond {k}: unmixing distance {distance}, {n_sig_assembly[k]} assemblies from {W_.shape[0]} clusters, '
                      f'after {clusterless} times increasing distance & {clustermore} times reducing distance')

                ica_weights = W_[:n_sig_assembly[k], :]
                b['weights'][k] = ica_weights @ pca_weights  # weights to project neurons into ICA space

                # Calculate coactivation strength
                ncells = np.shape(z_mat[k])[1]
                nbins = np.shape(z_mat[k])[0]
                diag_ind = np.diag_indices(ncells)
                b['assemb_strength'][k] = np.zeros((n_sig_assembly[k], nbins))
                for n in range(n_sig_assembly[k]):
                    P = np.outer(b['weights'][k][n, :], b['weights'][k][n, :])
                    P[diag_ind] = 0   # set diagonal to zeros to remove contribution from a single cell
                    R = (z_mat[k] @ P) * (z_mat[k])
                    b['assemb_strength'][k][n, :] = np.sum(R, axis=1)

                mean_strength = np.mean(b['assemb_strength'][k], axis=0)
                plt.plot(mean_strength)
                # add opto on period to the plot
                if k in opto_on_conds:
                    opto_on_z = (self.opto_summary.loc[k, 'opto on frame'] - self.opto_summary.loc[k, 'switch frame']) * \
                                b['frame_scale'][k]
                    opto_off_z = (self.opto_summary.loc[k, 'opto off frame'] - self.opto_summary.loc[k, 'switch frame']) * \
                                b['frame_scale'][k]
                    smax = np.max(mean_strength)
                    ssd = np.std(mean_strength)
                    plt.plot([opto_on_z, opto_off_z], [smax + ssd, smax + ssd], color='red')
                env = self.opto_summary.loc[k, 'env']

                plt.title(f'{self.mouse} {k}: {env} mean assembly strength binwindow {key}')
                plt.show()

                # identify neurons with significant contribution to assembly
                z_weights = stats.zscore(b['weights'][k], axis = 1)
                n_assemb, n_sig_neuron = np.where((z_weights > z_thresh) | (z_weights < -z_thresh))
                b['assemb_neuron'][k] = {}
                for n in range(max(n_assemb)+1):
                    b['assemb_neuron'][k][n] = n_sig_neuron[n_assemb == n]

        self.save_to_file('assemb_bin')

    def check_assemb(self, assemb, cond):
        """ plot assembly strength against z_mat of all neurons identified with the assemble to double check the results

        :param assemb: assembles (eg. np.arange(10) or [1,3,5])
        :param cond: which experimental conditions to check eg. [0,1]
        """

        opto_on_conds = self.opto_summary['opto on frame'].dropna().index.to_list()  # conds with opto on
        for key, b in self.assemb_bin.items():
            for k in cond:
                if k in opto_on_conds:
                    opto_on_frame = (self.opto_summary.loc[k, 'opto on frame'] - self.opto_summary.loc[k, 'switch frame'])*\
                                    b['frame_scale'][k]
                    opto_off_frame = (self.opto_summary.loc[k, 'opto off frame'] - self.opto_summary.loc[k, 'switch frame']) * \
                                    b['frame_scale'][k]
                else:
                    opto_on_frame = None

                for i in assemb:
                    titleName = f'Cond {k}, coactivation pattern {i} binwindow {key}'
                    neurons = b['assemb_neuron'][k][i]
                    strength = b['assemb_strength'][k][i, :]
                    z_neurons = b['z_mat'][k][:, neurons]

                    fig, axs = plt.subplots(2, 1, sharex=True)
                    axs = axs.ravel()
                    sns.heatmap(ax=axs[0], data=z_neurons.transpose(), cbar=False, cmap=plt.cm.gray_r, yticklabels=neurons)
                    axs[0].set_title(titleName)
                    axs[0].set_ylabel('neurons in ensemble')
                    axs[1].plot(strength)
                    axs[1].set_title('assembly strength')
                    axs[1].set_xlabel('frames')

                    if opto_on_frame is not None:
                        smax = np.max(strength)
                        ssd = np.std(strength)
                        axs[1].plot([opto_on_frame, opto_off_frame], [smax+ssd, smax+ssd], color = 'red')

                    fig.tight_layout()
                    plt.show()

    def draw_coactive_events(self, binwindow, cond, assemb, slidewindow=5):
        """ plot coactivation events detected in assemb in cond through local maxima of assemb strength """

        sep_frame = np.concatenate(([0], self.sep_frame))

        # loop thru each condition
        for b in binwindow:
            for k in cond:
                assemb_strength = self.assemb_bin[b]['assemb_strength'][k][assemb, :]
                sd = np.std(assemb_strength, axis=1)
                thresh = np.percentile(assemb_strength, 90, axis=1)

                # loop thru each assembly
                for n in range(len(assemb)):
                    time, _ = find_peaks(assemb_strength[n, :], prominence=2*sd[n]+thresh[n], distance=5)
                    plt.plot(assemb_strength[n, :])
                    plt.scatter(time, assemb_strength[n, time], c='r', marker = 'x')
                    plt.ylabel('assembly strength')
                    plt.title(f'assembly {str(assemb[n])} in cond {str(cond[k])} assembly strength binwindow {b}')
                    neurons = self.assemb_bin[b]['assemb_neuron'][k][assemb[n]]
                    plt.show()
                    count = 0

                    # loop thru each coactivation event
                    for t in time:
                        if count == 0:
                            fig, axs = plt.subplots(3, 3)
                            axs = axs.ravel()

                        scale = self.assemb_bin[b]['frame_scale'][k]
                        startframe = np.max((0,int(np.floor((t-slidewindow)/scale))))
                        endframe = np.min((int(np.ceil((t + slidewindow) / scale)), np.shape(self.mat)[0]))
                        event = self.mat[sep_frame[k]+startframe:sep_frame[k]+endframe, neurons]      # time * cell
                        # sns.heatmap(event.transpose())
                        # plt.xlabel(f'frames {startframe} to {endframe}')
                        # plt.ylabel(f'neurons in assemb {assemb[n]}')
                        midline = int(np.floor(np.shape(event)[0]/2))
                        # plt.plot([midline, midline], [0, len(neurons)+1], c = 'r', linestyle='dashed')
                        # plt.title(f'cond {k} strength {np.round(assemb_strength[n, t], 4)}')
                        # plt.show()
                        active_cells = np.sum(event, axis=0)>0
                        event_plot = event[:, active_cells]
                        ncells = np.shape(event_plot)[1]
                        peak_loc = np.nanargmax(event_plot, axis=0)

                        for cells in range(ncells):
                            axs[count].plot(event_plot[:, cells]+2*cells, c='k')
                            axs[count].scatter(peak_loc[cells], event_plot[peak_loc[cells], cells]+2*cells, c='orange', marker='x')
                        axs[count].set_title(f'{t} weight {np.round(assemb_strength[n, t], 3)}')
                        axs[count].plot([midline, midline], [0, 2*ncells+1], c = 'r', linestyle='dashed')
                        axs[count].plot([midline-b/scale/2, midline-b/scale/2], [0, 2*ncells+1],
                                        c='grey', linestyle='dashed')
                        axs[count].plot([midline + b / scale / 2, midline + b/ scale / 2],
                                        [0, 2 * ncells + 1], c='grey', linestyle='dashed')

                        if count == 8:
                            for ax in axs.flat:
                                ax.set(xlabel='frames', ylabel='dF/F')
                                ax.label_outer()
                            plt.show()
                            count = -1

                        count = count +1
                    plt.show()

    def draw_neurons_in_assemb(self, cond, n_dim, binwindow = None, foldername='PFs in assemble', save=0):
        """ draw all neurons activity (with behavior) in each assemble"""

        if save == 1:
            print('saving in folder' + foldername)
            path = os.path.join(self.path, foldername)
            if not os.path.exists(path):
                print('creating' +foldername +'folder')
                os.mkdir(os.path.join(path))
        else:
            path = self.path

        if binwindow is None:
            b = np.min(self.binwindow)

        for k in cond:
            env = self.opto_summary.loc[k, 'env']
            if env in self.laps:
                laps = self.laps[env]
            else:
                env = f'control{env[-5:]}'
                laps = self.laps[env]
            for n in range(n_dim):
                assemb_folder_name = f'Cell heatmaps cond {k} assemble {n}'
                neurons = self.assemb_bin[b]['assemb_neuron'][k][n]
                print(f'Cond {k}: {env}, assemb {n} includes neurons {neurons}')
                cell_heatmap(self.mean_activity[slice(laps[0], laps[-1]+1),:,neurons], path, np.arange(len(neurons)), assemb_folder_name, save)

    def draw_neuron_weights(self, n_dim, cond, binwindow):
        """ draw weights of neurons in each neural assembly for the first n_dim assembly"""

        for k in cond:
            for n in range(n_dim):
                for b in binwindow:
                    titleName = f'cond {k} assemble {str(n)} binwindow {b}'
                    data = self.assemb_bin[b]['weights'][k][n, :]
                    sig_neuron = self.assemb_bin[b]['assemb_neuron'][k][n]
                    plt.scatter(np.arange(self.ncells), data)
                    plt.scatter(sig_neuron, data[sig_neuron], c= 'r')
                    plt.title(titleName)
                    plt.xlabel('neurons')
                    plt.ylabel('weights')
                    plt.show()

    def assemb_mean_activity(self, binwindow, cond_assemb = None):
        """ heatmap of mean activity of all neurons associated with the assembly
        :param cond_assemb: dict{cond: assemb} eg. plot cond2 assemb[10,23,25] and cond3 assemb[2,3,10] {2: [10,23,25], 3:[2,3,10]}, default plot all
        """

        # plot all
        if cond_assemb is None:
            assert self.assemb_bin[binwindow]['n_sig_assembly'] is not None, 'run ICA first'
            assembs = [np.arange(self.assemb_bin[binwindow]['n_sig_assembly'][x])
                       for x in range(len(self.assemb_bin[binwindow]['n_sig_assembly']))]   # plot all assembs
            cond_assemb = dict(enumerate(assembs))

        for k in cond_assemb:
            figcount = 0
            assemb = cond_assemb[k]    # assemblies in cond
            env = self.opto_summary.loc[k, 'env']
            if env in ['control_first_day1', 'control_later_day1', 'control_first_day2', 'control_later_day2']:
                env = env[:7]+env[-5:]
            laps = self.laps[env]

            # loop thru each assembly
            for n in assemb:
                neurons = self.assemb_bin[binwindow]['assemb_neuron'][k][n]   # neurons in each assemb
                map = self.mean_activity[slice(laps[0], laps[-1]+1), :, neurons]
                mean_map = np.nanmean(map, axis=2)

                if figcount == 0:
                    fig, axs = plt.subplots(3, 3, sharex=True, sharey=True)
                    axs = axs.ravel()

                sns.heatmap(ax=axs[figcount], data=mean_map, cbar=False, xticklabels=False, yticklabels=10)
                axs[figcount].set_title(f'assemb {n}')

                if figcount == 8:
                    for ax in axs.flat:
                        ax.set(xlabel='location', ylabel='laps')
                        ax.label_outer()
                    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
                    fig.suptitle(f'{self.mouse} cond {k} {env} binwindow {binwindow}')
                    plt.show()

                figcount = (figcount + 1) % 9
            fig.suptitle(f'{self.mouse} cond {k} {env} binwindow {binwindow}')
            plt.show()

    def n_neurons_per_assemb(self):
        """ count # of neurons every assembly in all envs by binwindow, plot hist to compare opto and control"""

        n_neuron_assemb = {}
        for b in self.binwindow:
            n_neuron_assemb[b] = {}
            for k in self.assemb_bin[b]['assemb_neuron']:
                n_neuron_assemb[b][k] = []
                for n in self.assemb_bin[b]['assemb_neuron'][k]:
                    n_neuron_assemb[b][k].append(len(self.assemb_bin[b]['assemb_neuron'][k][n]))

        opto_envs = ['opto_first_day1', 'opto_first_day2', 'opto_later_day1', 'opto_later_day2']
        env_w_opto = self.opto_summary.loc[self.opto_summary['opto on frame'] >=0]['env'].to_list()

        for e in np.intersect1d(opto_envs, env_w_opto):
            control_env = 'control'+e[4:]
            cond_list = self.opto_summary.loc[self.opto_summary['env'] == e].index.to_list()
            control_cond_list =self.opto_summary.loc[self.opto_summary['env'] == control_env].index.to_list()

            for b in self.binwindow:
                fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
                for k in range(len(cond_list)):
                    axs[0].hist(n_neuron_assemb[b][cond_list[k]], density=True, histtype='step', fill=False, alpha=0.5)
                    axs[0].set_title(e)
                    axs[0].set_ylabel('proportion')
                    axs[1].set_xlabel('# cells in each assembly')
                    axs[1].hist(n_neuron_assemb[b][control_cond_list[k]], density=True, histtype='step', fill=False, alpha=0.5)
                    axs[1].set_title(control_env)
                plt.suptitle(f'binwindow {b}')
                axs[0].legend(cond_list)
                axs[1].legend(control_cond_list)
                plt.show()

        return n_neuron_assemb

    def compare_conds(self, cond1, cond2, b, heatmap = 1):
        """ cosine similarity comparison between weights in cond1 and cond2. El-Gaby et al 2021 for reference"""

        self._inactive_neuron_zero()

        cos_sim = cosine_similarity(self.assemb_bin[b]['weights'][cond1], self.assemb_bin[b]['weights'][cond2])
        max_sim = np.max(cos_sim, axis=1)

        # sort cos_sim and plot heatmap of cos_sim
        if heatmap == 1:
            best_match = np.argmax(cos_sim, axis=1)
            _, idx = np.unique(best_match, return_index=True)
            sort_ind = best_match[np.sort(idx)]
            rest_ind = np.setdiff1d(np.arange(cos_sim.shape[1]), sort_ind)
            cos_sim_sorted = np.copy(cos_sim)
            cos_sim_sorted = cos_sim_sorted[:, np.concatenate((sort_ind, rest_ind))]
            sns.heatmap(cos_sim_sorted, cmap=plt.cm.gray_r)
            env1 = self.opto_summary.loc[cond1, 'env']
            env2 = self.opto_summary.loc[cond2, 'env']
            plt.xlabel(f'{cond2}: {env2} assemblies sorted')
            plt.ylabel(f'{cond1}: {env1} assemblies')
            plt.title(f'{self.mouse} {cond1}: {env1} vs. {cond2}: {env2} cosine similarity sorted, bin={b}')
            plt.show()

        return cos_sim, max_sim

    def assemb_conds_envs_days(self, b, plot_hist = 1):

        envs = self.opto_summary['env'].to_list()

        opto = ['opto_first_day1', 'opto_first_day2', 'opto_later_day1', 'opto_later_day2']
        env_w_opto = self.opto_summary.loc[self.opto_summary['opto on frame'] >= 0]['env'].to_list()
        opto_envs = np.intersect1d(opto, env_w_opto)        # envs with opto in opto_summary
        max_sim_conds = {}
        figtitle = f'{self.mouse} assemblies max cos sim binwindow={b}'
        c = 0

        # compare overday stability during opto off for opto_first two days
        if all(np.isin(['opto_first_day1', 'opto_first_day2', 'control_first_day1','control_first_day2'], envs)):
            first_day1_list = self.opto_summary.loc[self.opto_summary['env'] == 'opto_first_day1'].index.to_list()
            first_day2_list = self.opto_summary.loc[self.opto_summary['env'] == 'opto_first_day2'].index.to_list()
            control_day1_list = self.opto_summary.loc[self.opto_summary['env'] == 'control_first_day1'].index.to_list()
            control_day2_list = self.opto_summary.loc[self.opto_summary['env'] == 'control_first_day2'].index.to_list()
            _, max_sim = self.compare_conds(first_day2_list[1], first_day1_list[1], b, 0)
            _, max_sim_control = self.compare_conds(control_day2_list[1], control_day1_list[1], b, 0)
            max_sim_conds[('opto_first_stability')] = max_sim
            max_sim_conds[('control_first_stability')] = max_sim_control

            if plot_hist == 1:
                self.plot_hist([max_sim, max_sim_control], ['opto_first_stability', 'control_first_stability'],
                               figtitle, 'max cos sim', '% assemblies')
                plt.show()

        # compare overday stability during initial opto off for opto_later two days
        if all(np.isin(['opto_later_day1', 'opto_later_day2', 'control_later_day1','control_later_day2'], envs)):
            first_day1_list = self.opto_summary.loc[self.opto_summary['env'] == 'opto_later_day1'].index.to_list()
            first_day2_list = self.opto_summary.loc[self.opto_summary['env'] == 'opto_later_day2'].index.to_list()
            control_day1_list = self.opto_summary.loc[self.opto_summary['env'] == 'control_later_day1'].index.to_list()
            control_day2_list = self.opto_summary.loc[self.opto_summary['env'] == 'control_later_day2'].index.to_list()
            _, max_sim = self.compare_conds(first_day2_list[0], first_day1_list[0], b, 0)
            _, max_sim_control = self.compare_conds(control_day2_list[0], control_day1_list[0], b, 0)
            max_sim_conds[('opto_later_stability')] = max_sim
            max_sim_conds[('control_later_stability')] = max_sim_control

            if plot_hist == 1:
                self.plot_hist([max_sim, max_sim_control], ['opto_later_stability', 'control_later_stability'],
                               figtitle, 'max cos sim', '% assemblies')
                plt.show()

        # check opto on vs. opto off assemblies within day
        for env in opto_envs:
            day = int(env[-1])
            cond_list = self.opto_summary.loc[self.opto_summary['env'] == env].index.to_list()
            control_env = f'control_{env[5:]}'
            control_list = self.opto_summary.loc[self.opto_summary['env'] == control_env].index.to_list()
            print(env, control_env, cond_list, control_list)

            # account for missing envs
            if len(control_list) != len(cond_list):
                display(self.opto_summary)
                c = input(f'{env} length: {len(cond_list)} and {control_env} length: {len(control_list)} not matched \n'
                          f'press 2 for opto_first or press 3 for opto_later, any other key to exit')

            # opto first
            if len(cond_list) == 2 or int(c) == 2:
                print('opto list', cond_list, 'control', control_list)
                _, max_sim = self.compare_conds(cond_list[1], cond_list[0], b, 0)
                _, max_sim_control = self.compare_conds(control_list[1], control_list[0], b, 0)

                if day==1:
                    max_sim_conds[('opto_first_day1', 'off')] = max_sim
                    max_sim_conds[('control_first_day1', 'off')] = max_sim_control
                elif day==2:
                    max_sim_conds[('opto_first_day2', 'off')] = max_sim
                    max_sim_conds[('control_first_day2', 'off')] = max_sim_control

                if plot_hist == 1:
                    self.plot_hist([max_sim, max_sim_control], [env, control_env], figtitle, 'max cos sim', '% assemblies')
                    plt.show()

            # opto later
            elif len(cond_list) == 3 or int(c) == 3:
                print('opto list', cond_list, 'control', control_list)
                # before vs. opto on
                _, max_sim_before = self.compare_conds(cond_list[0], cond_list[1], b, 0)
                _, max_sim_before_control = self.compare_conds(control_list[0], control_list[1], b, 0)

                if day==1:
                    max_sim_conds[('before_opto_day1', 'opto_later')] = max_sim_before
                    max_sim_conds[('before_control_day1', 'control_later')] = max_sim_before_control
                elif day==2:
                    max_sim_conds[('before_opto_day2', 'opto_later')] = max_sim_before
                    max_sim_conds[('before_control_day2', 'control_later')] = max_sim_before_control

                if len(cond_list) == len(control_list):
                    # after vs. opto on
                    _, max_sim_after = self.compare_conds(cond_list[2], cond_list[1], b, 0)
                    _, max_sim_after_control = self.compare_conds(control_list[2], control_list[1], b, 0)

                    # before vs. after (reliability)
                    _, max_sim_reliab = self.compare_conds(cond_list[2], cond_list[0], b, 0)
                    _, max_sim_reliab_control = self.compare_conds(control_list[2], control_list[0], b, 0)

                    if day==1:
                        max_sim_conds[('opto_later_day1', 'after_opto')] = max_sim_after
                        max_sim_conds[('control_later_day1', 'after_control')] = max_sim_after_control
                        max_sim_conds[('reliability_opto_day1')] = max_sim_reliab
                        max_sim_conds[('reliability_control_day1')] = max_sim_reliab_control
                    elif day==2:
                        max_sim_conds[('opto_later_day2', 'after_opto')] = max_sim_after
                        max_sim_conds[('control_later_day2', 'after_control')] = max_sim_after_control
                        max_sim_conds[('reliability_opto_day2')] = max_sim_reliab
                        max_sim_conds[('reliability_control_day2')] = max_sim_reliab_control

                    if plot_hist == 1:
                        self.plot_hist([max_sim_after, max_sim_after_control], ['after vs opto', 'control'],
                                       figtitle, 'max cos sim', '% assemblies')
                        plt.show()

                        self.plot_hist([max_sim_reliab, max_sim_reliab_control], ['reliability', 'control'], figtitle,
                                       'max cos sim', '% assemblies')
                        plt.show()

                if plot_hist == 1:
                    self.plot_hist([max_sim_before, max_sim_before_control], ['before vs opto', 'control'],
                                   figtitle, 'max cos sim', '% assemblies')
                    plt.show()

            else:
                print(f'exiting {self.mouse} now')
                return max_sim_conds

        return max_sim_conds

    def plot_hist(self, data_list, leg=None, titlename=None, xlabelname=None, ylabelname=None):

        for data in data_list:
            plt.hist(data, weights = np.ones(len(data))/ len(data), histtype='step', fill=False, alpha=0.5)
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
        if leg is not None:
            plt.legend(leg)
        if titlename is not None:
            plt.title(titlename)
        if xlabelname is not None:
            plt.xlabel(xlabelname)
        if ylabelname is not None:
            plt.ylabel(ylabelname)

    def _inactive_neuron_zero(self):
        """ assign inactive neurons to zero weights/z_mat to compare between conditions"""

        for b in self.binwindow:
            inactive_neuron_dict = self.assemb_bin[b]['inactive_neuron']
            weights_dict = self.assemb_bin[b]['weights']
            zmat_dict = self.assemb_bin[b]['z_mat']

            for cond in inactive_neuron_dict:
                insert_ind = inactive_neuron_dict[cond] - np.arange(len(inactive_neuron_dict[cond]))
                if zmat_dict[cond].shape[1] < self.ncells:
                    print(f'{self.mouse} {len(inactive_neuron_dict[cond])} inactive neuron found in bin {b} cond {cond}'
                          f', overwrite inactive neurons to 0 in weights and z_mat')
                    self.assemb_bin[b]['weights'][cond] = np.insert(weights_dict[cond], [*insert_ind], 0, axis=1)
                    self.assemb_bin[b]['z_mat'][cond] = np.insert(zmat_dict[cond], [*insert_ind], 0, axis=1)
                elif zmat_dict[cond].shape[1] > self.ncells:
                    print(f'check {self.mouse} binwindow {b} cond {cond} zmat length')
                    return None

    def assemb_strength_conds(self, original_cond, test_conds, b):

        self._inactive_neuron_zero()

        # calculate assembly strength
        weights = self.assemb_bin[b]['weights'][original_cond]
        diag_ind = np.diag_indices(self.ncells)
        nbins_per_cond = [self.assemb_bin[b]['z_mat'][k].shape[0] for k in test_conds]
        total_bins = sum(nbins_per_cond)
        n_assemb = weights.shape[0]
        assemb_strength = np.zeros((n_assemb, total_bins))
        start_ind = 0

        # apply weights to z_mat to calculate assembly strength in test_conds
        for conds in test_conds:
            z_mat = self.assemb_bin[b]['z_mat'][conds]
            nbins = z_mat.shape[0]
            end_ind = start_ind + nbins

            for n in range(n_assemb):
                P = np.outer(weights[n, :], weights[n, :])
                P[diag_ind] = 0  # set diagonal to zeros to remove contribution from a single cell
                R = (z_mat @ P) * (z_mat)
                assemb_strength[n, start_ind:end_ind] = np.sum(R, axis=1)
            start_ind = end_ind

        mean_strength = np.mean(assemb_strength, axis=0)
        plt.plot(mean_strength)
        switch_frame = np.cumsum(nbins_per_cond)
        envs = []
        # add lines to indicate switch of conditions
        for n in range(len(switch_frame)):
            plt.plot([switch_frame[n], switch_frame[n]], [0, np.max(mean_strength)], c='grey', linestyle = 'dashed')
            envs.append(self.opto_summary.loc[test_conds[n], 'env'])
        plt.ylabel('assembly strength in a.u.')
        env = ', '.join(np.unique(envs))
        plt.xlabel(f'in conds {test_conds}: {env}')
        plt.title(f'cond {original_cond} average assembly strength')
        plt.show()

        return assemb_strength

    @staticmethod
    def cos_sim_cond(cos_sim, env_dict):
        """ cosine similarity matrix by envs

        :param cos_sim: 3d mat of laps*laps*assemb
        :param env_dict: {env: slice(laps)} for all envs to combine without replacement. eg. {'fam':slice(0,20), 'nov':slice(20,60)}
        return: mean cosine similarity of laps for all assembles (n_assembly * n_combo) in dataframe
        """

        n_assemb = np.shape(cos_sim)[2]
        mean_cos_sim = [np.mean(cos_sim[env_dict[x[0]], env_dict[x[1]]], axis = (0,1))
                        for x in combinations_with_replacement(env_dict.keys(),2)]   # mean cos sim of laps b/t envs
        columnName = [' & '.join(x) for x in combinations_with_replacement(env_dict.keys(), 2)]
        n_combo = len(columnName)
        mean_cos_sim = np.concatenate(mean_cos_sim).reshape(n_assemb, n_combo)
        cos_sim_env = pd.DataFrame(data=mean_cos_sim, columns=columnName)

        return cos_sim_env


class Cell_struct_run(Cell_struct):
    """ Cell_struct using only imaging frames when the mouse is running"""

    def __init__(self, mouse, env, day, binwindow):
        savetitle = 'ensemble_run'
        self.ybinned = None
        self.velocity = None
        self.run_ind = None

        Cell_struct.__init__(self, mouse, env, day, binwindow, savetitle)

        if self.ybinned is None:
            self.find_pause()

    def find_pause(self, pause_frame=40, dist_thresh=0.05, plt_frames=5000):
        """ align with behavior and remove frames when animal is not running

        :param pause_frame: 2x duration of frames not moving to be considered as pauses
        :param dist_thresh: distance smaller than this will be considered not moving
        :return: run ind,  opto on ind while running (if applicable)
        """

        logging.info(f'Step 0: Using only active period to find coactivity patterns. '
                     f'Not moving at least {dist_thresh} for {pause_frame} frames is considered as pausing')

        beh_mat = load_py_var_mat(self.path, 'cond.mat')
        print(f'loading behavior {self.mouse} from {self.path}')
        ybinned = beh_mat['behavior']['ybinned'][0][0].transpose()
        velocity = beh_mat['behavior']['velocity'][0][0].transpose()
        self.ybinned = ybinned
        self.velocity = velocity
        beh_frames = ybinned.shape[0]
        img_frames = self.mat.shape[0]
        assert beh_frames == img_frames, f'{self.mouse} behavior and imaging of different size: beh {beh_frames}, img {img_frames}'

        # find ind for pauses
        trackend = 0.605
        trackstart = 0.015
        vr_ind = (ybinned > trackstart) & (ybinned < trackend)
        # ybinned_vr = ybinned[vr_ind]
        v_vr = velocity[vr_ind]
        v_thresh = np.quantile(v_vr, 0.05)
        v0_ind = np.where(velocity < v_thresh)[0]  # find indices for when velocity is lower than 5% of all velocity
        v0_dis = ybinned[np.minimum(v0_ind + pause_frame, beh_frames-1)] \
                 -ybinned[np.maximum(v0_ind - pause_frame, 0)]  # double check with frames and distance
        pause_ind = v0_ind[v0_dis[:, 0] < dist_thresh]
        run_ind = np.setdiff1d(np.arange(img_frames), pause_ind)
        print(f'{self.mouse} {len(pause_ind)} frames pausing, {len(run_ind)} frames running, {beh_frames} frames in total')

        # update attributes
        self.run_ind = run_ind
        nframes = len(run_ind)
        self.raw = self.raw[run_ind, :]
        self.mat = np.copy(self.raw)
        switch_ind = [bisect.bisect_left(run_ind, x) for x in self.switch_frame]
        self.switch_frame = np.array(switch_ind)
        self.switch = np.copy(self.switch_frame)

        plt.plot(ybinned[run_ind[:plt_frames]])
        plt.title(f'{self.mouse} Double check: behav trace after removing pauses, first {plt_frames} frames')
        plt.show()

            # separate opto on and opto off
        if hasattr(self, 'opto_summary'):
            opto_on_frame = self.opto_on_frame
            opto_off_frame = self.opto_off_frame
            opto_true = list(map(lambda x, y: list(range(x, y)), opto_on_frame, opto_off_frame))
            opto_true = np.array(sum(opto_true, [])).astype(int)
            opto_on_run = np.intersect1d(run_ind, opto_true)
            # opto_off_run = np.setdiff1d(run_ind, opto_on_run)
            self.opto_on_frame = np.array([bisect.bisect_left(self.run_ind, x) for x in opto_on_frame])
            self.opto_off_frame = np.array([bisect.bisect_left(self.run_ind, x) for x in opto_off_frame])
            print(f'{len(opto_on_run)} frames running under opto stim')
            self.opto_on = np.copy(self.opto_on_frame)
            self.opto_off = np.copy(self.opto_off_frame)
