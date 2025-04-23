import os.path

import pandas as pd
from matplotlib.ticker import PercentFormatter
from sklearn.naive_bayes import GaussianNB
import sklearn.metrics
from itertools import compress
from sklearn.metrics import r2_score
import pickle
import numpy as np
import matplotlib.pyplot as plt
from opto_analysis.place_cell_opto import *

def tsplot(data,**kw):
    x = np.arange(data.shape[1])
    est = np.nanmean(data, axis=0)
    sd = np.nanstd(data, axis=0)
    cis = (est - sd, est + sd)
    plt.fill_between(x,cis[0],cis[1],alpha=0.2, **kw)
    plt.plot(x,est,**kw)
    plt.margins(x=0)

class BayesDecoder:

    def __init__(self, nshuffle=500, nlaps=45, maxlaps=50, opto_later_align_to=10):
        self.decoder = GaussianNB()
        self.nshuffle = nshuffle
        self.nlaps = nlaps
        self.maxlaps = maxlaps
        self.opto_later_align = opto_later_align_to
        self.nbins = 40
        self.train_y = np.tile(np.arange(40), nlaps)

    def fit(self, class_data: LoadData, PF_summary: pd.DataFrame):
        decoder_env = {}

        for cond in class_data.params['opto_on_env']:
            env = map_string(cond)

            cells = PF_summary.loc[PF_summary['env'] == env]['cell'].unique()
            data = class_data.mean_activity[:, :, cells]

            env_laps = class_data.params['env_laps'][env]

            train_laps = np.setdiff1d(env_laps, class_data.params['opto_on_env'][cond], assume_unique=True)
            decoder_env[cond] = np.zeros((self.nshuffle, self.maxlaps))
            # n_test_laps = len(env_laps) - self.nlaps
            error_lap_shuffle = np.empty((self.nshuffle, len(env_laps))) * np.nan

            for n in range(self.nshuffle):
                selected_laps = np.random.choice(train_laps, self.nlaps)
                test_laps = np.setdiff1d(env_laps, selected_laps, assume_unique=True)
                self.decoder.fit(np.vstack(data[selected_laps, :, :]), self.train_y)

                # calculate mean (min sq error) per lap
                test_y = self.decoder.predict(np.vstack(data[test_laps, :, :]))
                true_y = np.tile(np.arange(self.nbins), len(test_laps))
                error_y = (test_y - true_y) ** 2
                error_add = (test_y - true_y + self.nbins) ** 2
                error_min = (test_y - true_y - self.nbins) ** 2
                min_error_mat = np.minimum(error_y, np.minimum(error_add, error_min))

                min_error_lap = np.reshape(min_error_mat, (-1, 40))

                #min_error_lap = np.reshape(min_error_mat, (40, -1), 'C')
                mean_error_lap = np.mean(min_error_lap, axis=1)
                error_lap_shuffle[n, test_laps-env_laps[0]] = mean_error_lap

            first_lap_selected = np.max((env_laps[0], class_data.params['opto_on_env'][cond][0]-self.opto_later_align))-env_laps[0]
            print(cond, first_lap_selected)
            n_laps = np.min((len(env_laps), self.maxlaps+first_lap_selected))
            print(n_laps)
            error_mat = np.empty((self.nshuffle, self.maxlaps)) * np.nan
            error_mat[:, :n_laps-first_lap_selected] = error_lap_shuffle[:, first_lap_selected:n_laps]
            decoder_env[cond] = error_mat

        return decoder_env



#
# class BayesDecoder:
#     """ laod pickle file from PF_analysis and use Bayesian decoder to quantify the effect of opto"""
#
#     def __init__(self, mouse, day):
#         self.mouse = mouse
#         self.day = day
#         self.save_path = os.path.join('D:\\Opto\\Analysis', mouse, day)
#
#         # check for cache in the directory first
#         pickle_file = os.path.join(self.save_path, f'{self.mouse}_PF.pickle')
#         if os.path.exists(pickle_file):
#             print(f'Loading {pickle_file} from cache')
#             self.load(pickle_file)
#         else:
#             print(f'no cache, run PF_analysis first')
#
#     def load(self, pickle_file):
#         """ load previously saved pickle file"""
#
#         file = open(pickle_file, 'rb')
#         print(f'Loading stored PF analysis from cache: {pickle_file}')
#         temp_dict = pickle.load(file)
#         file.close()
#         self.__dict__.update(temp_dict)
#
#     def _bayes_mat(self, train_laps, test_laps, cells):
#         """ construct X:(n_samples, n_features) and Y:(n_samples) matrix for bayesian decoder training and testing
#
#         :param train_laps: select laps to train. eg. np.arange(25,40)
#         """
#
#         train_mat = self.mean_activity[train_laps[:, None], :, cells]
#         # reorganize matrix as inputs to bayesian decoder (X.shape = nlaps*nbins, ncells; len(Y) = nlaps*nbins)
#         X_train = train_mat.transpose((2, 1, 0)).reshape((len(cells), -1)).transpose()
#         Y_train = np.tile(np.arange(self.nbins), (len(train_laps),1)).transpose().reshape((-1, 1))
#
#         test_mat = self.mean_activity[test_laps[:, None], :, cells]
#         X_test = test_mat.transpose((2, 1, 0)).reshape((len(cells), -1)).transpose()
#         Y_true = np.tile(np.arange(self.nbins), (len(test_laps),1)).transpose().reshape((-1, 1))
#
#         return X_train, Y_train, X_test, Y_true
#
#     def _mean_sq_error_lap(self, Y_pred, Y_true):
#         """ calculate mean squared error by each lap"""
#
#         error = np.abs(Y_pred - Y_true[:, 0])
#         error_add = np.abs(Y_pred - Y_true[:, 0] + self.nbins)  # eg. bin 0 is the same as bin 40
#         error_minus = np.abs(Y_pred - Y_true[:, 0] - self.nbins)  # eg. bin 39 is the same as bin -1
#         sq_error_adj = np.amin([error, error_minus, error_add],
#                                axis=0) ** 2  # use the smallest error to quantify decoder performance
#         sq_error_lap = np.reshape(sq_error_adj, (self.nbins, -1))  # reshape sq error to (bin * lap) mat
#         mean_sq_error_lap = np.mean(sq_error_lap, axis=0)
#
#         return mean_sq_error_lap
#
#     def _bayes_laps(self, train_env, test_env, opto_laps=0, train_test_ratio=0.5):
#         """ randomly select training and testing laps given env. training laps will not contain opto on laps"""
#
#         all_laps = self.laps[train_env]  # all laps in training env
#         opto_off_laps = np.setdiff1d(all_laps, opto_laps)   # all opto off laps in training env
#         n_train = int(np.round(len(opto_off_laps) * train_test_ratio, 0))
#         train_laps = np.random.choice(opto_off_laps, n_train, replace=False)
#
#         if train_env == test_env:
#             test_laps = np.setdiff1d(all_laps, train_laps)
#         else:
#             test_laps = self.laps[test_env]
#
#         return np.sort(train_laps), np.sort(test_laps)
#
#     def bayes_fit(self, cells, train_env, test_env, opto_on_train_env=0, nshuffle=100):
#         """ fit Naive Bayes decoder, plot mean sq error by laps and confusion matrix for all laps
#         :param opto_on_train_env: if in train_env opto is on. default off
#
#         """
#
#         title_name = 'fitted in ' + train_env + ' tested in ' + test_env
#         score = np.zeros((nshuffle, 1))
#         # C_norm = np.zeros((self.nbins, self.nbins, nshuffle))
#         all_laps = len(self.laps[test_env]) # all possible laps in test env
#         mean_sq_error_lap = np.zeros((nshuffle, all_laps))
#         mean_sq_error_lap[:] = np.nan
#         #r2 = np.zeros((nshuffle, 1))
#
#         for n in range(nshuffle):
#             bayes = GaussianNB()
#             if opto_on_train_env == 1:
#                 boo = np.isin(self.opto_off_lap, self.laps[train_env])
#                 opto_off_lap = list(compress(self.opto_off_lap, boo))[0]
#                 opto_on_lap = list(compress(self.opto_on_lap, boo))[0]
#                 opto_on_laps = np.arange(opto_on_lap,opto_off_lap)
#                 train_laps, test_laps = self._bayes_laps(train_env, test_env, opto_on_laps)
#             else:
#                 train_laps, test_laps = self._bayes_laps(train_env, test_env)
#
#             X_train, Y_train, X_test, Y_true = self._bayes_mat(train_laps, test_laps, cells)
#             bayes.fit(X_train, Y_train)
#             #print('fitting Bayesian decoder in env', train_env)
#             Y_pred = bayes.predict(X_test)
#             #print('testing Bayesian decoder in env', test_env)
#             score[n] = bayes.score(X_test, Y_true)
#
#             # squared error
#             test_laps_idx = test_laps - np.min(test_laps)
#             mean_sq_error_lap[n, test_laps_idx] = self._mean_sq_error_lap(Y_pred, Y_true)
#             #r2[n] = r2_score(Y_pred, Y_true)
#
#         # fig, axs = plt.subplots(2,1)
#         # axs = axs.ravel()
#
#         # plot mean squared error
#         mean_over_lap = np.nanmean(mean_sq_error_lap, axis=0)
#         sd = np.std(mean_over_lap)
#         plt.scatter(self.laps[test_env], mean_over_lap)
#         plt.ylim(0, np.max(mean_over_lap)+sd)
#         plt.xlabel('laps')
#         plt.ylabel('mean squared error')
#         plt.title(title_name)
#
#         # axs[0].scatter(self.laps[test_env], mean_over_lap)
#         # axs[0].set_ylim(0, max(mean_over_lap)+sd)
#         # axs[0].set_xlabel('laps')
#         # axs[0].set_ylabel('mean squared error')
#         # axs[0].set_title(title_name)
#
#         # add opto on laps
#         if hasattr(self, 'opto_on_lap'):
#             lap_on_ind = np.isin(self.opto_on_lap, test_laps)
#             if sum(lap_on_ind) >= 1:
#                 plt.plot([self.opto_on_lap[lap_on_ind], self.opto_off_lap[lap_on_ind]],
#                          [max(mean_over_lap)+sd/2, max(mean_over_lap)+sd/2], c='r', linewidth=4)
#         plt.show()
#
#         # # plot r2
#         # axs[1].scatter(self.laps[test_env], )
#
#         # # plot confusion matrix
#         # C = sklearn.metrics.confusion_matrix(Y_true[:, 0], Y_pred)
#         # C_norm[:,:,n] = C.astype('float')/C.sum(axis=1)[:,np.newaxis]
#         # C_norm_mean = np.mean(C_norm, axis=2)
#         # sns.heatmap(C_norm_mean, vmin=0, vmax=1, cmap=plt.cm.gray_r, square=True)
#         # plt.ylabel('True location')
#         # plt.xlabel('Predicted location')
#         # plt.title(title_name)
#         # plt.show()
#
#         return mean_sq_error_lap, score
#
#
