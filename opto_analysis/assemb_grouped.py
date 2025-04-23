import os
import pickle
import numpy as np
import pandas as pd
from scipy import stats
from matplotlib.ticker import PercentFormatter
import matplotlib.pyplot as plt

class Assemb_group:
    """
    group individual mice together and perform stats tests. works as a wrapper for PF_analysis
    """

    def __init__(self, group, mice_list):
        """ take in individual class objects in list and label them as group
        :param group: eg. 'left' or 'right'
        :param mice_list: eg. [eL121, eL123, eL113], each argument is a class object from PF_analysis
        """

        self.group = group
        self.mice = mice_list
        self.mouse_env = {}
        self.opto_env = None

        self.save_path = os.path.join('D:\\Opto\\Analysis', group)
        if not os.path.exists(self.save_path):
            print(f'creating path {self.save_path}')
            os.mkdir(os.path.join(self.save_path))

        for m in self.mice:
            print(m.mouse, m.env)
            self.mouse_env[m.mouse] = set(m.env)
        self.all_env = set.intersection(*self.mouse_env.values())
        print(f'all mice have env {self.all_env}')

    def __repr__(self):
        return f'{self.group} includes {[*self.mouse_env.keys()]}'

    def save_to_file(self, var, save_name):
        data_path = os.path.join(self.save_path, f'{self.group}_{save_name}.pickle')
        print(f'Saving {save_name} to file at {data_path}')
        with open(data_path, 'wb') as output_file:
            pickle.dump(var, output_file, pickle.HIGHEST_PROTOCOL)

    def max_sim_cond_compare(self):

        max_sim_cond = {}
        for m in self.mice:
            max_sim_cond[m.mouse] = {}
            for b in m.binwindow:
                max_sim_cond[m.mouse][b] = m.assemb_conds_envs_days(b)

        return max_sim_cond

    def unpack_nested_dict(self, dict_to_unpack, dict_keys = None):

        cat_dict = {}

        if dict_keys is None:
            # assume all the dict keys are the same
            dict_keys = [*dict_to_unpack[self.mice[0].mouse][self.mice[0].binwindow[0]].keys()]
        binwindow = self.mice[0].binwindow

        for b in binwindow:
            cat_dict[b] = {}
            for key in dict_keys:
                var = np.concatenate([dict_to_unpack[mouse.mouse][b][key] for mouse in self.mice])
                cat_dict[b][key] = var

        return cat_dict

    @staticmethod
    def plot_max_sim_hist(cat_dict: {}, alpha = 0.05):

        binwindow = [*cat_dict.keys()]
        conds = [[('opto_first_day1', 'off'), ('control_first_day1', 'off')], [('before_opto_day1', 'opto_later'),
         ('before_control_day1', 'control_later')], [('opto_later_day1', 'after_opto'), ('control_later_day1', 'after_control')],
         ['reliability_opto_day1', 'reliability_control_day1'], [('opto_first_day2', 'off'), ('control_first_day2', 'off')],
         [('before_opto_day2', 'opto_later'), ('before_control_day2', 'control_later')],
         #[('opto_later_day2', 'after_opto'), ('control_later_day2', 'after_control')],
         #['reliability_opto_day2', 'reliability_control_day2'],
         ['opto_first_stability', 'control_first_stability'],
         ['opto_later_stability', 'control_later_stability']]
        # conds = [['opto_first_stability', 'control_first_stability'],
        #          [('opto_first_day1', 'off'), ('control_first_day1', 'off')],
        #          [('opto_first_day2', 'off'), ('control_first_day2', 'off')],
        #          [('before_opto_day1', 'opto_later'), ('before_control_day1', 'control_later')],
        #          [('opto_later_day1', 'after_opto'), ('control_later_day1', 'after_control')],
        #          ['reliability_opto_day1', 'reliability_control_day1']]

        fig, axs = plt.subplots(len(binwindow), len(conds), sharey=True, figsize=(25, 25))
        plt.subplots_adjust(top = 0.99, bottom=0.01, hspace=0.3, wspace=0.3)

        # sidak adjustment for family-wise error
        alpha_adj = 1- (1-alpha)**(1/len(conds))

        for b in range(len(binwindow)):
            for c in range(len(conds)):
                if conds[c][0] in cat_dict[binwindow[b]] or conds[c][1] in cat_dict[binwindow[b]]:
                    data0 = cat_dict[binwindow[b]][conds[c][0]]
                    axs[b,c].hist(data0, weights=np.ones(len(data0)) / len(data0), alpha=0.5)
                    data1 = cat_dict[binwindow[b]][conds[c][1]]
                    axs[b, c].hist(data1, weights=np.ones(len(data1)) / len(data1), alpha=0.5)
                    _, p = stats.kruskal(data0, data1)
                    axs[b, c].set(ylabel = f'{binwindow[b]}')
                    axs[b, c].legend(['opto', 'control'])
                    if p < alpha_adj:
                        axs[b, c].set_title(f'{conds[c][0]}', color = 'red')
                    else:
                        axs[b, c].set_title(f'{conds[c][0]}')

### not usable yet
    @staticmethod
    def run_kruskal(dict_var, cond_list, alpha=0.05):
        """ run kruskal wallis test on dict_var[cond_list] """

        l = []
        for cond in cond_list:
            l.append(list(dict_var[cond].reshape(-1)))

        _, p = stats.kruskal(*l, nan_policy='omit')
        thresh = 'significant' if p < alpha else 'not significant'
        print(f'Kruskal wallis test p value is: {p}, {thresh}')

    @staticmethod
    def plot_hist(dict_var, bins):
        """ plot histogram from dict_var[envs]
        :param dict_var: {env: value} eg. output from unpack_dict or unpack_dict_var
        :param envs: list of envs, must be the keys of dict_var
        """

        for b in bins:
            plt.hist(dict_var[b], weights=np.ones(len(dict_var[e])) / len(dict_var[e]), alpha=0.5)
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
        plt.ylabel('cell count')
        plt.legend(envs)

        PF_group.run_kruskal(dict_var, envs)