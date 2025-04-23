import os
import pickle
import numpy as np
import pandas as pd
from scipy import stats
from matplotlib.ticker import PercentFormatter
import matplotlib.pyplot as plt
import seaborn as sns
import dabest
import random

class PF_group:
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
        self.mice_str = [m.name for m in mice_list]   # eg. ['eL121', 'eL123'] etc.
        self.mouse_env = {}
        self.opto_env = None
        self.str_to_class = dict(zip(self.mice_str, self.mice))

        self.save_path = os.path.join('D:\\Opto\\Analysis', group)
        if not os.path.exists(self.save_path):
            print(f'creating path {self.save_path}')
            os.mkdir(os.path.join(self.save_path))

        for m in self.mice:
            print(m.name, m.env)
            self.mouse_env[m.name] = set(m.env)
        self.all_env = set.intersection(*self.mouse_env.values())
        print(f'all mice have env {self.all_env}')

    def __repr__(self):
        return f'{self.group} includes {[*self.mouse_env.keys()]}'

    def save_to_file(self, var, save_name):
        data_path = os.path.join(self.save_path, f'{self.group}_{save_name}.pickle')
        print(f'Saving {save_name} to file at {data_path}')
        with open(data_path, 'wb') as output_file:
            pickle.dump(var, output_file, pickle.HIGHEST_PROTOCOL)

    def load_file(self, save_name):
        data_path = os.path.join(self.save_path, f'{self.group}_{save_name}.pickle')
        print(f'Loading {self.group}_{save_name}')
        file = open(data_path, 'rb')
        pickle_file = pickle.load(file)
        return pickle_file

    def check_PF(self):
        """ first step before running any analysis. check_PF_peak in PF_analysis"""

        for m in self.mice:
            for e in m.env:
                m.check_PF_peak(e)
            m.save_to_file()

    def check_opto(self, same_thresh=5, off_laps='after'):
        """ opto_check in PF_analysis"""

        opto_dict = {}
        for c in self.mice:
            print(f'analyzing {c.mouse}')
            for e in c.env:
                opto_d = c.opto_check(e, same_thresh=same_thresh, off_laps=off_laps)
                if opto_d is not None:
                    opto_dict[c.mouse] = opto_d

        self.opto_env = [*opto_dict[c.mouse].keys()]
        self.save_to_file(opto_dict, 'opto_dict_by_mice')

        return opto_dict

    def check_PF_features(self, PF_features = None):
        """ combine PF_summary_peak from each mouse and compare PF features """

        # load individual parquet files from mice
        parq_envs = ['opto_first_day1', 'opto_first_day2', 'opto_later_day1', 'opto_later_day2',
                     'control_opto_first_day1', 'control_opto_first_day2',
                     'control_opto_later_day1', 'control_opto_later_day2']
        df_list = []

        for m in self.mice:
            for env in parq_envs:
                filename = os.path.join(m.save_path, 'opto', f'{m.mouse}_{env}_after.parquet')
                if os.path.exists(filename):
                    df = pd.read_parquet(filename, engine='fastparquet')
                    df['env'] = env
                    df_list.append(df)
                else:
                    print(f'{m.mouse} does not have {env} parquet file')
        PF_all_mice = pd.concat(df_list)

        # combine and save PF_summary_peak
        # print(PF_all_mice.columns)
        if PF_features is None:
            PF_features = ["precision_on", "precision_off", "ratio_on", "ratio_off", "-log(p)_on", "-log(p)_off",
                           "out in ratio_off", "out in ratio_on"]
        PF_all_mice = PF_all_mice.astype(dict(zip(PF_features, ['float'] * len(PF_features))))

        opto_pq = os.path.join(self.save_path, f'PF_opto_summary_{self.group}.parquet')
        PF_all_mice.to_parquet(opto_pq, compression='gzip')  # save to parquet

        # stats test correct for multiple comparison sidak correction
        a = 0.05
        alpha_adj = 1 - (1 - a) ** (1 / len(PF_features))

        # dabest plotting precision and ratio
        dab_comparison = (("control_opto_first_day1", "opto_first_day1"), ("control_opto_first_day2", "opto_first_day2"),
                          ("control_opto_later_day1", "opto_later_day1"), ("control_opto_later_day2", "opto_later_day2"))
        for f in PF_features:
            dab = dabest.load(data = PF_all_mice, x="env", y=f, idx=dab_comparison)
            dab.mean_diff.plot(raw_marker_size=1.75);
            sig_rows = dab.mean_diff.statistical_tests['pvalue_permutation'].to_numpy() < alpha_adj
            conds = dab.mean_diff.statistical_tests.loc[sig_rows, 'test'].to_list()
            print(f'{f} significantly different in: {conds}')
            plt.savefig(os.path.join(self.save_path, f))
            plt.show()

        return PF_all_mice

    def check_stability(self, envs = None, nlaps = 40, min_laps = 10, mice_list = None, corr_bw_days = None, corr_wi_days = None):
        """ stability in PF_analysis
        :param envs: env between days to check.
        """

        if envs is None:
            envs = ['opto_first', 'opto_later', 'control']
        envs_day1 = [f'{e}_day1' for e in envs]
        envs_day2 = [f'{e}_day2' for e in envs]

        if corr_bw_days is None:
            corr_bw_days = {}

        if corr_wi_days is None:
            corr_wi_days = {}

        if mice_list is None:
            mice_list = self.mice

            # first check if all envs exist in mice condition
            if not all(np.isin([*envs_day1, *envs_day2], list(self.all_env))):
                print('not all mice have all envs')
                missing_mouse = []
                for c in mice_list:
                    missing_env = np.setdiff1d([*envs_day1, *envs_day2], c.env)
                    if len(missing_env) > 0:
                        print(f'{c.mouse} miss env: {missing_env}')
                        missing_mouse.append(c.mouse)
                print(f'first run {np.setdiff1d(self.mice_str, missing_mouse)} in missing envs')
                return None

        # find PFs using first day and check stability between days
        for c in mice_list:
            corr_bw_days[c.name] = {}
            for day in range(len(envs)):
                print(f'{c.name}: comparing stability b/w {envs_day1[day]} and {envs_day2[day]}')
                # use first nlaps # of laps in envs to calculate mean place map of the day
                corr_day1 = c.remapping(c.PFs[envs_day1[day]], c.laps[envs_day1[day]][0], np.min((c.laps[envs_day1[day]][0]+nlaps, c.laps[envs_day1[day]][-1])))
                corr_day2 = c.remapping(c.PFs[envs_day1[day]], c.laps[envs_day2[day]][0], np.min((c.laps[envs_day2[day]][0]+nlaps, c.laps[envs_day2[day]][-1])))
                corr_bw_days[c.name][envs[day]] = c.remapping_corr(corr_day1, corr_day2)
                plt.title(f'{c.name} corr between {envs_day1[day]} and {envs_day2[day]}')
                plt.show()

        # check stability within env
        for c in mice_list:
            corr_wi_days[c.name] = {}

            for e in ['opto_later_day1', 'opto_later_day2', 'opto_first_day1', 'opto_first_day2']:
                if (e in ['opto_later_day1', 'opto_later_day2']) & (e in c.opto_env):
                    print(f'{c.name} in {e} w/i day')
                    early_laps = np.arange(np.max((c.opto_env[e][0] - min_laps,c.laps[e][0])), c.opto_env[e][0])
                    late_laps = np.arange(c.opto_env[e][-1]+1, np.min((c.opto_env[e][-1]+1+min_laps, c.laps[e][-1])))

                elif (e in ['opto_first_day1', 'opto_first_day2']) & (e in c.opto_env):
                    print(f'{c.name} in {e} w/i day')
                    early_laps = np.arange(c.opto_env[e][0], c.opto_env[e][-1]+1)
                    late_laps = np.arange(c.opto_env[e][-1]+1+min_laps, np.min((c.opto_env[e][-1]+1+min_laps*2, c.laps[e][-1])))

                else:
                    print(f'{c.name} does not have {e}')
                    continue

                print(f'{c.name} in {e}: {len(early_laps)} {early_laps}, {len(late_laps)} {late_laps}')
                corr_early = c.remapping(c.PFs[e], early_laps[0], early_laps[-1])
                corr_late = c.remapping(c.PFs[e], late_laps[0], late_laps[-1])
                corr_wi_days[c.name][e] = c.remapping_corr(corr_early, corr_late)
                early_laps = early_laps - c.laps[e][0]
                late_laps = late_laps - c.laps[e][0]

                # control using same laps
                print(f'{c.name} in control_{e}')
                control_env = f'control_{e[-4:]}'
                corr_early = c.remapping(c.PFs[control_env], early_laps[0]+c.laps[control_env][0], np.min((early_laps[-1]+c.laps[control_env][0], c.laps[control_env][-1])))
                corr_late = c.remapping(c.PFs[control_env], late_laps[0]+c.laps[control_env][0], np.min((late_laps[-1]+c.laps[control_env][0], c.laps[control_env][-1])))
                corr_wi_days[c.name][f'control_{e}'] = c.remapping_corr(corr_early, corr_late)

        return corr_bw_days, corr_wi_days

    def check_reliability(self, opto_later_lap = 10 , end_lap = 100):
        """ reliability in PF_analysis
        :param opto_later_lap: # laps before opto on in opto_later conditions
        :param end_lap: concatenate first end_lap together (due to different laps ran) between mice. Use a big number
        if want to use min # laps run of all mice.
        """

        reliab = {}
        envs = ['opto_first_day1', 'opto_first_day2', 'opto_later_day1', 'control_day1','opto_later_day2',  'control_day2']
        minlap = dict(zip(envs, [end_lap]*len(envs)))
        minlap.update({'control_later_day1':end_lap, 'control_later_day2':end_lap})

        for c in self.mice:
            reliab[c.mouse] = {}
            temp_later_lap = None
            for env in c.env:
                temp = c.reliability(env, 10, 30)
                if env in ['opto_first_day1', 'opto_first_day2']:
                    reliab[c.mouse][env] = temp
                    minlap[env] = np.min((minlap[env], temp.shape[1]))
                elif env in ['opto_later_day1', 'opto_later_day2']:
                    if env in c.opto_env:
                        temp_later_lap = c.opto_env[env][0] - c.laps[env][0]          # opto on lap in opto_later env
                        temp_day = env[-4:]
                        reliab[c.mouse][env] = temp[:, temp_later_lap-opto_later_lap:]
                        minlap[env] = np.min((minlap[env], reliab[c.mouse][env].shape[1]))
                    else:
                        temp_later_lap = None
                elif env in ['control_day1', 'control_day2']:
                    reliab[c.mouse][env] = temp
                    minlap[env] = np.min((minlap[env], temp.shape[1]))
                    if temp_later_lap is not None:
                        control_day = f'control_later_{temp_day}'
                        reliab[c.mouse][control_day] = temp[:, temp_later_lap-opto_later_lap:]
                        nlaps = reliab[c.mouse][control_day].shape[1]
                        minlap[control_day] = np.min((minlap[control_day], nlaps))
                plt.show()

        # find common env keys that all mice have
        envs.extend(['control_later_day1', 'control_later_day2'])
        all_env = set(envs)
        missing_env = {}
        for c in self.mice:
            dict_keys = [*reliab[c.mouse].keys()]
            if len(dict_keys) != len(envs):
                miss_env = np.setdiff1d(np.array(envs), np.array(dict_keys)).tolist()  # missing env
                missing_env[c.mouse] = miss_env
                all_env = all_env.intersection(set(dict_keys))
                print(f'{c.mouse} miss env: {miss_env}')

        # unpack dict based on minlap
        reliab_var = {}
        mouse_list = [c.mouse for c in self.mice]
        for e in list(all_env):
            var_all_mice = np.concatenate([reliab[mouse][e][:, :minlap[e]] for mouse in mouse_list])
            reliab_var[e] = var_all_mice
        if len(missing_env) > 0:
            # reverse missing_env dict to {env: mouse}
            concat_map = {}
            for k, v in missing_env.items():
                for env in v:
                    concat_map[env] = concat_map.get(env, []) + [k]
            for miss_env in concat_map:
                exist_mouse = np.setdiff1d(np.array(mouse_list), np.array(concat_map[miss_env])).tolist()
                var_all_mice = np.concatenate([reliab[mouse][miss_env][:, :minlap[miss_env]] for mouse in exist_mouse])
                reliab_var[miss_env] = var_all_mice

        return reliab, reliab_var

    def check_emerge_lap(self, mice_list=None):

        # plot emerge lap histogram for each mouse
        laps_dict = {}
        if mice_list is None:
            mice_list = self.mice
        for c in mice_list:
            laps_dict[c.mouse] = {}
            for e in c.env:
                laps = c.emerge_lap_cumhist(e)
                laps_dict[c.mouse][e] = laps.astype(int)
            plt.xlabel('emerge lap')
            plt.title(c.mouse)
            plt.legend(c.env, loc = 'lower right')
            plt.show()

        # laps_var = self.unpack_dict(laps_dict, mouse_list= mouse)
        return laps_dict  #, laps_var

    def check_overday_PF(self, mice_list=None, max_lap=30):
        envs = [['opto_first_day1', 'opto_first_day2'], ['control_day1', 'control_day2'], ['opto_later_day1', 'opto_later_day2']]

        laps_dict_overday = {}
        reliab_dict_day2 = {}

        if mice_list is None:
            mice_list = self.mice
        for c in mice_list:
            laps_dict_overday[c.mouse] = {}
            reliab_dict_day2[c.mouse] = {}
            for e in envs:
                _, d, reliab_dict = c.overday_PF_match(e[0], e[1], max_lap=max_lap)
                laps_dict_overday[c.mouse].update(d)
                reliab_dict_day2[c.mouse].update(reliab_dict)
                plt.show()

        laps_var_overday = self.unpack_dict(laps_dict_overday, mice_list=mice_list)

        return laps_dict_overday, laps_var_overday, reliab_dict_day2

    def check_same_PF_opto(self, mice_list=None):

        if mice_list is None:
            mice_list = self.mice

        for c in mice_list:
            c.PF_summary_opto = c.PF_summary_opto[0:0]
            for e in c.env:
                c.opto_check(e)
        PF_summary_opto = pd.concat([c.PF_summary_opto for c in mice_list], ignore_index=True)
        PF_summary_opto.reset_index(drop=True, inplace=True)

        opto_pq = os.path.join(self.save_path, f'PF_summary_opto_timing_{self.group}.parquet')
        PF_summary_opto.astype(str).to_parquet(opto_pq, compression='gzip')  # save to parquet

        # for e in ['opto_later_day1', 'opto_later_day2', 'control_opto_later_day1', 'control_opto_later_day2']:
        #     print(f'{e}')
        #     for f in features:
        #         df_env = PF_summary_opto.loc[PF_summary_opto['env']==e]
        #         dab = dabest.load(data=df_env, x="opto lap", y=f, idx=tuple(['before', 'during', 'after']))
        #         dab.mean_diff.plot(raw_marker_size=2, swarm_label = f'{e}: {f}');

        return PF_summary_opto

    def check_shift_xcorr(self):
        lag_dict = {}
        emerge_lag_dict = {}

        for c in self.mice:
            lag_dict[c.mouse] = {}
            emerge_lag_dict[c.mouse] = {}
            for e in c.env:
                lag_mat, xcorr_mat, emerge_lag = c.shift_xcorr(e)
                lag_dict[c.mouse][e] = lag_mat
                emerge_lag_dict[c.mouse][e] = emerge_lag

        return lag_dict, emerge_lag_dict

    def check_remapping(self, min_lap = 10):

        remapping_heatmap_on = {}
        remapping_heatmap_off = {}

        env = ['opto_first_day1', 'opto_first_day2', 'opto_later_day1', 'opto_later_day2']
        mouse_list = [m.mouse for m in self.mice]
        df = pd.DataFrame(index = mouse_list, columns = env)

        for m in self.mice:
            remapping_heatmap_on[m.mouse] = {}
            remapping_heatmap_off[m.mouse] = {}
            for e in m.opto_env:
                last_lap_env = m.laps[e][-1]
                last_lap_opto = m.opto_env[e][-1]+1
                if last_lap_env - last_lap_opto > min_lap:
                    first_lap_opto = m.opto_env[e][0]
                    print(f'{m.mouse} in {e} opto on and off')
                    remapping_heatmap_on[m.mouse][e] = m.remapping(m.PFs[e], first_lap_opto, last_lap_opto)
                    remapping_heatmap_off[m.mouse][e] = m.remapping(m.PFs[e], last_lap_opto, last_lap_env)
                    df.loc[m.mouse, e] = 1
                else:
                    print(f'{m.mouse} not enough laps in {e}')

        remapping_on = self.unpack_dict_missing_env(remapping_heatmap_on, full_env= env)
        remapping_off = self.unpack_dict_missing_env(remapping_heatmap_off, full_env= env)

        for e in env:
            on = remapping_on[e]
            off = remapping_off[e]
            max_bin = np.argmax(off, axis=1)
            ind = np.argsort(max_bin)
            remapping_on[e] = on[ind, :]
            remapping_off[e] = off[ind, :]

            sns.heatmap(on[ind, :], xticklabels=5, yticklabels=50, vmin=0, vmax=1)
            plt.xlabel('location on track')
            plt.title(f'{self.group}: {e} opto on')
            plt.savefig(os.path.join(self.save_path, f'{self.group} {e} opto on'))
            plt.show()

            sns.heatmap(off[ind, :], xticklabels=5, yticklabels=50, vmin=0, vmax=1)
            plt.xlabel('location on track')
            plt.title(f'{self.group}: {e} opto off')
            plt.savefig(os.path.join(self.save_path, f'{self.group} {e} opto off'))
            plt.show()

        return remapping_on, remapping_off

    def plot_opto_feature(self, df, feature, col, loop_y, x_label, orders, ylim):
        """ pointplot for dataframe
        :param: df: data in dataframe format
        :param: feature: column name in df that's plotted on the y-axis
        :param: col: under column in df that values equal to loop_y to subsets the df
        :param: loop_y: under column in df that values equal to loop_y to subsets the df
        :param: x_label: column name in df that's plotted on the x-axis, orders are values within column name
        :param: orders: values in x_label column
        """

        for o in loop_y:
            df_o = df.loc[df[col] == o]
            sns.pointplot(x=x_label, y=feature, data=df_o, errorbar='sd', order=orders, hue = 'mouse', alpha = 0.5, dodge = True)
            titlename = f'{feature} {o}'
            plt.ylim(ylim)
            plt.title(titlename)
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.savefig(os.path.join(self.save_path, f'{self.group} {titlename}'), bbox_inches='tight')
            plt.show()

    def plot_opto_feature_combined(self, df, feature, x_label, orders, color, ylim):
        """ pointplot for dataframe animals within the same group combined together
        :param: df: data in dataframe format
        :param: feature: column name in df that's plotted on the y-axis
        :param: x_label: column name in df that's plotted on the x-axis, orders are values within column name
        :param: orders: values in x_label column
        :param: color: column name to plot different colors on the graph
        """

        sns.pointplot(x=x_label, y=feature, data=df, errorbar='sd', order=orders, hue=color, alpha = 0.5, dodge = True)
        plt.ylim(ylim)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        n = len(df['mouse'].unique())
        titlename = f'{self.group} {feature} combined n={n}'
        plt.title(titlename)
        plt.savefig(os.path.join(self.save_path, titlename), bbox_inches='tight')

    @staticmethod
    def dict_to_df(dict_to_unpack, feature_name, envs = None):

        if envs is None:
            envs = [*dict_to_unpack.keys()]

        # organize data into dabest dataframe format
        data = np.concatenate([dict_to_unpack[e] for e in envs])
        cell_pairs = [len(x) for x in dict_to_unpack.values()]
        data_env = [cell_pairs[x] * [envs[x]] for x in range(len(cell_pairs))]
        data_env = sum(data_env, [])
        df = pd.DataFrame(data, columns=[feature_name])
        df['env'] = data_env

        return df

    def dict_dabest_by_mouse(self, corr_bw_days, feature_name, order):

        mouse_df = pd.DataFrame()

        for k in corr_bw_days:
            mouse_dict = corr_bw_days[k]
            print(k)
            # mouse_df_local = self.dict_to_df(mouse_dict, feature_name=feature_name)
            # mouse_df_local['mouse'] = k
            # mouse_df = pd.concat([mouse_df, mouse_df_local], ignore_index=True)
            # sns.swarmplot(data=mouse_df_local, y="stability", x="env", alpha = 0.5, order = order)
            # plt.title(k)
            # plt.show()
            _, df = self.dict_dabest(mouse_dict, feature_name, envs= order)
            df['mouse'] = k
            mouse_df = pd.concat([mouse_df, df], ignore_index=True)
            plt.show()

        # sns.swarmplot(data=mouse_df, y="stability", x="env", alpha=0.5, order=order, size = 1.5)
        # plt.title('all mice combined')
        # plt.show()

        return mouse_df

    def dict_dabest(self, dict_to_dabest, feature_name, envs = None, size = 1.5, alpha = 0.05):
        """ convert dict to dataframe to visualize using dabest package
        :param dict_to_dabest: dict that contains data from all animals combined. eg. {'opto_first_day1': all data}
        """
        if envs is None:
            envs = [*dict_to_dabest.keys()]

        df = self.dict_to_df(dict_to_dabest, feature_name)

        # dabest plotting
        dab = dabest.load(data=df, x="env", y=feature_name, idx=tuple(envs))
        dab.mean_diff.plot(raw_marker_size=size);
        summary = dab.mean_diff.statistical_tests
        alpha_adj = 1 - (1 - alpha) ** (1 / len(envs))
        if sum(summary['pvalue_mann_whitney'] < alpha_adj) > 0:
            sig_envs = summary.loc[summary['pvalue_mann_whitney'] < alpha_adj, ['control', 'test']].values.tolist()
            print(f'{feature_name} is sig diff between: {sig_envs}')

        return dab, df

    @staticmethod
    def shuffle_sig(df, feature, group, nshuffle=600):
        """ non-parametric test of significance thru shuffling
        :param df: overall dataframe with data, within and between group information
        :param feature: a column in dataframe that contains data to test for significance
        :param group: a column in dataframe that labels between group variable
        output: shuffled matrix, original data mean, groups for comparison
        """

        data = df[feature].dropna().to_numpy()
        n = len(data)
        shuffle_mat = np.zeros((nshuffle, n))
        population = range(n)

        all_groups = df.groupby([group, 'env'])
        counts = all_groups.count()[feature].to_numpy()
        inds = np.cumsum(np.array(counts))
        if inds[-1] != n:
            print(f'warning: count ind {inds[-1]} not matched to # data {n}')

        for s in range(nshuffle):
            r = random.sample(population, n)
            shuffle_mat[s, :] = data[r]

        array_sec = np.split(shuffle_mat, inds, axis=1)[0:-1]
        shuffle_mean = np.zeros((nshuffle, len(counts)))
        for n in range(len(array_sec)):
            shuffle_mean[:, n] = np.mean(array_sec[n], axis=1)

        df_mean = all_groups.mean()[feature]
        print('group mean:')
        print(df_mean.round(3).reset_index())

        return shuffle_mean, df_mean.to_numpy(), [*all_groups.groups.keys()]

    @staticmethod
    def sig_wi(shuffle_mat, data_mat, columns1, columns0, envs, alpha = 0.01):
        """
        :param shuffle_mat: shuffled matrix of nshuffles * nenvs (output from shuffle_sig())
        :param data_mat: original data matrix (output from shuffle_sig())
        :param columns1: within group comparison experimental group indices, eg. [0, 1, 5,6]
        :param columns0: within group comparison control group indices, eg. [2, 3, 7,8]
        :param envs: envs corresponding to each column of data (output from shuffle_sig())
        """

        # adjust for multiple comparison
        alpha_adj = 1 - (1 - alpha) ** (1 / len(columns1))

        # within group comparison
        shuffle_diff = shuffle_mat[:, np.array(columns1)] - shuffle_mat[:, np.array(columns0)]
        sig_thresh_wi = np.zeros((2, len(columns0)))
        sig_thresh_wi[0, :] = np.quantile(shuffle_diff, 1-alpha_adj, axis=0)
        sig_thresh_wi[1, :] = np.quantile(shuffle_diff, alpha_adj, axis=0)

        data_diff = data_mat[np.array(columns1)] - data_mat[np.array(columns0)]
        for n in range(len(data_diff)):
            if (data_diff[n] > sig_thresh_wi[0, n]) or (data_diff[n] < sig_thresh_wi[1, n]):
                print(f'sig diff between {envs[columns1[n]]} & {envs[columns0[n]]}')

        return sig_thresh_wi, data_diff

    def unpack_dict_missing_env(self, dict_to_unpack, full_env = None):
        """ flatten nested dict, identify mouse with missing envs, and combine data between mice
        :param dict_to_unpack: nested dict to unpack in the format of {mouse: {envs: data}}
        :param full_env: full envs to compare with envs in each mouse
        """

        if full_env is None:
            full_env = ['opto_first_day1', 'opto_first_day2', 'opto_later_day1', 'opto_later_day2', 'control_day1',
                        'control_day2']

        # dataframe to organize mouse with corresponding envs
        mice_list = [*dict_to_unpack.keys()]
        #print(mice_list)
        mouse_env_df = pd.DataFrame(index = mice_list, columns = full_env)

        for m in dict_to_unpack:
            envs = [*dict_to_unpack[m].keys()]
            col_envs = np.intersect1d(envs, full_env)
            mouse_env_df.loc[m, col_envs] = 1

        complete_env = mouse_env_df.columns[~mouse_env_df.isnull().any()].tolist()   # envs that all mice have
        missing_env = mouse_env_df.columns[mouse_env_df.isnull().any()].tolist()     # envs that some mice miss

        dict_combined = self.unpack_dict(dict_to_unpack, env = complete_env, mice_list = mice_list)
        display(mouse_env_df)

        for n in missing_env:
            mouse = mouse_env_df[mouse_env_df[n].notnull()].index.tolist()
            print(f'adding {mouse} in env {n}')
            dict_combined = self.unpack_dict(dict_to_unpack, env=[n], mice_list=mouse, dict_combined=dict_combined)

        return dict_combined

    def unpack_dict(self, dict_to_unpack, env=None, mice_list=None, dict_combined=None):
        """ flatten nested dict and combine data between mice
        :param dict_to_unpack: nested dict of mouse, env (eg. output from check_stability)
        :param env: must be the same as a subset of keys in dict_to_unpack
        """

        if dict_combined is None:
            dict_combined = {}

        if env is None:
            env = [*dict_to_unpack[self.mice[0].mouse].keys()]  # assume envs for all mice are same
        if mice_list is None:
            mice_list = [*dict_to_unpack.keys()]

        for e in env:
            dict_combined[e] = np.concatenate([dict_to_unpack[mouse][e] for mouse in mice_list])

        return dict_combined

    def unpack_dict_var(self, dict_to_unpack, var, output_dict=None, env = None, mice_list=None):
        """ flatten nested dict and combine data between mice
        :param dict_to_unpack: nested dict of mouse, env, var (eg. output from check_opto)
        :param var: var to combine within env and between mice
        """
        if output_dict is None:
            output_dict = {}

        if mice_list is None:
            mice_list = self.mice

        if env is None:
            env = [*dict_to_unpack[self.mice[0].mouse].keys()]   # assume envs for all mice are same

        for e in env:
            output_dict[e] = np.concatenate([dict_to_unpack[mouse.mouse][e][var] for mouse in mice_list])

        return output_dict

    def unpack_PF_dict(self, dict_to_unpack, output_dict={}, env = None, mice_list=None):
        """ flatten PF_dict from opto_dict (output from self.check_opto())
        format: eg. opto_dict[mouse][env]['PF_dict']['on_only']
        """
        if mice_list is None:
            mice_list = self.mice

        if env is None:
            env = [*dict_to_unpack[mice_list[0].mouse].keys()]   # assume envs for all mice are same

        for e in env:
            output_dict[e] = {}
            for k in ['on only', 'off only', 'both same COM', 'both shift COM']:
                var_all_mice = np.concatenate([dict_to_unpack[mouse.mouse][e]['PF_dict'][k] for mouse in mice_list])
                output_dict[e][k] = var_all_mice

        self.save_to_file(output_dict, 'PF_dict')

        for e in env:
            field_count = np.array([len(output_dict[e]['on only']), len(output_dict[e]['off only']),
                                    len(output_dict[e]['both same COM']), len(output_dict[e]['both shift COM'])])
            pie_labels = ['on only', 'off only', 'both same COM', 'both shift COM']
            plt.pie(field_count, labels=pie_labels)
            pie_title = f'{self.group} (n={len(mice_list)}) in {e} opto effects on place fields'
            plt.title(pie_title)
            plt.savefig(os.path.join(self.save_path, pie_title))
            plt.show()

        return output_dict

    @staticmethod
    def run_kruskal(dict_var, cond_list, alpha=0.05):
        """ run kruskal wallis test on dict_var[cond_list] """

        l = []
        for cond in cond_list:
            l.append(list(dict_var[cond].reshape(-1)))

        _, p = stats.kruskal(*l, nan_policy='omit')
        thresh = 'significant' if p < alpha else 'not significant'
        print(f'Kruskal wallis test comparing {cond_list} p value is: {np.round(p, 3)}, e^{np.round(np.log10(p),1)}, {thresh}')

    def run_kw_slide(self, dict_var, cond_list, opto_on, opto_laps=13, alpha=0.05, max_lap = 100):
        """ run kw test on dict_var[cond_list] for each lap, corrected for multiple comparison, use for reliability.
        shaded by SEM.

        :param dict_var: {env: value} eg. output from unpack_dict or unpack_dict_var
        :param cond_list: eg. ['control_day1', 'opto_first_day1']. control first
        :param alpha: adjust alpha level to mark significance
        :param n: # of mice in the group
        :param max_lap: plot at most # laps
        """

        nlaps = min([dict_var[e].shape[1] for e in cond_list])
        nlaps = min([nlaps, max_lap])
        n = len(self.mice)

        # plotting mean reliability
        fig, axs = plt.subplots(2,1, sharex=True, gridspec_kw={'height_ratios': [2, 1]})
        for env in cond_list:
            mean_r = np.nanmean(dict_var[env][:, :nlaps+1], axis=0)
            std_r = np.nanstd(dict_var[env][:, :nlaps+1], axis=0)
            axs[0].plot(mean_r)
            axs[0].fill_between(np.arange(len(mean_r)), mean_r + std_r/np.sqrt(n)/2, mean_r - std_r/np.sqrt(n)/2, alpha=0.2)

        axs[0].legend(cond_list, loc='lower right', framealpha=0.5)
        axs[0].plot([opto_on, opto_on+opto_laps], [np.max(mean_r)+0.1, np.max(mean_r)+0.1], '--', c='r')
        axs[1].set_xlabel('laps')
        axs[0].set_ylabel('reliability')
        axs[0].set_title(f'{self.group}: lap-by-lap reliability')

        # sidak adjustment for family-wise error
        alpha_adj = 1- (1-alpha)**(1/nlaps)

        # run kw test on each lap
        p = np.zeros((nlaps, 1))
        diff = np.zeros((nlaps, len(cond_list)))
        for n in range(nlaps):
            l = []
            for cond in cond_list:
                l.append(list(dict_var[cond][:, n]))
            _, p[n] = stats.kruskal(*l, nan_policy='omit')
            diff[n, :] = [np.nanmean(reb) for reb in [*l]]

        # plot mean difference and significance level
        for n in range(1, len(cond_list)):
            data = diff[:, n] - diff[:, 0]
            axs[1].plot(data, c='k')
            axs[1].text(len(data), data[-1], f'{cond_list[n]} - control')
        axs[1].set_ylabel('delta')
        axs[1].set_ylim([-0.25, 0.2])

        sig_lap = np.where(p<alpha_adj)[0]
        axs[1].scatter(sig_lap, data[sig_lap]+np.nanstd(data)*2, c=0-np.log10(p[sig_lap]), cmap='Reds', marker='x')
        #plt.show()

        return p, diff

    def plot_all_cells(self):
        for m in self.mice:
            for e in m.env:
                print(f'{m.mouse} in {e}')
                m.plot_all_cells(e, save=1, autoscale=0)

    @staticmethod
    def plot_hist(dict_var, envs):
        """ plot histogram from dict_var[envs]
        :param dict_var: {env: value} eg. output from unpack_dict or unpack_dict_var
        :param envs: list of envs, must be the keys of dict_var
        """

        for e in envs:
            plt.hist(dict_var[e], weights=np.ones(len(dict_var[e])) / len(dict_var[e]), alpha=0.5)
        plt.legend([envs])
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
        plt.ylabel('cell count')
        plt.legend([envs])

        PF_group.run_kruskal(dict_var, envs)

    @staticmethod
    def plot_exp_dist(dict_var, envs, max_data = 100, legendloc = 'lower right'):

        for e in envs:
            data = dict_var[e]
            plt.hist(data[data<max_data], density=True, cumulative=True, alpha=0.5)
            P = stats.expon.fit(data[data<max_data])
            rX = np.linspace(0, np.max(data[data<max_data]), 100)
            rP = stats.expon.cdf(rX, *P)
            plt.plot(rX, rP)

        plt.ylabel('cell count')
        plt.legend(envs)

        #PF_group.run_kruskal(dict_var, envs)

    @staticmethod
    def plot_cumhist(dict_var, envs, max_data = 100, legendloc = 'upper left', alpha=0.05):
        """ cumulative histogram for clarity """
        l = []

        for env in envs:
            data = dict_var[env]
            data = data[data<max_data]
            l.append(data)
            plt.hist(data, density=True, histtype='step', cumulative=True)

        plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
        plt.legend(envs, loc=legendloc)
        plt.ylabel('% cells')

        _, p = stats.kruskal(*l, nan_policy='omit')
        thresh = 'significant' if p < alpha else 'not significant'
        print(f'Kruskal wallis test comparing {envs} p value is: {np.round(p, 3)}, '
              f'e^{np.round(np.log10(p), 1)}, {thresh}')