import os
import numpy as np
import pandas as pd
import scipy
from scipy.signal import find_peaks, peak_widths
import seaborn as sns


def load_beh(path):

    df = []
    f = [f for f in os.listdir(path) if f.endswith('beh.mat')]
    #print(f)

    for files in f:
        print(files)
        mouse = files.split('-')[1]
        env = files.split('-')[2]
        opto = files.split('-')[3]
        beh_file = scipy.io.loadmat(os.path.join(path, files))
        ybinned = beh_file['behavior']['ybinned'][0][0].transpose()
        velocity = beh_file['behavior']['velocity'][0][0].transpose()
        rewards = beh_file['behavior']['reward'][0][0].transpose()
        lick = beh_file['behavior']['lick'][0][0].transpose()
        lap = beh_file['E'][0]
        beh_df = pd.DataFrame(dict(zip(['y', 'velocity', 'rewards', 'lick', 'lap'], [ybinned.flatten(), velocity.flatten(), rewards.flatten(), lick.flatten(), lap])))
        beh_df['mouse'] = mouse
        beh_df['env'] = env
        beh_df['opto'] = opto
        beh_df.loc[beh_df['lick'] < 2, 'lick'] = 0
        beh_df.loc[beh_df['lick'] > 2, 'lick'] = 1
        beh_df.loc[beh_df['rewards'] < 6, 'rewards'] = 0
        beh_df.loc[beh_df['rewards'] > 6, 'rewards'] = 1
        df.append(beh_df)

    return pd.concat(df).reset_index(drop=True)


def get_pre_reward_rows(group, rows_above=155, rows_below=62):
    # Find indices where rewards is 1
    reward_rows = group[group['rewards'] == 1]

    # If there is no reward in this group, return an empty DataFrame
    if reward_rows.empty:
        return pd.DataFrame()

    # Get the first occurrence
    first_reward_index = reward_rows.index[0]

    # Get the position of this row within the group
    pos = group.index.get_loc(first_reward_index)

    # Determine the start position for the 5 rows preceding the reward row
    start = max(0, pos - rows_above)
    last_row = min(len(group), pos + rows_below)
    group['timing'] = np.arange(len(group)) - pos
    # print(start, last_row)

    # Select the 5 rows immediately before the first rewards==1 row
    return group.iloc[start:last_row]


def remove_lick(df: pd.DataFrame, width_thresh=11, abs_width=32, distance_thresh=0.075, lower_thresh=-0.545):

    lick = df.reset_index(drop=True)
    s = np.diff(lick['lick'].astype('int').to_numpy())
    ind_5 = np.flatnonzero(s > 0)
    ind_neg5 = np.flatnonzero(s < 0)
    if len(ind_5) > len(ind_neg5):
        ind_neg5 = np.concatenate((ind_neg5, [len(lick) - 1]))
    width = (ind_neg5 - ind_5)
    lick_width = pd.DataFrame(
        {'start ind':ind_5+1, 'end ind': ind_neg5+1,'start_y': lick.loc[ind_5, 'y'].to_numpy(),
         'end_y': lick.loc[ind_neg5, 'y'].to_numpy(), 'width': width,
         'mouse': lick.loc[ind_5, 'mouse'].to_numpy(), 'env': lick.loc[ind_5, 'env'].to_numpy(),
         'opto': lick.loc[ind_5, 'opto'].to_numpy(), 'lap': lick.loc[ind_5, 'lap'].to_numpy()})
    lick_width['distance'] = lick_width['end_y'] - lick_width['start_y']
    lick_width['remove'] = False
    lick_width.loc[(lick_width['width'] > width_thresh) & (lick_width['distance']>distance_thresh), 'remove'] = True
    lick_width.loc[(lick_width['width'] > width_thresh) & (lick_width['distance']<0-distance_thresh) & (lick_width['distance']>lower_thresh), 'remove'] = True
    lick_width.loc[(lick_width['width'] > abs_width), 'remove'] = True
    lick_width.loc[lick_width['distance'] >distance_thresh, 'remove'] = True

    drop_ind = np.concatenate([np.arange(start, end) for start, end in
                               zip(lick_width.loc[lick_width['remove'] == True, 'start ind'],
                                   lick_width.loc[lick_width['remove'] == True, 'end ind'])])

    return lick_width, drop_ind


def remove_artifact(fam_beh: pd.DataFrame, column_name='lick', thresh=7):

    peaks, _ = find_peaks(fam_beh[column_name].to_numpy())
    results_full = peak_widths(fam_beh[column_name].to_numpy(), peaks, rel_height=1)
    fam_beh['length'] = np.nan
    fam_beh = fam_beh.reset_index(drop=True)

    for n in range(len(results_full[0])):
        left = results_full[2][n] + 1
        right = results_full[3][n]
        fam_beh.loc[left:right, 'length'] = results_full[0][n] - 1

    print(f'{len(fam_beh.loc[fam_beh["length"] > thresh])} rows removed')
    fam_beh.loc[fam_beh['length'] > thresh, column_name] = 0

    return fam_beh.drop(columns=['length'])

def binarize_location(df: pd.DataFrame, column_name='y', start_y=0.02, end_y=0.64, nbins=100):

    binEdges = np.linspace(start_y, end_y, nbins + 1)
    fam_beh = df.copy()
    fam_beh['spatial bin'] = pd.cut(fam_beh[column_name], binEdges)
    bin_map = dict(zip(fam_beh.sort_values(by=column_name)['spatial bin'].unique()[1:], np.arange(nbins)))
    fam_beh['spatial bin'] = fam_beh['spatial bin'].map(bin_map)
    fam_beh['spatial bin'] = fam_beh['spatial bin'].astype('float')

    return fam_beh, bin_map

def lick_raster(df, groupby_var = ('mouse', 'opto', 'lap'), column_name = 'spatial bin'):
    var_list = list(groupby_var)
    var_list.append(column_name)
    lick = df.groupby(var_list)['lick'].sum().reset_index()
    lick_df = lick.pivot(index=var_list[:-1], columns=column_name, values='lick').reset_index()
    lick_mat = lick_df.sort_values(by=var_list[:-1]).iloc[:, 3:].fillna(0).to_numpy()

    sns.heatmap(lick_mat, vmin=0.01, vmax=1, cmap='gray_r')

    return lick_df