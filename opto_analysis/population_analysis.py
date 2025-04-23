from opto_analysis.place_cell_opto import LoadData
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import seaborn as sns
from sklearn import manifold
from sklearn.decomposition import PCA
from sklearn.metrics import euclidean_distances
from scipy.ndimage import uniform_filter1d


def rsa(mouse: LoadData):

    data = mouse.mean_activity[:, :, :]
    lap_by_lap = data.reshape(data.shape[0], -1, order='F')
    lap_corr = np.corrcoef(lap_by_lap)
    dis_mat = 1-lap_corr

    sns.heatmap(dis_mat)
    plt.title(f'{mouse.name}')
    plt.savefig(os.path.join(mouse.path, f'{mouse.name} lap by lap dissimilarity'))
    plt.show()

    return dis_mat


def nmds_reduce(dis_mat, n_dim = 2):

    seed = np.random.RandomState(seed=3)

    mds = manifold.MDS(
        n_components=n_dim,
        max_iter=3000,
        random_state=seed,
        dissimilarity="precomputed",
        n_jobs=1,
    )
    pos = mds.fit(dis_mat).embedding_

    nmds = manifold.MDS(
        n_components=2,
        metric=False,
        max_iter=3000,
        dissimilarity="precomputed",
        random_state=seed,
        n_jobs=1,
        n_init=1,
    )
    npos_fit = nmds.fit(dis_mat, init=pos)
    stress = npos_fit.stress_
    npos = npos_fit.embedding_

    clf = PCA(n_components=n_dim)
    npos = clf.fit_transform(npos)

    mov_mean = np.zeros_like(npos)
    for n in range(n_dim):
        mov_mean[:, n] = uniform_filter1d(npos[:, n], size=5)

    plt.scatter(npos[:, 0], npos[:, 1], c=np.arange(npos.shape[0]), lw=0, label="NMDS")
    plt.colorbar(label='lap')
    plt.plot(mov_mean[:, 0], mov_mean[:, 1], alpha=0.2, color='black')
    plt.text(1, 1, f"stress: {np.round(stress, 4)}", ha='right', va='top', wrap=True)
    plt.axis('scaled')
    #plt.show()

    return npos, mov_mean
