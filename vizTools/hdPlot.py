import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


# Visualize high dimensional data
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from utils.utils import PCAComp


def visualize3D(X, colors=0):
    if colors is visualize3D.__defaults__[0]:
        colors = np.zeros(X.shape[0], dtype=int)
    fig = plt.figure(1, figsize=(8, 8))
    plt.clf()
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=120)
    ax.set_xlabel('Component-1', fontsize=15)
    ax.set_ylabel('Component-2', fontsize=15)
    ax.set_zlabel('Component-3', fontsize=15)
    X2 = X
    if X.shape[1] > 3:
        X2 = PCAComp(X, 3)
    # ax.scatter(X2[:, 0], X2[:, 1], X2[:, 2], cmap=plt.cm.nipy_spectral, edgecolor='k')
    ax.scatter(X2[:, 0], X2[:, 1], X2[:, 2], c=colors, cmap='winter', edgecolor='k')
    plt.show()


def visualize2D(X, colors=0):
    if colors is visualize3D.__defaults__[0]:
        colors = np.zeros(X.shape[0], dtype=int)
    X2 = X
    if X.shape[1] > 2:
        X2 = PCAComp(X, 2)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Component-1', fontsize=15)
    ax.set_ylabel('Component-2', fontsize=15)
    ax.scatter(X2[:, 0]
               , X2[:, 1]
               , c=colors
               , s=50)
    ax.grid()
    plt.show()


def plot_tsne_pca(data, labels):
    max_label = max(labels)
    max_items = np.random.choice(range(data.shape[0]), size=3000, replace=False)

    pca = PCA(n_components=2).fit_transform(data[max_items, :].todense())
    tsne = TSNE().fit_transform(PCA(n_components=50).fit_transform(data[max_items, :].todense()))

    idx = np.random.choice(range(pca.shape[0]), size=300, replace=False)
    label_subset = labels[max_items]
    label_subset = [cm.hsv(i / max_label) for i in label_subset[idx]]

    f, ax = plt.subplots(1, 2, figsize=(14, 6))

    ax[0].scatter(pca[idx, 0], pca[idx, 1], c=label_subset)
    ax[0].set_title('PCA Cluster Plot')

    ax[1].scatter(tsne[idx, 0], tsne[idx, 1], c=label_subset)
    ax[1].set_title('TSNE Cluster Plot')


