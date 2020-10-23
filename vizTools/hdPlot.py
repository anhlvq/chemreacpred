import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn.decomposition import PCA


# Visualize high dimensional data
def visualize3D(X, colors=0):
    if colors is visualize3D.__defaults__[0]:
        colors = np.zeros(X.shape[0], dtype=int)
    fig = plt.figure(1, figsize=(8, 8))
    plt.clf()
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=120)
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_zlabel('Principal Component 3', fontsize=15)
    plt.cla()
    pca3 = PCA(n_components=3)
    pca3.fit(X)
    X2 = pca3.transform(X)
    # ax.scatter(X2[:, 0], X2[:, 1], X2[:, 2], cmap=plt.cm.nipy_spectral, edgecolor='k')
    ax.scatter(X2[:, 0], X2[:, 1], X2[:, 2], c=colors, cmap='winter', edgecolor='k')
    plt.show()


def visualize2D(X, colors=0):
    if colors is visualize3D.__defaults__[0]:
        colors = np.zeros(X.shape[0], dtype=int)
    pca2 = PCA(n_components=2)
    pca2.fit(X)
    X2 = pca2.transform(X)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.scatter(X2[:, 0]
               , X2[:, 1]
               , c=colors
               , s=50)
    ax.grid()
    plt.show()
