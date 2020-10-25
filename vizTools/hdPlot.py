import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


# Visualize high dimensional data
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
