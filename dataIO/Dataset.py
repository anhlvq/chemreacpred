import glob

from dataIO.loader import loadTrainingDataFeatures, readNumpyArrayFile, writeNumpyArrayFile
from utils.fileSystemUtils import getBaseName, checkExists, getFullDataPath
from utils.utils import TSNEComp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class FeatureDataset:
    dataSetName = None
    filePath = None
    idList = None
    features = None

    def __init__(self, fname, isNormalized=True):
        self.dataSetName = getBaseName(fname)
        self.filePath = fname
        self.idList, self.features = loadTrainingDataFeatures(self.filePath, isNormalized=isNormalized)

    def tsne2Comp(self):
        file = self.filePath + ".tsne2Comp"
        if checkExists(file):
            return readNumpyArrayFile(file)
        else:
            X = TSNEComp(self.features, 2)
            writeNumpyArrayFile(file, X)
            return X

    def tsne3Comp(self):
        file = self.filePath + ".tsne3Comp"
        if checkExists(file):
            return readNumpyArrayFile(file)
        else:
            X = TSNEComp(self.features, 3)
            writeNumpyArrayFile(file, X)
            return X

    def plot2D(self, colors="blue", ax=None):
        X2 = self.tsne2Comp()
        ax1 = ax
        if ax1 is None:
            fig = plt.figure(figsize=(8, 8))
            ax1 = fig.add_subplot(1, 1, 1)
            ax1.set_title(self.dataSetName)
        ax1.set_xlabel('Component-1', fontsize=12)
        ax1.set_ylabel('Component-2', fontsize=12)
        ax1.scatter(X2[:, 0]
                    , X2[:, 1]
                    , c=colors
                    , s=50)
        ax1.grid()
        if ax is None:
            plt.show()

    def plot3D(self, colors="blue", ax=None):
        X3 = self.tsne3Comp()
        ax1 = ax
        if ax1 is None:
            fig = plt.figure(1, figsize=(8, 8))
            plt.clf()
            ax1 = Axes3D(fig, rect=[0., 0., .95, 1.], elev=48, azim=120)
            ax1.set_title(self.dataSetName)
        ax1.set_xlabel('Component-1', fontsize=12)
        ax1.set_ylabel('Component-2', fontsize=12)
        ax1.set_zlabel('Component-3', fontsize=12)
        # ax.scatter(X2[:, 0], X2[:, 1], X2[:, 2], cmap=plt.cm.nipy_spectral, edgecolor='k')
        ax1.scatter(X3[:, 0], X3[:, 1], X3[:, 2], c=colors, cmap='winter', edgecolor='k')
        if ax is None:
            plt.show()

    def plot(self, colors="blue", addText=None):
        fig = plt.figure(figsize=(12, 6))
        if addText is not None:
            fig.suptitle(self.dataSetName + " - " + addText)
        else:
            fig.suptitle(self.dataSetName)
        ax1 = fig.add_subplot(1, 2, 1)
        self.plot2D(colors=colors, ax=ax1)
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        self.plot3D(colors=colors, ax=ax2)
        plt.show()


def LoadAllFeatureDataSets(filepattern="_feature*.csv", isNormalized=True):
    all_files = glob.glob(getFullDataPath(filepattern))
    ds_list = list()
    for fname in all_files:
        ds_list.append(FeatureDataset(fname, isNormalized))
    return ds_list

# ds = FeatureDataset(getFullDataPath('_featureSBHA.csv'))
# lst = LoadAllFeatureDataSets()
# lst[0].plot2D()

# for ds in lst:
#    X_2Comp = ds.tsne2Comp()
#    X_3Comp = ds.tsne3Comp()
