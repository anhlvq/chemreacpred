import os

from utils.config import path_dir_data


def getFullDataPath(fname):
    return os.path.join(path_dir_data, fname)


def getFullDataPath(fname):
    return os.path.join(path_dir_data, fname)


def getBaseName(fname):
    return os.path.basename(fname)


def checkExists(filename):
    return os.path.isfile(filename)
