import ast
import glob
import os
from sklearn import preprocessing

from utils.config import path_dir_data
import pandas as pd
import numpy as np


def loadTrainingDataFeatures(fname='features.csv', isNormalized=True):
    file = fname
    if fname is loadTrainingDataFeatures.__defaults__[0]:
        file = os.path.join(path_dir_data, fname)
    df = pd.read_csv(file)
    idList = df[['id']].values
    df1 = df.drop(columns={'id'})
    X = df1.values  # return a numpy array
    if isNormalized:
        scaler = preprocessing.StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X = X_scaled
    return idList, X


def createSubSet(X, nsample=1000):
    if nsample > X.shape[0]:
        nsample = X.shape[0]
    idx = np.random.randint(X.shape[0], size=nsample)
    return X[idx, :], idx


def from_np_array(array_string):
    array_string = ','.join(array_string.replace('[ ', '[').split())
    return np.array(ast.literal_eval(array_string))


def loadTrainingDataOutput(fname='output.csv'):
    file = fname
    if fname is loadTrainingDataOutput.__defaults__[0]:
        file = os.path.join(path_dir_data, fname)
    df = pd.read_csv(file, converters={'coef': from_np_array})
    idList = df[['id']].values[:, 0]
    df1 = df.drop(columns={'id'})
    coefs = df1.values[:, 0]  # return a numpy array
    return idList, coefs


