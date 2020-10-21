import glob
import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn import preprocessing
import numpy as np

from utils.config import path_dir_descriptor, path_dir_data, path_dir_tsOutput, Poly_degree, path_file_mapping


# Load descriptor
def loadAllCsv(path, pattern):
    all_files = glob.glob(os.path.join(path, pattern))
    data = pd.concat((pd.read_csv(f) for f in all_files))
    return data


def combineData():
    # 1_Substrate
    df1 = loadAllCsv(path_dir_descriptor, "1*.csv")
    df1 = df1.drop("Compound", axis=1)
    # 2_Base or Conjugate
    df2 = loadAllCsv(path_dir_descriptor, "2*.csv")
    df2 = df2.drop("Compound", axis=1)
    # 3_Hydroxyamine or Oxoammonium
    df3 = loadAllCsv(path_dir_descriptor, "3*.csv")
    df3 = df3.drop("Compound", axis=1)
    # 4_Anti or Syn Ligand
    df4 = loadAllCsv(path_dir_descriptor, "4*.csv")
    df4 = df4.drop("Compound", axis=1)
    df1["tmp"] = 1
    df2["tmp"] = 1
    df3["tmp"] = 1
    df4["tmp"] = 1

    df1.rename(columns={"Name": "id"}, inplace=True)
    df = pd.merge(df1, df2, on=['tmp'])
    df["id"] = df["id"] + "_" + df["Name"]
    df = df.drop(columns={"Name"})
    df = pd.merge(df, df3, on=['tmp'])
    df["id"] = df["id"] + "_" + df["Name"]
    df = df.drop(columns={"Name"})
    df = pd.merge(df, df4, on=['tmp'])
    df["id"] = df["id"] + "_" + df["Name"]
    df = df.drop(columns={"Name"})
    df = df.drop('tmp', axis=1)
    return df


def makeTraingDataFeatures(fname="features.csv"):
    df = combineData()
    df.to_csv(os.path.join(path_dir_data, fname), index=False)


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


def PolyRegression(degree, x, y):
    x_ = PolynomialFeatures(degree=degree, include_bias=True).fit_transform(x)
    model = LinearRegression(fit_intercept=False).fit(x_, y)
    return model.coef_


# load TimeseriesCsv and convert to Polynomial Regression coefficients
def loadTimeSeriesCsv(fname):
    df = pd.read_csv(fname)
    ncols = df.shape[1]
    x_ = np.array(df.iloc[:, 0])
    x_ = x_.reshape(-1, 1)
    coef_df = pd.DataFrame()
    for col in range(1, ncols):
        y_ = np.array(df.iloc[:, col])
        id_ = df.columns.values[col]
        coef_ = PolyRegression(degree=Poly_degree, x=x_, y=y_)
        row = pd.DataFrame()
        row["id"] = [id_]
        row["coef"] = [coef_]
        coef_df = coef_df.append(row)
    return coef_df


def loadDataOutput():
    all_files = glob.glob(os.path.join(path_dir_tsOutput, "*.csv"))
    coef_df = pd.DataFrame()
    for fname in all_files:
        coef_ = loadTimeSeriesCsv(fname)
        coef_df = coef_df.append(coef_)
    return coef_df


def makeTrainingDataOutput(fname="output.csv"):
    df = loadDataOutput()
    # Load Mapping
    mapping = pd.read_csv(path_file_mapping)
    mapping = mapping.set_index("id").T
    di = mapping.to_dict(orient="list")
    df["id"].replace(di, inplace=True)
    df.to_csv(os.path.join(path_dir_data, fname), index=False)
