import glob
import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from utils.config import path_dir_descriptor, path_dir_data, path_dir_tsOutput, Poly_degree


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


def loadTimeSeriesCsv(fname):
    df = pd.read_csv(fname)
    ncols = df.shape()[1]
    tsList = []
    for col in range(1, ncols):
        df1 = df[[0, col]]
        tsList = tsList.append(df1)
    return tsList


def PolyRegression(degree, x, y):
    x_ = PolynomialFeatures(degree=degree, include_bias=True).fit_transform(x)
    model = LinearRegression(fit_intercept=False).fit(x_, y)
    return model.coef_


def makeTrainingDataOutput(fname="output.csv"):
    all_files = glob.glob(os.path.join(path_dir_tsOutput, "*.csv"))
    for fname in all_files:
        tsList = loadTimeSeriesCsv(fname)
        for ts in tsList:
            coef = PolyRegression(degree=Poly_degree)
            ts.to_csv(os.path.join(path_dir_data, fname))
    # df = pd.read_csv("")
