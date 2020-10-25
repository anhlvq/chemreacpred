import glob
import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

from utils.config import path_dir_descriptor, path_dir_data, path_dir_tsOutput, Poly_degree, path_file_mapping


# Load descriptor
def loadAllCsv(path, pattern):
    all_files = glob.glob(os.path.join(path, pattern))
    data = pd.concat((pd.read_csv(f) for f in all_files))
    return data


def combineData(path=path_dir_descriptor, pattern_1="1*.csv", pattern_2="2*.csv",
                pattern_3="3*.csv", pattern_4="4*.csv"):
    # 1_Substrate
    df1 = loadAllCsv(path, pattern_1)
    df1 = df1.drop("Compound", axis=1)
    # 2_Base or Conjugate
    df2 = loadAllCsv(path, pattern_2)
    df2 = df2.drop("Compound", axis=1)
    # 3_Hydroxyamine or Oxoammonium
    df3 = loadAllCsv(path, pattern_3)
    df3 = df3.drop("Compound", axis=1)
    # 4_Anti or Syn Ligand
    df4 = loadAllCsv(path, pattern_4)
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


def makeTraingDataFeatures(fname="features.csv", pattern_1="1*.csv", pattern_2="2*.csv",
                           pattern_3="3*.csv", pattern_4="4*.csv"):
    df = combineData(path=path_dir_descriptor, pattern_1=pattern_1, pattern_2=pattern_2, pattern_3=pattern_3,
                     pattern_4=pattern_4)

    df.to_csv(os.path.join(path_dir_data, fname), index=False)


# Make all training dataset
makeTraingDataFeatures(fname="_featureSBHA.csv", pattern_2="2_B*.csv", pattern_3="3_H*.csv", pattern_4="4_Anti*.csv")
makeTraingDataFeatures(fname="_featureSBHS.csv", pattern_2="2_B*.csv", pattern_3="3_H*.csv", pattern_4="4_Syn*.csv")
makeTraingDataFeatures(fname="_featureSBOA.csv", pattern_2="2_B*.csv", pattern_3="3_O*.csv", pattern_4="4_Anti*.csv")
makeTraingDataFeatures(fname="_featureSBOS.csv", pattern_2="2_B*.csv", pattern_3="3_O*.csv", pattern_4="4_Syn*.csv")

makeTraingDataFeatures(fname="_featureCBHA.csv", pattern_2="2_C*.csv", pattern_3="3_H*.csv", pattern_4="4_Anti*.csv")
makeTraingDataFeatures(fname="_featureCBHS.csv", pattern_2="2_C*.csv", pattern_3="3_H*.csv", pattern_4="4_Syn*.csv")
makeTraingDataFeatures(fname="_featureCBOA.csv", pattern_2="2_C*.csv", pattern_3="3_O*.csv", pattern_4="4_Anti*.csv")
makeTraingDataFeatures(fname="_featureCBOS.csv", pattern_2="2_C*.csv", pattern_3="3_O*.csv", pattern_4="4_Syn*.csv")


def PolyRegression(degree, x, y):
    x_ = PolynomialFeatures(degree=degree, include_bias=True).fit_transform(x)
    model = LinearRegression(fit_intercept=False).fit(x_, y)
    return model.coef_


# load TimeseriesCsv
def loadTimeSeriesCsv(fname):
    df = pd.read_csv(fname)
    return df


def loadTimeSeriesOutput():
    files = glob.glob(os.path.join(path_dir_tsOutput, "*.csv"))
    df_list = list()
    for fname in files:
        df = loadTimeSeriesCsv(fname)
        df_list.append(df)
    return df_list


def extractTimeSeries(df, col):
    x_ = np.array(df.iloc[:, 0])
    y_ = np.array(df.iloc[:, col])
    return x_, y_


# convert to Polynomial Regression coefficients
def convertToPolynomialRegression(df, degree=Poly_degree):
    ncols = df.shape[1]
    x_ = np.array(df.iloc[:, 0])
    x_ = x_.reshape(-1, 1)
    coef_df = pd.DataFrame()
    for col in range(1, ncols):
        y_ = np.array(df.iloc[:, col])
        id_ = df.columns.values[col]
        coef_ = PolyRegression(degree=degree, x=x_, y=y_)
        row = pd.DataFrame()
        row["id"] = [id_]
        row["coef"] = [coef_]
        coef_df = coef_df.append(row)
    return coef_df


def loadDataOutput():
    all_files = glob.glob(os.path.join(path_dir_tsOutput, "*.csv"))
    coef_df = pd.DataFrame()
    for fname in all_files:
        df = loadTimeSeriesCsv(fname)
        coef_ = convertToPolynomialRegression(df, degree=Poly_degree)
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
