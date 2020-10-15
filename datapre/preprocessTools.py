import glob
import os
import pandas as pd

from utils.config import path_dir_descriptor, path_dir_data


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

def makeTrainingDataOutput(fname="output.csv"):
    df = pd.read_csv("")