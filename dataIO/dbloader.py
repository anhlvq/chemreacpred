import sqlite3
import pandas as pd


def create_connection(db_file):
    """ create a database connection to a SQLite database """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        print(sqlite3.version)
    except sqlite3.Error as e:
        print(e)
    return conn


def combineFeature(db_conn, pattern_1="1_Substrate", pattern_2="2_Base",
                   pattern_3="3_Hydroxylamine", pattern_4="4_Anti_Ligand", hilow_1=1, hilow_2=1, hilow_3=1, hilow_4=1):
    conn = db_conn
    # 1_Substrate
    df1 = pd.read_sql_query("SELECT * FROM '" + pattern_1 + "'", conn)
    df1 = df1.drop("Name", axis=1)
    # 2_Base or Conjugate
    df2 = pd.read_sql_query('SELECT * FROM "' + pattern_2 + '"', conn)
    df2 = df2.drop("Name", axis=1)
    # 3_Hydroxyamine or Oxoammonium
    df3 = pd.read_sql_query('SELECT * FROM "' + pattern_3 + '"', conn)
    df3 = df3.drop("Name", axis=1)
    # 4_Anti or Syn Ligand
    df4 = pd.read_sql_query('SELECT * FROM "' + pattern_4 + '"', conn)
    df4 = df4.drop("Name", axis=1)
    df1["hilow_1"] = hilow_1
    df2["hilow_2"] = hilow_2
    df3["hilow_3"] = hilow_3
    df4["hilow_4"] = hilow_4

    df1["tmp"] = 1
    df2["tmp"] = 1
    df3["tmp"] = 1
    df4["tmp"] = 1

    df1.rename(columns={"Compound": "id"}, inplace=True)
    df = pd.merge(df1, df2, on=['tmp'])
    df["id"] = df["id"] + "_" + df["Compound"]
    df = df.drop(columns={"Compound"})
    df = pd.merge(df, df3, on=['tmp'])
    df["id"] = df["id"] + "_" + df["Compound"]
    df = df.drop(columns={"Compound"})
    df = pd.merge(df, df4, on=['tmp'])
    df["id"] = df["id"] + "_" + df["Compound"]
    df = df.drop(columns={"Compound"})
    df = df.drop('tmp', axis=1)
    dsname = pattern_1+'_'+pattern_2+'_'+pattern_3+'_'+pattern_4
    return dsname, df


def generateAllDataSets(db_file="../data/3_processed/data.sqlite"):
    conn = create_connection(db_file=db_file)
    db_lst = []
    lst1 = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type ='table' AND name LIKE '1_%'", conn).values[:,0]
    lst2 = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type ='table' AND name LIKE '2_%'", conn).values[:,0]
    lst3 = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type ='table' AND name LIKE '3_%'", conn).values[:,0]
    lst4 = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type ='table' AND name LIKE '4_%'", conn).values[:,0]
    for t1 in lst1:
        for t2 in lst2:
            for t3 in lst3:
                for t4 in lst4:
                    dbname, df = combineFeature(db_conn=conn, pattern_1=t1, pattern_2=t2, pattern_3=t3, pattern_4=t4)
                    db_lst.append({dbname: df})
    return db_lst


lst = generateAllDataSets()