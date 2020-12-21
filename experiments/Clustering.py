from sklearn_extra.cluster import KMedoids
from sklearn.cluster import KMeans
from sklearn import metrics
import pandas as pd
from dataIO.Dataset import FeatureDataset, LoadAllFeatureDataSets, LoadAllFeatureDataSetsDB
from dataIO.loader import readNumpyArrayFile, writeNumpyArrayFile
from utils.fileSystemUtils import checkExists


def searchBestNumberOfClusters(ds, method='KMedoids', min_silhouette_score=0.05):
    X = ds.features
    df = pd.DataFrame(columns=['nclusters', 'silhouettescore'])
    for i in range(2, 50):
        labels = None
        if method == 'KMedoids':
            kmedoids = KMedoids(n_clusters=i, random_state=0).fit(X)
            labels = kmedoids.labels_
        else:  # Kmean
            km = KMeans(i)
            labels = km.fit_predict(X)
        # score1 = metrics.calinski_harabasz_score(X, labels)
        # score2 = metrics.davies_bouldin_score(X, labels)
        score3 = metrics.silhouette_score(X, labels, metric='euclidean')
        if score3 >= min_silhouette_score:
            new_row = {'nclusters': i, 'silhouettescore': score3}
            df = df.append(new_row, ignore_index=True)
    df.sort_values(by=['silhouettescore'], ascending=False, inplace=True)
    df.to_csv('Results/'+ds.dataSetName+'silhouette_score_' + method + '.csv', index=False)
    return df


def doCluster(ds, k, method='KMeans'):
    file = ds.filePath + "." + str(k) + "." + method
    if checkExists(file):
        labels = readNumpyArrayFile(file)
    else:
        X = ds.features
        if method == 'KMeans':
            km = KMeans(k)
            labels = km.fit_predict(X)
        elif method == 'KMedoids-Euclidean':
            kmedoids = KMedoids(n_clusters=k, metric='euclidean').fit(X)
            labels = kmedoids.labels_
        elif method == 'KMedoids-Cosine':
            kmedoids = KMedoids(n_clusters=k, metric='cosine').fit(X)
            labels = kmedoids.labels_
        elif method == 'KMedoids-Manhattan':
            kmedoids = KMedoids(n_clusters=k, metric='manhattan').fit(X)
            labels = kmedoids.labels_
        else:
            labels = {i for i in range(0, X.shape[1])}
        writeNumpyArrayFile(file, labels)
    return labels


ds_list = LoadAllFeatureDataSetsDB(isNormalized=True)

for ds in ds_list:
    print(ds)
    doCluster(ds, 2, method='KMeans')
    #print(searchBestNumberOfClusters(ds, method='Kmean'))
    #print(searchBestNumberOfClusters(ds, method='KMedoids'))

#idList = ds.idList
#print(idList.shape)
#X = ds.features
#print(X.shape)
