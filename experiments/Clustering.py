from sklearn_extra.cluster import KMedoids
from sklearn.cluster import KMeans
from sklearn import metrics
import pandas as pd
from dataIO.Dataset import FeatureDataset, LoadAllFeatureDataSets


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


ds_list = LoadAllFeatureDataSets(isNormalized=True)

for ds in ds_list:
    print(searchBestNumberOfClusters(ds, method='Kmean'))
    print(searchBestNumberOfClusters(ds, method='KMedoids'))

#idList = ds.idList
#print(idList.shape)
#X = ds.features
#print(X.shape)
