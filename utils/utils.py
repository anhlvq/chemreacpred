import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def get_nulls(training, testing):
    print("Training Data:")
    print(pd.isnull(training).sum())
    print("Testing Data:")
    print(pd.isnull(testing).sum())


def PCAComp(X, n_components=3):
    pca = PCA(n_components=n_components)
    pca.fit(X)
    return pca.transform(X)


def TSNEComp(X, n_components=3):
    tsne = TSNE(n_components=n_components, verbose=1, perplexity=40, n_iter=300)
    return tsne.fit_transform(X)


