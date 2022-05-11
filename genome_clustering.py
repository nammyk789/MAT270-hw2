from scipy.io import loadmat
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn import metrics

if __name__ == "__main__":
    mat_contents = loadmat("genomedata.mat")
    raw_data = mat_contents["X"]

    # get rid of all the list nesting
    unnested_data = np.array([x[0][0] for x in raw_data])

    # split on tab characters and remove whitespace
    no_whitespace = [x.replace(" ", "") for x in unnested_data]
    split_data = np.array([x.split("\t") for x in no_whitespace])
    
    # remove last whitespace entry
    clean_data = np.delete(split_data, -1, axis=1)

    # one-hot encoding
    enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
    encoded_data = enc.fit_transform(clean_data)
    print("data encoded")

    pca = PCA(n_components=20)
    pca_data = pca.fit_transform(encoded_data)

    # knn clustering
    clusterer = KMeans(n_clusters=10)
    sk_labels = clusterer.fit_predict(pca_data)

    #sillhouette score ranges from -1 to 1, where 1 is best and 0 indicates cluster overlap
    ss = metrics.silhouette_score(pca_data, sk_labels, metric='euclidean')
    print("Sillhouette score:", ss)
    # variance ratio criterion-- how tightly clustered (higher is better)
    chs = metrics.calinski_harabasz_score(pca_data, sk_labels)
    print("Calinski-Harabasz Index:", chs)
    # similarity between clusters (lower is better)
    dbs = metrics.davies_bouldin_score(pca_data, sk_labels)   
    print("Davies-Bouldin Index:", dbs)

