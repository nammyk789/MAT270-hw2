from scipy.io import loadmat
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics


class DiffusionMap():
    def __init__(self, dim=10, sigma=0.5):
        self.dim = dim
        self.sigma = sigma

    def map(self, data):
        self.data = data
        self.get_matrix()
        
        # get eigenvalues and vectors of S
        w, V = np.linalg.eigh(self.matrix)

        # get top k eigenvectors based on values
        idx = w.argsort()[::-1]
        V = V[:,idx]

        # PHI = D^-1/2 V
        self.map = np.matmul(self.__degree_matrix_to_the_power_of(-1/2), V)
        self.map = self.map[:, :self.dim]

        return self.map

    def get_matrix(self):
        """ calculates the diffusion matrix """
        # create W
        self.__create_gaussian_weight_matrix()
        
        # M = D^-1 W
        M = np.matmul(self.__degree_matrix_to_the_power_of(-1), self.weight)

        # S = D^1/2 M D^-1/2
        m_d_half = np.matmul(M, self.__degree_matrix_to_the_power_of(-1/2))
        self.matrix = np.matmul(self.__degree_matrix_to_the_power_of(1/2), m_d_half)

    def __degree_matrix_to_the_power_of(self, exponent):
        # get list of degrees
        self.degree = np.sum(self.weight, axis=1)
        return np.diag(self.degree ** exponent)


    def __create_gaussian_weight_matrix(self):
        """
        creates a weight matrix using gaussian kernel
        """
        distances = squareform(pdist(self.data, 'sqeuclidean'))
        self.weight = np.exp(-distances / self.sigma)


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

    map = DiffusionMap()
    reduced_data = map.map(encoded_data)

    clusterer = KMeans(n_clusters=5)
    sk_labels = clusterer.fit_predict(reduced_data)

    print(metrics.silhouette_score(reduced_data, sk_labels, metric='euclidean'))