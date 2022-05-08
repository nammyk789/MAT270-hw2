import numpy as np
from scipy.io import loadmat
import kmeans

class spectralClustering:
    def __init__(self, sigma=5):
        self.sigma = sigma
    
    def rand_score(self):
        return self.model.rand_score()
    
    def accuracy(self):
        return self.model.accuracy()
    
    def cluster(self, data, labels):
        self.data = data

        # get weight and degree of data
        self.__create_weight_matrix()
        self.degree = np.diag(self.weight.sum(axis=1))
        self.graph_Laplacian = self.degree - self.weight

        # eigenvalues and eigenvectors
        vals, vecs = np.linalg.eig(self.graph_Laplacian)

        # sort these based on the eigenvalues
        vecs = vecs[:,np.argsort(vals)]
        vals = vals[np.argsort(vals)]

        self.model = kmeans.kMeans()
        self.model.train(vecs[:,1:10], labels)



    def __create_weight_matrix(self):
        """
        creates a weight matrix using gaussian kernel
        """
        self.weight = np.ndarray(shape=(self.data.shape[0], self.data.shape[0]), dtype="float")
        for i in range(self.data.shape[0]):
            for j in range(self.data.shape[0]):
                distance = np.linalg.norm(self.data[i] - self.data[j], ord=2)
                self.weight[i, j] = np.exp(-distance**2/self.sigma)
        np.fill_diagonal(self.weight, 0)

if __name__ == "__main__":
    M = loadmat('MNIST_digit_data.mat')
    images_train,images_test,labels_train,labels_test= M['images_train'],M['images_test'],M['labels_train'],M['labels_test']
    #randomly permute data points
    inds = np.random.permutation(images_train.shape[0])
    images_train = images_train[inds]
    labels_train = labels_train[inds]

    model = spectralClustering()
    model.cluster(images_train[:2000], labels_train[:2000])
    print("accuracy:", model.accuracy())
    print("rand score:", model.rand_score())