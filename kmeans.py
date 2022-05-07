import numpy as np
from scipy.io import loadmat
from scipy import stats as st

class Hyperparameters:
    def __init__(self, threshold=0.001):
        self.threshold = threshold

class kMeans:
    def __init__(self, hyperparameters=None):
        if not hyperparameters:
            hyperparameters = Hyperparameters()
        self.threshold = hyperparameters.threshold
        self.num_centers = 10
    
    def vote(self, cluster):
        return st.mode(self.labels[cluster])

    def accuracy(self):
        clusters = self.__get_clusters()
        accuracy = 0
        for cluster in clusters:
            cluster_labels = self.labels[cluster]
            # find what percent of the labels in the cluster are the vote
            label_homogeneity = (cluster_labels == st.mode(cluster_labels)).sum() / len(cluster_labels)
            accuracy += label_homogeneity / self.num_centers
        return accuracy

    def train(self, data, labels):
        self.data = data
        self.labels = labels
        self.__initialize_centroids()
        new_centroids = self.__update_centroids()
        # while self.__is_outside_of_threshold(new_centroids):  # update until change is small
        #     self.centers = new_centroids
        #     new_centroids = self.__update_centroids()
        for _ in range(50):  # only 61%
            self.centers = self.__update_centroids()
    
    def __initialize_centroids(self):
        """
        initialize 10 random data points to be 
        the first 10 centers
        """
        # creates random clusters
        rand_clusters = np.random.randint(low=0, high=256, size=(self.num_centers, self.data.shape[1]))
        self.centers = [0] * self.num_centers
        for i in range(self.num_centers):
            self.centers[i] = np.average(rand_clusters[i], axis=0)
    
    def __get_clusters(self):
        """
        sorts indices of training data into clusters, returns a 2D array of clusters
        """
        clusters = []
        for _ in range(self.num_centers):
            clusters.append([])  # initialize list of clusters
        for i in range(len(self.data)):
            closest_center = 0
            min_distance = np.linalg.norm(self.data[i] - self.centers[0], ord=2)
            for j in range(1, self.num_centers):
                distance = np.linalg.norm(self.data[i] - self.centers[j], ord=2)
                if distance < min_distance:
                    min_distance = distance
                    closest_center = j
            clusters[closest_center].append(i) 
        clusters = np.array([np.array(cluster) for cluster in clusters], dtype='object')  # convert to numpy array
        return clusters 
    
    def __update_centroids(self):
        clusters = self.__get_clusters()
        new_centers = [0] * self.num_centers
        for i in range(self.num_centers):
            new_centers[i] = np.average(self.data[clusters[i]], axis=0)
        return np.array(new_centers)
    
    def __is_outside_of_threshold(self, new_centers):
        # check that update change is greater than threshold
        delta = np.subtract(new_centers, self.centers)
        if np.average(delta) > self.threshold:
            return True
        return False


if __name__ == "__main__":
    M = loadmat('MNIST_digit_data.mat')
    images_train,images_test,labels_train,labels_test= M['images_train'],M['images_test'],M['labels_train'],M['labels_test']
    #randomly permute data points
    inds = np.random.permutation(images_train.shape[0])
    images_train = images_train[inds]
    labels_train = labels_train[inds]


    inds = np.random.permutation(images_test.shape[0])
    images_test = images_test[inds]
    labels_test = labels_test[inds]
    model = kMeans()
    model.train(images_train, labels_train)
    print(model.accuracy())