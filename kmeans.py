import numpy as np
from scipy.io import loadmat
from scipy import stats as st
import sklearn.metrics

class kMeans:
    def __init__(self, k = 10, threshold=0, iterations=100):
        self.threshold = threshold
        self.num_centers = k
        self.iterations = iterations
    
    def vote(self, cluster):
        return st.mode(self.labels[cluster])
    
    def get_cluster_labels(self):
        clusters = self.get_clusters(self.centers)
        labels = [0] * self.num_centers
        for i in range(self.num_centers):
            labels[i] = st.mode(self.labels[clusters[i]])[0][0][0]
        print(labels) 
    
    def cluster_counts(self):
        cluster_counts = []
        for cluster in self.get_clusters(self.centers):
            cluster_counts.append(len(cluster))
        return cluster_counts

    def rand_score(self):
        clusters = self.get_clusters(self.centers)
        label_pred = [0] * len(self.labels)
        for cluster in clusters:
            cluster_vote = np.nan
            if len(cluster) > 0:
                cluster_vote = st.mode(self.labels[cluster])[0][0][0]
            for idx in cluster:
                label_pred[idx] = cluster_vote
        return sklearn.metrics.rand_score(self.labels.flatten(), np.array(label_pred))

    def accuracy(self):
        clusters = self.get_clusters(self.centers)
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
        while True:
            self.centers, points_changed = self.__update_centroids()
            if points_changed <= self.threshold: 
                break
            print("Updating. {} points changed.".format(points_changed))

    
    def __initialize_centroids(self):
        """
        initialize 10 random data points to be 
        the first 10 centers
        """
        init_centers = np.random.randint(low=0, high=self.data.shape[0], size=self.num_centers)
        self.centers = self.data[init_centers]
      
    
    def get_clusters(self, centers):
        """
        sorts indices of training data into clusters, returns a 2D array of clusters
        """
        clusters = []
        for _ in range(self.num_centers):
            clusters.append([])  # initialize list of clusters
        for i in range(len(self.data)):
            closest_center = 0
            min_distance = np.linalg.norm(self.data[i] - centers[0], ord=2)
            for j in range(1, self.num_centers):
                distance = np.linalg.norm(self.data[i] - centers[j], ord=2)
                if distance < min_distance:
                    min_distance = distance
                    closest_center = j
            clusters[closest_center].append(i) 
        clusters = np.array([np.array(cluster) for cluster in clusters], dtype='object')  # convert to numpy array
        return clusters 
    
    def __update_centroids(self):
        """ 
        returns new centers for the clusters
        and the number of points that changed clusters
        in the update
        """
        clusters = self.get_clusters(self.centers)
        new_centers = [0] * self.num_centers
        for i in range(self.num_centers):
            if len(clusters[i] > 0):
                cluster = self.data[clusters[i]]
            else:
                cluster = []
            new_centers[i] = np.average(cluster, axis=0)
        # find how many points changed
        new_clusters = self.get_clusters(new_centers)
        points_changed = 0
        for cluster in range(len(new_clusters)):
            for point in new_clusters[cluster]:
                if point not in clusters[cluster]:
                    points_changed += 1
        return np.array(new_centers), points_changed
    
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

    images_train = np.array([x.flatten() for x in images_train])
    model = kMeans()
    model.train(images_train, labels_train)
    print(model.rand_score())
    print(model.cluster_counts())