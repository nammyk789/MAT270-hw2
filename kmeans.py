import numpy as np
from scipy.io import loadmat

class kMeans:
    def __init__(self, num_centers=10):
        self.num_centers = num_centers

    def train(self, train_data, train_labels):
        self.train_data = train_data
        self.train_labels = train_labels
        self.__initialize_centers()
  
    def __initialize_centers(self):
        """
        initialize 10 random data points to be 
        the first 10 centers
        """
        init_centers = np.random.randint(low=0, high=self.train_data.shape[0], size=self.num_centers)
        self.centers = self.train_data[init_centers]
    
    def __get_clusters(self):
        """
        sorts indices of training data into clusters, returns a 2D array of clusters
        """
        clusters = [[]] * 10
        for i in range(len(self.train_data)):
            closest_center = 0
            min_distance = np.linalg.norm(self.train_data[i] - self.centers[0], ord=2)
            for j in range(self.num_centers):
                distance = np.linalg.norm(self.train_data[i]point - self.centers[j], ord=2)
                if distance < min_distance:
                    min_distance = distance
                    closest_center = j
            clusters[closest_center].append(i)
        return np.array(clusters)  # convert to numpy array
    
    def __update_clusters(self):
        clusters = self.__get_clusters()
        for i in range(self.num_centers):
            self.centers[i] = np.mean(clusters[i], axis=0) # CHECK TO MAKE SURE THIS IS CORRECT

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
