import numpy as np

class DiffusionMap():
    def __init__(self, dim=10):
        self.dim = dim
    
    def reduce_data(self, data):
        w, v = np.linalg.eig(data)
        map = []
        for i in range(len(data)):
            new_line = [0]*self.dim
            for j in range(len(w)):
                new_line.append(w[j]**self.dim * v[j][i])
            map.append(np.array(new_line))
