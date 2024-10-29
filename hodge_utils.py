import numpy as np

class hodge_rank:
    def __init__(self, data, n):
        '''
        N: total pair wise comparison samples,
        n: number of instances to be ranked.
        '''
        self.data = data
        self.N = len(data)
        self.n = n
        self.edge_set = []
        for k in range(self.N):
            if (self.data[k][0], self.data[k][1]) not in self.edge_set:
                self.edge_set.append((self.data[k][0], self.data[k][1]))
        self.E = len(self.edge_set)
       
        self.edge_set_dict = {}
        for k in range(len(self.edge_set)):
            self.edge_set_dict[self.edge_set[k]] = k
        self.row = [i[0] for i in self.edge_set]
        self.col = [i[1] for i in self.edge_set]

        self.get_graph()

    def get_delta_matrix(self):
        delta_matrix = np.zeros((self.E,self.n))
        for k in range(len(self.row)):
            delta_matrix[k, self.row[k]] = 1
            delta_matrix[k, self.col[k]] = -1
        return delta_matrix

    def get_graph(self):
        self.value_matrix = np.zeros((self.E,))
        self.count_matrix = np.zeros((self.E,))
        self.norm_matrix = np.zeros((self.E,))
        
        for k in range(self.N):
            self.value_matrix[self.edge_set_dict[(self.data[k][0], self.data[k][1])]] += 1
            self.norm_matrix[self.edge_set_dict[(self.data[k][0], self.data[k][1])]] += 1
            if (self.data[k][1], self.data[k][0]) in self.edge_set_dict.keys():
                self.norm_matrix[self.edge_set_dict[(self.data[k][1], self.data[k][0])]] += 1
                self.value_matrix[self.edge_set_dict[(self.data[k][1], self.data[k][0])]] -= 1
            self.count_matrix[self.edge_set_dict[(self.data[k][0], self.data[k][1])]] += 1
        self.value_matrix = self.value_matrix/ (self.norm_matrix)

    def get_global_rank(self):
        ## Y E  Y= delta \theta +\epsilon  delta : Exn
        ## 1/2  ||Y- D \theta||_W^2
        ## theta = (WDTD)-1 WDTY
        D = self.get_delta_matrix()
        W = np.diag(self.count_matrix)
        Y = self.value_matrix
        Mat1 = np.matmul(np.matmul(D.transpose(), W), D)
        Mat2 = np.matmul(np.matmul(D.transpose(),W), Y)
        theta = np.linalg.lstsq(Mat1, Mat2, rcond=1e-5)[0]
        return theta
