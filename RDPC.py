import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial.distance as dis
from matplotlib.widgets import RectangleSelector


class RDPC:
    """
    Parameter
    ----------
    metric: string, default='mutual'
        'mutual' means to compute the distance between objects by the mutual
          reachability distance.
        An alternative option is 'euclidean', which means to compute the distance
          between objects by the Euclidean distance.
    """
    __threshold = 1
    __display_index = []
    __row_num = 0
    __d_c_ratio = 0.02
    __density = None
    __density_index = None
    __delta = None
    __centers = None

    def __init__(self, distance_measure='mutual'):
        self.distance_measure = distance_measure

    def line_select_callback(self, eclick, erelease):
        x1, y1 = eclick.xdata, eclick.ydata
        centers = []
        for index in reversed(self.__display_index):
            if self.__density[index] >= x1:
                if self.__delta[index] >= y1:
                    centers.append(index)
            else:
                break
        self.__centers = centers
        print(centers)
        return

    def plot_decision(self, density, delta):    #generating decision graph
        fig1, ax1 = plt.subplots()
        density = density - np.amin(density)
        self.__density = density
        delta = delta - np.amin(delta)
        self.__delta = delta

        # Filter objects with very low density
        densityNZ = []
        deltaNZ = []
        max_delta = 0
        for i, index in enumerate(self.__density_index):
            if density[index] > self.__threshold:
                self.__display_index.append(index)
                densityNZ.append(density[index])
                if i == self.__row_num - 1:
                    deltaNZ.append(max_delta * 1.1)
                    break
                deltaNZ.append(delta[index])
                if delta[index] > max_delta:
                    max_delta = delta[index]
        ax1.scatter(densityNZ, deltaNZ)

        ax1.set_title(
            "Click and drag until the rectangle covers all maximum objects.\n")
        RS = RectangleSelector(ax1, self.line_select_callback,
                               drawtype='box', useblit=True,
                               button=[1, 3],  # disable middle button
                               minspanx=5, minspany=5,
                               spancoords='pixels',
                               interactive=True)
        plt.show()

    def fit_predict(self, X):
        row_num, col_num = X.shape
        self.__row_num = row_num
        dist_matrix, dist_sort = self.get_distance(X)
        density, density_index = self.get_density(dist_matrix, dist_sort, row_num)
        delta, parent = self.get_delta(density_index, dist_matrix, row_num)
        self.plot_decision(density, delta)
        labels = self.get_labels(parent, density_index, row_num)
        return labels

    def get_distance(self, X):
        if self.distance_measure == 'euclidean':
            distance = dis.pdist(X)
            distance_matrix = dis.squareform(distance)
            distance_sort = np.sort(distance_matrix, axis=1)
            return distance_matrix, distance_sort
        elif self.distance_measure == 'mutual':
            dim = X.shape[0]
            distance = dis.pdist(X)
            b_matrix = dis.squareform(distance)
            b_sort = np.sort(b_matrix, axis=1)
            distance_matrix = np.zeros([dim, dim])
            for i in range(dim):
                for j in range(i + 1, dim):
                    distance_matrix[i, j] = max(
                        [b_sort[i, round(dim * 0.015)], b_sort[j, round(dim * 0.015)], b_matrix[i, j]])
                    distance_matrix[j, i] = distance_matrix[i, j]
            distance_sort = np.sort(distance_matrix, axis=1)
            return distance_matrix, distance_sort
        else:
            print("metric error")
            return None, None


    def get_density(self, dist, dist_sort, row_num):     # computing density_rdpc
        density = np.zeros(row_num)
        area = np.mean(dist_sort[:, round(row_num * self.__d_c_ratio)])
        for i in range(row_num - 1):
            for j in range(i + 1, row_num):
                density[i] = density[i] + math.exp(- (dist[i][j] * dist[i][j]) / (area * area))
                density[j] = density[j] + math.exp(- (dist[i][j] * dist[i][j]) / (area * area))
        density_index = np.argsort(density)
        self.__density_index = density_index.copy()
        return density, density_index

    def get_delta(self, density_index, dist_matrix, row_num):  # computing delta_rdpc
        index_map = [0 for i in range(row_num)]
        for i in range(row_num):
            index_map[density_index[i]] = i
        candidates_index = [density_index[i + 1:] for i in range(row_num)]
        parent = [0 for i in range(row_num)]
        delta = np.zeros(row_num)
        max_delta = 0
        for i, indexi in enumerate(density_index):
            min_dist = float('inf')
            candidate = -1
            for indexj in candidates_index[i]:
                if dist_matrix[indexi][indexj] < min_dist:
                    min_dist = dist_matrix[indexi][indexj]
                    candidate = indexj
            if candidate != -1:
                if min_dist > max_delta:
                    max_delta = min_dist
                delta[indexi] = min_dist
                parent[indexi] = candidate
                j = index_map[candidate]
                for indexk in density_index[i + 1:j]:
                    if dist_matrix[indexi][indexk] < dist_matrix[indexk][candidate]:
                        dist_matrix[indexk][candidate] = dist_matrix[indexi][indexk]
                for indexk in candidates_index[j]:
                    if dist_matrix[indexi][indexk] < dist_matrix[candidate][indexk]:
                        dist_matrix[candidate][indexk] = dist_matrix[indexi][indexk]
            else:
                delta[indexi] = max_delta * 1.1
                break
        return delta, parent

    def get_labels(self, parent, density_index, row_num):
        label = np.zeros(row_num, dtype=np.int8)
        density_index = density_index[::-1]
        i = 1
        for c in self.__centers:
            label[c] = i
            i = i + 1
        for index in density_index:
            if label[index] == 0:
                label[index] = label[parent[index]]
        return label
