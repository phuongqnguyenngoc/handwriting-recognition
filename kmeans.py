'''kmeans.py
Performs K-Means clustering
Phuong Nguyen Ngoc
CS 252 Mathematical Data Analysis Visualization, Spring 2021
'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, rc
from IPython.display import HTML
import palettable
from palettable import cartocolors
import random
import palettable
import math


class KMeans():
    def __init__(self, data=None):
        '''KMeans constructor

        (Should not require any changes)

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features)
        '''

        # k: int. Number of clusters
        self.k = None
        # centroids: ndarray. shape=(k, self.num_features)
        #   k cluster centers
        self.centroids = None
        # data_centroid_labels: ndarray of ints. shape=(self.num_samps,)
        #   Holds index of the assigned cluster of each data sample
        self.data_centroid_labels = None

        # inertia: float.
        #   Mean squared distance between each data sample and its assigned (nearest) centroid
        self.inertia = None

        # data: ndarray. shape=(num_samps, num_features)
        self.data = data
        # num_samps: int. Number of samples in the dataset
        self.num_samps = None
        # num_features: int. Number of features (variables) in the dataset
        self.num_features = None
        if data is not None:
            self.num_samps, self.num_features = data.shape

    def set_data(self, data):
        '''Replaces data instance variable with `data`.

        Reminder: Make sure to update the number of data samples and features!

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features)
        '''
        self.data = data
        self.num_samps = data.shape[0]
        self.num_features = data.shape[1]

    def get_data(self):
        '''Get a COPY of the data

        Returns:
        -----------
        ndarray. shape=(num_samps, num_features). COPY of the data
        '''
        return np.copy(self.data)

    def get_centroids(self):
        '''Get the K-means centroids

        (Should not require any changes)

        Returns:
        -----------
        ndarray. shape=(k, self.num_features).
        '''
        return self.centroids

    def get_data_centroid_labels(self):
        '''Get the data-to-cluster assignments

        (Should not require any changes)

        Returns:
        -----------
        ndarray of ints. shape=(self.num_samps,)
        '''
        return self.data_centroid_labels

    def dist_pt_to_pt(self, pt_1, pt_2):
        '''Compute the Euclidean distance between data samples `pt_1` and `pt_2`

        Parameters:
        -----------
        pt_1: ndarray. shape=(num_features,)
        pt_2: ndarray. shape=(num_features,)

        Returns:
        -----------
        float. Euclidean distance between `pt_1` and `pt_2`.

        NOTE: Implement without any for loops (you will thank yourself later since you will wait
        only a small fraction of the time for your code to stop running)
        '''
        distance = np.sqrt(np.sum((pt_1 - pt_2)**2))
        return distance

    def dist_pt_to_centroids(self, pt, centroids):
        '''Compute the Euclidean distance between data sample `pt` and and all the cluster centroids
        self.centroids

        Parameters:
        -----------
        pt: ndarray. shape=(num_features,)
        centroids: ndarray. shape=(C, num_features)
            C centroids, where C is an int.

        Returns:
        -----------
        ndarray. shape=(C,).
            distance between pt and each of the C centroids in `centroids`.

        NOTE: Implement without any for loops (you will thank yourself later since you will wait
        only a small fraction of the time for your code to stop running)
        '''
        distance = np.sqrt(np.sum((centroids-pt)**2, axis=1))
        return distance

    def initialize(self, k):
        '''Initializes K-means by setting the initial centroids (means) to K unique randomly
        selected data samples

        Parameters:
        -----------
        k: int. Number of clusters

        Returns:
        -----------
        ndarray. shape=(k, self.num_features). Initial centroids for the k clusters.

        NOTE: Can be implemented without any for loops
        '''
        centroids = self.data[np.random.randint(self.data.shape[0], size=k), :]
        return centroids

    def initialize_plusplus(self, k):
        '''Initializes K-means by setting the initial centroids (means) according to the K-means++
        algorithm

        (LA section only)

        Parameters:
        -----------
        k: int. Number of clusters

        Returns:
        -----------
        ndarray. shape=(k, self.num_features). Initial centroids for the k clusters.

        TODO:
        - Set initial centroid (i = 0) to a random data sample.
        - To pick the i-th centroid (i > 0)
            - Compute the distance between all data samples and i-1 centroids already initialized.
            - Create the distance-based probability distribution (see notebook for equation).
            - Select the i-th centroid by randomly choosing a data sample according to the probability
            distribution.
        '''
        data = np.copy(self.data)
        centroids = np.zeros((k, data.shape[1]))
        index1 = np.random.randint(data.shape[0])
        c1 = data[index1]
        centroids[0] = c1
        data = np.delete(data, index1, axis=0)

        for i in range(1, k):
            d = np.zeros((data.shape[0], i))
            for j in range(i):
                di = self.dist_pt_to_centroids(centroids[j], data)
                d[:, j] = di
            D = np.min(d, axis=1)
            prob_square = (D)**2 / np.sum(D**2, axis=0)
            p = np.random.choice(np.arange(data.shape[0]), p=prob_square)
            centroids[i] = data[p]
            data = np.delete(data, p, axis=0)

        return centroids

    def cluster(self, k=2, tol=1e-2, max_iter=1000, verbose=False, init_method='random'):
        '''Performs K-means clustering on the data

        Parameters:
        -----------
        k: int. Number of clusters
        tol: float. Terminate K-means if the difference between all the centroid values from the
        previous and current time step < `tol`.
        max_iter: int. Make sure that K-means does not run more than `max_iter` iterations.
        verbose: boolean. Print out debug information if set to True.

        Returns:
        -----------
        self.inertia. float. Mean squared distance between each data sample and its cluster mean
        int. Number of iterations that K-means was run for

        TODO:
        - Initialize K-means variables
        - Do K-means as long as the max number of iterations is not met AND the difference
        between every previous and current centroid value is > `tol`
        - Set instance variables based on computed values.
        (All instance variables defined in constructor should be populated with meaningful values)
        - Print out total number of iterations K-means ran for
        '''
        self.k = k
        if init_method == 'random':
            self.centroids = self.initialize(k)
        elif init_method == 'kmeans++':
            self.centroids = self.initialize_plusplus(k)

        diff = 100000
        iterations = 0

        while iterations < max_iter and np.all(diff > tol):
            self.data_centroid_labels = self.update_labels(self.centroids)
            self.centroids, diff = self.update_centroids(
                k, self.data_centroid_labels, self.centroids)
            diff = np.absolute(diff)
            self.inertia = self.compute_inertia()
            iterations += 1
        if verbose == True:
            print(f'number of iterations:{iterations-1}')
            print(f'inertia: {self.inertia}')
            print(f'centroids: {self.centroids}')
        return self.inertia, iterations

    def cluster_batch(self, k=2, n_iter=1, verbose=False, init_method="random"):
        '''Run K-means multiple times, each time with different initial conditions.
        Keeps track of K-means instance that generates lowest inertia. Sets the following instance
        variables based on the best K-mean run:
        - self.centroids
        - self.data_centroid_labels
        - self.inertia

        Parameters:
        -----------
        k: int. Number of clusters
        n_iter: int. Number of times to run K-means with the designated `k` value.
        verbose: boolean. Print out debug information if set to True.
        '''
        inertia = math.inf
        centroids = None
        data_centroid_labels = None
        sum_iter = 0
        for i in range(n_iter):
            self.inertia, iterations = self.cluster(
                k, tol=1e-6, max_iter=1000, verbose=verbose, init_method=init_method)
            sum_iter += iterations
            if self.inertia < inertia:
                inertia = self.inertia
                centroids = self.centroids
                data_centroid_labels = self.data_centroid_labels
        self.intertia = inertia
        self.centroids = centroids
        self.data_centroid_labels = data_centroid_labels
        mean_iter = sum_iter/n_iter
        return k, n_iter, mean_iter

    def update_labels(self, centroids):
        '''Assigns each data sample to the nearest centroid

        Parameters:
        -----------
        centroids: ndarray. shape=(k, self.num_features). Current centroids for the k clusters.

        Returns:
        -----------
        ndarray of ints. shape=(self.num_samps,). Holds index of the assigned cluster of each data
            sample. These should be ints (pay attention to/cast your dtypes accordingly).

        Example: If we have 3 clusters and we compute distances to data sample i: [0.1, 0.5, 0.05]
        labels[i] is 2. The entire labels array may look something like this: [0, 2, 1, 1, 0, ...]
        '''
        labels = np.zeros(self.data.shape[0])
        for i in range(self.data.shape[0]):
            labels[i] = np.argmin(
                self.dist_pt_to_centroids(self.data[i], centroids))
        return labels

    def dist_data_to_centroids(self, centroids, data=None):
        '''(Helper method) Compute the distance between each data point to each centroid

        Parameters:
        -----------
        centroids: ndarray. shape=(k, self.num_features). Current centroids for the k clusters.

        Returns:
        -----------
        ndarray of float. shape=(self.num_samps, centroids.shape[0]). Each row holds the distance
            between the corresponding data point and the centroids.
            Eg: row 1 = [1, 2, 4] means that there are 3 centroids, the distance between
            data point indexed at 1 and centroid 0 is 1, the distance to centroid 1 is 2 and the distance to centroid 2 is 4
        '''
        if data is None:
            data = self.data
        transformed_data = np.tile(
            data, (1, centroids.shape[0]))
        transformed_centroids = np.reshape(centroids,
                                           (1, centroids.shape[0]*centroids.shape[1]))
        distance = transformed_data - transformed_centroids
        distance = np.reshape(
            distance, (distance.shape[0]*centroids.shape[0], centroids.shape[1]))
        distance = np.sqrt(np.sum(distance**2, axis=1))
        distance = np.reshape(
            distance, (data.shape[0], centroids.shape[0]))
        return distance

    def update_centroids(self, k, data_centroid_labels, prev_centroids):
        '''Computes each of the K centroids (means) based on the data assigned to each cluster

        Parameters:
        -----------
        k: int. Number of clusters
        data_centroid_labels. ndarray of ints. shape=(self.num_samps,)
            Holds index of the assigned cluster of each data sample
        prev_centroids. ndarray. shape=(k, self.num_features)
            Holds centroids for each cluster computed on the PREVIOUS time step

        Returns:
        -----------
        new_centroids. ndarray. shape=(k, self.num_features).
            Centroids for each cluster computed on the CURRENT time step
        centroid_diff. ndarray. shape=(k, self.num_features).
            Difference between current and previous centroid values
        '''
        new_centroid = np.zeros(prev_centroids.shape)
        for c in range(k):
            sample_indicies = np.where(data_centroid_labels == c)
            sample = self.data[np.ix_(sample_indicies[0])]
            new_centroid[c] = np.mean(sample, axis=0)
        centroid_diff = new_centroid - prev_centroids
        return new_centroid, centroid_diff

    def compute_inertia(self):
        '''Mean squared distance between every data sample and its assigned (nearest) centroid

        Parameters:
        -----------
        None

        Returns:
        -----------
        float. The average squared distance between every data sample and its assigned cluster centroid.
        '''
        sum_distance = 0
        for i in range(self.data.shape[0]):
            distance = self.dist_pt_to_pt(
                self.data[i], self.centroids[int(self.data_centroid_labels[i])])
            sum_distance += distance**2
        J = sum_distance/self.data.shape[0]
        return J

    def plot_clusters(self):
        '''Creates a scatter plot of the data color-coded by cluster assignment.

        TODO:
        - Plot samples belonging to a cluster with the same color.
        - Plot the centroids in black with a different plot marker.
        - The default scatter plot color palette produces colors that may be difficult to discern
        (especially for those who are colorblind). Make sure you change your colors to be clearly
        differentiable.
            You should use a palette Colorbrewer2 palette. Pick one with a generous
            number of colors so that you don't run out if k is large (e.g. 10).
        '''
        brewer_colors = palettable.scientific.diverging.Roma_10.mpl_colormap
        plt.scatter(self.data[:, 0], self.data[:, 1],
                    c=self.data_centroid_labels, cmap=brewer_colors)
        plt.scatter(self.centroids[:, 0],
                    self.centroids[:, 1], marker="D", color="#000000")
        plt.show()

    def elbow_plot(self, max_k):
        '''Makes an elbow plot: cluster number (k) on x axis, inertia on y axis.

        Parameters:
        -----------
        max_k: int. Run k-means with k=1,2,...,max_k.

        TODO:
        - Run k-means with k=1,2,...,max_k, record the inertia.
        - Make the plot with appropriate x label, and y label, x tick marks.
        '''
        cluster = [i+1 for i in range(max_k)]
        inertia = []
        for i in cluster:
            self.cluster(i)
            inertia.append(self.inertia)
        plt.plot(cluster, inertia)
        plt.xlabel("# clusters")
        plt.ylabel("Inertia")
        plt.title("Elbow plot")
        plt.xticks([i+1 for i in range(max_k)])

    def replace_color_with_centroid(self):
        '''Replace each RGB pixel in self.data (flattened image) with the closest centroid value.
        Used with image compression after K-means is run on the image vector.

        Parameters:
        -----------
        None

        Returns:
        -----------
        None
        '''
        replaced_colors = np.zeros(self.data.shape)
        for i in range(self.k):
            cluster = np.where(self.data_centroid_labels == i)
            replaced_colors[np.ix_(cluster[0])] = self.centroids[i]
        self.data = np.clip(replaced_colors.astype(int), 0, 256)

    def leader(self, threshold):
        self.centroids = np.zeros(self.data.shape)
        self.data_centroid_labels = np.zeros(self.data.shape[0])
        data = np.copy(self.data)
        np.random.shuffle(data)
        leaders = []
        l1_index = 0
        l1 = data[l1_index]
        self.data_centroid_labels[0] = 0
        leaders.append(l1)
        for i in range(1, data.shape[0]):
            distance_to_leaders = self.dist_pt_to_centroids(
                data[i], np.array(leaders))
            nearest_leader = np.argmin(distance_to_leaders)
            if distance_to_leaders[nearest_leader] < threshold:
                self.data_centroid_labels[i] = nearest_leader
            else:
                self.data_centroid_labels[i] = len(leaders)
                leaders.append(data[i])

        for i in range(len(leaders)):
            cluster_mem_indices = np.where(self.data_centroid_labels == i)[0]
            cluster = data[np.ix_(cluster_mem_indices)]
            centroid = np.mean(cluster, axis=0)
            self.centroids[i] = centroid
        self.centroids = self.centroids[:len(leaders)]
        print(self.centroids.shape)
        self.k = len(leaders)
        self.data = data


class Animation():
    def __init__(self, kmeans, k=2, max_iter=1000, tol=1e-2, xlabel="", ylabel="", title="", figsize=None):
        '''
        Parameter:
            kmeans: a KMeans object
            k: int. number of clusters to run kmeans clustering
            max_iter: int. kmean clustering will stop after the number of iterations reaches max_iter
            tol: int. threshold for difference between previous and current position of centroids.
                if difference < tol, kmean clustering will stop
        '''
        self.kmeans = kmeans
        self.k = k
        self.kmeans.centroids = self.kmeans.initialize(self.k)
        self.max_iter = max_iter
        self.tol = tol
        self.diff = 1000000
        self.iteration = 0
        self.brewer_colors = palettable.scientific.diverging.Roma_10.mpl_colormap

        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.min_vals = np.min(self.kmeans.data, axis=0)
        self.max_vals = np.max(self.kmeans.data, axis=0)

        self.xlabel = xlabel
        self.ylabel = ylabel
        self.title = title
        self.i = -1

    def gen(self):
        while self.iteration <= self.max_iter and np.all(self.diff > self.tol):
            self.i += 1
            yield self.i

    def animate(self, i):
        self.kmeans.data_centroid_labels = self.kmeans.update_labels(
            self.kmeans.centroids)
        self.kmeans.centroids, diff = self.kmeans.update_centroids(
            self.k, self.kmeans.data_centroid_labels, self.kmeans.centroids)
        self.diff = np.absolute(diff)
        self.kmeans.inertia = self.kmeans.compute_inertia()
        self.ax.clear()
        self.ax.scatter(self.kmeans.data[:, 0], self.kmeans.data[:, 1],
                        c=self.kmeans.data_centroid_labels, cmap=self.brewer_colors)
        self.ax.scatter(
            self.kmeans.centroids[:, 0], self.kmeans.centroids[:, 1], marker="D", color="#000000")
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        self.ax.text(self.max_vals[0], self.min_vals[1], f'iteration {self.iteration}', fontsize=16,
                     verticalalignment='bottom', horizontalalignment='right', bbox=props)
        self.ax.set_xlabel(self.xlabel)
        self.ax.set_ylabel(self.ylabel)
        self.ax.set_title(self.title)
        self.iteration += 1
        return

    def run_animation(self):
        anim = animation.FuncAnimation(
            self.fig, self.animate, frames=self.gen, interval=1000, blit=False, repeat=False)
        return HTML(anim.to_html5_video())
