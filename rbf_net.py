'''rbf_net.py
Radial Basis Function Neural Network
Phuong Nguyen Ngoc
'''
import numpy as np
import kmeans
import scipy.linalg
import scipy


class RBF_Net:
    def __init__(self, num_hidden_units, num_classes):
        '''RBF network constructor

        Parameters:
        -----------
        num_hidden_units: int. Number of hidden units in network. NOTE: does NOT include bias unit
        num_classes: int. Number of output units in network. Equals number of possible classes in
            dataset

        TODO:
        - Define number of hidden units as an instance variable called `k` (as in k clusters)
            (You can think of each hidden unit as being positioned at a cluster center)
        - Define number of classes (number of output units in network) as an instance variable
        '''
        # prototypes: Hidden unit prototypes (i.e. center)
        #   shape=(num_hidden_units, num_features)
        self.prototypes = None
        # sigmas: Hidden unit sigmas: controls how active each hidden unit becomes to inputs that
        # are similar to the unit's prototype (i.e. center).
        #   shape=(num_hidden_units,)
        #   Larger sigma -> hidden unit becomes active to dissimilar inputs
        #   Smaller sigma -> hidden unit only becomes active to similar inputs
        self.sigmas = None
        # wts: Weights connecting hidden and output layer neurons.
        #   shape=(num_hidden_units+1, num_classes)
        #   The reason for the +1 is to account for the bias (a hidden unit whose activation is always
        #   set to 1).
        self.wts = None
        self.k = num_hidden_units
        self.num_classes = num_classes

    def get_prototypes(self):
        '''Returns the hidden layer prototypes (centers)

        (Should not require any changes)

        Returns:
        -----------
        ndarray. shape=(k, num_features).
        '''
        return self.prototypes

    def get_num_hidden_units(self):
        '''Returns the number of hidden layer prototypes (centers/"hidden units").

        Returns:
        -----------
        int. Number of hidden units.
        '''
        return self.k

    def get_num_output_units(self):
        '''Returns the number of output layer units.

        Returns:
        -----------
        int. Number of output units
        '''
        return self.num_classes

    def avg_cluster_dist(self, data, centroids, cluster_assignments, kmeans_obj):
        '''Compute the average distance between each cluster center and data points that are
        assigned to it.

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features). Data to learn / train on.
        centroids: ndarray. shape=(k, num_features). Centroids returned from K-means.
        cluster_assignments: ndarray. shape=(num_samps,). Data sample-to-cluster-number assignment from K-means.
        kmeans_obj: KMeans. Object created when performing K-means.

        Returns:
        -----------
        ndarray. shape=(k,). Average distance within each of the `k` clusters.

        Hint: A certain method in `kmeans_obj` could be very helpful here!
        '''

        avg_dist = np.zeros(self.k)
        for i in range(centroids.shape[0]):
            data_clustered_ind = np.where(cluster_assignments == i)[0]
            data_clustered = data[np.ix_(data_clustered_ind)]
            # print(data_clustered.shape)
            # print(centroids[i].shape)
            dist = kmeans_obj.dist_pt_to_centroids(
                centroids[i], data_clustered)
            avg_dist[i] = np.mean(dist)
        return avg_dist

    def initialize(self, data):
        '''Initialize hidden unit centers using K-means clustering and initialize sigmas using the
        average distance within each cluster

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features). Data to learn / train on.

        TODO:
        - Determine `self.prototypes` (see constructor for shape). Prototypes are the centroids
        returned by K-means. It is recommended to use the 'batch' version of K-means to reduce the
        chance of getting poor initial centroids.
            - To increase the chance that you pick good centroids, set the parameter controlling the
            number of iterations > 1 (e.g. 5)
        - Determine self.sigmas as the average distance between each cluster center and data points
        that are assigned to it. Hint: You implemented a method to do this!
        '''
        print("Start initializing")
        kmeans_obj = kmeans.KMeans(data)
        kmeans_obj.cluster_batch(self.k, n_iter=5, init_method='kmeans++')
        cluster_assignments = kmeans_obj.get_data_centroid_labels()
        self.prototypes = kmeans_obj.get_centroids()
        self.sigmas = self.avg_cluster_dist(
            data, self.prototypes, cluster_assignments, kmeans_obj)
        print("Finish initializing")

    def qr_decomposition(self, A):
        '''Performs a QR decomposition on the matrix A. Make column vectors orthogonal relative
        to each other. Uses the Gram–Schmidt algorithm

        (Week 2)

        Parameters:
        -----------
        A: ndarray. shape=(num_data_samps, num_ind_vars+1).
            Data matrix for independent variables.

        Returns:
        -----------
        Q: ndarray. shape=(num_data_samps, num_ind_vars+1)
            Orthonormal matrix (columns are orthogonal unit vectors — i.e. length = 1)
        R: ndarray. shape=(num_ind_vars+1, num_ind_vars+1)
            Upper triangular matrix

        TODO:
        - Q is found by the Gram–Schmidt orthogonalizing algorithm.
        Summary: Step thru columns of A left-to-right. You are making each newly visited column
        orthogonal to all the previous ones. You do this by projecting the current column onto each
        of the previous ones and subtracting each projection from the current column.
            - NOTE: Very important: Make sure that you make a COPY of your current column before
            subtracting (otherwise you might modify data in A!).
        Normalize each current column after orthogonalizing.
        - R is found by equation summarized in notebook
        '''
        num_rows, num_cols = np.shape(A)
        Q = np.ones((num_rows, num_cols))
        for i in range(num_cols):
            col_i_copied = np.copy(A[:, i])
            for j in range(i):
                col_i_copied -= np.squeeze(
                    np.squeeze(A[:, i].T@Q[:, j]))*Q[:, j]
            col_i_copied = col_i_copied/np.linalg.norm(col_i_copied)
            Q[:, i] = col_i_copied
        R = Q.T@A
        return (Q, R)

    def SVD(self, A):
        '''
        Return the reduced components of singular value decomposition of matrix A

        Parameters:
        A: ndarray. shape(n,m)

        Return:
        Ur: ndarray shape (n, r). Reduced left singular vector of A
        D: ndarray shape (r, r). Reduced singular value matrix of A
        Vr: ndarray shape (m, r). Reduced right singluar vector of A
        '''
        e_vals, right_singular_vecs = np.linalg.eig(A.T@A)
        indices = np.flip(np.argsort(e_vals))
        e_vals = e_vals[indices]
        right_singular_vecs = ((right_singular_vecs.T[indices]).T)
        right_singular_vecs = right_singular_vecs*(-1)
        r = len(np.where(e_vals > float(0))[0])
        Vr = right_singular_vecs[:, :r]
        D = np.zeros((r, r))
        Ur = np.zeros((A.shape[0], r))
        for i in range(r):
            D[i, i] = np.sqrt(e_vals[i])
            ui = A@(Vr[:, i])/D[i, i]
            ui = ui.reshape(-1)
            Ur[:, i] = ui
        return Ur, D, Vr

    def pinverse(self, A):
        '''
        Compute the pseudo inverse of A using singular value decomposition

        Parameter:
        A: ndarray

        Returns:
        A_pinverse: pseudo inverse of A
        '''
        Ur, D, Vr = self.SVD(A)
        D_inv = scipy.linalg.inv(D)
        A_pinverse = Vr@D_inv@(Ur.T)
        return A_pinverse

    def linear_regression(self, A, y, mode="qr"):
        '''Performs linear regression
        CS251: Adapt your SciPy lstsq code from the linear regression project.
        CS252: Adapt your QR-based linear regression solver

        Parameters:
        -----------
        A: ndarray. shape=(num_data_samps, num_features).
            Data matrix for independent variables.
        y: ndarray. shape=(num_data_samps, 1).
            Data column for dependent variable.
        mode: (for extension) qr or pinverse

        Returns
        -----------
        c: ndarray. shape=(num_features+1,)
            Linear regression slope coefficients for each independent var AND the intercept term

        NOTE: Remember to handle the intercept ("homogenous coordinate")
        '''
        homo = np.ones(A.shape[0])[:, np.newaxis]
        A = np.hstack([A, homo])
        if mode == "qr":
            Q, R = self.qr_decomposition(A)
            c = scipy.linalg.solve_triangular(R, Q.T@y)
        elif mode == "pinverse":
            A_pinverse = self.pinverse(A)
            c = A_pinverse@y
        return c

    def hidden_act(self, data):
        '''Compute the activation of the hidden layer units

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features). Data to learn / train on.

        Returns:
        -----------
        ndarray. shape=(num_samps, k).
            Activation of each unit in the hidden layer to each of the data samples.
            Do NOT include the bias unit activation.
            See notebook for refresher on the activation equation
        '''
        H = np.zeros((data.shape[0], self.k))
        for i in range(self.prototypes.shape[0]):
            dist = np.sum((data-self.prototypes[i])**2, axis=1)
            H[:, i] = np.exp(-dist/(2*(self.sigmas[i]**2)+1e-8))
        return H

    def output_act(self, hidden_acts):
        '''Compute the activation of the output layer units

        Parameters:
        -----------
        hidden_acts: ndarray. shape=(num_samps, k).
            Activation of the hidden units to each of the data samples.
            Does NOT include the bias unit activation.

        Returns:
        -----------
        ndarray. shape=(num_samps, num_output_units).
            Activation of each unit in the output layer to each of the data samples.

        NOTE:
        - Assumes that learning has already taken place
        - Can be done without any for loops.
        - Don't forget about the bias unit!
        '''
        homo = np.ones(hidden_acts.shape[0])[:, np.newaxis]
        hidden_acts = np.hstack([hidden_acts, homo])
        output_act = hidden_acts@self.wts
        return output_act

    def train(self, data, y, mode="qr"):
        '''Train the radial basis function network

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features). Data to learn / train on.
        y: ndarray. shape=(num_samps,). Corresponding class of each data sample.
        mode: (for extension)

        Goal: Set the weights between the hidden and output layer weights (self.wts) using
        linear regression. The regression is between the hidden layer activation (to the data) and
        the correct classes of each training sample. To solve for the weights going FROM all of the
        hidden units TO output unit c, recode the class vector `y` to 1s and 0s:
            1 if the class of a data sample in `y` is c
            0 if the class of a data sample in `y` is not c

        Notes:
        - Remember to initialize the network (set hidden unit prototypes and sigmas based on data).
        - Pay attention to the shape of self.wts in the constructor above. Yours needs to match.
        - The linear regression method handles the bias unit.
        '''
        self.initialize(data)
        hidden_act = self.hidden_act(data)

        self.wts = np.zeros((hidden_act.shape[1]+1, self.num_classes))
        decode_y = np.zeros((y.shape[0], self.num_classes))
        for i in range(y.shape[0]):
            decode_y[i, y[i]] = 1
        for i in range(self.num_classes):
            self.wts[:, i] = self.linear_regression(
                hidden_act, decode_y[:, i], mode)

    def predict(self, data):
        '''Classify each sample in `data`

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features). Data to predict classes for.
            Need not be the data used to train the network

        Returns:
        -----------
        ndarray of nonnegative ints. shape=(num_samps,). Predicted class of each data sample.

        TODO:
        - Pass the data thru the network (input layer -> hidden layer -> output layer).
        - For each data sample, the assigned class is the index of the output unit that produced the
        largest activation.
        '''
        hidden_act = self.hidden_act(data)
        output_act = self.output_act(hidden_act)
        y_pred = np.argmax(output_act, axis=1)
        return y_pred

    def accuracy(self, y, y_pred):
        '''Computes accuracy based on percent correct: Proportion of predicted class labels `y_pred`
        that match the true values `y`.

        Parameters:
        -----------
        y: ndarray. shape=(num_data_sams,)
            Ground-truth, known class labels for each data sample
        y_pred: ndarray. shape=(num_data_sams,)
            Predicted class labels by the model for each data sample

        Returns:
        -----------
        float. Between 0 and 1. Proportion correct classification.

        NOTE: Can be done without any loops
        '''
        correct_pred = (y == y_pred)
        true_pred_index = np.where(correct_pred == True)
        true_pred_total = len(true_pred_index[0])
        return true_pred_total/y.shape[0]


class RBF_Reg_Net(RBF_Net):
    '''RBF Neural Network configured to perform regression
    '''

    def __init__(self, num_hidden_units, num_classes, h_sigma_gain=5):
        '''RBF regression network constructor

        Parameters:
        -----------
        num_hidden_units: int. Number of hidden units in network. NOTE: does NOT include bias unit
        num_classes: int. Number of output units in network. Equals number of possible classes in
            dataset
        h_sigma_gain: float. Multiplicative gain factor applied to the hidden unit variances

        TODO:
        - Create an instance variable for the hidden unit variance gain
        '''
        super().__init__(num_hidden_units, num_classes)
        self.g = h_sigma_gain

    def hidden_act(self, data):
        '''Compute the activation of the hidden layer units

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features). Data to learn / train on.

        Returns:
        -----------
        ndarray. shape=(num_samps, k).
            Activation of each unit in the hidden layer to each of the data samples.
            Do NOT include the bias unit activation.
            See notebook for refresher on the activation equation

        TODO:
        - Copy-and-paste your classification network code here.
        - Modify your code to apply the hidden unit variance gain to each hidden unit variance.
        '''
        H = np.zeros((data.shape[0], self.k))
        for i in range(data.shape[0]):
            dist = np.sum((self.prototypes-data[i])**2, axis=1)
            H[i] = np.exp(-dist/(2*self.g*(self.sigmas**2)+1e-8))
        return H

    def train(self, data, y):
        '''Train the radial basis function network

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features). Data to learn / train on.
        y: ndarray. shape=(num_samps,). Corresponding class of each data sample.

        Goal: Set the weights between the hidden and output layer weights (self.wts) using
        linear regression. The regression is between the hidden layer activation (to the data) and
        the desired y output of each training sample.

        Notes:
        - Remember to initialize the network (set hidden unit prototypes and sigmas based on data).
        - Pay attention to the shape of self.wts in the constructor above. Yours needs to match.
        - The linear regression method handles the bias unit.

        TODO:
        - Copy-and-paste your classification network code here, modifying it to perform regression on
        the actual y values instead of the y values that match a particular class. Your code should be
        simpler than before.
        - You may need to squeeze the output of your linear regression method if you get shape errors.
        '''
        self.initialize(data)
        hidden_act = self.hidden_act(data)
        self.wts = self.linear_regression(hidden_act, y)

    def predict(self, data):
        '''Classify each sample in `data`

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features). Data to predict classes for.
            Need not be the data used to train the network

        Returns:
        -----------
        ndarray. shape=(num_samps, num_output_neurons). Output layer neuronPredicted "y" value of
            each sample in `data`.

        TODO:
        - Copy-and-paste your classification network code here, modifying it to return the RAW
        output neuron activaion values. Your code should be simpler than before.
        '''
        hidden_act = self.hidden_act(data)
        y_pred = self.output_act(hidden_act)
        return y_pred

    def mean_sse(self, y, y_pred):
        '''Computes the mean sum-of-squares error in the predicted y compared the actual y values.
        See notebook for equation.

        Returns:
        -----------
        float. Mean sum-of-squares error

        Hint: Make use of self.compute_residuals
        '''
        residuals = y - y_pred
        E = np.sum(np.square(residuals), axis=0)/y.shape[0]
        E = np.squeeze(E)
        return E
