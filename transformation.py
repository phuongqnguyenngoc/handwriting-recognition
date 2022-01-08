'''transformation.py
Perform projections, translations, rotations, and scaling operations on Numpy ndarray data.
Phuong Nguyen Ngoc
CS 251 Data Analysis Visualization, Spring 2021
'''
import numpy as np
import matplotlib.pyplot as plt
import palettable
import analysis
import data
from data import *
from analysis import *
import math


class Transformation(analysis.Analysis):

    def __init__(self, orig_dataset, data=None):
        '''Constructor for a Transformation object

        Parameters:
        -----------
        orig_dataset: Data object. shape=(N, num_vars).
            Contains the original dataset (only containing all the numeric variables,
            `num_vars` in total).
        data: Data object (or None). shape=(N, num_proj_vars).
            Contains all the data samples as the original, but ONLY A SUBSET of the variables.
            (`num_proj_vars` in total). `num_proj_vars` <= `num_vars`

        TODO:
        - Pass `data` to the superclass constructor.
        - Create an instance variable for `orig_dataset`.
        '''
        self.data = data
        self.orig_dataset = orig_dataset

    def project(self, headers):
        '''Project the original dataset onto the list of data variables specified by `headers`,
        i.e. select a subset of the variables from the original dataset.
        In other words, your goal is to populate the instance variable `self.data`.

        Parameters:
        -----------
        headers: Python list of str. len(headers) = `num_proj_vars`, usually 1-3 (inclusive), but
            there could be more.
            A list of headers (strings) specifying the feature to be projected onto each axis.
            For example: if headers = ['hi', 'there', 'cs251'], then the data variables
                'hi' becomes the 'x' variable,
                'there' becomes the 'y' variable,
                'cs251' becomes the 'z' variable.
            The length of the list matches the number of dimensions onto which the dataset is
            projected — having 'y' and 'z' variables is optional.

        TODO:
        - Create a new `Data` object that you assign to `self.data` (project data onto the `headers`
        variables). Determine and fill in 'valid' values for all the `Data` constructor
        keyword arguments (except you dont need `filepath` because it is not relevant here).
        '''
        new_headers = []
        header2col = {}
        idx = 0
        for var in self.orig_dataset.get_headers():
            if var in headers:
                new_headers.append(var)
                header2col[var] = idx
                idx += 1
        projected = self.orig_dataset.select_data(new_headers)

        projected_data = Data(headers=new_headers,
                              data=projected, header2col=header2col)
        self.data = projected_data

    def get_data_homogeneous(self):
        '''Helper method to get a version of the projected data array with an added homogeneous
        coordinate. Useful for homogeneous transformations.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars+1). The projected data array with an added 'fake variable'
        column of ones on the right-hand side.
            For example: If we have the data SAMPLE (just one row) in the projected data array:
            [3.3, 5.0, 2.0], this sample would become [3.3, 5.0, 2.0, 1] in the returned array.

        NOTE:
        - Do NOT update self.data with the homogenous coordinate.
        '''
        homo = np.ones(self.data.get_num_samples())
        homo = homo[:, np.newaxis]
        return np.hstack([self.data.data, homo])

    def translation_matrix(self, magnitudes):
        ''' Make an M-dimensional homogeneous transformation matrix for translation,
        where M is the number of features in the projected dataset.

        Parameters:
        -----------
        magnitudes: Python list of float.
            Translate corresponding variables in `headers` (in the projected dataset) by these
            amounts.

        Returns:
        -----------
        ndarray. shape=(num_proj_vars+1, num_proj_vars+1). The transformation matrix.

        NOTE: This method just creates the translation matrix. It does NOT actually PERFORM the
        translation!
        '''
        translation = np.eye(self.data.get_num_dims() + 1)
        translation[:-1, -1] = np.array(magnitudes)
        return translation

    def scale_matrix(self, magnitudes):
        '''Make an M-dimensional homogeneous scaling matrix for scaling, where M is the number of
        variables in the projected dataset.

        Parameters:
        -----------
        magnitudes: Python list of float.
            Scale corresponding variables in `headers` (in the projected dataset) by these amounts.

        Returns:
        -----------
        ndarray. shape=(num_proj_vars+1, num_proj_vars+1). The scaling matrix.

        NOTE: This method just creates the scaling matrix. It does NOT actually PERFORM the scaling!
        '''
        scale = np.eye(self.data.get_num_dims() + 1)
        for i in range(len(magnitudes)):
            scale[i, i] = magnitudes[i]
        return scale

    def translate(self, magnitudes):
        '''Translates the variables `headers` in projected dataset in corresponding amounts specified
        by `magnitudes`.

        Parameters:
        -----------
        magnitudes: Python list of float.
            Translate corresponding variables in `headers` (in the projected dataset) by these amounts.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The translated data (with all variables in the projected).
            dataset. NOTE: There should be NO homogenous coordinate!

        TODO:
        - Use matrix multiplication to translate the projected dataset, as advertised above.
        - Update `self.data` with a NEW Data object with the SAME `headers` and `header2col`
        dictionary as the current `self.data`, but DIFFERENT data (set to the data you
        transformed in this method). NOTE: The updated `self.data` SHOULD NOT have a homogenous
        coordinate!
        '''
        translate_matrix = self.translation_matrix(magnitudes)
        data_matrix = self.get_data_homogeneous()
        translated_data = (translate_matrix@(data_matrix.T)).T[:, :-1]
        self.data = Data(headers=self.data.headers,
                         data=translated_data, header2col=self.data.header2col)
        return translated_data

    def scale(self, magnitudes):
        '''Scales the variables `headers` in projected dataset in corresponding amounts specified
        by `magnitudes`.

        Parameters:
        -----------
        magnitudes: Python list of float.
            Scale corresponding variables in `headers` (in the projected dataset) by these amounts.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The scaled data (with all variables in the projected).
            dataset. NOTE: There should be NO homogenous coordinate!

        TODO:
        - Use matrix multiplication to scale the projected dataset, as advertised above.
        - Update `self.data` with a NEW Data object with the SAME `headers` and `header2col`
        dictionary as the current `self.data`, but DIFFERENT data (set to the data you
        transformed in this method). NOTE: The updated `self.data` SHOULD NOT have a
        homogenous coordinate!
        '''
        scale_matrix = self.scale_matrix(magnitudes)
        data_matrix = self.get_data_homogeneous()
        scaled_data = (scale_matrix@(data_matrix.T)).T[:, :-1]
        self.data = Data(headers=self.data.headers,
                         data=scaled_data, header2col=self.data.header2col)
        return scaled_data

    def transform(self, C):
        '''Transforms the PROJECTED dataset by applying the homogeneous transformation matrix `C`.

        Parameters:
        -----------
        C: ndarray. shape=(num_proj_vars+1, num_proj_vars+1).
            A homogeneous transformation matrix.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The projected dataset after it has been transformed by `C`

        TODO:
        - Use matrix multiplication to apply the compound transformation matix `C` to the projected
        dataset.
        - Update `self.data` with a NEW Data object with the SAME `headers` and `header2col`
        dictionary as the current `self.data`, but DIFFERENT data (set to the data you
        transformed in this method). NOTE: The updated `self.data` SHOULD NOT have a homogenous
        coordinate!
        '''
        data_matrix = self.get_data_homogeneous()
        transformed_data = (C@(data_matrix.T)).T[:, :-1]
        self.data = Data(headers=self.data.headers,
                         data=transformed_data, header2col=self.data.header2col)
        return transformed_data

    def normalize_together(self):
        '''Normalize all variables in the projected dataset together by translating the global minimum
        (across all variables) to zero and scaling the global range (across all variables) to one.

        You should normalize (update) the data stored in `self.data`.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The normalized version of the projected dataset.

        NOTE: Given the goal of this project, for full credit you should implement the normalization
        using matrix multiplications (matrix transformations).
        '''
        analysis = Analysis(self.data)
        max_val = max(analysis.max(self.data.get_headers()))
        min_val = min(analysis.min(self.data.get_headers()))
        range_val = max_val - min_val
        num_of_var = self.data.get_num_dims()

        scale_vector = [1/(max_val - min_val) for i in range(num_of_var)]
        translation_vector = [-min_val for i in range(num_of_var)]

        C = self.scale_matrix(
            scale_vector)@self.translation_matrix(translation_vector)
        normalized_data = self.transform(C)
        self.data = Data(headers=self.data.headers,
                         data=normalized_data, header2col=self.data.header2col)
        return normalized_data

    def normalize_separately(self):
        '''Normalize each variable separately by translating its local minimum to zero and scaling
        its local range to one.

        You should normalize (update) the data stored in `self.data`.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The normalized version of the projected dataset.

        NOTE: Given the goal of this project, for full credit you should implement the normalization
        using matrix multiplications (matrix transformations).
        '''
        analysis = Analysis(self.data)
        max_vals = analysis.max(self.data.get_headers())
        min_vals = analysis.min(self.data.get_headers())
        scale_vector = [1/(max_vals[i] - min_vals[i])
                        for i in range(len(max_vals))]
        translation_vector = [-min_vals[i] for i in range(len(max_vals))]
        C = self.scale_matrix(
            scale_vector)@self.translation_matrix(translation_vector)
        normalized_data = self.transform(C)
        self.data = Data(headers=self.data.headers,
                         data=normalized_data, header2col=self.data.header2col)
        return normalized_data

    def rotation_matrix_3d(self, header, degrees):
        '''Make an 3-D homogeneous rotation matrix for rotating the projected data
        about the ONE axis/variable `header`.

        Parameters:
        -----------
        header: str. Specifies the variable about which the projected dataset should be rotated.
        degrees: float. Angle (in degrees) by which the projected dataset should be rotated.

        Returns:
        -----------
        ndarray. shape=(4, 4). The 3D rotation matrix with homogenous coordinate.

        NOTE: This method just creates the rotation matrix. It does NOT actually PERFORM the rotation!
        '''
        rotation_matrix = np.eye(self.data.get_num_dims()+1)
        rotate_radian = math.radians(degrees)
        if self.data.get_header_indices([header])[0] == 0:
            rotation_matrix[1][1] = math.cos(rotate_radian)
            rotation_matrix[1][2] = -math.sin(rotate_radian)
            rotation_matrix[2][1] = math.sin(rotate_radian)
            rotation_matrix[2][2] = math.cos(rotate_radian)
        elif self.data.get_header_indices([header])[0] == 1:
            rotation_matrix[0][0] = math.cos(rotate_radian)
            rotation_matrix[0][2] = math.sin(rotate_radian)
            rotation_matrix[2][0] = -math.sin(rotate_radian)
            rotation_matrix[2][2] = math.cos(rotate_radian)
        elif self.data.get_header_indices([header])[0] == 2:
            rotation_matrix[0][0] = math.cos(rotate_radian)
            rotation_matrix[0][1] = -math.sin(rotate_radian)
            rotation_matrix[1][0] = math.sin(rotate_radian)
            rotation_matrix[1][1] = math.cos(rotate_radian)
        return rotation_matrix

    def rotate_3d(self, header, degrees):
        '''Rotates the projected data about the variable `header` by the angle (in degrees)
        `degrees`.

        Parameters:
        -----------
        header: str. Specifies the variable about which the projected dataset should be rotated.
        degrees: float. Angle (in degrees) by which the projected dataset should be rotated.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The rotated data (with all variables in the projected).
            dataset. NOTE: There should be NO homogenous coordinate!

        TODO:
        - Use matrix multiplication to rotate the projected dataset, as advertised above.
        - Update `self.data` with a NEW Data object with the SAME `headers` and `header2col`
        dictionary as the current `self.data`, but DIFFERENT data (set to the data you
        transformed in this method). NOTE: The updated `self.data` SHOULD NOT have a
        homogenous coordinate!
        '''
        rotation_matrix = self.rotation_matrix_3d(header, degrees)
        data_matrix = self.get_data_homogeneous()
        rotated_data = (rotation_matrix@(data_matrix.T)).T[:, :-1]
        self.data = Data(headers=self.data.headers,
                         data=rotated_data, header2col=self.data.header2col)
        return rotated_data

    def scatter_color(self, ind_var, dep_var, c_var, title=None):
        '''Creates a 2D scatter plot with a color scale representing the 3rd dimension.

        Parameters:
        -----------
        ind_var: str. Header of the variable that will be plotted along the X axis.
        dep_var: Header of the variable that will be plotted along the Y axis.
        c_var: Header of the variable that will be plotted along the color axis.
            NOTE: Use a ColorBrewer color palette (e.g. from the `palettable` library).
        title: str or None. Optional title that will appear at the top of the figure.
        '''
        plt.figure(figsize=(6, 6))
        plt.xlabel(ind_var)
        plt.ylabel(dep_var)
        ind_var_data = self.data.select_data([ind_var])
        dep_var_data = self.data.select_data([dep_var])
        c_var_data = self.data.select_data([c_var])
        brewer_colors = palettable.cartocolors.sequential.Teal_7.mpl_colormap
        scat = plt.scatter(ind_var_data, dep_var_data,
                           c=c_var_data, cmap=brewer_colors, edgecolors='#808080')
        cbar = plt.colorbar(scat)
        cbar.set_label(c_var)
        if title is not None:
            plt.title(title)

    def heatmap(self, headers=None, title=None, cmap="gray"):
        '''Generates a heatmap of the specified variables (defaults to all). Each variable is normalized
        separately and represented as its own row. Each individual is represented as its own column.
        Normalizing each variable separately means that one color axis can be used to represent all
        variables, 0.0 to 1.0.

        Parameters:
        -----------
        headers: Python list of str (or None). (Optional) The variables to include in the heatmap.
            Defaults to all variables if no list provided.
        title: str. (Optional) The figure title. Defaults to an empty string (no title will be displayed).
        cmap: str. The colormap string to apply to the heatmap. Defaults to grayscale
            -- black (0.0) to white (1.0)

        Returns:
        -----------
        fig, ax: references to the figure and axes on which the heatmap has been plotted
        '''

        # Create a doppelganger of this Transformation object so that self.data
        # remains unmodified when heatmap is done
        data_clone = data.Data(headers=self.data.get_headers(),
                               data=self.data.get_all_data(),
                               header2col=self.data.get_mappings())
        dopp = Transformation(self.data, data_clone)
        dopp.normalize_separately()

        fig, ax = plt.subplots()
        if title is not None:
            ax.set_title(title)
        ax.set(xlabel="Individuals")

        # Select features to plot
        if headers is None:
            headers = dopp.data.headers
        m = dopp.data.select_data(headers)

        # Generate heatmap
        hmap = ax.imshow(m.T, aspect="auto", cmap=cmap, interpolation='None')

        # Label the features (rows) along the Y axis
        y_lbl_coords = np.arange(m.shape[1]+1) - 0.5
        ax.set_yticks(y_lbl_coords, minor=True)
        y_lbls = [""] + headers
        ax.set_yticklabels(y_lbls)
        ax.grid(linestyle='none')

        # Create and label the colorbar
        cbar = fig.colorbar(hmap)
        cbar.ax.set_ylabel("Normalized Features")

        return fig, ax

    def scatter_color_size(self, ind_var, dep_var, c_var, s_var, title=None):
        '''Creates a 2D scatter plot with a color scale representing the 3rd dimension
            and the marker's size representing the 4th dimension

        Parameters:
        -----------
        ind_var: str. Header of the variable that will be plotted along the X axis.
        dep_var: Header of the variable that will be plotted along the Y axis.
        c_var: Header of the variable that will be plotted along the color axis.
        s_var: Header of the variable that will be plotted along the size axis
            NOTE: Use a ColorBrewer color palette (e.g. from the `palettable` library).
        title: str or None. Optional title that will appear at the top of the figure.
        '''
        plt.figure(figsize=(12, 12))
        plt.xlabel(ind_var)
        plt.ylabel(dep_var)
        ind_var_data = self.data.select_data([ind_var])
        dep_var_data = self.data.select_data([dep_var])
        c_var_data = self.data.select_data([c_var])
        s_var_data = self.data.select_data([s_var])*500
        # print(s_var_data)
        # print(c_var)
        brewer_colors = palettable.cartocolors.sequential.Teal_7.mpl_colormap
        scat = plt.scatter(ind_var_data, dep_var_data,
                           c=c_var_data, cmap=brewer_colors, s=s_var_data, edgecolors='#808080')
        cbar = plt.colorbar(scat)
        cbar.set_label(c_var)
        legend_elements = [plt.Line2D(
            [0], [0], lw=0, label=f"Marker size represents {s_var}")]
        plt.legend(handles=legend_elements)
        if title is not None:
            plt.title(title)

    def scatter_color_size_shape(self, ind_var, dep_var, c_var, s_var, shape_var, shape_label=None, title=None):
        '''Creates a 2D scatter plot with a color scale representing the 3rd dimension
            marker's size representing the 4th dimension,
            and marker's shape representing the 5th dimension

        Parameters:
        -----------
        ind_var: str. Header of the variable that will be plotted along the X axis.
        dep_var: str. Header of the variable that will be plotted along the Y axis.
        c_var: str. Header of the variable that will be plotted along the color axis.
        s_var: str. Header of the variable that will be plotted along the size axis.
        shape_var: str. Header of the variable that will be plotted along the shape axis.
        shape_label: Python list of string. label of type string for the variable mapped on the shape axis whose value is currently numeric
            NOTE: Use a ColorBrewer color palette (e.g. from the `palettable` library).
        title: str or None. Optional title that will appear at the top of the figure.
        '''
        plt.figure(figsize=(12, 12))
        plt.xlabel(ind_var)
        plt.ylabel(dep_var)
        ind_var_data = self.data.select_data([ind_var])
        dep_var_data = self.data.select_data([dep_var])
        c_var_data = self.data.select_data([c_var])
        s_var_data = self.data.select_data([s_var])*400
        shape_var_data = self.data.select_data([shape_var])
        brewer_colors = palettable.cartocolors.sequential.Teal_7.mpl_colormap
        shape = ["o", "s", "v"]
        scat = ""
        for i in range(len(shape_label)):
            curr_label = shape_var_data == i
            scat = plt.scatter(ind_var_data[curr_label], dep_var_data[curr_label],
                               c=c_var_data[curr_label], marker=shape[i], cmap=brewer_colors, s=s_var_data[curr_label], edgecolors='#808080', label=shape_label[i])
        cbar = plt.colorbar(scat)
        cbar.set_label(c_var)

        if title is not None:
            plt.title(title)
        legend1 = plt.legend(labelspacing=1, loc=4)

        legend_elements = [plt.Line2D(
            [0], [0], lw=0, label=f"Marker size represents {s_var}")]
        legend2 = plt.legend(handles=legend_elements, loc=2, frameon=True)
        plt.gca().add_artist(legend1)
        plt.gca().add_artist(legend2)

    def whiten_PCA(self, headers, set_data=False):
        data = self.data.select_data(headers)
        means = self.mean(headers)
        data = data - means
        cov_mat = self.covariance_matrix(headers)
        # calculate egeinvalues and eigenvectors of the covariance matrix
        eigenvalues, eigenvectors = np.linalg.eig(cov_mat)
        scale_matrix = self.scale_matrix(
            1/(eigenvalues**0.5))[:len(headers), :len(headers)]
        compound_mat = scale_matrix@(eigenvectors.T)
        whitened_data = compound_mat@(data.T)
        whiten_cov = np.cov(whitened_data.T, rowvar=False, bias=True)
        print("Covariance matrix of whitened data:")
        print(whiten_cov)
        if len(headers) == self.data.get_num_dims() and set_data:
            self.data = Data(headers=self.data.headers,
                             data=whitened_data.T, header2col=self.data.header2col)
        return whitened_data
