'''analysis.py
Run statistical analyses and plot Numpy ndarray data
Phuong Nguyen Ngoc
CS 251 Data Analysis Visualization, Spring 2021
'''
import numpy as np
import matplotlib.pyplot as plt
import math


class Analysis:
    def __init__(self, data):
        '''

        Parameters:
        -----------
        data: Data object. Contains all data samples and variables in a dataset.
        '''
        self.data = data

        # Make plot font sizes legible
        plt.rcParams.update({'font.size': 18})

    def set_data(self, data):
        self.data = data
        '''Method that re-assigns the instance variable `data` with the parameter.
        Convenience method to change the data used in an analysis without having to create a new
        Analysis object.

        Parameters:
        -----------
        data: Data object. Contains all data samples and variables in a dataset.
        '''

    def min(self, headers, rows=[]):
        '''Computes the minimum of each variable in `headers` in the data object.
        Possibly only in a subset of data samples (`rows`) if `rows` is not empty.
        (i.e. the minimum value in each of the selected columns)

        Parameters:
        -----------
        headers: Python list of str.
            One str per header variable name in data
        rows: Python list of int.
            Indices of data samples to restrict computation of min over, or over all indices
            if rows=[]

        Returns
        -----------
        mins: ndarray. shape=(len(headers),)
            Minimum values for each of the selected header variables

        NOTE: Loops are forbidden!
        '''
        return np.min(self.data.select_data(headers, rows), axis=0)

    def max(self, headers, rows=[]):
        '''Computes the maximum of each variable in `headers` in the data object.
        Possibly only in a subset of data samples (`rows`) if `rows` is not empty.

        Parameters:
        -----------
        headers: Python list of str.
            One str per header variable name in data
        rows: Python list of int.
            Indices of data samples to restrict computation of max over, or over all indices
            if rows=[]

        Returns
        -----------
        maxs: ndarray. shape=(len(headers),)
            Maximum values for each of the selected header variables

        NOTE: Loops are forbidden!
        '''
        return np.max(self.data.select_data(headers, rows), axis=0)

    def range(self, headers, rows=[]):
        '''Computes the range [min, max] for each variable in `headers` in the data object.
        Possibly only in a subset of data samples (`rows`) if `rows` is not empty.

        Parameters:
        -----------
        headers: Python list of str.
            One str per header variable name in data
        rows: Python list of int.
            Indices of data samples to restrict computation of min/max over, or over all indices
            if rows=[]

        Returns
        -----------
        mins: ndarray. shape=(len(headers),)
            Minimum values for each of the selected header variables
        maxes: ndarray. shape=(len(headers),)
            Maximum values for each of the selected header variables

        NOTE: Loops are forbidden!
        '''
        return (np.min(self.data.select_data(headers, rows), axis=0), np.max(self.data.select_data(headers, rows), axis=0))

    def mean(self, headers, rows=[]):
        '''Computes the mean for each variable in `headers` in the data object.
        Possibly only in a subset of data samples (`rows`).

        Parameters:
        -----------
        headers: Python list of str.
            One str per header variable name in data
        rows: Python list of int.
            Indices of data samples to restrict computation of mean over, or over all indices
            if rows=[]

        Returns
        -----------
        means: ndarray. shape=(len(headers),)
            Mean values for each of the selected header variables

        NOTE: You CANNOT use np.mean here!
        NOTE: Loops are forbidden!
        '''
        return np.sum(self.data.select_data(headers, rows), axis=0)/float(self.data.get_num_samples()) if rows == [] else np.sum(self.data.select_data(headers, rows), axis=0)/len(rows)

    def var(self, headers, rows=[]):
        '''Computes the variance for each variable in `headers` in the data object.
        Possibly only in a subset of data samples (`rows`) if `rows` is not empty.

        Parameters:
        -----------
        headers: Python list of str.
            One str per header variable name in data
        rows: Python list of int.
            Indices of data samples to restrict computation of variance over, or over all indices
            if rows=[]

        Returns
        -----------
        vars: ndarray. shape=(len(headers),)
            Variance values for each of the selected header variables

        NOTE: You CANNOT use np.var or np.mean here!
        NOTE: Loops are forbidden!
        '''
        return np.sum((self.data.select_data(headers, rows) - self.mean(headers, rows))*(self.data.select_data(headers, rows) - self.mean(headers, rows)), axis=0)/(self.data.get_num_samples()-1) if rows == [] \
            else np.sum((self.data.select_data(headers, rows) - self.mean(headers, rows))*(self.data.select_data(headers, rows) - self.mean(headers, rows)), axis=0)/(len(rows)-1)

    def std(self, headers, rows=[]):
        '''Computes the standard deviation for each variable in `headers` in the data object.
        Possibly only in a subset of data samples (`rows`) if `rows` is not empty.

        Parameters:
        -----------
        headers: Python list of str.
            One str per header variable name in data
        rows: Python list of int.
            Indices of data samples to restrict computation of standard deviation over,
            or over all indices if rows=[]

        Returns
        -----------
        vars: ndarray. shape=(len(headers),)
            Standard deviation values for each of the selected header variables

        NOTE: You CANNOT use np.var, np.std, or np.mean here!
        NOTE: Loops are forbidden!
        '''
        return np.sqrt(np.sum((self.data.select_data(headers, rows) - self.mean(headers, rows))*(self.data.select_data(headers, rows) - self.mean(headers, rows)), axis=0)/(self.data.get_num_samples()-1)) if rows == [] \
            else np.sqrt(np.sum((self.data.select_data(headers, rows) - self.mean(headers, rows))*(self.data.select_data(headers, rows) - self.mean(headers, rows)), axis=0)/(len(rows)-1))

    def show(self):
        '''Simple wrapper function for matplotlib's show function.

        (Does not require modification)
        '''
        plt.show()

    def scatter(self, ind_var, dep_var, title):
        '''Creates a simple scatter plot with "x" variable in the dataset `ind_var` and
        "y" variable in the dataset `dep_var`. Both `ind_var` and `dep_var` should be strings
        in `self.headers`.

        Parameters:
        -----------
        ind_var: str.
            Name of variable that is plotted along the x axis
        dep_var: str.
            Name of variable that is plotted along the y axis
        title: str.
            Title of the scatter plot

        Returns:
        -----------
        x. ndarray. shape=(num_data_samps,)
            The x values that appear in the scatter plot
        y. ndarray. shape=(num_data_samps,)
            The y values that appear in the scatter plot

        NOTE: Do not call plt.show() here.
        '''
        plt.xlabel(ind_var)
        plt.ylabel(dep_var)
        ind_var = self.data.select_data([ind_var])
        dep_var = self.data.select_data([dep_var])
        plt.scatter(ind_var, dep_var)
        if title is not None:
            plt.title(title)
        return (ind_var, dep_var)

    def pair_plot(self, data_vars, fig_sz=(12, 12), title=''):
        '''Create a pair plot: grid of scatter plots showing all combinations of variables in
        `data_vars` in the x and y axes.

        Parameters:
        -----------
        data_vars: Python list of str.
            Variables to place on either the x or y axis of the scatter plots
        fig_sz: tuple of 2 ints.
            The width and height of the figure of subplots. Pass as a paramter to plt.subplots.
        title. str. Title for entire figure (not the individual subplots)

        Returns:
        -----------
        fig. The matplotlib figure.
            1st item returned by plt.subplots
        axes. ndarray of AxesSubplot objects. shape=(len(data_vars), len(data_vars))
            2nd item returned by plt.subplots

        TODO:
        - Make the len(data_vars) x len(data_vars) grid of scatterplots
        - The y axis of the first column should be labeled with the appropriate variable being
        plotted there.
        - The x axis of the last row should be labeled with the appropriate variable being plotted
        there.
        - There should be no other axis or tick labels (it looks too cluttered otherwise!)

        Tip: Check out the sharex and sharey keyword arguments of plt.subplots.
        Because variables may have different ranges, pair plot columns usually share the same
        x axis and rows usually share the same y axis.
        '''
        fig, axs = plt.subplots(len(data_vars), len(
            data_vars), sharex='col', sharey='row', figsize=fig_sz)
        for i in range(len(data_vars)):
            for j in range(len(data_vars)):
                axs[i, j].scatter(self.data.select_data(
                    [data_vars[j]]), self.data.select_data([data_vars[i]]))
                axs[i, j].set_ylabel(data_vars[i])
                axs[i, j].set_xlabel(data_vars[j])
        for ax in fig.get_axes():
            ax.label_outer()
        fig.suptitle(title)
        return (fig, axs)

    def histogram(self, var, title, bins=10, stacked=False):
        var = self.data.select_data(var)
        plt.hist(var, bins=bins, stacked=stacked)
        plt.title(title)
        return (var)

    def color_coded_scatter(self, ind_var, dep_var, colored_var, title):
        plt.xlabel(ind_var)
        plt.ylabel(dep_var)
        ind_var = self.data.select_data([ind_var])  # len N
        dep_var = self.data.select_data([dep_var])  # len N
        colored_var = self.data.select_data([colored_var])  # len N
        labels = ["No", "Yes"]
        colors = ['r', 'b']
        for outcome in [0, 1]:
            curr_outcome = colored_var == outcome  # len N: each value True or False
            plt.scatter(ind_var[curr_outcome], dep_var[curr_outcome], c=colors[outcome],
                        label=labels[outcome])  # labels len 2: pick ONE thing out
        if title is not None:
            plt.title(title)
        plt.legend()
        return (ind_var, dep_var)

    def color_coded_pair_plot(self, data_vars, colored_var, fig_sz=(12, 12), title=''):
        fig, axs = plt.subplots(len(data_vars), len(
            data_vars), sharex='col', sharey='row', figsize=fig_sz)
        labels = ["No", "Yes"]
        colors = ['r', 'b']
        colored_var = self.data.select_data([colored_var])
        for i in range(len(data_vars)):
            for j in range(len(data_vars)):
                for outcome in [0, 1]:
                    curr_outcome = colored_var == outcome
                    axs[i, j].scatter(self.data.select_data([data_vars[j]])[curr_outcome], self.data.select_data([data_vars[i]])[curr_outcome], c=colors[outcome],
                                      label=labels[outcome])
                    axs[i, j].set_ylabel(data_vars[i])
                    axs[i, j].set_xlabel(data_vars[j])
                if i == 0 and j == 0:
                    axs[i, j].legend()
        for ax in fig.get_axes():
            ax.label_outer()

        return (fig, axs)

    def covariance_matrix(self, headers):
        data = self.data.select_data(headers)
        means = self.mean(headers)
        data = data - means
        covariance = (data.T)@data/(data.shape[0]-1)
        return covariance
