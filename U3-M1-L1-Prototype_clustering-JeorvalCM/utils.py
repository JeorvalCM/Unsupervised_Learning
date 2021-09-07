import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib as mpl


def plotProgresskMeans(i, X, centroid_history, idx_history):
    """
    A helper function that displays the progress of k-Means as it is running. It is intended for use
    only with 2D data. It plots data points with colors assigned to each centroid. With the
    previous centroids, it also plots a line between the previous locations and current locations
    of the centroids.

    Parameters
    ----------
    i : int
        Current iteration number of k-means. Used for matplotlib animation function.

    X : array_like
        The dataset, which is a matrix (m x n). Note since the plot only supports 2D data, n should
        be equal to 2.

    centroid_history : list
        A list of computed centroids for all iteration.

    idx_history : list
        A list of computed assigned indices for all iterations.
    """
    K = centroid_history[0].shape[0]
    plt.gcf().clf()
    cmap = plt.cm.rainbow
    norm = mpl.colors.Normalize(vmin=0, vmax=2)

    for k in range(K):
        current = np.stack([c[k, :] for c in centroid_history[:i+1]], axis=0)
        plt.plot(current[:, 0], current[:, 1],
                    '-Xk',
                    mec='k',
                    lw=2,
                    ms=10,
                    mfc=cmap(norm(k)),
                    mew=2)

        plt.scatter(X[:, 0], X[:, 1],
                       c=idx_history[i],
                       cmap=cmap,
                       marker='o',
                       s=8**2,
                       linewidths=1,)
    plt.grid(False)
    plt.title('Iteration number %d' % (i+1))


def runkMeans(X, centroids, findClosestCentroids, computeCentroids,
              max_iters=10, plot_progress=False):
    """
    Runs the K-means algorithm.

    Parameters
    ----------
    X : array_like
        The data set of size (m, n). Each row of X is a single example of n dimensions. The
        data set is a total of m examples.

    centroids : array_like
        Initial centroid location for each clusters. This is a matrix of size (K, n). K is the total
        number of clusters and n is the dimensions of each data point.

    findClosestCentroids : func
        A function (implemented by student) reference which computes the cluster assignment for
        each example.

    computeCentroids : func
        A function(implemented by student) reference which computes the centroid of each cluster.

    max_iters : int, optional
        Specifies the total number of interactions of K-Means to execute.

    plot_progress : bool, optional
        A flag that indicates if the function should also plot its progress as the learning happens.
        This is set to false by default.

    Returns
    -------
    centroids : array_like
        A (K x n) matrix of the computed (updated) centroids.
    idx : array_like
        A vector of size (m,) for cluster assignment for each example in the dataset. Each entry
        in idx is within the range [0 ... K-1].

    anim : FuncAnimation, optional
        A matplotlib animation object which can be used to embed a video within the jupyter
        notebook. This is only returned if `plot_progress` is `True`.
    """
    K = centroids.shape[0]
    idx = None
    idx_history = []
    centroid_history = []

    for i in range(max_iters):
        idx = findClosestCentroids(X, centroids)

        if plot_progress:
            idx_history.append(idx)
            centroid_history.append(centroids)

        centroids = computeCentroids(X, idx, K)

    if plot_progress:
        fig = plt.figure()
        anim = FuncAnimation(fig, plotProgresskMeans,
                             frames=max_iters,
                             interval=500,
                             repeat_delay=2,
                             fargs=(X, centroid_history, idx_history))
        return centroids, idx, anim

    return centroids, idx


def runkMedoids(X, centroids, findClosestCentroids, computeCentroids,
                max_iters=10, plot_progress=False):
    K = centroids.shape[0]
    idx = None
    idx_history = []
    centroid_history = []

    for i in range(max_iters):
        idx = findClosestCentroids(X, centroids)

        if plot_progress:
            idx_history.append(idx)
            centroid_history.append(X[centroids])

        centroids = computeCentroids(X, idx, K)

    if plot_progress:
        fig = plt.figure()
        anim = FuncAnimation(fig, plotProgresskMeans,
                             frames=max_iters,
                             interval=500,
                             repeat_delay=2,
                             fargs=(X, centroid_history, idx_history))
        return centroids, idx, anim

    return centroids, idx
