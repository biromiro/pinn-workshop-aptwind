###########################################################################
#               Physics-Informed Machine Learning                         #
#                             SS 2021                                     #
#                                                                         #
#                           Exercise 8                                    #
#                                                                         #
# NOTICE: Sharing and distribution of any course content, other than      #
# between individual students registered in the course, is not permitted  #
# without permission.                                                     #
#                                                                         #
###########################################################################

import scipy.io
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

# repoPath = os.path.join(".", "PINNs")  # "." for Colab/VSCode, and ".." for GitHub
# repoPath = os.path.join("..", "PINNs")
# utilsPath = os.path.join(repoPath, "Utilities")
# dataPath = os.path.join(repoPath, "main", "Data")
# appDataPath = os.path.join(repoPath, "appendix", "Data")

sys.path.append("utils")

__all__ = ['lhs']


def lhs(n, samples=None):
    """
    Generate a latin-hypercube design
    
    Parameters
    ----------
    n : int
        The number of factors to generate samples for
    
    Optional
    --------
    samples : int
        The number of samples to generate for each factor (Default: n)
    
    Returns
    -------
    H : 2d-array
        An n-by-samples design matrix that has been normalized so factor values
        are uniformly spaced between zero and one.
    
    """

    cut = np.linspace(0, 1, samples + 1)

    # Fill points uniformly in each interval
    u = np.random.rand(samples, n)
    a = cut[:samples]
    b = cut[1:samples + 1]
    rdpoints = np.zeros_like(u)
    for j in range(n):
        rdpoints[:, j] = u[:, j] * (b - a) + a

    # Make the random pairings
    H = np.zeros_like(rdpoints)
    for j in range(n):
        order = np.random.permutation(range(samples))
        H[:, j] = rdpoints[order, j]

    return H


def prep_data(path, data_size, collocation_size):
    # Reading external data [t is 100x1, usol is 256x100 (solution), x is 256x1]
    data = scipy.io.loadmat(path)

    # Flatten makes [[]] into [], [:,None] makes it a column vector
    t = data['t'].flatten()[:, None]  # T x 1
    x = data['x'].flatten()[:, None]  # N x 1

    # Keeping the 2D data for the solution data (real() is maybe to make it float by default, in case of zeroes)
    u_target_mesh = np.real(data['usol']).T  # T x N

    # Meshing x and t in 2D (256,100)
    X, T = np.meshgrid(x, t)

    # Preparing the inputs x and t (meshed as X, T) for predictions in one single array, as X_star
    X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))

    # Preparing the testing u_star
    u_star = u_target_mesh.flatten()[:, None]

    # Noiseless data
    idx = np.random.choice(X_star.shape[0], data_size, replace=False)
    X_u_train = X_star[idx, :]
    u_train = u_star[idx, :]

    # Domain bounds (lower bounds upper bounds) [x, t], which are here ([-1.0, 0.0] and [1.0, 1.0])
    lb = X_star.min(axis=0)
    ub = X_star.max(axis=0)
    # Getting the initial conditions (t=0)
    xx1 = np.hstack((X[0:1, :].T, T[0:1, :].T))
    uu1 = u_target_mesh[0:1, :].T
    # Getting the lowest boundary conditions (x=-1) 
    xx2 = np.hstack((X[:, 0:1], T[:, 0:1]))
    uu2 = u_target_mesh[:, 0:1]
    # Getting the highest boundary conditions (x=1) 
    xx3 = np.hstack((X[:, -1:], T[:, -1:]))
    uu3 = u_target_mesh[:, -1:]
    # print("uu1", uu1)
    # plt.plot(xx1[:,0], uu1)
    # plt.show()
    # print("uu2", uu2)
    # print("uu3", uu3)
    # exit()
    # Stacking them in multidimensional tensors for training (X_u_train is for now the continuous boundaries)
    X_u_train = np.vstack([xx1, xx2, xx3])
    u_train = np.vstack([uu1, uu2, uu3])

    # Generating the x and t collocation points for f, with each having a collocation_size size
    # We pointwise add and multiply to spread the LHS over the 2D domain
    X_f_train = lb + (ub - lb) * lhs(2, collocation_size)

    # Generating a uniform random sample from ints between 0, and the size of x_u_train, of size data_size (initial data size) and without replacement (unique)
    idx = np.random.choice(X_u_train.shape[0], data_size, replace=False)
    # Getting the corresponding X_u_train (which is now scarce boundary/initial coordinates)
    X_u_train = X_u_train[idx, :]
    # Getting the corresponding u_train
    u_train = u_train[idx, :]

    return x, t, X, T, u_target_mesh, X_star, u_star, X_u_train, u_train, X_f_train
