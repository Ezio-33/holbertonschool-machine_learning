#!/usr/bin/env python3
"""
Project Dimensionality reduction
By Ced+
"""
import numpy as np


def pca(X, var=0.95):
    """
    Takes X a matrix,
    return Wr (reduction)
    """
    n = X.shape[0]  # number of data points
    d = X.shape[1]  # number of dimensions

    eigv_norm = list()
    W = np.ones((d, d), dtype=float)

    # covariance matrix
    cov = X.T @ X
    # finding eigenvalues and eigenvectors

    eigv, W = np.linalg.eig(cov)
    # eigv = np.linalg.eigvals(cov)

    # sorting elements
    zipW = list(zip(eigv, W.T))
    sorted_zip = sorted(zipW, key=lambda x: x[0], reverse=True)

    # get nd wich is the domension reduction
    summation = 0
    i = 0
    threshold = sum(eigv) * var
    while summation < threshold:
        summation += sorted_zip[i][0]
        i += 1
    nd = i + 1

    W_r = np.zeros((d, nd), dtype=float)

    # sorted zip a 2 dimension les deuxieme donne eigen
    for i in range(nd):
        W_r[:, i] = pow(-1, i) * sorted_zip[i][1]
    return W_r
