"""
Name     : ledoit.py
Author   : Yinsen Miao
 Contact : yinsenm@gmail.com
Time     : 7/1/2021
Desc     : Compute Ledoit's robust covariance estimator
"""

import pandas as pd
import numpy as np
from numpy import linalg as LA
from gerber import is_psd_def

def ledoit(rets: np.array):
    """
    compute Ledoit covariance Statistics
    :param rets: assets return matrix of dimension n x p
    :return: Ledoit covariance matrix of p x p, shrinkage parameter
    """
    x = rets.copy()
    t, n = x.shape
    _mean = np.tile(x.mean(axis=0), (t, 1))

    # de-mean the returns
    x -= _mean

    # compute sample covariance matrix
    sample = (1 / t) * x.transpose() @ x

    # compute the prior
    _var = np.diag(sample)
    sqrt_var = np.sqrt(_var).reshape((-1, n))
    rBar = (np.sum(sample / (np.tile(sqrt_var, (n, 1)).T * np.tile(sqrt_var, (n, 1)))) - n) / (n * (n - 1))
    prior = rBar * (np.tile(sqrt_var, (n, 1)).T * np.tile(sqrt_var, (n, 1)))
    prior[np.diag_indices_from(prior)] = _var.tolist()

    # compute shrinkage parameters and constant
    # what we call pi-hat
    y = x ** 2
    phiMat = y.T @ y / t - 2 * (x.T @ x) * sample / t + sample ** 2
    phi = np.sum(phiMat)

    # what we call rho-hat
    term1 = (x ** 3).T @ x / t
    help = (x.T @ x) / t
    helpDiag = np.diag(help).reshape((n, 1))
    term2 = np.tile(helpDiag, (1, n)) * sample
    term3 = help * np.tile(_var.reshape(n, 1), (1, n))
    term4 = np.tile(_var.reshape(n, 1), (1, n)) * sample
    thetaMat = term1 - term2 - term3 + term4
    thetaMat[np.diag_indices_from(thetaMat)] = np.zeros(n)
    rho = np.sum(np.diag(phiMat)) + rBar * np.sum(((1 / sqrt_var.T).dot(sqrt_var)) * thetaMat)

    # what we call gamma-hat
    gamma = LA.norm(sample - prior, 'fro') ** 2

    # compute shrinkage costant
    kappa = (phi - rho) / gamma
    shrinkage = max(0, min(1, kappa / t))

    # compute the estimator
    covMat = shrinkage * prior + (1 - shrinkage) * sample

    return covMat, shrinkage



# test gerber_cov_stat1 and gerber_cov_stat2
if __name__ == "__main__":
    bgn_date = "2018-01-01"
    end_date = "2020-01-01"
    nassets = 4
    file_path = "../data/prcs.csv"
    rets_df = pd.read_csv(file_path, parse_dates=['Date'], index_col=["Date"]).pct_change()[bgn_date: end_date].iloc[:, 0: nassets]    
    rets = rets_df.values

    covMat, shrinkage = ledoit(rets)
    is_psd_def(covMat)
    shrinkage
    print(covMat)
    print(shrinkage)



