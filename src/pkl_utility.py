'''
Utility programs for

SPaiK   - A scalable pairwise kernel learning software (version 0.1)                    
                                                                       
The work was financially supported by the Research Council of Finland (Project No. #345804 and #345805).


Created on 13 Mar 2023, last modified on 22 Aug 2024
The SPaiK software is covered by the MIT license.

'''

import numpy as np
import os
from rlscore.utilities.pairwise_kernel_operator import PairwiseKernelOperator


def file_exists(filename):
    """ Check if the given file exists in the current working directory.

    Args:
        filename (str): The name of the file to check for.

    Returns:
        bool: True if the file exists, False otherwise.
    """
    if os.path.exists(filename):
        return True
    else:
        return False


def naive_kronecker_kernel(K1, K2, rows1, cols1, rows2, cols2):
    """ Compute explicit Knonecker kernel matrix K.                 
    
    In case of complete data, naive_kronecker_kernel == np.kron

    Args:
        K1,K2:  input kernel matrices (e.g. drug and target kernels), 
                shape = [n_samples#, n_samples#].
                In addition, X1,X2 (shape= [n_samples#, n_features#] 
                can be used but the dimensions must match.
        rows1,col1,row2,col2: row and column indices of label matrix 
                (e.g. rows# are drug indices and cols# are target indices),
                shape = [n_train_pairs]
    
    Returns:
        K:      An explicit Kronecer kernel matrix, shape = [n_train_pairs,n_train_pairs]
    """
    assert len(rows1) == len(cols1) 
    assert len(rows2) == len(cols2)
    o = len(rows1)
    p = len(rows2)
    K = np.zeros((o, p))
    for i in range(o):
        for j in range(p):
            k_ij = K1[rows1[i], rows2[j]]
            g_ij = K2[cols1[i], cols2[j]]
            val = k_ij * g_ij
            K[i,j] = val
    return K

def pko_kronecker(K1, K2, rows1, cols1, rows2, cols2):
    """ Compute pairwise Knonecker kernel operator. """

    pko = PairwiseKernelOperator([K1], [K2], [rows1], [cols1], [rows2], [cols2], weights=[1.0])
    return pko


def pko_linear(K1, K2, rows1, cols1, rows2, cols2):
    """ Compute linear kernel operator. """
    
    n, d = K1.shape#[0]
    m, k = K2.shape#[0]
    O1 = np.ones((n, d))
    O2 = np.ones((m, k))
    pko = PairwiseKernelOperator([K1, O1], [O2, K2], [rows1, rows1], [cols1, cols1], [rows2, rows2], [cols2, cols2], weights=[1.0, 1.0])
    return pko


def K_to_dense(K, row_inds, col_inds):
    """ Transform K into a dense matrix, such that column and row indices are surjective """

    rows, rows_inverse = np.unique(row_inds, return_inverse=True)
    cols, cols_inverse = np.unique(col_inds, return_inverse=True)
    K = np.array(K[np.ix_(rows, cols)])
    row_indices = np.arange(len(rows))
    col_indices = np.arange(len(cols))
    row_inds = np.array(row_indices[rows_inverse])
    col_inds = np.array(col_indices[cols_inverse])
    return K, row_inds, col_inds


def group_performance(measure, y, y_predicted, group_ids):
    """ Compute row- or columnwise C-index
    
        The group_ids is either drug_inds or target_inds depending on whether a row or a column index is needed.
        The return value is the average of row or column indices.
    """

    performances = []
    for i in set(group_ids):
        y_subset = y[group_ids == i]
        y_predicted_subset = y_predicted[group_ids == i]
        if len(set(y_subset)) > 1:
            performances.append(measure(y_subset, y_predicted_subset))
    performances_average = np.mean(performances)
    return(performances_average)


if __name__ == "__main__":
    print("Warning: pkl_utility.py only consists utility programs for stochPKL.\n It is not supposed to run separately.")
    
