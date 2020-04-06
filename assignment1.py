"""Basic code for assignment 1."""

import numpy as np
import pandas as pd
from math import e
from scipy import nanmean

def load_unicef_data():
    """Loads Unicef data from CSV file.

    Retrieves a matrix of all rows and columns from Unicef child mortality
    dataset.

    Args:
      none

    Returns:
      Country names, feature names, and matrix of values as a tuple (countries, features, values).

      countries: vector of N country names
      features: vector of F feature names
      values: matrix N-by-F
    """
    fname = 'SOWC_combined_simple.csv'

    # Uses pandas to help with string-NaN-numeric data.
    data = pd.read_csv(fname, na_values='_', encoding='latin1')
    # Strip countries title from feature names.
    features = data.axes[1][1:]
    # Separate country names from feature values.
    countries = data.values[:,0]
    values = data.values[:,1:]
    # Convert to numpy matrix for real.
    values = np.asmatrix(values,dtype='float64')

    # Modify NaN values (missing values).
    mean_vals = nanmean(values, axis=0)
    inds = np.where(np.isnan(values))
    values[inds] = np.take(mean_vals, inds[1])
    return (countries, features, values)


def normalize_data(x):
    """Normalize each column of x to have mean 0 and variance 1.
    Note that a better way to normalize the data is to whiten the data (decorrelate dimensions).  This can be done using PCA.

    Args:
      input matrix of data to be normalized

    Returns:
      normalized version of input matrix with each column with 0 mean and unit variance

    """
    mvec = x.mean(0)
    stdvec = x.std(axis=0)
    
    return (x - mvec)/stdvec
    


def linear_regression(x, t, basis, reg_lambda, bias, degree=0, mu=0, s=1):
    """Perform linear regression on a training set with specified regularizer lambda and basis

    Args:
      x is training inputs
      t is training targets
      reg_lambda is lambda to use for regularization tradeoff hyperparameter
      basis is string, name of basis to use
      degree is degree of polynomial to use (only for polynomial basis)
      mu,s are parameters of Gaussian basis

    Returns:
      w vector of learned coefficients
      train_err RMS error on training set
      """


    # Construct the design matrix.
    # Pass the required parameters to this function
    phi = design_matrix(x,basis,degree,s,bias)
    phi_inv = np.linalg.pinv(phi)

    # Learning Coefficients
    if reg_lambda > 0:
        w = np.linalg.inv(reg_lambda * np.identity(phi.shape[1]) + np.transpose(phi).dot(phi)) \
            .dot(np.transpose(phi)).dot(t)
        y_train= phi.dot(w)

        t_train_error = t - y_train
        train_err = np.sqrt(np.mean(np.square(t_train_error)))

    else:
        w = phi_inv.dot(t)
        y_train = phi.dot(w)

        t_train_error = t - y_train
        train_err = np.sqrt(np.mean(np.square(t_train_error)))



    # Measure root mean squared error on training data.
    # train_err =
    #
    return (w, train_err)



def design_matrix(input,basis,degree,s,bias):
    """ Compute a design matrix Phi from given input datapoints and basis.
    Args:
        ?????

    Returns:
      phi design matrix
    """
    if basis == 'polynomial':
        if bias == 1:
            if degree==1:
                a = np.ones((len(input), 1), dtype=int)
                b = np.array(input, dtype=float)
                phi_matrix = np.concatenate((a, b), axis=1)
            elif degree>1:
                a = np.ones((len(input), 1), dtype=float)
                b = np.array(input, dtype=float)
                phi_matrix = np.concatenate((a,b), axis=1)
                for i in range(2, degree + 1):
                    phi_matrix = np.concatenate((phi_matrix, np.power(b, i)), axis=1)

            phi = phi_matrix

        elif bias== 0:
            if degree == 1:
                #a = np.ones((len(input), 1), dtype=int)
                b = np.array(input, dtype=float)
                phi_matrix = b
            elif degree > 1:
                #a = np.ones((len(input), 1), dtype=float)
                b = np.array(input, dtype=float)
                phi_matrix = b
                for i in range(2, degree+1):
                    phi_matrix = np.concatenate((phi_matrix, np.power(b, i)), axis=1)

        phi = phi_matrix


    elif basis == 'sigmoid':
        read_x = np.array(input, dtype=np.float64)
        phi_matrix1 = [] #initializing 2 empty lists to hold the phi values
        phi_matrix2 = []
        for i in np.nditer(read_x):
            phi_calc1 = np.exp(-np.square((i - 100)/(np.sqrt(2)*s)))
            phi_calc2 = np.exp(-np.square((i - 10000)/(np.sqrt(2)*s)))
            phi_matrix1.append(phi_calc1)
            phi_array1 = np.asmatrix(phi_matrix1)
            phi_x1 = np.transpose(phi_array1)
            phi_matrix2.append(phi_calc2)
            phi_array2 = np.asmatrix(phi_matrix2)
            phi_x2 = np.transpose(phi_array2)
        phi_12 = np.concatenate((phi_x1,phi_x2),axis=1)
        phi_one = np.ones((len(input),1),dtype = int)
        phi_matrix = np.concatenate((phi_one,phi_12),axis=1)
        phi = phi_matrix

    else: 
        assert(False), 'Unknown basis %s' % basis

    return phi


def evaluate_regression(x,t,w,basis,reg_lambda,bias,degree=0, mu=0, s=1):
    """Evaluate linear regression on a dataset.

    Args:
      ?????

    Returns:
      t_est values of regression on inputs
      err RMS error on training set if t is not None
      """
    phi = design_matrix(x,basis,degree,s,bias)

    if reg_lambda > 0:
        y_test = phi.dot(w)
        t_test_error = t - y_test
        err = np.sqrt(np.mean(np.square(t_test_error)))
    else:
        y_test = phi.dot(w)
        t_test_error = t - y_test
        err = np.sqrt(np.mean(np.square(t_test_error)))

    return err


