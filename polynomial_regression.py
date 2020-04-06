#!/usr/bin/env python

import assignment1 as a1
import numpy as np
import matplotlib.pyplot as plt

def unnormalized_data(degree, reg_lambda,x,targets,bias):
    x_train = x[0:N_TRAIN, :]
    x_test = x[N_TRAIN:, :]
    t_train = targets[0:N_TRAIN]
    t_test = targets[N_TRAIN:]
    bias = bias
    degree = degree
    reg_lambda = reg_lambda
    # Complete the linear_regression and evaluate_regression functions of the assignment1.py
    # Pass the required parameters to these functions

    list_t_err = []
    list_test_err = []
    for i in range(1, degree + 1):
        (w, tr_err) = a1.linear_regression(x_train, t_train, 'polynomial', reg_lambda, bias, degree=i)
        list_t_err.append(tr_err)

        err = a1.evaluate_regression(x_test, t_test, w, 'polynomial', reg_lambda, bias, degree=i)
        list_test_err.append(err)

    print(*list_test_err)
    x = np.arange(1, degree + 1)
    # Produce a plot of results.
    plt.rcParams.update({'font.size': 15})
    plt.plot(x, list_t_err)
    plt.plot(x, list_test_err)
    plt.ylabel('RMS')
    plt.legend(['Training error', 'Testing error'])
    plt.title('Fit with polynomials with unnormalized data')
    plt.xlabel('Polynomial degree')
    plt.show()


def normalized_data(degree, reg_lambda,x,targets,bias):
    x = a1.normalize_data(x)

    x_train = x[0:N_TRAIN, :]
    x_test = x[N_TRAIN:, :]
    t_train = targets[0:N_TRAIN]
    t_test = targets[N_TRAIN:]
    degree = degree
    reg_lambda = reg_lambda
    bias = bias
    # Complete the linear_regression and evaluate_regression functions of the assignment1.py
    # Pass the required parameters to these functions

    list_t_err = []
    list_test_err = []
    for i in range(1, degree + 1):
        (w, tr_err) = a1.linear_regression(x_train, t_train, 'polynomial', reg_lambda, bias, degree=i)
        list_t_err.append(tr_err)

        err = a1.evaluate_regression(x_test, t_test, w, 'polynomial', reg_lambda, bias, degree=i)
        list_test_err.append(err)

    print(*list_test_err)
    x = np.arange(1, degree + 1)
    # Produce a plot of results.
    plt.rcParams.update({'font.size': 15})
    plt.plot(x, list_t_err)
    plt.plot(x, list_test_err)
    plt.ylabel('RMS')
    plt.legend(['Training error', 'Testing error'])
    plt.title('Fit with polynomials, with normalized inputs')
    plt.xlabel('Polynomial degree')
    plt.show()


if __name__ == '__main__':
    (countries, features, values) = a1.load_unicef_data()

    N_TRAIN = 100
    targets = values[:, 1]
    x = values[:, 7:]
    unnormalized_data(6,0,x,targets,1)
    normalized_data(6,0,x,targets,1)



