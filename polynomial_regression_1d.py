#!/usr/bin/env python

import assignment1 as a1
import numpy as np
import matplotlib.pyplot as plt


def biased_output(targets,values,degree,reg_lambda,bias,s):
    list_t_err = []
    list_test_err = []
    reg_lambda = reg_lambda

    for i in range(7,15):
        x = values[:,i]
        #x = a1.normalize_data(x)

        N_TRAIN = 100
        x_train = x[0:N_TRAIN,:]
        x_test = x[N_TRAIN:,:]
        t_train = targets[0:N_TRAIN]
        t_test = targets[N_TRAIN:]


    # Complete the linear_regression and evaluate_regression functions of the assignment1.py
    # Pass the required parameters to these functions

        (w, tr_err) = a1.linear_regression(x_train,t_train,'polynomial',reg_lambda, bias, degree)
        list_t_err.append(tr_err)

        err = a1.evaluate_regression(x_test, t_test,w, 'polynomial', reg_lambda, bias, degree)
        list_test_err.append(err)

        if i in (10, 11, 12):
            x2_ev = np.linspace(np.asscalar(min(x_train)), np.asscalar(max(x_train)), num=500)
            # x2_ev = np.linspace(np.asscalar(min(p.x_train), min(p.x_test))),
            # np.asscalar(max(max(p.x_train), max(p.x_test))), num=500)
            x2_ev = x2_ev.reshape(500, 1)
            phi_poly_train = a1.design_matrix(x2_ev, 'polynomial', degree, s, bias)
            y2_ev = phi_poly_train.dot(w)

            plt.plot(x_train, t_train, 'yo')
            plt.plot(x_test,t_test,'ro')
            plt.plot(x2_ev, y2_ev, 'g-o')
            plt.xlabel('Feature '+str(i+1))
            plt.ylabel('Model')
            plt.legend(['Train Data Points','Test Data Points','Learned Model'])
            plt.show()

    print("for biased, the training error is", list_t_err)
    print("for biased, the test error is", list_test_err)
    x = np.arange(8)
    width = 0.3
    # Produce a plot of results.
    plt.rcParams.update({'font.size': 15})
    plt.ylabel('RMS')
    plt.bar(x + width / 2, list_t_err, width)
    plt.bar(x - width / 2, list_test_err, width)
    plt.xticks(x)
    plt.legend(['Training error', 'Testing error'])
    plt.title('Fit with 8 features, with the biased parameter')
    plt.xlabel('Feature Parameter')
    plt.show()


def unbiased_output(target, values, degree,reg_lambda, bias,s):
    list_t_err = []
    list_test_err = []
    reg_lambda = reg_lambda
    degree=degree

    targets = values[:, 1]
    for i in range(7,15):
        x = values[:, i]
        # x = a1.normalize_data(x)

        N_TRAIN = 100
        x_train = x[0:N_TRAIN, :]
        x_test = x[N_TRAIN:, :]
        t_train = targets[0:N_TRAIN]
        t_test = targets[N_TRAIN:]

        # Complete the linear_regression and evaluate_regression functions of the assignment1.py
        # Pass the required parameters to these functions

        (w, tr_err) = a1.linear_regression(x_train, t_train, 'polynomial', reg_lambda, bias, degree)
        list_t_err.append(tr_err)

        err = a1.evaluate_regression(x_test, t_test, w, 'polynomial', reg_lambda, bias, degree)
        list_test_err.append(err)

        if i in (10, 11, 12):
            x2_ev = np.linspace(np.asscalar(min(x_train)), np.asscalar(max(x_train)), num=500)
            # x2_ev = np.linspace(np.asscalar(min(p.x_train), min(p.x_test))),
            # np.asscalar(max(max(p.x_train), max(p.x_test))), num=500)
            x2_ev = x2_ev.reshape(500, 1)
            phi_poly_train = a1.design_matrix(x2_ev, 'polynomial', degree, s, bias)
            y2_ev = phi_poly_train.dot(w)

            plt.plot(x_train, t_train, 'yo')
            plt.plot(x_test, t_test, 'ro')
            plt.plot(x2_ev, y2_ev, 'g-o')
            plt.xlabel('Feature ' + str(i + 1))
            plt.ylabel('Model')
            plt.legend(['Train Data Points', 'Test Data Points', 'Learned Model'])
            plt.show()
    print("for unbiased, the training error is", list_t_err)
    print("for unbiased, the test error is", list_test_err)
    x = np.arange(8)
    width = 0.3
    # Produce a plot of results.
    plt.rcParams.update({'font.size': 15})
    plt.ylabel('RMS')
    plt.bar(x + width / 2, list_t_err, width)
    plt.bar(x - width / 2, list_test_err, width)
    plt.xticks(x)
    plt.legend(['Training error', 'Testing error'])
    plt.title('Fit with 8 features, without bias parameter')
    plt.xlabel('Feature Parameter')
    plt.show()


if __name__ == '__main__':
    (countries, features, values) = a1.load_unicef_data()

    targets = values[:,1]
    biased_output(targets, values, 3, 0, 1, 1)
    unbiased_output(targets, values, 3, 0, 0, 1)





