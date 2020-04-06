#!/usr/bin/env python

import assignment1 as a1
import numpy as np
import matplotlib.pyplot as plt

(countries, features, values) = a1.load_unicef_data()
""",0.01,0.1,1,10,100,1000,10000"""

list_t_err = []
list_test_err = []
average = {}

targets = values[:, 1]
x = values[:, 7:]
x = a1.normalize_data(x)
bias = 1

N_TRAIN = 100
x = x[0:N_TRAIN, :]
targets = targets[0:N_TRAIN]

def lambda_regulator(reg_lambda, degree):

    for i in range(0,100,10):
        if i>0:
            x_train = x[(i+10):,:]
            t_train = targets[(i+10):,:]
            x_train = np.concatenate((x[0:i,:],x_train),axis=0)
            t_train = np.concatenate((targets[0:i,:],t_train),axis=0)
        x_valid = x[i:(i+10),:]   #Validation set
        t_valid = targets[i:(i+10),:]
        if i==0:
            x_train = x[(i+10):,:]   #Training set
            t_train = targets[(i+10):,:]
        #print(concat)

        # # Complete the linear_regression and evaluate_regression functions of the assignment1.py
        # # Pass the required parameters to these functions
        #for reg_lambda in lambda_values:
        #print(reg_lambda)
        (w, tr_err) = a1.linear_regression(x_train,t_train,'polynomial',reg_lambda,bias, degree)
        err = a1.evaluate_regression(x_valid, t_valid, w, 'polynomial',reg_lambda,bias, degree)
        list_test_err.append(err)
    average_rms = np.mean(list_test_err)
    average.update({reg_lambda:average_rms})
lambda_regulator(0,2)
list_test_err=[]
lambda_regulator(0.01,2)
list_test_err=[]
lambda_regulator(0.1,2)
list_test_err=[]
lambda_regulator(1,2)
list_test_err=[]
lambda_regulator(10,2)
list_test_err=[]
lambda_regulator(100,2)
list_test_err=[]
lambda_regulator(1000,2)
list_test_err=[]
lambda_regulator(10000,2)



print(average)


lists = sorted(average.items()) # sorted by key, return a list of tuples
print(lists)

x, y = zip(*lists) # unpack a list of pairs into two tuples

# Produce a plot of results.
plt.semilogx(x,y)
plt.ylabel('Validation Test Error')
plt.legend(['Validation Test error'])
plt.title('Regularization with degree 2(with 10-fold Cross Validation)')
plt.xlabel('Lambda values')
plt.show()
