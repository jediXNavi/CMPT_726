#!/usr/bin/env python

import assignment1 as a1
import numpy as np
import matplotlib.pyplot as plt
import polynomial_regression_1d as p


# Plot a curve showing learned function.
# # Use linspace to get a set of samples on which to evaluate
# x1_ev = np.linspace(np.asscalar(min(min(si.x_train), min(si.x_test))),
#                         np.asscalar(max(max(si.x_train), max(si.x_test))), num=500)
# x1_ev.reshape(500, 1)
# phi_train = a1.design_matrix(x1_ev, 'sigmoid', degree=0, mu=10000, s=2000)
# y1_ev = phi_train.dot(si.w)

x2_ev = np.linspace(np.asscalar(min(p.x_train_11)), np.asscalar(max(p.x_train_11)), num=500)
# x2_ev = np.linspace(np.asscalar(min(p.x_train), min(p.x_test))),
#                    np.asscalar(max(max(p.x_train), max(p.x_test))), num=500)
x2_ev = x2_ev.reshape(500,1)
phi_poly_train = a1.design_matrix(x2_ev, 'polynomial', degree=3, s=0)
y2_ev = phi_poly_train.dot(p.w)

#x1_ev = np.linspace(np.asscalar(min(si.x_train)), np.asscalar(max(si.x_train)), num=500)

#x1_ev = np.linspace(0, 10, num=500)
#x2_ev = np.linspace(0, 10, num=50)

# TO DO::
# Perform regression on the linspace samples.
# # Put your regression estimate here in place of y_ev.
#y1_ev = np.random.random_sample(x1_ev.shape)
#y2_ev = np.random.random_sample(x2_ev.shape)
# y1_ev = 100*np.sin(x1_ev)
# y2_ev = 100*np.sin(x2_ev)

plt.plot(p.x_train, p.t_train, 'yo')
plt.plot(x2_ev, y2_ev, 'g-o')
plt.xlabel('GNU Per Capita(mu-10000)')
plt.ylabel('Model')
plt.show()