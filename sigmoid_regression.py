
import assignment1 as a1
import numpy as np
import matplotlib.pyplot as plt

(countries, features, values) = a1.load_unicef_data()

#Run the normalized or the unnormalized code:

targets = values[:,1]
x = values[:,10]
#x = a1.normalize_data(x)

N_TRAIN = 100
x_train = x[0:N_TRAIN,:]
x_test = x[N_TRAIN:,:]
t_train = targets[0:N_TRAIN]
t_test = targets[N_TRAIN:]
degree = 1
reg_lambda = 0

(w, train_err) = a1.linear_regression(x_train, t_train, 'sigmoid',reg_lambda, degree,s=2000)
err = a1.evaluate_regression(x_test, t_test, w,'sigmoid',reg_lambda, degree , s=2000)
print(train_err)
print(err)

x1_ev = np.linspace(np.asscalar(min(min(x_train), min(x_test))),
                     np.asscalar(max(max(x_train), max(x_test))), num=500)
x1_ev.reshape(500, 1)
phi_train = a1.design_matrix(x1_ev, 'sigmoid', degree, s=2000, bias=1)
y1_ev = phi_train.dot(w)

plt.plot(x_train, t_train, 'yo')
plt.plot(x_test, t_test, 'bo')
plt.plot(x1_ev, y1_ev, 'g-o')
plt.xlabel('GNU Per Capita')
plt.ylabel('Model')
plt.show()

