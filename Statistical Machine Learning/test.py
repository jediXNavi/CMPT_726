import numpy as np

a = np.random.randn(10)

b = np.random.choice([0,1],size=4,p=[0.75,0.25])
# print(a)
# print(b)

list_val = [[0,1] for _ in range(4)]
# print(list_val)

yTilde = np.array([[1, 1, 1, 84, 1, 1]]).reshape(6,1)