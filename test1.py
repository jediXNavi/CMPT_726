import pandas as pd
import numpy as np
from scipy import nanmean

a = np.array([[1, 1], [2, 2], [3, 3]])
for i in range(0,5):
    b = np.insert(a, 2,pow(2,i),axis=1)
    b=a
print(b)