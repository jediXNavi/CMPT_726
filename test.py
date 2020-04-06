import pandas as pd
import numpy as np
from scipy import nanmean

fname = 'SOWC_combined_simple.csv'

# Uses pandas to help with string-NaN-numeric data.
data = pd.read_csv(fname, na_values='_', encoding='latin1')
# Strip countries title from feature names.
features = data.axes[1][1:]
# Separate country names from feature values.
countries = data.values[:, 0]
values = data.values[:, 1:]
# Convert to numpy matrix for real.
values = np.asmatrix(values, dtype='float64')

# Modify NaN values (missing values).
mean_vals = nanmean(values, axis=0)
inds = np.where(np.isnan(values))
values[inds] = np.take(mean_vals, inds[1])

i = np.array([np.ones((100,1), dtype=int)])
# x1= np.insert(i,)
