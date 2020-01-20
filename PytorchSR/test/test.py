from __future__ import print_function
import torch
import numpy as np

a = torch.randn(82,1,1)
b = np.diff(a)
print(np.shape(b))
print(torch.cuda.is_available())

device =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')
a = torch.ones((1, 1)).to(device)

#Additional Info when using cuda
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')