# -*- coding: utf-8 -*-
"""
Created on Wed Oct 15 21:04:18 2014

@author: carlos
"""

import numpy as np
import scipy as sc
import matplotlib.pylab as plt
from scipy import ndimage, misc

#%%dibujar un seno
ts = 0.02
f = 1/5.
time_vec = np.arange(0, 20, ts)
print(len(time_vec))
sig = np.sin(2 * np.pi*f * time_vec) + 0.5 * np.random.randn(time_vec.size)
plt.figure()
plt.plot(time_vec,sig)
plt.xlabel('Time[s]')
plt.ylabel('Amplitude')

#%% Figura Lena
plt.figure()
lena = misc.lena()
type(lena)
plt.imshow(lena, cmap = cm.gray)

#%% PCA
from sklearn.datasets import load_boston

data = load_boston()
print data.keys()
print data.data.shape
print data.target.shape

plt.hist(data.target)
plt.xlabel('price ($1000)')
plt.ylabel('count')