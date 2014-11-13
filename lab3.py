# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 19:18:38 2014

@author: carlos
"""
import numpy as np
import os as os
import csv as csv
import pandas as pd
from sklearn.decomposition import PCA
import sklearn.preprocessing as prepro
import statsmodels.formula.api as sm
import statsmodels.stats.outliers_influence as vif
import matplotlib.pylab as plt
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
import sklearn.metrics as metrics
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.cross_decomposition import PLSRegression
import pydna
from Bio.Restriction import BamHI
#pydna.pcr



plt.ioff()
fname = 'hitters_data.csv'
wild_boar_data = pd.read_csv(fname,delimiter = ";") # this reads the data using panda
#print(wild_boar_data)

#remove the first column which is only an id, a lo mejor la tengo que eliminar a pelo
#wild_boar_data = wild_boar_data.drop('Unnamed: 0', 1)

wb_data = wild_boar_data.as_matrix()
#print(wb_data)

X = wb_data[:,2:]
#print (X)
print "------------------------------------"
Y = wb_data[:,0]
#print(Y)

pca_ex = PCA()
pca_ex.fit(X)
#Z = pca_ex.transform(X)


print "hola mundo"
