# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 19:18:38 2014

@author: carlos
"""

#import numpy as np
import os as os
#import csv as csv
import pandas as pd
from sklearn.decomposition import PCA
#import sklearn.preprocessing as prepro
#import statsmodels.formula.api as sm
#import statsmodels.stats.outliers_influence as vif
#import matplotlib.pylab as plt
from sklearn.cross_validation import train_test_split
#from sklearn.grid_search import GridSearchCV
#import sklearn.metrics as metrics
#from sklearn.svm import SVR
#from sklearn.ensemble import GradientBoostingRegressor
#from sklearn.cross_decomposition import PLSRegression
#import pydna
#from Bio.Restriction import BamHI
#pydna.pcr
from sklearn import datasets, linear_model
from sklearn import cross_validation

#plt.ioff()
path = '/home/carlos/TFG-carlos-biedma-tapia'
os.chdir(path)
fname = 'hitters_data2.csv'
#wild_boar_data = pd.read_csv(fname,delimiter = ";") # this reads the data using panda
wild_boar_data = pd.read_table(fname,delimiter = ";")
#print str(wild_boar_data)


wb_data = wild_boar_data.as_matrix()
#print(wb_data)

#Scatter plots for each variable
#for i in wild_boar_data.columns:
 #   pd.tools.plotting.scatter_plot(wild_boar_data,i,'AtBat')
#%%
X = wb_data[:,2:]
#print (X)

#print "------------------------------------"
Y = wb_data[:,0]
print(Y)


#1) Create a PCA object
pca_ex = PCA()

#2) Let's compute real pca
pca_ex.fit(X)

Z = pca_ex.transform(X)
#%%
#print Z

scores = list()
scores_std = list()

n_features = np.shape(Z)[1]
#print(n_features)
#print np.shape(Z)



#%%
#for over all n_features
for m in range(n_features):
    #print Z[:m+1]
    print "----------------------------------------------------------------------------"
    #Let compute a linear regression Y = w(T)Z using the first n_features
    clf  = linear_model.LinearRegression()
    
    this_scores = cross_validation.cross_val_score(clf,Z[:,:m+1],Y,n_jobs = -1)
    #print this_scores
    #Estimate the score using cross validation. 
    #You should check which is the score used
    scores.append(np.mean(this_scores))
    scores_std.append(np.std(this_scores)) #desviación estándar
    #print scores_std

plot(scores)


#Now plot all the scores and select which is the number of components





