# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 11:21:10 2014

@author: obarquero
"""

##Here, Carlos, you should import the modules needed.
import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
from sklearn import cross_validation, linear_model

fname = 'wild_boar_age_2.csv'
wild_boar_data = pd.read_csv(fname,delimiter = ";") # this reads the data using panda
#print str(wild_boar_data)

wb_data = wild_boar_data.as_matrix()
#print wb_data
X = wb_data[:,2:]
print X
Y = wb_data[:,0]
print Y
#%%
##Let's suppose data are in X

#1) Create a PCA object

pca_ex = PCA()


#2) Let's compute real pca

pca_ex.fit(X)

#You should look at the help of pca from sklearn page and verify where are the projected
#data. I think these are called scores in the book.

Z = pca_ex.transform(X)

#Now let's compute PCR by cross-validation

scores = list()
scores_std = list()

n_features = np.shape(Z)[1]

#for over all n_features
for m in range(n_features):
    
    #Let compute a linear regression Y = w(T)Z using the first n_features
    clf = linear_model.LinearRegression()
    this_scores = cross_validation.cross_val_score(clf,Z[:,:m+1],Y,n_jobs = -1)
    
    #Estimate the score using cross validation. 
    #You should check which is the score used
    
    scores.append(np.mean(this_scores))
    scores_std.append(np.std(this_scores))

#Now plot all the scores and select which is the number of components
    
        


