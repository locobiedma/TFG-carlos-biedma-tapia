# -*- coding: utf-8 -*-
"""
Created on Tue Jan 27 23:01:40 2015

@author: carlos
"""

#import numpy as np
#import csv as csv
#import sklearn.preprocessing as prepro
#import statsmodels.formula.api as sm
#import statsmodels.stats.outliers_influence as vif
#from sklearn.grid_search import GridSearchCV
#import sklearn.metrics as metrics
#from sklearn.svm import SVR
#from sklearn.ensemble import GradientBoostingRegressor
#from sklearn.cross_decomposition import PLSRegression
#import pydna
#from Bio.Restriction import BamHI
#pydna.pcr
import os as os
import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pylab as plt
from sklearn.cross_validation import train_test_split
from sklearn import datasets, linear_model
from sklearn import cross_validation
import sklearn.preprocessing as prepro
from sklearn import metrics
from matplotlib.pyplot import *
from sklearn.cross_decomposition import PLSRegression

#plt.ioff()
path = '/home/carlos/TFG-carlos-biedma-tapia'
#path = '/home/carlos/TFG-carlos-biedma-tapia'
os.chdir(path)
#fname = 'hitters_data2.csv'
#fname = 'hitters_data3.csv'
fname = 'hitters_data_original.csv'
wild_boar_data = pd.read_csv(fname,delimiter = ",") # this reads the data using panda
#wild_boar_data = pd.read_table(fname,delimiter = ";")
#print str(wild_boar_data)

#Let's codify the qualititatve variables as numerics.
#League: N-0; A-1; Division and New League
league_map = {'A':0,'N':1}
wild_boar_data['League'] = wild_boar_data.League.map(league_map)

division_map = {'W':0,'E':1}
wild_boar_data['Division'] = wild_boar_data.Division.map(division_map)

newleague_map = {'A':0,'N':1}
wild_boar_data['NewLeague'] = wild_boar_data.NewLeague.map(newleague_map)


wb_data = wild_boar_data.as_matrix()
#Normalizar datos
scaler = prepro.StandardScaler()
scaler.fit(wb_data)
wb_data = scaler.transform(wb_data)
#%%

X = np.concatenate((wb_data[:,0:-2],wb_data[:,0:-1:]),axis = 1) #¿Con esto te quitas la última columna?
#print(X)
X2 = wb_data[:,:]
#print X2
#%%
#print "------------------------------------"
Y = wb_data[:,-2]
#print(Y)
#%%
pls_wild_b = PLSRegression(n_components = 20) #No sé que número poner.
pls_wild_b.fit(X2,Y)
Z = pls_wild_b.transform(X2)


scores = list() 
scores_std = list()

n_features = np.shape(Z)[1]
print(n_features) #¿Aquí no debería ser un numero menor de 20?
print np.shape(Z)



#%%
#for over all n_features
for m in range(n_features):
    print m
    #print Z[:m+1]
    print "----------------------------------------------------------------------------"
    #Let compute a linear regression Y = w(T)Z using the first n_features
    clf  = linear_model.LinearRegression()
    
    this_scores = cross_validation.cross_val_score(clf,Z[:,:m+1],Y,scoring = 'mean_squared_error',n_jobs = -1)
      
      
    #scores = cross_validation.cross_val_score(svr, diabetes.data, diabetes.target, cv=5, scoring='mean_squared_error')  
    #hay que pasarle tres vectores... El tercero es opcional  
              #prueba = mean_squared_error(Z[:,:m+1],Y)
    #score(X, y[, sample_weight])
    #print this_scores
    #Estimate the score using cross validation. 
    #You should check which is the score used
    scores.append(np.mean(this_scores))
    print "scores es: " + str(np.mean(this_scores))
    scores_std.append(np.std(this_scores)) #desviación estándar
    #print scores_std
#%%
plt.plot(scores)
xlabel('Componentes')
ylabel("$MSE$")
title("6.7.2")
plt.show()

print "% Variance Explained (cumulative)"
print np.cumsum(pca_ex.explained_variance_ratio_)


pca_ex.explained_variance_ratio_[0]



