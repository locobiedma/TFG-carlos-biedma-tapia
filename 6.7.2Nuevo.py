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
from sklearn.cross_validation import KFold
from sklearn import datasets, linear_model
from sklearn import cross_validation
import sklearn.preprocessing as prepro
from sklearn import metrics
from matplotlib.pyplot import *
from sklearn.cross_decomposition import PLSRegression

#plt.ioff()
path = '/home/carlos/TFG-carlos-biedma-tapia'
#path = '/home/carlos/TFG-carlos-biedma-tapia'
path = './'
os.chdir(path)
#fname = 'hitters_data2.csv'
#fname = 'hitters_data3.csv'
#fname = 'hitters_data_original.csv'
fname = 'hitters.csv'
wild_boar_data = pd.read_csv(fname,delimiter = ";") # this reads the data using panda
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
Y = wb_data[:,-2]
#Normalizar datos
scaler = prepro.StandardScaler()
scaler.fit(wb_data)
wb_data = scaler.transform(wb_data)
#%%

X = np.concatenate((wb_data[:,0:-2],wb_data[:,-1:]),axis = 1) #¿Con esto te quitas la última columna?
#print(X)
X2 = wb_data[:,:]
#print X2
#%%
#print "------------------------------------"
#Y = wb_data[:,-2]
#print(Y)
#%%
pls_wild_b = PLSRegression(n_components = 19) #No sé que número poner.
pls_wild_b.fit(X,Y)
Z = pls_wild_b.transform(X)


scores = list() 
scores_std = list()

n_features = np.shape(X)[1]

print(n_features) #¿Aquí no debería ser un numero menor de 20?
print np.shape(X)

X_orig = X.copy()
Y_orig = Y.copy()

X,X_test_tot, Y, Y_test_tot = cross_validation.train_test_split(X,Y,test_size = 0.5,random_state = 0)
N = np.shape(X)[0]
#%%
#for over all n_features
for num_comp in range(n_features):
    
    kf = KFold(N,n_folds=10)
    aux_scores = list()
    for train, test in kf:
        X_train, X_test, y_train, y_test = X[train], X[test], Y[train], Y[test]
          
        if num_comp == 0:
            y_pred = np.mean(y_test)
            y_pred = y_pred* np.ones(np.shape(y_test))
            aux_scores.append(metrics.mean_squared_error(y_test,y_pred))
        
        else:
            pls_foo = PLSRegression(n_components = num_comp)                        
            pls_foo.fit(X_train,y_train)
            y_pred = pls_foo.predict(X_test)
        
            #obtaing the score
            this_score = metrics.mean_squared_error(y_test,y_pred)
            aux_scores.append(this_score)
        
    #compute the total score for this number of components
    scores.append(np.mean(aux_scores))
    scores_std.append(np.std(aux_scores))
    #desviación estándar
    #print scores_std
#%%
plt.plot(scores)
xlabel('Componentes')
ylabel("$MSE$")
title("6.7.2")
plt.show()

num_comp = np.argmin(scores)

pls_pred = PLSRegression(n_components =2)
pls_pred.fit(X,Y)
y_pred_test = pls_pred.predict(X_test_tot)

print "MSE test = " + str(metrics.mean_squared_error(Y_test_tot,y_pred_test))
#print "% Variance Explained (cumulative)"
#print np.cumsum(pca_ex.explained_variance_ratio_)


#pca_ex.explained_variance_ratio_[0]



