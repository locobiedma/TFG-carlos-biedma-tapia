# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 19:18:38 2014

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

#plt.ioff()
#path = '/home/carlos/TFG-carlos-biedma-tapia'
#path = '/home/carlos/TFG-carlos-biedma-tapia'
path = '/Users/obarquero/Escritorio/tfg_carlos_biedma/TFG-carlos-biedma-tapia/'
path = './'
os.chdir(path)
#fname = 'hitters_data2.csv'
#fname = 'hitters_data3.csv'
fname = 'hitters.csv'
wild_boar_data = pd.read_csv(fname,delimiter = ";") # this reads the data using panda
#wild_boar_data = pd.read_table(fname,delimiter = ";")
print str(wild_boar_data)

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


#Scatter plots for each variable
#for i in wild_boar_data.columns:
 #   pd.tools.plotting.scatter_plot(wild_boar_data,i,'AtBat')
#%%

X = np.concatenate((wb_data[:,0:-2],wb_data[:,-1:]),axis = 1) #¿Con esto te quitas la última columna?
#print(X)
X2 = wb_data[:,:]
print X2
#%%
#print "------------------------------------"
#Y = wb_data[:,-2]
print(Y)
#%%

#1) Create a PCA object
pca_ex = PCA()

#2) Let's compute real pca
pca_ex.fit(X) #Este creo que no lo uso para nada; Fit the model with X.

Z = pca_ex.transform(X) #Apply the dimensionality reduction on X; 
#X is projected on the first principal components previous extracted from a training set.
#%%
#print Z

scores = list() 
scores_std = list()

n_features = np.shape(Z)[1]
print(n_features) 
print np.shape(Z)



#%%
#for over all n_features
for m in range(n_features+1):
    print m
    #print Z[:m+1]
    print "----------------------------------------------------------------------------"
    #Let compute a linear regression Y = w(T)Z using the first n_features
    clf  = linear_model.LinearRegression()
    if m == 0:
        unos = unos = np.ones((np.shape(Y)[0],1))
        this_scores = cross_validation.cross_val_score(clf,unos,Y,scoring = 'mean_squared_error',n_jobs = -1)
    else:
        this_scores = cross_validation.cross_val_score(clf,Z[:,:m],Y,scoring = 'mean_squared_error',n_jobs = -1)
      
      
    #scores = cross_validation.cross_val_score(svr, diabetes.data, diabetes.target, cv=5, scoring='mean_squared_error')  
    #hay que pasarle tres vectores... El tercero es opcional  
              #prueba = mean_squared_error(Z[:,:m+1],Y)
    #score(X, y[, sample_weight])
    #print this_scores
    #Estimate the score using cross validation. 
    #You should check which is the score used
    scores.append(np.mean(-this_scores))
    print "scores es: " + str(np.mean(-this_scores))
    scores_std.append(np.std(-this_scores)) #desviación estándar
    #print scores_std
#%%
plt.plot(scores)
xlabel('Componentes')
ylabel("$MSE$")
title("lab3")


print "% Variance Explained (cumulative)"
print np.cumsum(pca_ex.explained_variance_ratio_)
plt.show()

#Algo no estamos haciendo bien
#
#1) 
#Acuerdate de poner nombres a los ejes
#2)
#No está claro que los resultados sean como los que se obtienen en el lab3. 
#si te fijas en su caso dice que M=7 components es lo que se elije utilizando CV
#en la gráfica que tu presentas siempre se desciende. De todas formas no sé qué 
#es lo que se representa en el eje Y. Date cuenta de que el valor es negativo,
# tendrías que verificar que score está utilizando clf y modificarlo para que 
#sea RMSE, igual que lo que están utilizando en el libro para que podamos comparar.
#Aquí te ayudo yo: si miras la ayuda del sklearn verás que el score que utiliza 
#linear regression por defecto es 
#score(X, y[, sample_weight]) 	Returns the coefficient of determination R^2 of the prediction.

#3)También sería conveniente primero que verificasemos que los valores en Training
#son equivalentes a los que obtienen en el lab3. Si miras los resultados de % variance
#explained verás que usando una componente la varianza explicada es 38,31, sin embargo lo que
# te da a tí es
pca_ex.explained_variance_ratio_[0]
#99,9
#Algo está mal
#4) Mi hipótesis es que hay primero que transformar los datos: centrarlos y normalizarlos
#si le echas un vistazo al script que te adjunté yo verás cómo se realizar con un objeto
#de sklearn que se llama preprocessing.



#There should make pca in training data






