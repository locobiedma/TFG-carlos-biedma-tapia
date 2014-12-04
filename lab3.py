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
import numpy as np
#import sklearn.preprocessing as prepro
#import statsmodels.formula.api as sm
#import statsmodels.stats.outliers_influence as vif
import matplotlib.pylab as plt
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
path = '/Users/obarquero/Qsync/TFGs/TFG_Carlos_Biedma/python_code/TFG-carlos-biedma-tapia/'
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

plt.plot(scores)
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
pca_ex.pca_ex.explained_variance_ratio_[0]
#99,9
#Algo está mal
#4) Mi hipótesis es que hay primero que transformar los datos: centrarlos y normalizarlos
#si le echas un vistazo al script que te adjunté yo verás cómo se realizar con un objeto
#de sklearn que se llama preprocessing.





