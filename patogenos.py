# -*- coding: utf-8 -*-
"""
Created on Mon Feb 16 19:17:12 2015

@author: carlos
"""

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
from pandas import concat

#path = '/home/carlos/TFG-carlos-biedma-tapia'
path = '/Users/obarquero/Documents/TFGs/TFG-carlos-biedma-tapia'
os.chdir(path)
fname = 'Respiratorio.csv'
animales = pd.read_csv(fname,delimiter = ",", index_col=0) # this reads the data using panda

Cambio = {'Positivo':1,'Negativo':0}
animales['ELISAInfluenza'] = animales.ELISAInfluenza.map(Cambio)
animales['ELISAPRRS'] = animales.ELISAPRRS.map(Cambio)
animales['ELISAADV'] = animales.ELISAADV.map(Cambio)
animales['PCRMycoPul'] = animales.PCRMycoPul.map(Cambio)
animales['PCRHaemoPul'] = animales.PCRHaemoPul.map(Cambio)
animales['PCRAPPPul'] = animales.PCRAPPPul.map(Cambio)
animales['PCRPCVPul'] = animales.PCRPCVPul.map(Cambio)
animales['Metastronguilus_Clas'] = animales.Metastronguilus_Clas.map(Cambio)
 
edad = animales['Edad']
ELISAInfluenza = animales['ELISAInfluenza']
ELISAPRRS = animales['ELISAPRRS']
ELISAADV = animales['ELISAADV']
PCRMycoPul = animales['PCRMycoPul']
PCRHaemoPul = animales['PCRHaemoPul']
PCRAPPPul = animales['PCRAPPPul']
PCRPCVPul = animales['PCRPCVPul']
Metastronguilus_Clas = animales['Metastronguilus_Clas']
NIntersticial = animales['NIntersticial']
Pleuritis = animales['Pleuritis']
Peribronquitis = animales['Peribronquitis']
Bronquitis = animales['Bronquitis']
Necrosis = animales['Necrosis']

#Esto lo hago porque si quito los NAN en la matriz entera se queda vacía
animales = concat([edad, ELISAInfluenza, ELISAPRRS, ELISAADV, PCRMycoPul, PCRHaemoPul, PCRAPPPul, PCRPCVPul, Metastronguilus_Clas, NIntersticial, Pleuritis, Peribronquitis, Bronquitis, Necrosis], axis=1) 
animales = animales.dropna()


print animales.describe()
print animales.mean()
print animales.var()
animales.std()
animales.corr()
animales.cov()
animales.median()
animales.mode()
animales.min()
animales.max()
animales.quantile()
animales.head()
animales.count() #devuelve los no Nulos
animales.columns
#%%
animalesMatriz = animales.as_matrix()

Y = animalesMatriz[:,-5:]

scaler = prepro.StandardScaler()
scaler.fit(animalesMatriz)
animalesMatriz = scaler.transform(animalesMatriz)

X = animalesMatriz[:,:-5]
#%%
def hacerPCR(X,Y):
    pca_ex = PCA()
    pca_ex.fit(X) 
    Z = pca_ex.transform(X) 
    
    scores = list() 
    scores_std = list()
    
    n_features = np.shape(Z)[1]
    print(n_features) 
    print np.shape(Z)
    
    for m in range(n_features+1):
        print "--------------------------------------"
        #Let compute a linear regression Y = w(T)Z using the first n_features
        clf  = linear_model.LinearRegression()
        if m == 0:
            unos = unos = np.ones((np.shape(Y)[0],1))
            this_scores = cross_validation.cross_val_score(clf,unos,Y,scoring = 'mean_squared_error',n_jobs = -1)
        else:
            this_scores = cross_validation.cross_val_score(clf,Z[:,:m],Y,scoring = 'mean_squared_error',n_jobs = -1)
  
        scores.append(np.mean(-this_scores))
        scores_std.append(np.std(-this_scores)) #desviación estándar
    
        plt.plot(scores)
        xlabel('Componentes')
        ylabel("$MSE$")
        title("Animales PCR")
        print "% Variance Explained (cumulative)"
        print np.cumsum(pca_ex.explained_variance_ratio_)
        plt.show()
#%%        
def hacerPLS(X,Y):
    pls_wild_b = PLSRegression(n_components = 9) 
    pls_wild_b.fit(X,Y)
    Z = pls_wild_b.transform(X)
    scores = list() 
    scores_std = list()
    n_features = np.shape(X)[1]
    
    X,X_test_tot, Y, Y_test_tot = cross_validation.train_test_split(X,Y,test_size = 0.5,random_state = 0)
    N = np.shape(X)[0]
    
    for num_comp in range(n_features):
        kf = KFold(N,n_folds = 10)
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
                
        scores.append(np.mean(aux_scores))
        scores_std.append(np.std(aux_scores))
    
    plt.plot(scores)
    xlabel('Componentes')
    ylabel("$MSE$")
    title("Animales PLS")
    plt.show()
    
    num_comp = np.argmin(scores)
    
    pls_pred = PLSRegression(n_components =2)
    pls_pred.fit(X,Y)
    y_pred_test = pls_pred.predict(X_test_tot)
    
    print "MSE test = " + str(metrics.mean_squared_error(Y_test_tot,y_pred_test))
    #print "% Variance Explained (cumulative)"
    #print np.cumsum(pca_ex.explained_variance_ratio_)
    
    
    #pca_ex.explained_variance_ratio_[0]
    
#%%
hacerPCR(X,Y)

#%%


#nFilas = len(animalesMatriz) #numero de filas
#nColumnas = len(animalesMatriz[0]) #numero de columnas
        












