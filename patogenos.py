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
import matplotlib.pyplot as plt

import statsmodels.api as sm_g
import scipy.stats as st
import statsmodels.formula.api as sm


#path = '/home/carlos/TFG-carlos-biedma-tapia'
#path = '/home/carlos/TFG-carlos-biedma-tapia'
path = '/Users/obarquero/Documents/TFGs/TFG-carlos-biedma-tapia/Patogenos_work/data/'
os.chdir(path)
fname = 'tfg_respiratorio.csv'
animales = pd.read_csv(fname,delimiter = ";", index_col=0) # this reads the data using panda



Cambio = {'Positivo':1,'Negativo':0}
Cambio2 = {'Macho':1,'Hembra':0}

#animales['ELISAPCV2'] = animales.ELISAPCV2.map(Cambio)
#animales['ELISAADV'] = animales.ELISAADV.map(Cambio)
#animales['Sexo'] = animales.Sexo.map(Cambio2)
#animales['ELISAPPV'] = animales.ELISAPPV.map(Cambio)
#animales['ELISAPCV'] = animales.ELISAPCV.map(Cambio)
#animales['TB'] = animales.TB.map(Cambio)
#
#Densidad = animales['Densidad'] 
#edad = animales['Edad']
#CondicionCorporal = animales['CondicionCorporal']
#
#ELISAPCV2 = animales['ELISAPCV2']
#Sexo = animales['Sexo']
#ELISAADV = animales['ELISAADV']
#ELISAPPV = animales['ELISAPPV']
#ELISAPCV = animales['ELISAPCV']
#TB = animales['TB']

#Working with the new ddbb
animales['Mhyo'] = animales.Mhyo.map(Cambio)
animales['ADV'] = animales.ADV.map(Cambio)
animales['SIV'] = animales.SIV.map(Cambio)
animales['HPS'] = animales.HPS.map(Cambio)
animales['APP'] = animales.APP.map(Cambio)
animales['PCV'] = animales.PCV.map(Cambio)
animales['PM'] = animales.PM.map(Cambio)
animales['Sexo'] = animales.Sexo.map(Cambio2)


Sexo = animales['Sexo']
edad = animales['Edad']
mhyo = animales['Mhyo']
adv = animales['ADV']
siv = animales['SIV']
hps = animales['HPS']
app = animales['APP']
pcv = animales['PCV']
pm = animales['PM']
metast = animales['Metastrongylus']
pi = animales['Peribronquitis']

#old data
#animales = concat([Densidad, edad, CondicionCorporal, ELISAPCV2, Sexo, ELISAADV, ELISAPPV, ELISAPCV, TB], axis=1) 
#new data
animales = concat([Sexo,edad, mhyo, adv, siv, hps, app, pcv, pm,metast,pi], axis=1) 
animales = animales.dropna()

#ELISAInfluenza = animales['ELISAInfluenza']
#ELISAPRRS = animales['ELISAPRRS']
#ELISAADV = animales['ELISAADV']
#PCRMycoPul = animales['PCRMycoPul']
#PCRHaemoPul = animales['PCRHaemoPul']
#PCRAPPPul = animales['PCRAPPPul']
#PCRPCVPul = animales['PCRPCVPul']
#Metastronguilus_Clas = animales['Metastronguilus_Clas']
#NIntersticial = animales['NIntersticial']
#Pleuritis = animales['Pleuritis']
#Peribronquitis = animales['Peribronquitis']
#Bronquitis = animales['Bronquitis']
#Necrosis = animales['Necrosis']

#animales = concat([Densidad, edad, CondicionCorporal, ELISAPCV2, Sexo, ELISAADV, ELISAPPV, ELISAPCV, TB], axis=1) 
#animales = animales.dropna()


Sexo = pd.Categorical.from_array(animales['Sexo'])
animales['Sexo'] = Sexo.labels #inset in dataframe
#ELISAPCV2 = pd.Categorical.from_array(animales['ELISAPCV2'])
#animales['ELISAPCV2'] = ELISAPCV2.labels # insert in dataframe

Mhyo = pd.Categorical.from_array(animales['Mhyo'])
animales['Mhyo'] = Mhyo.labels #inset in dataframe
#Sexo = pd.Categorical.from_array(animales['Sexo'])
#animales['Sexo'] = Sexo.labels # insert in dataframe

adv = pd.Categorical.from_array(animales['ADV'])
animales['ADV'] = adv.labels #inset in dataframe
#ELISAADV = pd.Categorical.from_array(animales['ELISAADV'])
#animales['ELISAADV'] = ELISAADV.labels # insert in dataframe

siv= pd.Categorical.from_array(animales['SIV'])
animales['SIV'] = siv.labels #inset in dataframe
#ELISAPPV = pd.Categorical.from_array(animales['ELISAPPV'])
#animales['ELISAPPV'] = ELISAPPV.labels # insert in dataframe

hps = pd.Categorical.from_array(animales['HPS'])
animales['HPS'] = hps.labels #inset in dataframe
#ELISAPCV = pd.Categorical.from_array(animales['ELISAPCV'])
#animales['ELISAPCV'] = ELISAPCV.labels # insert in dataframe

app = pd.Categorical.from_array(animales['APP'])
animales['APP'] = app.labels #inset in dataframe
pcv = pd.Categorical.from_array(animales['PCV'])
animales['PCV'] = pcv.labels #inset in dataframe
pm = pd.Categorical.from_array(animales['PM'])
animales['PM'] = pm.labels #inset in dataframe
#TB = pd.Categorical.from_array(animales['TB']) # default order: alphabetical
#animales['TB'] = TB.labels # insert in dataframe

periobronquitis = animales['Peribronquitis']
animales = animales.dropna()

#%%

#histograma de la salida

#Let's work by groups

g_s = animales.groupby('Sexo')
print g_s.Peribronquitis.describe()

#age plot scatter plot
pd.tools.plotting.scatter_plot(animales,'Edad','Peribronquitis')

g_mh = animales.groupby('Mhyo')
print g_mh.Peribronquitis.describe()

g_adv = animales.groupby('ADV')
print g_adv.Peribronquitis.describe()

#make the same for the rest of the binary variables

pd.tools.plotting.scatter_plot(animales,'Metastrongylus','Peribronquitis')
#############BOXPLOT#############
#¿No debería hacer tb los boxplot sin enfrentar las variables,
#sino haciéndolo independientes?
#animales.boxplot(column = 'Edad',by = 'TB', grid=True)
animales.boxplot(column = 'Peribronquitis',by = 'Sexo', grid=True)
animales.boxplot(column = 'Peribronquitis',by = 'Mhyo', grid=True)
animales.boxplot(column = 'Peribronquitis',by = 'ADV', grid=True)

#make the same for the rest of binary variables



#%%
##############TABLAS CONTINGENCIA#############
#pd.crosstab(animales['TB'], animales['ELISAPCV2'])
#pd.crosstab(animales['TB'], animales['Sexo'])
#pd.crosstab(animales['TB'], animales['ELISAADV'])
#pd.crosstab(animales['TB'], animales['ELISAPPV'])
#pd.crosstab(animales['TB'], animales['ELISAPCV'])


#%%
#############Matriz Correlaciones#############
for i in animales.columns:
    pd.tools.plotting.scatter_plot(animales,i,'Peribronquitis') #esto es un ejemplo

#Estas gráficas salen fatal no se aprecian los detalles.
#No sé como cambiar la posición de los ejes.    
#pd.tools.plotting.scatter_matrix(animales, figsize=(20,10), diagonal='kde', rotation = 45)
pd.tools.plotting.scatter_matrix(animales, figsize=(20, 20))

matriz_correlacion = animales.corr()
print matriz_correlacion

#%%


#%%
#Al hacer esto me dice que hay mucha correlación
est = sm.ols(formula = 'Peribronquitis ~ Sexo + Edad+Mhyo + ADV + SIV + HPS+ APP + PCV + PM + Metastrongylus',data = animales).fit()
#est = est.fit()
print est.summary()

sm_g.graphics.plot_partregress_grid(est)

#%%

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

############Change this to be adequate the new data
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
        













