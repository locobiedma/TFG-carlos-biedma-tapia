# -*- coding: utf-8 -*-
"""
Created on Thu Aug  7 09:33:06 2014

Script containing all the analysis to compare brt and svm estimating wild boar 
age from dental measurements. 


@author: Óscar Barquero Pérez mail to: oscar.barquero@urjc.es   
         Rebeca Goya Esteban mail to: rebeca.goyaesteban@urjc.es
"""


#########################
# Importing modules
#########################

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

plt.ioff()
###############################################################################
# First step is to read the data from the text file. The first line inclues the names
# of the variables.
###############################################################################

#change d
path = '/home/carlos/TFG-carlos-biedma-tapia'
os.chdir(path)

fname = 'wild_boar_age.csv'
data = np.genfromtxt(fname,delimiter = ";",skip_header = 1)

#read the header LO HE QUITADO
f = open(fname, 'rU')
reader = csv.reader(f,delimiter = ";")
headers = reader.next()


#Create a dic with the header names and a data fiel
wild_boar_ddbb = {"header":headers,'data':data}  #LO HE QUITADO

#print(wild_boar_ddbb)

###############################################################################
#
# Exploratory analysis
#
###############################################################################

#Some plots and distributions
wild_boar_data =pd.read_csv(fname,delimiter = ";") # this reads the data using panda
#remove the first column which is only an id
wild_boar_data = wild_boar_data.drop('Unnamed: 0', 1)
print(wild_boar_data.describe())
#%%
#Scatter matrix
pd.tools.plotting.scatter_matrix(wild_boar_data)

#Scatter plots for each variable
for i in wild_boar_data.columns:
    pd.tools.plotting.scatter_plot(wild_boar_data,i,'Edad')

#Boxplot with zone
wild_boar_data.boxplot(column = ['Edad'],by = 'z')

#Correlation matrix between variables

correlation_matrix = wild_boar_data.corr()
print(correlation_matrix)


#Assessing collinearity using the condition number

model_fitted = sm.ols(formula = 'Edad ~ Abertura_A + Abertura_B + Raiz_A + Raiz_B + Superficie_A +Superficie_B ', data=wild_boar_data).fit() # this is the model

print model_fitted.summary() #shows OLS regression output

#Assessing multicollinearity using the variance inflation factor
wb_data = wild_boar_data.as_matrix() #LO HE QUITADO
X = wb_data[:,2:]
Y = wb_data[:,0]
zone = wb_data[:,1]
vif_wild_boar = []
for i in range(X.shape[1]):
    vif_wild_boar.append(vif.variance_inflation_factor(X,i))

print("######################################################################")
print("Variance inflation factor")
print vif_wild_boar


###############################################################################
#
# Split train and test
#
###############################################################################
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
###############################################################################
#
# PCA and PLSR analysis 
#
###############################################################################

#using pca to avoid collinearity

#There should make pca in training data
#Normalize data
scaler = prepro.StandardScaler()
scaler.fit(X_train)
X_train_prepro = scaler.transform(X_train)

pca_wild_b = PCA()
pca_wild_b.fit(X_train_prepro)

#keep the number of componets with 95% of the explained variance
n_comps = np.sum(np.cumsum(pca_wild_b.explained_variance_ratio_)<=0.94)
pca_wild_b = PCA(n_components = n_comps)
pca_wild_b.fit(X_train_prepro)
X_train_proj = pca_wild_b.transform(X_train_prepro)


#Some scatter plots
#First the loadings 
print("loadings")

#%%
for i in range(pca_wild_b.n_components):
    plt.figure()
    plt.bar(np.arange(np.shape(X_train_prepro)[1]), pca_wild_b.components_[i])
    if i == 0:
        plt.ylabel('1st component')
    elif i == 1:
        plt.ylabel('2nd component')
    else:
        plt.ylabel('3rd component')
    axis_c = plt.gca()
    axis_c.set_xticklabels(wild_boar_ddbb['header'][3:],fontsize = 7)
    axis_c.set_xticks(axis_c.get_xticks() + 0.5)
#%%
#sm.OLS()
#Select the number of components using CV

##PLSR
pls_wild_b = PLSRegression(n_components = 3)
pls_wild_b.fit(X_train_prepro,Y_train)
X_train_pls_proj = pls_wild_b.transform(X_train_prepro)
print("loadings")

for i in range(pls_wild_b.n_components):
    plt.figure()
    plt.bar(np.arange(np.shape(X_train_prepro)[1]), pls_wild_b.x_loadings_[:,i])
    if i == 0:
        plt.ylabel('PLS 1st component')
    elif i == 1:
        plt.ylabel('PLS2nd component')
    else:
        plt.ylabel('PLS 3rd component')
    axis_c = plt.gca()
    axis_c.set_xticklabels(wild_boar_ddbb['header'][3:],fontsize = 7)
    axis_c.set_xticks(axis_c.get_xticks() + 0.5)
    
#Select the number of components using CV

    
    
    
    
###############################################################################
#
# Regression models
#
###############################################################################    
regress_models = {'svm': SVR(kernel = 'rbf'),'brt':GradientBoostingRegressor(n_estimators=3000),
                  'svm_pca':SVR(kernel = 'rbf'),'brt_pca':GradientBoostingRegressor(n_estimators=3000),\
                  'svm_pcr':SVR(kernel = 'rbf'),'brt_pcr':GradientBoostingRegressor(n_estimators=3000),\
                  'svm_pls':SVR(kernel = 'rbf'),'brt_pls':GradientBoostingRegressor(n_estimators=3000),\
                  'svm_plsr':SVR(kernel = 'rbf'),'brt_plsr':GradientBoostingRegressor(n_estimators=3000)}
################################################################################
#
# SVM regression
#
################################################################################

#Tuning parameters of SVM
#tuned_parameters_svm = {'kernel': ['rbf'], 'gamma': np.logspace(-4, 2,40),'C': np.logspace(-2, 2, 40),'epsilon':np.logspace(-4,2,40)}
tuned_parameters_svm = {'kernel': ['rbf'], 'gamma':[1e-3,1e-4],'C': [0.1,100,1000],'epsilon':[0.001,0.1]}



 
#evaluation metrics
scorer = metrics.make_scorer(metrics.mean_squared_error,greater_is_better = False)

#instantiate the svr

clf = GridSearchCV(SVR(C=1), param_grid = tuned_parameters_svm, cv=5, scoring = scorer)
clf.fit(X_train_proj,Y_train)


print("Best parameters set found on development set:")
print()
print(clf.best_estimator_)
print()
print("Grid scores on development set:")
print()
for params, mean_score, scores in clf.grid_scores_:
    print("%0.3f (+/-%0.03f) for %r"% (mean_score, scores.std() / 2, params))
print()

print("Detailed classification report:")
print()
print("The model is trained on the full development set.")
print("The scores are computed on the full evaluation set.")
print()


##Tunning BRT parameters
param_grid_brt = {'learning_rate': [1e-3,1e-2,0.9],'max_depth': [2,3,4],'min_samples_leaf': [3, 5]}

clf_brt = GradientBoostingRegressor(n_estimators=3000)
brt_gs = GridSearchCV(clf_brt, param_grid_brt,scoring = scorer)
brt_gs.fit(X_train_proj,Y_train)

#Suggestion is now, keeping all the paremeter fixed, increase the number of estimators
#and perform grid search over learning rate
print("Best parameters set found on development set:")
print()
print(brt_gs.best_estimator_)
print()
print("Grid scores on development set:")
print()
for params, mean_score, scores in brt_gs.grid_scores_:
    print("%0.3f (+/-%0.03f) for %r"% (mean_score, scores.std() / 2, params))
print()

print("Detailed classification report:")
print()
print("The model is trained on the full development set.")
print("The scores are computed on the full evaluation set.")
print()

########################################################
#
# Obtaining test evaluations
#
########################################################
#conditioning the test set with the preprocessing of the training set
X_test_prepro = scaler.transform(X_test) #scaler has the mean an std from X_train
X_test_proj = pca_wild_b.transform(X_test_prepro) #pca_wild_b has eigenvectors those
#obtained from X_train_prepro

Y_true, Y_pred = Y_test, clf.predict(X_test_proj) # using SVM
Y_pred_brt = brt_gs.predict(X_test_proj)

print("R^2 for SVM approach")
print("SVM R^2 = %0.3f"%metrics.r2_score(Y_true, Y_pred))
print("############")

plt.show()