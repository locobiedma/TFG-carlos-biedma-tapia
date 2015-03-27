# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 21:38:51 2015

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

#old data
#animales = concat([Densidad, edad, CondicionCorporal, ELISAPCV2, Sexo, ELISAADV, ELISAPPV, ELISAPCV, TB], axis=1) 
#new data
animales = concat([Sexo,edad, mhyo, adv, siv, hps, app, pcv, pm,metast], axis=1) 
animales = animales.dropna()

pd.tools.plotting.scatter_matrix(animales, figsize=(6, 6))