# -*- coding: utf-8 -*-
"""
Created on Wed May 18 23:22:21 2022

@author: DELL3467
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import neighbors
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from warnings import simplefilter
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

simplefilter(action='ignore', category=FutureWarning)

url = 'diabetes.csv'
data = pd.read_csv(url)

# Tratamiento de la data

data.Pregnancies.replace(np.nan, 4, inplace=True)
rangos_pregnancies = [0, 7, 10, 15, 20]
nombres_pregnancies = [1, 2,3, 4]
data.Pregnancies = pd.cut(data.Pregnancies, rangos_pregnancies, labels=nombres_pregnancies)

data.Glucose.replace(np.nan, 123, inplace=True)
rangos_glucose = [0, 70, 150, 180, 200]
nombres_glucose = [1, 2,3, 4]
data.Glucose = pd.cut(data.Glucose, rangos_glucose, labels=nombres_glucose)

data.BloodPressure.replace(np.nan, 69, inplace=True)
rangos_bloodPressure = [0, 70, 150, 180]
nombres_bloodPressure = [1, 2, 3]
data.BloodPressure = pd.cut(data.BloodPressure, rangos_bloodPressure, labels=nombres_bloodPressure)

data.SkinThickness.replace(np.nan, 32, inplace=True)
rangos_skinThickness = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
nombres_skinThickness = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,]
data.SkinThickness = pd.cut(data.SkinThickness, rangos_skinThickness, labels=nombres_skinThickness)

data.Insulin.replace(np.nan, 80, inplace=True)
rangos_insulin = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
nombres_insulin = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,]
data.Insulin = pd.cut(data.Insulin, rangos_insulin, labels=nombres_insulin)

data.BMI.replace(np.nan, 36, inplace=True)
rangos_bmi = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
nombres_bmi = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,]
data.BMI = pd.cut(data.BMI, rangos_bmi, labels=nombres_bmi)

data.drop(['DiabetesPedigreeFunction'], axis=1, inplace=True)

data.Age.replace(np.nan, 38, inplace=True)
rangos = [0, 8, 15, 18, 25, 40, 60, 100]
nombres = ['1', '2', '3', '4', '5', '6', '7']
data.Age = pd.cut(data.Age, rangos, labels=nombres)


data.dropna(axis=0,how='any', inplace=True)

