# -*- coding: utf-8 -*-
"""
Created on Wed May 18 22:53:04 2022

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

url = 'bank-full.csv'
data = pd.read_csv(url)

# Tratamiento de la data
data.marital.replace(['married', 'divorced', 'single'],
                     [0, 1, 2], inplace=True)
data.education.replace(['primary', 'secondary', 'tertiary', 'unknown'], [
                       0, 1, 2, 3], inplace=True)
data.default.replace(['no', 'yes'], [0, 1], inplace=True)
data.housing.replace(['no', 'yes'], [0, 1], inplace=True)
data.loan.replace(['no', 'yes'], [0, 1], inplace=True)
data.poutcome.replace(['failure', 'other', 'success', 'unknown'], [
                      0, 1, 2, 3], inplace=True)

data.drop(['job', 'balance', 'contact', 'day', 'month', 'duration',
          'pdays', 'previous', 'campaign'], axis=1, inplace=True)

data.age.replace(np.nan, 41, inplace=True)
rangos = [0, 8, 15, 18, 25, 40, 60, 100]
nombres = ['1', '2', '3', '4', '5', '6', '7']
data.age = pd.cut(data.age, rangos, labels=nombres)

data.dropna(axis=0, how='any', inplace=True)
data.y.replace(['no', 'yes'], [0, 1], inplace=True)

# partir la data en dos

data_train = data[:728]
data_test = data[728:]

x = np.array(data_train.drop(['y'], 1))
y = np.array(data_train.y)  # 0 NO 1 Si

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
x_test_out = np.array(data_test.drop(['y'], 1))
y_test_out = np.array(data_test.y)  # 0 No 1 Si

# Regresión Logística

# Seleccionar un modelo
logreg = LogisticRegression(solver='lbfgs', max_iter=7600)

# REGRESIÓN LOGÍSTICA CON VALIDACIÓN CRUZADA

kfold = KFold(n_splits=10)

acc_scores_train_train = []
acc_scores_test_train = []
logreg = LogisticRegression(solver='lbfgs', max_iter=7600)

for train, test in kfold.split(x, y):
    logreg.fit(x[train], y[train])
    scores_train_train = logreg.score(x[train], y[train])
    scores_test_train = logreg.score(x[test], y[test])
    acc_scores_train_train.append(scores_train_train)
    acc_scores_test_train.append(scores_test_train)

y_pred = logreg.predict(x_test_out)

print('*' * 50)
print('Regresión Logística Validación cruzada')

# Accuracy de Entrenamiento de Entrenamiento
print(
    f'accuracy de Entrenamiento de Entrenamiento: {np.array(acc_scores_train_train).mean()}')

# Accuracy de Test de Entrenamiento
print(
    f'accuracy de Test de Entrenamiento: {np.array(acc_scores_test_train).mean()}')

# Accuracy de Validación
print(f'accuracy de Validación: {logreg.score(x_test_out, y_test_out)}')

# Matriz de confusión
print(f'Matriz de confusión: {confusion_matrix(y_test_out, y_pred)}')

matriz_confusion = confusion_matrix(y_test_out, y_pred)
plt.figure(figsize=(6, 6))
sns.heatmap(matriz_confusion)
plt.title("Mariz de confución")

precision = precision_score(y_test_out, y_pred, average=None).mean()
print(f'Precisión: {precision}')

recall = recall_score(y_test_out, y_pred, average=None).mean()
print(f'Re-call: {recall}')

f1_score_logred = f1_score(y_test_out, y_pred, average=None).mean()

print(f'f1: {f1_score_logred}')


# MAQUINA DE SOPORTE VECTORIAL
# Seleccionar un modelo


acc_scores_train_train_svc = []
acc_scores_test_train_svc = []
svc = SVC(gamma='auto')

for train, test in kfold.split(x, y):
    svc.fit(x[train], y[train])
    scores_train_train_svc = svc.score(x[train], y[train])
    scores_test_train_svc = svc.score(x[test], y[test])
    acc_scores_train_train_svc.append(scores_train_train_svc)
    acc_scores_test_train_svc.append(scores_test_train_svc)

y_pred_svc = svc.predict(x_test_out)

print('*' * 50)
print('Maquina de soporte vectorial con Validación cruzada')

# Accuracy de Entrenamiento de Entrenamiento
print(
    f'accuracy de Entrenamiento de Entrenamiento: {np.array(acc_scores_train_train_svc).mean()}')

# Accuracy de Test de Entrenamiento
print(
    f'accuracy de Test de Entrenamiento: {np.array(acc_scores_test_train_svc).mean()}')

# Accuracy de Validación
print(f'accuracy de Validación: {svc.score(x_test_out, y_test_out)}')

# Matriz de confusión
print(f'Matriz de confusión: {confusion_matrix(y_test_out, y_pred_svc)}')

matriz_confusion_svc = confusion_matrix(y_test_out, y_pred_svc)
plt.figure(figsize=(6, 6))
sns.heatmap(matriz_confusion_svc)
plt.title("Mariz de confución")

precision_svc = precision_score(y_test_out, y_pred_svc, average=None).mean()
print(f'Precisión: {precision_svc}')

recall_svc = recall_score(y_test_out, y_pred_svc, average=None).mean()
print(f'Re-call: {recall_svc}')

f1_score_svc = f1_score(y_test_out, y_pred_svc, average=None).mean()

print(f'f1: {f1_score_svc}')


# ARBOL DE DECISIÓN

acc_scores_train_train_arbol = []
acc_scores_test_train_arbol = []
arbol = DecisionTreeClassifier()

for train, test in kfold.split(x, y):
    arbol.fit(x[train], y[train])
    scores_train_train_arbol = arbol.score(x[train], y[train])
    scores_test_train_arbol = arbol.score(x[test], y[test])
    acc_scores_train_train_arbol.append(scores_train_train_arbol)
    acc_scores_test_train_arbol.append(scores_test_train_arbol)

y_pred_arbol = arbol.predict(x_test_out)

print('*' * 50)
print('Arbol de decisiòn con Validación cruzada')

# Accuracy de Entrenamiento de Entrenamiento
print(
    f'accuracy de Entrenamiento de Entrenamiento: {np.array(acc_scores_train_train_arbol).mean()}')

# Accuracy de Test de Entrenamiento
print(
    f'accuracy de Test de Entrenamiento: {np.array(acc_scores_test_train_arbol).mean()}')

# Accuracy de Validación
print(f'accuracy de Validación: {arbol.score(x_test_out, y_test_out)}')

# Matriz de confusión
print(f'Matriz de confusión: {confusion_matrix(y_test_out, y_pred_arbol)}')

matriz_confusion_arbol = confusion_matrix(y_test_out, y_pred_arbol)
plt.figure(figsize=(6, 6))
sns.heatmap(matriz_confusion_arbol)
plt.title("Mariz de confución")

precision_arbol = precision_score(
    y_test_out, y_pred_arbol, average=None).mean()
print(f'Precisión: {precision_arbol}')

recall_arbol = recall_score(y_test_out, y_pred_arbol, average=None).mean()
print(f'Re-call: {recall_arbol}')

f1_score_arbol = f1_score(y_test_out, y_pred_arbol, average=None).mean()

print(f'f1: {f1_score_arbol}')


# MODELO NAIVE BAYES

acc_scores_train_train_gnb = []
acc_scores_test_train_gnb = []
gnb = GaussianNB()

for train, test in kfold.split(x, y):
    gnb.fit(x[train], y[train])
    scores_train_train_gnb = gnb.score(x[train], y[train])
    scores_test_train_gnb = gnb.score(x[test], y[test])
    acc_scores_train_train_gnb.append(scores_train_train_gnb)
    acc_scores_test_train_gnb.append(scores_test_train_gnb)

y_pred_gnb = gnb.predict(x_test_out)

print('*' * 50)
print('Naive Bayes con Validación cruzada')

# Accuracy de Entrenamiento de Entrenamiento
print(
    f'accuracy de Entrenamiento de Entrenamiento: {np.array(acc_scores_train_train_gnb).mean()}')

# Accuracy de Test de Entrenamiento
print(
    f'accuracy de Test de Entrenamiento: {np.array(acc_scores_test_train_gnb).mean()}')

# Accuracy de Validación
print(f'accuracy de Validación: {gnb.score(x_test_out, y_test_out)}')

# Matriz de confusión
print(f'Matriz de confusión: {confusion_matrix(y_test_out, y_pred_gnb)}')

matriz_confusion_gnb = confusion_matrix(y_test_out, y_pred_gnb)
plt.figure(figsize=(6, 6))
sns.heatmap(matriz_confusion_gnb)
plt.title("Mariz de confución")

precision_gnb = precision_score(y_test_out, y_pred_gnb, average=None).mean()
print(f'Precisión: {precision_gnb}')

recall_gnb = recall_score(y_test_out, y_pred_gnb, average=None).mean()
print(f'Re-call: {recall_gnb}')

f1_score_gnb = f1_score(y_test_out, y_pred_gnb, average=None).mean()

print(f'f1: {f1_score_gnb}')


# MODELO KNN
knn = neighbors.KNeighborsClassifier()
acc_scores_train_train_knn = []
acc_scores_test_train_knn = []

for train, test in kfold.split(x, y):
    knn.fit(x[train], y[train])
    scores_train_train_knn = knn.score(x[train], y[train])
    scores_test_train_knn = knn.score(x[test], y[test])
    acc_scores_train_train_knn.append(scores_train_train_knn)
    acc_scores_test_train_knn.append(scores_test_train_knn)

y_pred_knn = knn.predict(x_test_out)

print('*' * 50)
print('Modelo KNN con Validación cruzada')

# Accuracy de Entrenamiento de Entrenamiento
print(
    f'accuracy de Entrenamiento de Entrenamiento: {np.array(acc_scores_train_train_knn).mean()}')

# Accuracy de Test de Entrenamiento
print(
    f'accuracy de Test de Entrenamiento: {np.array(acc_scores_test_train_knn).mean()}')

# Accuracy de Validación
print(f'accuracy de Validación: {knn.score(x_test_out, y_test_out)}')

# Matriz de confusión
print(f'Matriz de confusión: {confusion_matrix(y_test_out, y_pred_knn)}')

matriz_confusion_knn = confusion_matrix(y_test_out, y_pred_knn)
plt.figure(figsize=(6, 6))
sns.heatmap(matriz_confusion_knn)
plt.title("Mariz de confución")

precision_knn = precision_score(y_test_out, y_pred_knn, average=None).mean()
print(f'Precisión: {precision_knn}')

recall_knn = recall_score(y_test_out, y_pred_knn, average=None).mean()
print(f'Re-call: {recall_knn}')

f1_score_knn = f1_score(y_test_out, y_pred_knn, average=None).mean()

print(f'f1: {f1_score_knn}')
