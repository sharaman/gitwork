# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 22:45:43 2018

@author: Artem.Skachenko
"""

import pandas as pd


# Data
X = pd.read_csv("X_preproc.csv")
y = pd.read_csv("y.csv", header=None, names='y')

###################################################
import  numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

models = [
    LogisticRegression(),
    KNeighborsClassifier(),
    DecisionTreeClassifier(),
    LinearSVC(),
    RandomForestClassifier(n_estimators=100), 
    GradientBoostingClassifier(n_estimators=100)]


# Вот это надо переписать с кросс валидацией
models_scores=pd.DataFrame()
for model in models:
    score = pd.DataFrame({'model':[model],
                          'scr':[cross_val_score(model, X, np.ravel(y), 
                                                 cv=2).mean()]})
    models_scores = pd.concat([models_scores, score], axis=0)
    del score

del models
