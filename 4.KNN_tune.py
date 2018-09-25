# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 10:58:56 2018

@author: Artem.Skachenko
"""


import pandas as pd

# Data
X = pd.read_csv("X_feat_sel.csv")
y = pd.read_csv("y.csv", header=None, names='y')


#########################################################
import  numpy as np
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()



param_grid = [{'n_neighbors': np.arange(10,30,5),
               'weights':['uniform', 'distance']}]
gs3 = GridSearchCV(knn, param_grid, cv=2)

scores=pd.DataFrame()
for i in np.arange(1, X.shape[1]):
    X2 = SelectKBest(f_classif, k=i).fit_transform(X, np.ravel(y))
    gs3.fit(X2, np.ravel(y))
    score = pd.DataFrame({'n_feats':[i],'scr':[gs3.best_score_],
                          'n_neighbors':[gs3.best_estimator_.n_neighbors],
                          'weights':[gs3.best_estimator_.weights]})
    scores = pd.concat([scores, score], axis=0)
    del i,X2,score


gs3.best_params_
gs3.best_estimator_.max_depth
gs3.best_estimator_.criterion
gs3.best_estimator_.feature_importances_





