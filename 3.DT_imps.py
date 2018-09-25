# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 00:33:05 2018

@author: Artem.Skachenko
"""

import pandas as pd

# Data
X = pd.read_csv("X_preproc.csv")
y = pd.read_csv("y.csv", header=None, names='y')


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()

# Feature importance by DecisionTree
dt.fit(X_train, y_train)
#accuracy_score(dt.predict(X_test), y_test)
dt_imp = list(pd.concat([pd.DataFrame(list(X), columns=['feature']), 
                        pd.DataFrame(dt.feature_importances_, 
                                     columns=['importance'])],
                        axis=1).sort_values(by=['importance'], 
                              ascending = False).iloc[:9,0])

del X_train, X_test, y_train, y_test



# Select most important features
X1 = pd.DataFrame()
for i in dt_imp:
    x=X[i]
    X1 = pd.concat([X1, x], axis=1)
    del i,x

X1.to_csv('X_feat_sel.csv', index=False)

del dt_imp
