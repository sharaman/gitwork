# -*- coding: utf-8 -*-
"""
Created on Fri Sep  7 12:50:34 2018

@author: Artem.Skachenko
"""


import pandas as pd
import  numpy as np


# Data
df = pd.read_csv("banking.csv")


# Reduce amount of classes
df['education'].unique()
df['education']=np.where(df['education'] =='basic.9y', 'Basic', df['education'])
df['education']=np.where(df['education'] =='basic.6y', 'Basic', df['education'])
df['education']=np.where(df['education'] =='basic.4y', 'Basic', df['education'])
df['education'].unique()

# Deal with categorial variables
# Encoding and dummies
from sklearn.preprocessing import LabelEncoder
encod = LabelEncoder()
dummy=pd.DataFrame()
for i in df.select_dtypes(include=['object']):
    # Count lavels of variable
    a = df[i].value_counts().count()
    # Encode if 2, get dummy if more
    if a <= 2:
       one = pd.DataFrame(encod.fit_transform(df[i]), columns=[i])
    elif a > 2:
       one = pd.get_dummies(df[i], prefix=i)
    # Save to one DF
    dummy = pd.concat([dummy,one], axis=1)
    del i,a,one


df1 = pd.concat([df.select_dtypes(exclude=['object']),dummy], axis=1)
del dummy
df1.info()


y = df1.iloc[:,10]
X = pd.concat([df1.iloc[:,0:10], df1.iloc[:,11:]], axis=1)

import statsmodels.api as sm
X = sm.add_constant(X)


y.to_csv('y.csv', index=False)
X.to_csv('X_preproc.csv', index=False)

# hello
# new modification
