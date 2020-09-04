# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 08:26:54 2020

@author: CHEALE
"""

import numpy as np
import pandas as pd 
import matplotlib as plt

dataset2 = pd.read_csv("Data.csv")

X = dataset2.iloc[: , :-1].values
Y = dataset2.iloc[:,3].values

print(X)
print("-------------------------------------")
print(Y)

print(dataset2.describe())

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy = 'mean')
imputer = imputer.fit(X[:,1:3])
X[:, 1:3]= imputer.transform(X[:, 1:3])

