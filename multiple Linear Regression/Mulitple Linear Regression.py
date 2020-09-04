# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 02:58:17 2020

@author: CHEALE
"""

# Data Pre-processing

# 1 Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 2. Importing dataset
dataset = pd.read_csv("Org_data.csv")
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 4].values



# 3. Encoding categorical data

# Encoding independent variable
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer(transformers=[('one_hot_encoder',OneHotEncoder(categories='auto'),[3])],
                                     remainder='passthrough')

X = np.array(ct.fit_transform(X), dtype=np.float)

# Avoid dummy variables (always exclude one column after encoding the categorical data)
X = X[:, 1:]

# 4. Split dataset into train and test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.20,random_state=0)


#fitting Multiple Linear Regression to Training set
from sklearn.linear_model import LinearRegression
multiple_regressor = LinearRegression()
multiple_regressor.fit(X_train, Y_train)

from sklearn.externals import joblib
joblib.dump(multiple_regressor, 'multiple_regression.pkl')
saved_model = joblib.load('multiple_regression.pkl')


# predicting test set results
Y_predict = multiple_regressor.predict(X_test)


