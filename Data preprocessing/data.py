# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 12:48:42 2020

@author: CHEALE
"""


'''
DATA PREPROCESSING STEPS
1. Import the libraries
2. import the dataset
3. split into training and testing sets

'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#importing dataset
dataset = pd.read_csv("Data.csv")

#independent variables selecting 3 columns
X = dataset.iloc[: , :-1].values

#importing dependent variables
Y = dataset.iloc[:, 3].values

#prints the details of the dataset
#print(dataset.describe())

'''
#how to work with missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy='mean')
imputer = imputer.fit(X[:, 1:3])

#replace the missing values with the mean of colums
X[:, 1:3]= imputer.transform(X[:, 1:3])


#Encoding categorical data
#Encoding Independent variable
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# takes a tuple as name, transformer, column[s]
states = ColumnTransformer(transformers=[('one_hot_encoder', OneHotEncoder(categories='auto'),[0])]
                       , remainder='passthrough')

X = np.array(states.fit_transform(X) , dtype=np.float)


#Encoding dependent variables
# used label encoder because of yes/no answers
from sklearn.preprocessing import LabelEncoder
purchased = LabelEncoder()
Y = purchased.fit_transform(Y)
'''


#splitting dataset into training and testing set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20,random_state=0)

'''

# feature scaling when the distance between colums are large

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
'''