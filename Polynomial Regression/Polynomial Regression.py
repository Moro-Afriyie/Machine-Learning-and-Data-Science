# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 04:36:57 2020

@author: CHEALE
"""

# Data Pre-processing

# Importing libraries
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

# Importing dataset
dataset = pd.read_csv("Gaming_data.csv")
X = dataset.iloc[:, 0:1].values
Y = dataset.iloc[:, 1].values

# Split dataset into train and test set
#from sklearn.model_selection import train_test_split
#X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.20,random_state=0)


# fitting Linear Regression to dataset
from sklearn.linear_model import LinearRegression
linear_regression = LinearRegression()
linear_regression.fit(X,Y)

# fitting Polynomial regression to dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)

lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly, Y)



#visualizing Linear regression results
plt.scatter(X, Y)
# plot regressio line
plt.plot(X, linear_regression.predict(X), color='red')
plt.title('Gaming Data (Linear Regression)')
plt.xlabel('Gaming Steps')
plt.ylabel('Points')
plt.show()


#visualizing Polynomial regression results
plt.scatter(X, Y)
# plot regression line
plt.plot(X, lin_reg2.predict(poly_reg.fit_transform(X)), color='red')
plt.title('Gaming Data (Polynomial Regression)')
plt.xlabel('Gaming Steps')
plt.ylabel('Points')
plt.show()


# Predicting new result with linear Regression
print(linear_regression.predict([[7.5]]))

# Predicting new result with polynomial Regression
print(lin_reg2.predict(poly_reg.fit_transform([[7.5]])))
