#simple Linear regression


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# get the data
data = pd.read_csv('Salaries.csv')
X = data.iloc[:, 0].values
Y = data.iloc[: , 1].values

# split the data
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=1/3, random_state=0)


# Fitting simple linear regression to the training dataset

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
X_train = X_train.reshape(-1,1)  # converts it to a 2D array
Y_train = Y_train.reshape(-1,1)
regressor.fit(X_train, Y_train)  # uses the linear regression model




import pickle 
  
# Save the trained model as a pickle string. 
saved_model = pickle.dumps(regressor) 
  
# Load the pickled model 
Linear_regression_model = pickle.loads(saved_model) 
  



# predicting the Test set results
X_test = X_test.reshape(-1,1)
Y_predict = regressor.predict(X_test)



# Use the loaded pickled model to make predictions 
Linear_regression_predict =  Linear_regression_model .predict(X_test) 



# Saving the model to a file
from sklearn.externals import joblib 
  
# Save the model as a pickle in a file 
joblib.dump(regressor, 'LinearRegressionModel.pkl') 
  
# Load the model from the file 
Regressor_from_joblib = joblib.load('LinearRegressionModel.pkl')
  
# Use the loaded model to make predictions 
Joblib_model_predict =  Regressor_from_joblib .predict(X_test) 



'''
print(regressor.score(X_train, Y_train))
print(regressor.score(X_test, Y_test))
'''
# visualizing results: Training set
plt.scatter(X_train, Y_train)
# plots the best fit line
plt.plot(X_train, regressor.predict(X_train), color='red')
plt.title('Salary vs Experience (Training set results)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


# visualizing results: Test set
plt.scatter(X_test, Y_test)
# plots the best fit line
plt.plot(X_train, regressor.predict(X_train), color='red')
plt.title('Salary vs Experience (Testing set results)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

