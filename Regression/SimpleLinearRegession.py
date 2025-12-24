# -*- coding: utf-8 -*-
"""
Created on Fri Dec  5 22:24:08 2025

@author: arnav
"""


# data preprocessing 


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

dataset=pd.read_csv(r"C:\Users\arnav\Downloads\S01 - Machine-Learning-A-Z-Codes-Datasets\Machine-Learning-A-Z-Codes-Datasets\Part 2 - Regression\Section 4 - Simple Linear Regression\Python\Salary_Data.csv")
X= dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size =0.2, random_state = 0)

# now we will import the Linear Regression  class
# regression 
from sklearn.linear_model import LinearRegression
regressor = LinearRegression ()
regressor.fit(X_train , y_train )

# now we have to predict the test set results

y_pred =regressor.predict(X_test)
 # we have the predicted salary corresponding to the X_test 
 
plt.scatter(X_train,y_train, color='red') # it will plot the real points of X vs Y , plt here is the module 

# we now plot the regression line using plot() function 

plt.plot(X_train,regressor.predict(X_train),color ='blue')
plt.title('Salary vs experience (Training set)')
plt.xlabel('years of experience')
plt .ylabel('Salary')
plt.show()

'''
doing the same for the test set :
    we do not need to change the regression line to be plotted as we will be using the same line as it is unique. So, we will be anyhow getting the same line 
'''

plt.scatter(X_test,y_test, color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('Salary vs experience(test set )')
plt.xlabel('Years of experience ')
plt.ylabel('Salary')
plt.show()
 
#we see that the predicted salary is close enough to the test set

# but how do we now predict salary for a single employee ? 

print(regressor.predict([[12]]))

'''
we are using a 2D array format of 12 because the predict method expects a 2D array as an input 
# so the salary of an employee with 12 years of experience should be around  138531.00067138
'''


# now how do we get the coefficients of the regression line equation ?
'''
we call the attributes of coef_ and intercept_ to get those 
here attributes are single values and are different from methods
'''
print(regressor.coef_)
print(regressor.intercept_)






