# -*- coding: utf-8 -*-
"""
Created on Sun Dec 14 10:37:57 2025

@author: arnav
"""


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

dataset = pd.read_csv(r"C:\Users\arnav\Downloads\S01 - Machine-Learning-A-Z-Codes-Datasets\Machine-Learning-A-Z-Codes-Datasets\Part 2 - Regression\Section 8 - Decision Tree Regression\Python\Position_Salaries.csv")

X= dataset.iloc[:,1:-1].values
y=dataset.iloc[:,-1].values






from sklearn.ensemble import RandomForestRegressor 

regressor=RandomForestRegressor(n_estimators =10 , random_state = 0)
regressor.fit(X,y)
#n_estimator is the number of trees we are choosing 


# we are anyhow not going to see any good predictions as here too random forest works best for multidimensional dataset 


print(regressor.predict([[6.5]]))


X_grid = np.arange(min(X),max(X),0.1)
X_grid = X_grid.reshape(-1,1)

plt.scatter(X,y,color='red')
plt.plot(X_grid,regressor.predict(X_grid),color='blue')
plt.title('Random Forest(Truth or bluff)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()








