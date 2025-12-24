# -*- coding: utf-8 -*-
"""
Created on Fri Dec 12 20:36:27 2025

@author: arnav
"""
'''
DESICION TREE REGRESSOR : 
    
again here , we don't have to do any feature scaling '


Here we are only using one feature , but the following code will work for any dataset

'''

import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 

dataset=pd.read_csv(r"C:\Users\arnav\Downloads\S01 - Machine-Learning-A-Z-Codes-Datasets\Machine-Learning-A-Z-Codes-Datasets\Part 2 - Regression\Section 8 - Decision Tree Regression\Python\Position_Salaries.csv")

X=dataset.iloc[:,1:-1].values
y= dataset.iloc[:,-1].values

'''
according to our dataset we need to use OneHotEncoder in case we have categorical data . 
also if the categorical data has some order then we can use label encoder 
we can also split the data set into test set and train set according to our dataset 

but we don't have to use feature scaling as both random forest and decision tree regression  '

'''

from sklearn.tree import DecisionTreeRegressor 
regressor=DecisionTreeRegressor(random_state=0)
regressor.fit(X,y)

print(regressor.predict([[6.5]]))
#this doesn't really produce a good result cause as said , the decision tree regressor is good for more than one feature value 

X_grid = np.arange(min(X),max(X),0.1)
X_grid=X_grid.reshape(len(X_grid),1)
plt.plot(X_grid,regressor.predict(X_grid),color='blue')
plt.scatter(X,y,color='red')
plt.title('Truth or bluff(Decision Tree)')
plt.xlabel('Position level')
plt.ylabel('Salary')

plt.show()













