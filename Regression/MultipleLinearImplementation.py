# -*- coding: utf-8 -*-
"""
Created on Mon Dec  8 22:32:23 2025

@author: arnav
"""

'''
we would need to apply one hot encoding to the categorical variable column 
 
   
the dummy variables mentioned earlier is nothing but the one hot encoded column 


'''
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

dataset=pd.read_csv(r"C:\Users\arnav\Downloads\S01 - Machine-Learning-A-Z-Codes-Datasets\Machine-Learning-A-Z-Codes-Datasets\Part 2 - Regression\Section 5 - Multiple Linear Regression\Python\50_Startups.csv")

X= dataset.iloc[:,:-1].values

y=dataset.iloc[:,-1].values

from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OneHotEncoder

ct=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[3])],remainder='passthrough')

X= np.array(ct.fit_transform(X))


from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size =0.2,random_state=0)


'''
we do not have to apply feature scaling here as the coefficients of the independent variables will automatically scale the values 
here we do not need to check if the dataset has linear behaviours  as if it doesn't then the model we construct will just behave poorly 


we do not explicitly have to take care of the dummy variable trap 
the class we import for multiple linear regression will take care of it itself 

also we do not need to explicitly impliment the backward elimination 
the class will take care of it 

'''

from sklearn.linear_model import LinearRegression 
regressor=LinearRegression()

regressor.fit(X_train,y_train)

# here we do not plot the features graph

# we are going to display two vectors , one the real test profit and the other the predicted profit 

y_pred =regressor.predict(X_test)

np.set_printoptions(precision =2) # here inside the reshape function we pass first the number of rows we need which is the original array length , and second the number of columns here (1)
 # this will allow us to print values upto 2 decimal places 
 
print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),axis=1))

# here axis = 0 means vertical concatenation and 1 means horizontal concatenation 

print(regressor.coef_)
print(regressor.intercept_)


print(regressor.predict([[0,1,0,159874,199998,725678]]))










