# -*- coding: utf-8 -*-
"""
Created on Wed Dec 17 15:57:17 2025

@author: arnav
"""

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

dataset = pd.read_csv(r"C:\Users\arnav\Downloads\Data (3).csv")

X= dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values 


from sklearn.model_selection import train_test_split 
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)


#Multilinear regression
from sklearn.linear_model import LinearRegression

r1= LinearRegression()

r1.fit(X_train,y_train)

p1= r1.predict(X_test)

print(np.concatenate((y_test.reshape(len(y_test),1),p1.reshape(len(p1),1)),axis=1))

from sklearn.metrics import r2_score 
print(r2_score(y_test,p1))
# Decision Tree

from sklearn.tree import DecisionTreeRegressor

r2= DecisionTreeRegressor()

r2.fit(X_train,y_train)

p2=r2.predict(X_test)

print(np.concatenate((y_test.reshape(len(y_test),1),p2.reshape(len(p2),1)),axis=1 ))

print(r2_score(y_test,p2))
# Random forest 
from sklearn.ensemble import RandomForestRegressor 

r3= RandomForestRegressor()
r3.fit(X_train,y_train)

p3= r3.predict(X_test)

print(np.concatenate((y_test.reshape(len(y_test),1),p3.reshape(len(p3),1)),axis=1))

print(r2_score(y_test,p3))
#Polynomial Regression

from sklearn.preprocessing import PolynomialFeatures 

x_f= PolynomialFeatures(degree = 5)

r4= LinearRegression()

r4.fit(x_f.fit_transform(X_train),y_train)

p4= r4.predict(x_f.fit_transform(X_test))

print(np.concatenate((y_test.reshape(len(y_test),1),p4.reshape(len(p4),1)),axis=1))

print(r2_score(y_test,p4))
#SVR regression

from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
sc_y=StandardScaler()

y_train = y_train.reshape(-1,1)
X_train=sc_X.fit_transform(X_train)

y_train=sc_y.fit_transform(y_train)



X_tr =  sc_X.transform(X_test)


from sklearn.svm import SVR 

r5= SVR(kernel='rbf')
r5.fit(X_train,y_train)

p5=r5.predict(X_tr)

p5= sc_y.inverse_transform(p5.reshape(-1,1))
print(p5)
print(np.concatenate((y_test.reshape(-1,1),p5.reshape(-1,1)),axis = 1 ))

print(r2_score(y_test,p5))

'''
We want to now evaluate the models  using the r squared 

The closest the r2 to 1 the better the model is 
'''

'''
Therefore the best model is RandomForestRegression 
'''








