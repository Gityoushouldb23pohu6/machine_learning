# -*- coding: utf-8 -*-
"""
Created on Tue Dec 23 12:50:58 2025

@author: arnav
"""

'''
XGBOOST 
'''


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

dataset= pd.read_csv(r"C:\Users\arnav\Downloads\S01 - Machine-Learning-A-Z-Codes-Datasets\Machine-Learning-A-Z-Codes-Datasets\Part 10 - Model Selection and Boosting\Section 49 - XGBoost\Python\Data.csv")

X= dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values 

from sklearn.model_selection import train_test_split 

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder ()
y_train=le.fit_transform(y_train)
y_test=le.fit_transform(y_test)

from xgboost import XGBClassifier

classifier= XGBClassifier()

classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test) 


from sklearn.metrics import confusion_matrix , accuracy_score 

print(confusion_matrix(y_test,y_pred))
print(accuracy_score(y_test,y_pred))


'''

in the same way we can use XGBRegressor 

'''

from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator=classifier,X=X_train,y=y_train,cv=10,n_jobs=-1)

print(accuracies.mean())

print(accuracies.std())

