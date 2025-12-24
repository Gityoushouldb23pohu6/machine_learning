# -*- coding: utf-8 -*-
"""
Created on Thu Dec 18 22:54:35 2025

@author: arnav
"""

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

dataset=pd.read_csv(r"C:\Users\arnav\Downloads\Logistic_Regression\Final Folder\Dataset\breast_cancer.csv")

X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split 

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

from sklearn.preprocessing import StandardScaler

sc_X= StandardScaler()
sc_y=StandardScaler()



X_train=sc_X.fit_transform(X_train)
X_test =sc_X.transform(X_test)




from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state=0)
classifier.fit(X_train,y_train)


y_pred = classifier.predict(X_test)

print(np.concatenate((y_test.reshape(-1,1),y_pred.reshape(-1,1)),axis=1))


from sklearn.metrics import confusion_matrix ,accuracy_score

print(confusion_matrix(y_test,y_pred))


print(accuracy_score(y_test,y_pred))


