# -*- coding: utf-8 -*-
"""
Created on Thu Dec 18 09:15:44 2025

@author: arnav
"""

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

dataset = pd.read_csv(r"C:\Users\arnav\Downloads\S01 - Machine-Learning-A-Z-Codes-Datasets\Machine-Learning-A-Z-Codes-Datasets\Part 3 - Classification\Section 14 - Logistic Regression\Python\Social_Network_Ads.csv")
X= dataset.iloc[:,:-1].values
y= dataset.iloc[:,-1].values 


from sklearn.model_selection import train_test_split 

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0)

from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
X_train= sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.linear_model import LogisticRegression 

classifier= LogisticRegression(random_state= 0 )
classifier.fit(X_train,y_train)

print(classifier.predict(sc.transform([[30,87000]])))
# the predict method will predict the correct value only if the input values are scaled just the way(using same method as ) the training set 




y_pred = classifier.predict(X_test)

print(np.concatenate((y_pred.reshape(-1,1),y_test.reshape(-1,1)),axis=1))






'''
THE CONFUSION MATRIX 
will exactly tell us how many of the values predicted were correct and how many were not 

'''


from sklearn.metrics import confusion_matrix,accuracy_score 
 

cm = confusion_matrix(y_test,y_pred)

print(cm)


print(accuracy_score(y_test,y_pred))

'''
here seeing the plot , there are two partitions with different colors . if there are points which are of different color than the surrounding than it is a wrong prediction 
the prediction boundary is the separation between the green prediction and red prediction


Logistic regression is
 a linear classifier 

for a linear classifier the prediction boundary will always be a straight line 

in order to get a better classifier , we need to use a different shaped prediction curve 

'''







