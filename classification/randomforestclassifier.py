# -*- coding: utf-8 -*-
"""
Created on Sun Dec 21 15:32:08 2025

@author: arnav
"""
'''
ensemble learning -> multiple things combined 


RANDOM FOREST CLASSIFICATION 


1) pick at random k datapoints from training set 
2) build decision tree associated with these k points .
3) chosse n_trees you want to build and repeat previous steps 
4) for a new data point make each one of your n trees predict the category to which this data point belongs and based on majority vote classify the point 



basically we are using many decision trees 

'''

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

dataset = pd.read_csv(r'C:\Users\arnav\Downloads\S01 - Machine-Learning-A-Z-Codes-Datasets\Machine-Learning-A-Z-Codes-Datasets\Part 3 - Classification\Section 15 - K-Nearest Neighbors (K-NN)\Python\Social_Network_Ads.csv')


X= dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values


from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2,random_state=0)

from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
X_train= sc.fit_transform(X_train)
X_test  = sc.transform(X_test)

from sklearn.ensemble import RandomForestClassifier

classifier= RandomForestClassifier(n_estimators=10,criterion='entropy',random_state=0)


classifier.fit(X_train,y_train)


y_pred =classifier.predict(X_test)

print(np.concatenate((y_test.reshape(-1,1),y_pred.reshape(-1,1)),axis=1))

from sklearn.metrics import confusion_matrix,accuracy_score

print(confusion_matrix(y_test,y_pred))
print(accuracy_score(y_test,y_pred))


'''
The plot for the random _forest is also scattered like 

'''





