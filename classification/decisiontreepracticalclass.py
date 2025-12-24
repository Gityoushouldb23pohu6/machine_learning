# -*- coding: utf-8 -*-
"""
Created on Sun Dec 21 12:56:05 2025

@author: arnav

"""


import numpy as np 

import pandas as  pd 
import matplotlib.pyplot as plt 


dataset = pd.read_csv(r'C:\Users\arnav\Downloads\S01 - Machine-Learning-A-Z-Codes-Datasets\Machine-Learning-A-Z-Codes-Datasets\Part 3 - Classification\Section 15 - K-Nearest Neighbors (K-NN)\Python\Social_Network_Ads.csv')


X= dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values 



from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2,random_state=1)

from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train= sc.fit_transform(X_train)
X_test= sc.transform(X_test)

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion='entropy',random_state=0)
classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)


print(np.concatenate((y_test.reshape(-1,1),y_pred.reshape(-1,1)),axis=1))



from sklearn.metrics import confusion_matrix, accuracy_score 

cm= confusion_matrix(y_test,y_pred)
a= accuracy_score(y_test,y_pred)

print(cm)
print(a)


'''
the plot for decision tree classification is kind of scattered regions 

'''



