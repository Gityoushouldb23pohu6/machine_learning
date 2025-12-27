# -*- coding: utf-8 -*-
"""
Created on Thu Dec 25 21:56:06 2025

@author: arnav
"""

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

dataset= pd.read_csv(r"C:\Users\arnav\Downloads\S01 - Machine-Learning-A-Z-Codes-Datasets\Machine-Learning-A-Z-Codes-Datasets\Part 9 - Dimensionality Reduction\Section 43 - Principal Component Analysis (PCA)\Python\Wine.csv")


X= dataset.iloc[:,:-1].values 

y= dataset.iloc[:,-1].values  
from sklearn.model_selection import train_test_split 

X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2,random_state=0)



from sklearn.preprocessing import StandardScaler 

sc=StandardScaler()

X_train=sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.decomposition import KernelPCA 

kpca = KernelPCA(n_components=2,kernel= 'rbf')
X_train = kpca.fit_transform(X_train)
X_test = kpca.transform(X_test)


from sklearn.linear_model import LogisticRegression 
classifier = LogisticRegression()
classifier.fit(X_train,y_train)


y_pred = classifier.predict(X_test)


from sklearn.metrics import confusion_matrix,accuracy_score 

print(confusion_matrix(y_test,y_pred),accuracy_score(y_test,y_pred))

# it is possible that  a non kernel  model defeats a kernel model but its rare *****
