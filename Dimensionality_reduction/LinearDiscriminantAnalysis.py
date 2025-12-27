# -*- coding: utf-8 -*-
"""
Created on Wed Dec 24 23:29:17 2025

@author: arnav
"""

'''
Linear discriminant analysis 

- used in preprocessing step for pattern classifiaction 


it is similar to pca but in here in addition to finding the component axes we are also interested in axws 
that maximize the separartion between multiple classes 

here we project a feature space onto small subspace maintaining the class discriminatory information 


pca is unsupervised but lda  is supervised because of relation to dependent variable 

'''

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

dataset = pd.read_csv(r"C:\Users\arnav\Downloads\S01 - Machine-Learning-A-Z-Codes-Datasets\Machine-Learning-A-Z-Codes-Datasets\Part 9 - Dimensionality Reduction\Section 43 - Principal Component Analysis (PCA)\Python\Wine.csv")

X= dataset.iloc[:,:-1].values
y= dataset.iloc[:,-1].values 

from sklearn.model_selection import train_test_split 
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0 )

from sklearn.preprocessing import StandardScaler 

sc= StandardScaler()

X_train=sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA 
lda = LDA(n_components= 2)
X_train= lda.fit_transform(X_train,y_train)   # NOTE!! we also need to input the dependent variable also for lda unlike the pda 

X_test= lda.transform(X_test) # here we only need to have X_test as we can't enter y_test and we cant use it to train(its hidden)



from sklearn.linear_model import LogisticRegression 

classifier=LogisticRegression()

classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)


from sklearn.metrics import confusion_matrix , accuracy_score 

print(confusion_matrix(y_test,y_pred),accuracy_score(y_test,y_pred))









