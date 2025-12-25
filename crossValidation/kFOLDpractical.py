# -*- coding: utf-8 -*-
"""
Created on Tue Dec 23 07:22:26 2025

@author: arnav
"""
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 


dataset=pd.read_csv(r"C:\Users\arnav\Downloads\S01 - Machine-Learning-A-Z-Codes-Datasets\Machine-Learning-A-Z-Codes-Datasets\Part 3 - Classification\Section 19 - Decision Tree Classification\Python\Social_Network_Ads.csv")



X= dataset.iloc[:,:-1].values 
y=dataset.iloc[:,-1].values 

from sklearn.model_selection import train_test_split 

X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2,random_state=0)


from sklearn.preprocessing import StandardScaler 

sc= StandardScaler()

X_train=sc.fit_transform(X_train)

X_test=sc.transform(X_test)


from sklearn.svm import SVC 

classifier = SVC(kernel='rbf',random_state=0)

classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)


from sklearn.metrics import confusion_matrix, accuracy_score 

print(confusion_matrix(y_test,y_pred))
print(accuracy_score(y_test,y_pred))


#apply k fold cross validation 

from sklearn.model_selection import cross_val_score 
accuracies = cross_val_score(estimator=classifier,X=X_train,y=y_train,cv=10)
# this will return the 10 accuracies of models, from 10 fold models that have been created 
#cv is the number of folds we want to create 

# if we are importing a very large dataset we can use the n_jobs parameter and set it to -1 . 

accuracies.mean()
accuracies.std() # 4.8 percent std dev 




