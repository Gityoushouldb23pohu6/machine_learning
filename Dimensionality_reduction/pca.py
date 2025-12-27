# -*- coding: utf-8 -*-
"""
Created on Tue Dec 23 17:07:35 2025

@author: arnav
"""
'''
PCA -> principle  component analysis 

- identify patterns in data 
- detect the correlation between variables 


if there is any correlation between the variables we esentially reduce the dimentionality

- we reduce the dimensions of a d dimensional dataset by by projecting it onto a k dimensional subspace
where k< d 
 
pca is mostly concerned about learning about the relation between different variables 
- we do it by finding a list of principle axes 



-> pca is highly effected by the outliers in the data 

'''

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

# here since we will be reducing the dimension which is equivalent to features , so we are reducing features 

dataset= pd.read_csv(r"C:\Users\arnav\Downloads\S01 - Machine-Learning-A-Z-Codes-Datasets\Machine-Learning-A-Z-Codes-Datasets\Part 9 - Dimensionality Reduction\Section 43 - Principal Component Analysis (PCA)\Python\Wine.csv")

X= dataset.iloc[:,:-1].values 
y= dataset.iloc[:,-1].values 

from sklearn.model_selection import train_test_split 

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)


from sklearn.preprocessing import StandardScaler 

sc= StandardScaler()

X_train=sc.fit_transform(X_train)

X_test = sc.transform (X_test)

from sklearn.decomposition import PCA 
pca = PCA(n_components=2)
 # just like always we will have fit_transform on the train set but only transform on test set as we don't want information leakage as if we applu fit_transform again its like giving a hit to our test set , which we don't want 
 
X_train= pca.fit_transform(X_train)
X_test=pca.transform(X_test)


'''
here we are choosing 2 dimensions to be reduced to ,
how ever if the result doesn't turn out to be good as in if we get our predicted curve unsatisfactory or a lot of misplaced points then we can accordingly increase the dimensions .
another reason for choosing 2 is that for 2 dimensions we can easily plot a 2 d curve related to our result .
 
'''
from sklearn.linear_model import LogisticRegression 
classifier = LogisticRegression()
classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix,accuracy_score 

print(confusion_matrix(y_test,y_pred))
print(accuracy_score(y_test,y_pred))


# thus we see reducing dimensions  led to a higher accurate prediction 
# here we see a confusion matrix of 3 features as we have three categories of wine . The number of correct predictions lie along the diagonal 






