# -*- coding: utf-8 -*-
"""
Created on Tue Dec 23 11:38:28 2025

@author: arnav
"""
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

dataset=pd.read_csv(r"C:\Users\arnav\Downloads\S01 - Machine-Learning-A-Z-Codes-Datasets\Machine-Learning-A-Z-Codes-Datasets\Part 3 - Classification\Section 19 - Decision Tree Classification\Python\Social_Network_Ads.csv")

X= dataset.iloc[:,:-1].values 
y= dataset.iloc[:,-1].values 

from sklearn.model_selection import train_test_split 

X_train,X_test,y_train,y_test = train_test_split (X,y,test_size =0.2 , random_state=0)

from sklearn.preprocessing import StandardScaler 
sc= StandardScaler()

X_train=sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.svm import SVC
classifier= SVC(kernel='rbf',random_state= 0)
classifier.fit(X_train,y_train)

y_pred =classifier.predict(X_test)

print(np.concatenate((y_test.reshape(-1,1),y_pred.reshape(-1,1)),axis=1))


from sklearn.metrics import confusion_matrix,accuracy_score 

print(confusion_matrix(y_test,y_pred))
print(accuracy_score(y_test,y_pred))

from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator=classifier,X=X_train,y=y_train,cv=10)


print('Accuracy: {:.2f} %'.format(accuracies.mean()* 100))
print('Standard deviation : {:.2f} %'.format(accuracies.std()*100))

print(accuracies )

from sklearn.model_selection import GridSearchCV
# we will be implementing hyperparameter runing using grid search 

# we will first be passing all the parameteres we want to tune during the process to obtain better results , in the form of a python list 
parameters = [{'C':[0.25,0.5,0.75,1],'kernel':['linear']},
              {'C':[0.25,0.5,0.75,1],'kernel':['rbf'],'gamma':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]}]

grid_search= GridSearchCV(estimator=classifier,
                          param_grid=parameters,
                          scoring='accuracy',
                          cv=10,
                          n_jobs=-1)
grid_search.fit(X_train,y_train)

'''

We will be using two different dictionaries as the gamma parameter that we will use will only work with rbf and not linear svm 
# in such a case se will be having one non linear and one linear dict 
C is the regularisation parameter 


'''


best_accuracy= grid_search.best_score_

# from the grid_search object we are calling one attribute which is the best_score_
best_parameters=grid_search.best_params_

print('Best_Accuracy  : {:.2f} % '.format(best_accuracy*100))
print('Best parameters : ',best_parameters )













