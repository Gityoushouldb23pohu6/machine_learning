# -*- coding: utf-8 -*-
"""
Created on Sun Dec 21 15:53:35 2025

@author: arnav
"""
'''
accuracy rate is (total correct/total ) and error rate is (totalincorrect/total)

'''

import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 

dataset=pd.read_csv(r"C:\Users\arnav\Downloads\Data (4).csv")

X= dataset.iloc[:,:-1].values 
y=dataset.iloc[:,-1].values 

from sklearn.model_selection import train_test_split 

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

from sklearn.preprocessing import  StandardScaler 
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test= sc.transform(X_test)




from sklearn.linear_model import LogisticRegression 

classifier = LogisticRegression(random_state=0)

classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test) 

print(np.concatenate((y_test.reshape(-1,1),y_pred.reshape(-1,1)),axis=1))


from sklearn.metrics import confusion_matrix , accuracy_score 

print(confusion_matrix(y_test,y_pred))
print(accuracy_score(y_test,y_pred))

from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)
classifier.fit(X_train,y_train)

y_pred =classifier.predict(X_test)

print(np.concatenate((y_test.reshape(-1,1),y_pred.reshape(-1,1)),axis=1))


print(confusion_matrix(y_test,y_pred))
print(accuracy_score(y_test,y_pred))



from sklearn.svm import SVC 

classifier=SVC(kernel='rbf',random_state=0)

classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)


print(np.concatenate((y_test.reshape(-1,1),y_pred.reshape(-1,1)),axis=1))

print(confusion_matrix(y_test,y_pred))
print(accuracy_score(y_test,y_pred))




classifier = SVC(kernel='linear',random_state=0)

classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)


print(confusion_matrix(y_test,y_pred))
print(accuracy_score(y_test,y_pred))


from sklearn.naive_bayes import GaussianNB

classifier= GaussianNB()

classifier.fit(X_train,y_train)


y_pred = classifier.predict(X_test)

print(np.concatenate((y_test.reshape(-1,1),y_pred.reshape(-1,1)),axis=1))


print(confusion_matrix(y_test,y_pred))
print(accuracy_score(y_test,y_pred))


from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion='entropy',random_state=0)

classifier.fit(X_train,y_train)

y_pred =classifier.predict(X_test)


print(np.concatenate((y_test.reshape(-1,1),y_pred.reshape(-1,1)),axis=1))

print(confusion_matrix(y_test,y_pred))
print(accuracy_score(y_test,y_pred))


from sklearn.ensemble import RandomForestClassifier
classifier= RandomForestClassifier(n_estimators = 100, criterion= 'entropy', random_state = 0)
classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)

print(np.concatenate((y_test.reshape(-1,1),y_pred.reshape(-1,1)),axis=1))

print(confusion_matrix(y_test,y_pred))
print(accuracy_score(y_test,y_pred))




'''
the mangalorian phycology works for those who have been approved to stay within the certain restrictions 
'''




