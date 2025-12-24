# -*- coding: utf-8 -*-
"""
Created on Sat Dec 20 07:58:06 2025

@author: arnav
"""

'''
let's say we have two different groups of points Red and green '
if we want to add a new point to this plot , in which group should this point be classified to ? ->KNN

KNN

1) choose number k of the neighbours (usually k=5))
2) Take the k nearest neighbours of the new data point , according to euclidean distance (or any other distance such as manhattan )
3) among these k neighbours count the number of data points in each category 
4) assign the new data to the category where you counted the most neighbours 




'''

import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 

dataset=pd.read_csv(r"C:\Users\arnav\Downloads\S01 - Machine-Learning-A-Z-Codes-Datasets\Machine-Learning-A-Z-Codes-Datasets\Part 3 - Classification\Section 15 - K-Nearest Neighbors (K-NN)\Python\Social_Network_Ads.csv")

X= dataset.iloc[:,:-1].values
y= dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split 

X_train,X_test ,y_train,y_test= train_test_split(X,y,test_size=0.2,random_state=0)

from sklearn.preprocessing import StandardScaler 

sc= StandardScaler()

X_train = sc.fit_transform(X_train)
X_test  = sc.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier 

classifier= KNeighborsClassifier(n_neighbors=5,metric = 'minkowski',p=2)

classifier.fit(X_train,y_train)

y_pred= classifier.predict(X_test)


print(np.concatenate((y_test.reshape(-1,1),y_pred.reshape(-1,1)),axis=1))

from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(y_test,y_pred)
print(cm)
print(accuracy_score(y_test,y_pred))


# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = sc.inverse_transform(X_train), y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 1),
                     np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = 1))
plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('K-NN (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()




'''
here we notice that the prediction line is non linear and trys to keep the different point out of an area . 
Thus this is a more accurate classifier 



'''





# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = sc.inverse_transform(X_test), y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 1),
                     np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = 1))
plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('K-NN (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()




