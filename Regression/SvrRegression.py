# -*- coding: utf-8 -*-
"""
Created on Fri Dec 12 07:58:29 2025

@author: arnav
"""
'''
This time we are going to use support vector regression 
we are trying  to find the truth about the person earning 160000 dollars salary in his earlier company 

Here we don't have equation with coefficients that will take care of scaling  , that's why we do need to utilise feature scaling 

Here also we will be training our model on the whole data set without splitting 


'''
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

dataset = pd.read_csv(r"C:\Users\arnav\Downloads\S01 - Machine-Learning-A-Z-Codes-Datasets\Machine-Learning-A-Z-Codes-Datasets\Part 2 - Regression\Section 7 - Support Vector Regression (SVR)\Python\Position_Salaries.csv")

X=dataset.iloc[:,1].values 
y=dataset.iloc[:,-1].values

print(X)
print(y)


'''
earlier we only scaled the features as our dependent variable was taking values either 0 or 1 so we didn't need to scale it '
here we need to apply feature scaling to the salary section . 
also we do not have here an explicit equation like multiple linear regression .

we also apply feature scaling to the level column 
 
'''


# we need to have y as a 2d array because of the input type the standard scaler takes in only 2d arrays 


y= y.reshape(len(y),1)
X= X.reshape(len(X),1)
print(y)

from sklearn.preprocessing import StandardScaler
sc_1 = StandardScaler()

X= sc_1.fit_transform(X)
# we are not going to use the same scaler object for both the x and y because the means and std deviations for both are different . 

sc_2 = StandardScaler()

y=sc_2.fit_transform(y)# scale the dependent variable 

print(X)
print(y)


from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')# for the parameter it is always recommended to use rbf function 
regressor.fit(X,y)

# now how do we reverse the scaling and get the original scale ? 

# we need to get result in the original scaling , that's the reason we are using reverse scaling 

print(sc_2.inverse_transform(regressor.predict(sc_1.transform([[6.5]])).reshape(-1,1)))

plt.scatter(sc_1.inverse_transform(X),sc_2.inverse_transform(y),color='red')

# here the input of inverse transform is also a 2d array  
plt.plot(sc_1.inverse_transform(X),sc_2.inverse_transform(regressor.predict(X).reshape(-1,1)),color='blue')
plt.title("SVR REGRESSION PLOT")
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()




X_grid = np.arange(min(sc_1.inverse_transform(X)),max(sc_1.inverse_transform(X)),0.1)
X_grid=X_grid.reshape(len(X_grid),1)
plt.scatter(sc_1.inverse_transform(X),sc_2.inverse_transform(y),color='blue')
plt.plot(X_grid,sc_2.inverse_transform(regressor.predict(sc_1.transform(X_grid)).reshape(-1,1)),color='red')


plt.title('SVR REGRESSION PLOT')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
















 




