# -*- coding: utf-8 -*-
"""
Created on Wed Dec 10 10:15:54 2025

@author: arnav
"""
'''
here we are going to skip the step of splitting the training and testing model 
as we want to use maximum quantity of the dataset to predict future salary for some level between 6 and 7 (say 6.5)

'''
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 


dataset = pd.read_csv(r"C:\Users\arnav\Downloads\S01 - Machine-Learning-A-Z-Codes-Datasets\Machine-Learning-A-Z-Codes-Datasets\Part 2 - Regression\Section 6 - Polynomial Regression\Python\Position_Salaries.csv")

# here as we can notice the first column is already label encoded  , we do not further need to include it in our features .


X= dataset.iloc[:,1:-1].values 

y= dataset.iloc[:,-1 ].values 


from sklearn.linear_model import LinearRegression 
lin_reg=LinearRegression()

lin_reg.fit(X,y)


from sklearn.preprocessing  import PolynomialFeatures 

poly_reg=PolynomialFeatures(degree=4)


# here one of the parameters of the Polynomial features is n which will be the degree of the equation 
 
'''

 for the equation we will consider the features to be x1,x1^2 (n=2)
 
'''
X_poly = poly_reg.fit_transform( X) # the new matrix of features 

# now we will apply linear regression to these features 

lin_reg2= LinearRegression ()
lin_reg2.fit(X_poly,y)
# this is a polynomial regression model 


plt.scatter(X,y,color= 'blue')
plt.plot(X,lin_reg.predict(X),color='blue')
plt.title('Truth or bluff(Linear regression model) ')
plt.xlabel('Position level')
plt.ylabel('Salary')

plt.show()

plt.scatter(X,y,color='red')
plt.plot(X,lin_reg2.predict(X_poly),color='blue')
plt.title('Truth or bluff (Poly regression )')
plt.xlabel('Position level')
plt.ylabel('salary')

plt.show()

# for better resolution and smoother curve : 
    
X_grid = np.arange(min(X),max(X),0.1)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X,y,color='red')   
plt.plot(X_grid,lin_reg2.predict(poly_reg.fit_transform(X_grid)),color='blue')
plt.title ('Truth or Bluff (Polynomial Regression )')
plt.xlabel('Position level ')
plt.ylabel('Salary')
plt.show()    

pred=lin_reg.predict([[6.5]])  # the first bracket in here corresponds to the row and second corresponds to column  , we are covering both the dimensions 

print(pred )

pred2= lin_reg2.predict(poly_reg.fit_transform([[6.5]]))


print(pred2)







     






