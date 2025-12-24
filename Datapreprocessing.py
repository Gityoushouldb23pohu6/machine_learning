# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 22:37:54 2025

@author: arnav
"""
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 


# now we import the file / data 

dataset = pd.read_csv(r"C:\Users\arnav\Downloads\S01 - Machine-Learning-A-Z-Codes-Datasets\Machine-Learning-A-Z-Codes-Datasets\Part 1 - Data Preprocessing\Section 2 -------------------- Part 1 - Data Preprocessing --------------------\Python\data.csv")

# we first classify the column variables as dependent or independent 
# in most of the cases the last column is the dependent variable and rest all are independent 


# we will be making changes in the features or the dependent var not the dataset itself , that's why we are storing the values in different variables as follows

X= dataset.iloc[:,:-1 ].values # independent variables 

y= dataset.iloc[:,-1].values #dependent variables 
'''
iloc[a:b] means data from a to b excluding b index 

-1 index represents the last column 


with .values we are accessing the values which the columns have 

'''

print(X)
print(y)


'''

now we have to deal with the unfilled spaces in the dataset or NaN .

1) either we drop the entire column which may be okay if the column is just 1 percent of all the data 
   which might not even effect the future model 
   2 ) using mean strategy to fill up the missing values .
   for this we use imputer module of scikit learn (Simple Imputer class )
'''

from sklearn.impute import SimpleImputer # sklearn is the library , imputer is the module whereas Simple Imputer is the class 

imputer = SimpleImputer (missing_values =np.nan, strategy ='mean') # creating the object imputer of class Simple Imputer 

imputer.fit(X[:,1:3])

# most of the times when the data is very large we apply this on all the numerical features , for the model to be good 

X[:,1:3]= imputer.transform(X[:,1:3])  # we are making changes on the specific part so we are replacing the values with the new imputed values 

print(X)

'''
now the model won't be able to make much use of categorical data  , so we need to encode it 
1) one way is to assign values like 0 , 1 , 2 to different countries , but in that case the model will start treating the countries as something to which should be ordered according to the number assigned 
2) we use ONE HOT ENCODER which creates the binary vector equivalent of all the countries 

'''


from sklearn.compose       import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder 


ct = ColumnTransformer (transformers = [('encoder',OneHotEncoder(),[0])],remainder='passthrough')

X=np.array(ct.fit_transform(X)) # in default the fit_transform method doesn't return an array , so we convert it to an array and replace it with X 

print(X)

#in the same way we can encode the dependent variable y so that it facilitates the working of the model 

# for this we use something called Label Encoder 

from sklearn.preprocessing import LabelEncoder 

le=LabelEncoder ()
y=le.fit_transform(y)


'''
*** we have to apply feature scaling after splitting the data set into training set and testing set 
    

we split the data into training and testing so as to train our model to predict the y values for given x test and then compare it with the y test
 
where as scaling involves scaling all the feature so that have values in the same scale which would avoid dominating of one variable over the other for the ml model to work at the finest


-> The test set is not supposed to be changed in any way , as it is just used to test the model we build with the training set .

scaling the values makes it easier for the model to work on the features .
however, if we apply scaling before splitting , we would also scale the test set which would cause information leaking and is quite unnecessary and would cause resourse waste .

INFORMATION LEAKAGE ON THE TEST SET 

'''

'''
for train test split we will use a module called model selection 

we want X test , X train , y test and y train 
'''

from sklearn.model_selection import train_test_split 
# here the train_test_split is the function returning values of 4 variables 

X_train , X_test,y_train,y_test= train_test_split(X,y,test_size= 0.2 ,random_state = 1  )


# we want most of the data to go into training the model so recomended use of test_size is 20 percent 
# the test data is not nessecary the last 2 but can be any two and hence random 
# also we don't want our random selection to change always so we fix the random state to 1 . 

print(X_train)
print(X_test )
print(y_train)
print(y_test )


# here the y train corresponds to the y values of the x values in X train itself .

'''
now we are going to proceed with feature scaling

why ? 
we do not want some of the features to dominate the others
  However for many models we don't  even need to apply feature scaling 
  reason: 
      regression-> y= b0 + b1x1 + b2x2 +.... + bnxn
      for larger values of x the coefficient might be small 
      
      Thus , the coefficient might adjust themselves which would result in no need of feature scaling
      
      two methods of feature scaling : 
          1) Standardisation : 
            
            xstand=(x-mean)/ std deviation 
            the values will be between -3 to 3 
            
          2) Normalisation :
             xnorm= (x-min)/(max-min )
             all values between 0 and 1 .
             
    ** normalisation will be best if we are dealing with features that are normally distributed .
    ** Standardisation is best in all scenarios 
       
Therefore we will be using standardisation the most 
            
             
            we won't be applying feature scaling on x_test(which is given to us after the model is built ) as we want the x_test value to be something new to the model .'(if we are getting the new values of x test  later for our model testing)
            
'''


from sklearn.preprocessing import StandardScaler

sc= StandardScaler () # creating an instance of the Standard Scaler class 

'''
*** do we need to use standardisation to the dummy variables in our features ?

 ans -> we do not
since dummy variables already take values from -3 to 3 (0 or 1 precisely)
 if we apply feature scaling on dummy varibles it will be difficult here to identify the country with respect to the binary vector 
 
'''
X_train[:,3:]    = sc.fit_transform(X_train[:,3:])  #this will take all columns from 3 index 


# fit will calculate the mean and std deviation but tranform will actually apply the formula to calculate the standardised value 

# now since here we have x_test we will scale it  for better building of the model 


# we will apply only transform on x_test rather than fit_transform as we will use the same transformer for x test as was for x_train 
#inorder to make predictions congruent to the trained set , we would use the same transformer .
#if we would apply fit_transform on x_test as well then we would create a whole new transformer 

X_test[:,3:] = sc.transform(X_test[:,3:])
print(X_train)
print(X_test)















 

