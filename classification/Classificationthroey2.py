# -*- coding: utf-8 -*-
"""
Created on Sun Dec 21 16:48:07 2025

@author: arnav
"""

'''

accuracy paradox is where we do not use the model itself and have the same values predicted every time . This can lead to a false accuracy rate 

lets say we plot a graph between total contacted people and the customers you purchased 

say we  assume 10 percent of contacted poeple actually purchased 
we plot a linear graph passing through the origin(red) 

and in the second step we try to contact only those who have a high chance of purchasing based on the factor that they have the same customer behaviours as we have already encountered . 

based on this we plot another curve(blue) 

The area between these two curve , if large tells us that the model is good but if less tells us that the model needs improvement

THIS CURVE IS CALLED AS CAP CURVE .

we can then assess the models by using this cap graph  ,plotting the curve for various models 

the blue line is random model .


There is also the ideal line (crystal ball) this would happen if(more than 10 percent contacted ) all the contacted people acutually purchase 

cap -> CUMULATIVE ACCURACY PROFILE 
roc -> reciever operating characteristic 

'''



'''
CAP analysis 


->the closer the line to the perfect model line the better the model and the close to random is worse 
->  now the ar ratio is the area under perfect model curve divided by the area under the  model curve (the lower limit is the random model curve)
 AR= aR/aP
 
 
 
90 % <x< 100% -> too good  (may be caused by overfitting )
80%<x<90%  -> very good 
70% < x< 80 % -> good 
60%<x<70% -> poor  
x<60% ->rubbish
 


we don'pt plot it in a way that we get the most accuracy  

later we  continue to plot it till we have the  


'''
'''

What are the pros and cons of each model ?

How do I know which model to choose for my problem ?

How can I improve each of these models ?

Again, let's answer each of these questions one by one:

1. What are the pros and cons of each model ?

Please find attached at the bottom of this article a cheat-sheet that gives you all the pros and the cons of each classification model.

2. How do I know which model to choose for my problem ?

Same as for regression models, you first need to figure out whether your problem is linear or non linear. You will learn how to do that in Part 10 - Model Selection. Then:

If your problem is linear, you should go for Logistic Regression or SVM.

If your problem is non linear, you should go for K-NN, Naive Bayes, Decision Tree or Random Forest.

Then which one should you choose in each case ? You will learn that in Part 10 - Model Selection with k-Fold Cross Validation.

Then from a business point of view, you would rather use:

- Logistic Regression or Naive Bayes when you want to rank your predictions by their probability. For example if you want to rank your customers from the highest probability that they buy a certain product, to the lowest probability. Eventually that allows you to target your marketing campaigns. And of course for this type of business problem, you should use Logistic Regression if your problem is linear, and Naive Bayes if your problem is non linear.

- SVM when you want to predict to which segment your customers belong to. Segments can be any kind of segments, for example some market segments you identified earlier with clustering.

- Decision Tree when you want to have clear interpretation of your model results,

- Random Forest when you are just looking for high performance with less need for interpretation. 

3. How can I improve each of these models ?

Same answer as in Part 2: 

In Part 10 - Model Selection, you will find the second section dedicated to Parameter Tuning, that will allow you to improve the performance of your models, by tuning them. You probably already noticed that each model is composed of two types of parameters:

the parameters that are learnt, for example the coefficients in Linear Regression,

the hyperparameters.

The hyperparameters are the parameters that are not learnt and that are fixed values inside the model equations. For example, the regularization parameter lambda or the penalty parameter C are hyperparameters. So far we used the default value of these hyperparameters, and we haven't searched for their optimal value so that your model reaches even higher performance. Finding their optimal value is exactly what Parameter Tuning is about.
'''



