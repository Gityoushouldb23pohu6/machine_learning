# -*- coding: utf-8 -*-
"""
Created on Fri Dec 12 20:05:51 2025

@author: arnav
"""

'''
DECISION REGRESSION TREE 


Here in this case we have two independent variables plotted and the dependent variable is actully in the third dimension 
The independent variables are X1 and X2 


Our scatter plot would be splitted into two parts 
we do these splits 4 times 


-> our regression model will actually decide how these splits will take place 
-> the splits will be done such that we keep in mind the entropy . 
-> Information Entropy is how much information we are actually able to extract from  given points 
-> each split is called a leaf
-> The model would continue splitting till there is an extent of taking the data into consideration for our model 
->The final splits are termed as terminal leaves 


- making the corresponding decision tree.
-after the splits we actually  consider(decide using decision tree) where a new point should be added in scatter plot.


-> now the prediction works as the split or area in which the new point lies , the predicted y value for this point will be the avg y values of all the points lying in this split 










'''

