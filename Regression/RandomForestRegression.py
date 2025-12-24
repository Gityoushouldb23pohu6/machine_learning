# -*- coding: utf-8 -*-
"""
Created on Fri Dec 12 23:37:47 2025

@author: arnav
"""

'''
RANDOM FOREST INTUITION 

step1 : pick at random k data points from the Training set 
step2 : build the decision tree associated  to these k data points 
step3: choose the Ntree of trees  you want to build and repeat step 3 
step 4 : for a new data point , make each one of your Ntree tree predict the value of Y for the data point in question , and assign the new data point the avg of all predicted y values .
 


so in here we are basically using predictions using a forest of trees 

'''