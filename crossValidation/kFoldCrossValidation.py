# -*- coding: utf-8 -*-
"""
Created on Mon Dec 22 18:52:38 2025

@author: arnav
"""

'''
WE ANSWER THE QUESTION : how good our model is 

there is always a risk that at end when we test our model and we get accuracy quite high but we might as well be just lucky with our test set 

now for that we use KFOLD CROSS VALIDATION . 

we will be splitting our training set into k folds , (10 usually ) and then we are gonna first use the last fold as validation fold and the others to train the model .

-> similarly we then use the second last one as validation and the others to train the model . 
-> we keep on doing this till we use the first fold as validation fold 

so we will be testing our model 10 times and the probaility is quite less that we will end up being lucky each tim e


now during the whole process we will be keeping our hyperparameters the same so as to validify that these hyperparameters render the model to be good 

and at last we are going to train our model with all the folds and test it with out test set 

-> another approach might be to not have any test set and just use the training test and apply k fold cross validation on it . 

-> we can also choose not to either train it or test it and just apply k folds and pick one model (we are going to pick the model with the high accuracy .)


-> or  , we can split into training and test set and then apply k fold cross validation on training set and then again test with test set

 

'''