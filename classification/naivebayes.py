# -*- coding: utf-8 -*-
"""
Created on Sun Dec 21 10:20:22 2025

@author: arnav
"""

'''
Bayes theorm : 
    
    
    eqn of bayes theorm 
    
lets say we have two features with green category and red category
let's say that X= age and y= salary'

we add a new data point and want to classify it either to the green or the red category .
so for this new point given we know its age and corresponding salary we now have to classify it 

step 1 :  calculating posterior probability using likelihood , prior probability and (/) marginal likelihood  (given in sequence )
step 2 : after calculating the posterior probabilities for P((walks)|x) and P(Drives|x) we are going to compare them 


-> we are going to decide a particular radius and then we are going to calculate the P(X) which is the probability of any given random point to fall in this circle . 
-> P(X|walks) = what is the probability that a person walking will fall inside the decided circle 
 -> P(walks|X) - probability of the new point added walking given that it falls inside the decided circle 
 
 ->since  P(walks|X) > P(drives|X) we are going to classify the new point as red 
    
why naive? 
ans- we make several independence assumptions which often times aren't correct . So we are making the model with those naiive assumptions .

now , while comparing we can see that there's really no need to compute the value P (X) as while comparing we might just as well compute only the numerator 
'


now what if we have more than 2 classes 

-> if we had one we could have just calculated one probabiltiy , but now we would have to calculate atleast 2 probailities in order to compare the results 




    
'''