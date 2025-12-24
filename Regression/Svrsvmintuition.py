# -*- coding: utf-8 -*-
"""
Created on Fri Dec 12 06:43:45 2025

@author: arnav
"""
'''

SVM tells us how to find the boundary different groups of points 


but what if we can't  ? 

we use then kernel svm to  form the decision line to separate these points 
(Non linearly separable ).
We have to use something which is of higher dimensional space 

let's say we have  ***  +++ ***' separated in a one d line . 
->if we apply f=x-5 then everything shifts to the left 
-> again if we apply f=(x-5)^2 we see that we get a parabola of which the *'s and +'s are points of and we can then separate these points 
-> also for the 2 d problem if we use some function we can see that separating these points using an hyperplane is equivalent to separate these points using a non linear separater like a circle

- the draw back is mapping to a higher dimensional space can be highly compute intensive 
This is why we can  use : Kernel trick 

we use the gaussian rbf kernel function which is one of many types of kernel functions. 
 
The base point of the bump is called the landmark .
In the equation the x-l represents distance from this landmark to any point as shown . 
-> if the point is far like the * here then we see that the power of e becomes -ve high and so the overall value becomes something between 0 and 1 for the function output and the graph-
-> if the point is closer then we see the power becomes close to 0 and this the overall function has a value of 1 . 


so in the 2 d point distribution we basically find the landmark , so the circle around the function's base is actually the non - linear separator.'
all the points inside the circle are having the function value between 0 and 1 (closer to 0) , and the one's outside have closer to 1 '
Increasing or decreasing the circumference of the circle actually affects the number of points inside and outside the circle .
->circumference is here sigma 
so , depending on the value the function is taking we can actually separate the points from each other .

Types of kernel function : 

1) gaussian rbf kernel 
2) sigmoid kernel (everything to the left has small value and everything to the right has comparatively larger value )
3) Polynomial kernel     
    

'''
'''
non linear svr 

- if we try to separate these given points with linear svr , it is not possible 
- let's say all the points in here are enclosed in a blue box '.
- now we use the rbf function . 
- again the central function project to the top most point .
- we point different values of the function with respect to the 3d space . 
- now we use a hyperplane to actually separate these points .
- we are actually interested where does it intersect this function . 
- and the intersection pattern is basically projected , and that gives us the decision line 
- and for the one hyperplane we will have a hyperplane epsilon above and one hyperplane epsilon below .
- any points that are between the outmost hyperplane , error for these points are not considered . 
- the points which are out are abtually the support vectors  . 




'''