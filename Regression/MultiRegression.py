# -*- coding: utf-8 -*-
"""
Created on Sat Dec  6 12:15:04 2025

@author: arnav
"""

'''
multiple linear regression 
where the dependent variable depends on more than one parameter 

we can develop a model on the given data which can give many interpretations such as : 
    1) we can see which company is situated at closest proximity to the investors .
    2) the investors can look for companies which have low r&d spend or administrative spend and so on .
    3) we can understand what characteristics of companies lead them to have more profit 
    
    y^ = b0 +b1X1 +b2X3 +.... +bnXn
    
    
    There can be many cases in where linear regression can be misleading , in such cases we need to make sure that we see properly if the data distribution given is ideal for linear regression or not '
    
    Assumptions of linear regression : 
        
        1) linearity 
        2) Homoscedasticity (equivalent variance )
        3) multivariate normality
        4) Independence (no autocorrelation)
        5) Lack of multicollinearity
        - it renders the models prediction to be unreliable 
        
        6) outlier check 
  
    
    
->> 
now while building a model we would be interested in knowing the correlation between the feature variables 

y=b0 +b1*x1 +b2*x2 + b3*x3 + (what to do about state column ?)
 
-so , for each type of categorial data (for each state ) we create a new column and mark 1 wherever we see the state name or else mark 0 
these new columns are called dummy variables 
-so in order to complete the equation we use values from these dummy variables
- the dummy variables included in the equation work like light switches which will be included only when the value of the dummy variable is 1 
-at first the dummy variables might seem to be biased , but here in this case the coefficient of the dummy variables are somehow included in b0 already , and here when the newyork term disapperas as D1 is 0 by default california state becomes 1.
 so it works in an alternating way rather than in a baised way .
 
 
  /// DUMMY VARIABLE TRAP 
  
***** why it is suggested to not use all the dummy variables in the  equation ?****
 
if in our case we include both the dummy variables then it renders the equation being redundant 

here D2=1-D1

so if D1 is not activated in the equation that means D2 is automatically included 

The phenomenon where each dummy variable predicts one or more dummy variables is called multicollinearity 

- It is seen that we can't have both the constant b0 and two dummy variables in this case in the same equation 
- we should always try to eliminate one dummy variable 
 

in this dataset if we had one more column with categorical terms , we can do the same for it and also omit one dummy variable at the end 


'''


'''
p VALUES and Statistical significance:
     
    hypothesis: 
        H0 : Fair coin (hypothesis)
        H1 : This is not a fair coin (alternative hypothesis)
        
        assume H0 is true 
        
        
        as we get uneasy feeling about getting tails all the time this uneasiness is associated with statistical significance 
        
        statistical significance (aplha = 0.05) below which things start getting unlikely 
        we can set this alpha value to any value 
        
        we get 95 percent confident that the events further happening are unlikely 
        
        
        
        
    
'''

'''

We can now build a model 
1) we can't take too many variables otherwise the model will not be functionable 
    
2) we have to explain every variables (adding on )
    

    
    
    
    five methods for building models :
        1) All in 
    - throw in all of the variables , when we have previous knowledge that all the variables are crucial 
    - preparing for backward elimination .
    
    2) backward elimination 
      a ) choose a   significance level to stay in the model  (like sl=0.05)
      b) fit the model with all possible predictors 
      c) consider the predictor with highest p value . if p>SL then go to step 4 otherwise finish the model 
      d) remove the predictor 
      e) fill the model without this variable 
    
     after the e step we will again to c .
    
    3) forward selection
       
       a) significance level selection (like 0.05)
       b) build simple regression models with every single predictor and select the one which has the lowest p value 
       c) keep this variable(make the model with the current variable) and now we need to  make models with this one(s) and an extra one predictor ()
       d) consider the predictor with lowest p value. if p < sl go to step 3 , otherwise finish(keep the previous model)
       
    
    4) bidirectional elimination  (stepwise regression)
     a) select two significance level to enter and to stay in the model (eg. slenter=0.05,slstay=0.05) 
     b) next step of forward selection (new variable add on (p<slenter))
     c) backward elimination(old variables must have p<slstay to stay) , go to step2 again and repeat until there are no variables to be added or discarded 
     d) model ready
    
    ->ALL POSSIBLE MODELS
      a)select a criterion of goodness of fit(like akaike criterion)
      b) consider all possible regression models(2^N-1 combinations)
      c) one with best criterion 
      d) model ready
    
    5) score comparision
       
      
    
    we will for our practical use BACKWARD ELIMINATION
    
    
'''




















