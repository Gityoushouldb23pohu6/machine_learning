
import pandas as pd , numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score , mean_absolute_error , mean_squared_error
from sklearn.linear_model import LinearRegression , Ridge , Lasso , ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.datasets import fetch_california_housing

data = fetch_california_housing(as_frame=True)
df   = data.frame

X = df.drop( "MedHouseVal" , axis = 1 )
y = df["MedHouseVal"]

X_train , X_test , y_train , y_test = train_test_split(
X , y , test_size = 0.2 , random_state = 42
)

scaler = StandardScaler()
X_train = scaler.fit_transform( X_train )
X_test  = scaler.transform( X_test )

models = {
"Linear Regression" : LinearRegression() ,
"Ridge Regression"  : Ridge() ,
"Lasso Regression"  : Lasso() ,
"ElasticNet"        : ElasticNet() ,
"Decision Tree"     : DecisionTreeRegressor() ,
"Random Forest"     : RandomForestRegressor( n_estimators = 100 ) ,
"SVR"               : SVR() ,
"KNN"               : KNeighborsRegressor()
}

results = []

for name , model in models.items():
 model.fit( X_train , y_train )
 preds = model.predict( X_test )
 r2  = r2_score( y_test , preds )
 mae = mean_absolute_error( y_test , preds )
 mse = mean_squared_error( y_test , preds )
 results.append( [ name , r2 , mae , mse ] )

results_df = pd.DataFrame(
results ,
columns = [ "Model" , "R2 Score" , "MAE" , "MSE" ]
)

results_df = results_df.sort_values(
by = "R2 Score" , ascending = False
)

print( "\nRegression Model Comparison:\n" )
print( results_df )
