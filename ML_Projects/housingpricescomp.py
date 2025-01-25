import pandas as pd
import numpy as np
import sklearn as sk
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import os, requests, zipfile
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from matplotlib import pyplot as plt
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor
import xgboost as xgb

import time 

starttime = time.time()

# my_pipeline = Pipeline(steps=[('preprocessor', SimpleImputer()),
#                               ('model', RandomForestRegressor(n_estimators=50,
#                                                               random_state=42))
#                              ])



X = pd.read_csv(r'datasets\housingpricescompetition\home-data-for-ml-course\train.csv', index_col= "Id")
X.dropna(axis = 0, subset=['SalePrice'], inplace = True)
y = X.SalePrice
X.drop(['SalePrice'], axis = 1, inplace = True)
# Select numeric columns

numeric_cols = [cname for cname in X.columns if X[cname].dtype in ['int64', 'float64']]
my_cols = numeric_cols
X = X[my_cols]

X_train, X_valid, y_train, y_valid = train_test_split(X, y,random_state=42)

#Checking models
def get_score(n_estimators):
    my_pipeline = Pipeline(steps=[
        ('preprocessor', SimpleImputer()),
        ('model', RandomForestRegressor(n_estimators, random_state=42))
    ])
    scores = -1 * cross_val_score(my_pipeline, X_train, y_train,
                                  cv=5,
                                  scoring='neg_mean_absolute_error')
    return scores.mean()

# n=15
# n_estimators = []
# for i in range(1,n+1):
#     n_estimators.append(int(i*50))
# scores = []
# timescores = []
# times = time.time()
# for i in n_estimators:
#     scores.append(get_score(i))
#     t = time.time()- times
#     times = time.time()
#     print(t)
#     timescores.append(t)
    
# print(sum(timescores))

# plt.plot(n_estimators,scores)
# plt.show()

#Define model
model = RandomForestRegressor(n_estimators=600, random_state=42)
# model.fit(X_train,y_train)
# predictions = model.predict(X_valid)
# mae = mean_absolute_error(predictions, y_valid)
# print(f'Mean absolute error is {mae}')
rf_model_on_full_data = RandomForestRegressor(n_estimators=400,random_state =1 )
rf_model_on_full_data.fit(X,y)

test_data= pd.read_csv(r'datasets\housingpricescompetition\home-data-for-ml-course\test.csv')
# test_data.dropna(axis = 0, subset=['SalePrice'], inplace = True)
# test_data.drop(['SalePrice'], axis = 1, inplace = True)
test_X = test_data[my_cols]
test_preds = rf_model_on_full_data.predict(test_X)


folder_path = r'C:\Users\jorda\OneDrive\Documents\GitHub\JDMmessingaround\datasets\housingpricescompetition\home-data-for-ml-course'
file_name = 'submission.csv'
full_path = os.path.join(folder_path, file_name)

output = pd.DataFrame({'Id': test_data.Id,
                       'SalePrice': test_preds})

output.to_csv(full_path, index=False,)


print(f'Total run time {time.time()-starttime}')
print(((5310-1361)/5310)*100, ' Percentile')

