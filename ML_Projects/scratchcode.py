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

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()

X = pd.read_csv(r'datasets\housingpricescompetition\home-data-for-ml-course\train.csv')
#X.dropna(axis = 0, subset=['SalePrice'], inplace = True)
y = X.pop('SalePrice')
labels = X.pop('Id')


col = X.columns[1:]
print(col)
# for c in col:
#     if X[c].dtype != 'int' and  X[c].dtype != 'float':
#         X[c] = label_encoder.fit_transform(X[c])

# numeric_featuers = []
# for c in col:
#     if X[c].dtype != 'int' and  X[c].dtype != 'float':
#         numeric_featuers.append(c)
# print(numeric_featuers)

X = pd.get_dummies(X,columns=col)
print(X.head())

""" Feature Selection and Seperating Targets from Features"""
#print(X.columns)
#features = ['MSSubClass', 'LotArea','Condition1', 'OverallCond',  'YearBuilt', 'YearRemodAdd', 'TotalBsmtSF','HeatingQC', '1stFlrSF',
            # '2ndFlrSF', 'BedroomAbvGr', 'FullBath', 'HalfBath','GarageType','Functional','WoodDeckSF','YrSold', 'SaleType', 'SaleCondition',
            #  'PoolArea', 'PoolQC', 'Fence']
#features = X.columns


# num_of_features = 40

# correlation = []
# for c in X.columns:
#     correlation.append( X[c].corr(y))
# correlation = np.array([correlation])
# featureindices = np.argsort(correlation)[0][-1* num_of_features:]
# features = []
# for c in featureindices:
#     features.append(X.columns[c])

# print(features)
#print(X.columns)

"""Splitting Data"""
X_train, X_valid, y_train, y_valid = train_test_split(X, y,random_state=1)


""" Creating Pipeline"""
from sklearn.impute import KNNImputer

my_pipeline = Pipeline(steps=[
        ('preprocessor', KNNImputer()),
        #('preprocessor', SimpleImputer()),
        #('scaler', StandardScaler()),
        ('model', XGBRegressor())
    ])


from sklearn.model_selection import GridSearchCV

n=20
step_size = 50
n_estimators = []
for i in range(1,n+1):
    n_estimators.append(int(i*step_size))
neighbors= list(range(1,11))
rates = list(np.linspace(0.0001,.15,10))




print("Starting GridSearch")
param_grid = {'preprocessor__n_neighbors' :  neighbors,
     'model__n_estimators': n_estimators,
     'model__learning_rate': rates  
     }

grid_search = GridSearchCV( estimator=my_pipeline, param_grid = param_grid, cv = 4, scoring = 'neg_mean_absolute_error', verbose = 1)
grid_search.fit(X_train, y_train)
print(grid_search.best_params_)






# # """ Defining Pipeline model"""
# my_pipeline = Pipeline(steps=[
#         ('preprocessor', KNNImputer(n_neighbors= 4)),
#         #('preprocessor', SimpleImputer()),
#         ('scaler', StandardScaler()),
#         ('model', XGBRegressor(learning_rate =  0.08337777777777779, n_estimators= 350))
#     ])
# my_pipeline.fit(X,y)


# """ Processing Test Data"""
# test_X= pd.read_csv(r'datasets\housingpricescompetition\home-data-for-ml-course\test.csv')
# labels = test_X.pop('Id')
# #print(test_X.head())
# col = test_X.columns[1:]
# #print(col)


# test_X = pd.get_dummies(test_X,columns=col)



# # """Making Predictions"""
# # #test_preds = rf_model_on_full_data.predict(test_X)
# test_preds = my_pipeline.predict(test_X)
# test_X.insert(0,'Id',labels)
# #print(test_X.columns)

# # """ Estimates"""
# print(f'Total run time {time.time()-starttime}')
# print(f'Expected accuracy = {-1* cross_val_score(my_pipeline,X,y, scoring='neg_mean_absolute_error').mean()}')