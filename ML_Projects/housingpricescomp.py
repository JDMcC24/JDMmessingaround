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






"""Preprocessing X, changing categorical Data to integers"""

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()

X = pd.read_csv(r'datasets\housingpricescompetition\home-data-for-ml-course\train.csv')
#X.dropna(axis = 0, subset=['SalePrice'], inplace = True)
y = X.pop('SalePrice')
labels = X.pop('Id')
col = X.columns[1:]
#print(col)
for c in col:
    if X[c].dtype != 'int' and  X[c].dtype != 'float':
        X[c] = label_encoder.fit_transform(X[c])




""" Feature Selection and Seperating Targets from Features"""
#print(X.columns)
#features = ['MSSubClass', 'LotArea','Condition1', 'OverallCond',  'YearBuilt', 'YearRemodAdd', 'TotalBsmtSF','HeatingQC', '1stFlrSF',
            # '2ndFlrSF', 'BedroomAbvGr', 'FullBath', 'HalfBath','GarageType','Functional','WoodDeckSF','YrSold', 'SaleType', 'SaleCondition',
            #  'PoolArea', 'PoolQC', 'Fence']
#features = X.columns
num_of_features = 79

correlation = []
for c in X.columns:
    correlation.append( X[c].corr(y))
correlation = np.array([correlation])
featureindices = np.argsort(correlation)[0][-1* num_of_features:]
features = []
for c in featureindices:
    features.append(X.columns[c])



X= X[features]
# print(X.describe())


""" Splitting data"""
X_train, X_valid, y_train, y_valid = train_test_split(X, y,random_state=1)

from sklearn.impute import KNNImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler



""" Creating pipeline for XGBRegressor and cross validating"""
#Checking models
# def get_score(n_estimators):
#     my_pipeline = Pipeline(steps=[
#         ('preprocessor', KNNImputer(n_neighbors=5)),
#         #('preprocessor', SimpleImputer()),
#         ('scaler', StandardScaler()),
#         ('model', XGBRegressor(n_estimators=n_estimators, learning_rate = 0.05, random_state =1))
#     ])
#     scores = -1 * cross_val_score(my_pipeline, X_train, y_train,
#                                   cv=5,
#                                   scoring='neg_mean_absolute_error')
#     return scores.mean()
my_pipeline = Pipeline(steps=[
        ('preprocessor', KNNImputer()),
        #('preprocessor', SimpleImputer()),
        ('scaler', StandardScaler()),
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

print(rates)
print("Starting GridSearch")
param_grid = {'preprocessor__n_neighbors' :  neighbors,
     'model__n_estimators': n_estimators,
     'model__learning_rate': rates  
     }

grid_search = GridSearchCV( estimator=my_pipeline, param_grid = param_grid, cv = 4, scoring = 'neg_mean_absolute_error', verbose = 1)
grid_search.fit(X, y)
print(grid_search.best_params_)

# scores = []
# timescores = []
# times = time.time()
# for i in n_estimators:
#     s = get_score(i)
#     scores.append(s)
#     t = time.time()- times
#     times = time.time()
#     print(f' {i} = n_estimators, time taken {t} seconds. Accuracy is {s}')
#     timescores.append(t)
    
# print(sum(timescores))

# plt.plot(n_estimators,scores)
# plt.show()


""" Defining Random Forest Model"""
#Define model
#model = RandomForestRegressor(n_estimators=600, random_state=42)
# model.fit(X_train,y_train)
# predictions = model.predict(X_valid)
# mae = mean_absolute_error(predictions, y_valid)
# print(f'Mean absolute error is {mae}')
#rf_model_on_full_data = RandomForestRegressor(n_estimators=400,random_state =1 )
#rf_model_on_full_data.fit(X,y)

# # """ Defining Pipeline model"""
# my_pipeline = Pipeline(steps=[
#         ('preprocessor', KNNImputer(n_neighbors= 4)),
#         #('preprocessor', SimpleImputer()),
#         ('scaler', StandardScaler()),
#         ('model', XGBRegressor(learning_rate =  0.08337777777777779, n_estimators= 350))
#     ])
# my_pipeline.fit(X,y)


# # """ Processing Test Data"""
# test_X= pd.read_csv(r'datasets\housingpricescompetition\home-data-for-ml-course\test.csv')
# labels = test_X.pop('Id')
# #print(test_X.head())
# col = test_X.columns[1:]
# #print(col)
# for c in col:
#     if test_X[c].dtype != 'int' and  test_X[c].dtype != 'float':
#         test_X[c] = label_encoder.fit_transform(test_X[c])

# test_X = test_X[features]



# # """Making Predictions"""
# # #test_preds = rf_model_on_full_data.predict(test_X)
# test_preds = my_pipeline.predict(test_X)
# test_X.insert(0,'Id',labels)
# #print(test_X.columns)

# # """Saving submission"""
# folder_path = r'C:\Users\jorda\OneDrive\Documents\GitHub\JDMmessingaround\datasets\housingpricescompetition\home-data-for-ml-course'
# file_name = 'submission.csv'
# full_path = os.path.join(folder_path, file_name)

# output = pd.DataFrame({'Id': test_X.Id,
#                        'SalePrice': test_preds})

# output.to_csv(full_path, index=False,)


# # """ Estimates"""
# print(f'Total run time {time.time()-starttime}')
# print(f'Expected accuracy = {-1* cross_val_score(my_pipeline,X,y, scoring='neg_mean_absolute_error').mean()}')
# print(((5310-1361)/5310)*100, ' Percentile')

