import pandas as pd
import numpy as np
import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import os, requests, zipfile
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from matplotlib import pyplot as plt
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from Ufcdataclearnup import *
import joblib



#starttime = time.time()

data = pd.read_csv(r'C:\Users\jorda\OneDrive\Documents\GitHub\JDMmessingaround\UfcProject\CleanData.cvs')

y = data.pop('Result')
labelencoder = LabelEncoder()
y = labelencoder.fit_transform(y)
category_mapping = dict(zip(labelencoder.classes_, range(len(labelencoder.classes_))))
X = data.drop(columns = ['RedFighter', 'BlueFighter'])
X = pd.get_dummies(X)
Allfeatures = X.columns.to_list()
#print(X['RedStance_Open Stance'])
#print(X.head())

X_train, X_val, y_train, y_val = train_test_split(X, y)
nameless_fight_model = RandomForestClassifier()
nameless_fight_model.fit(X_train,y_train)

cv_score = cross_val_score(nameless_fight_model,X_train,y_train,scoring='accuracy', cv = 5)
print(f'Expected accuracy for simple model is {cv_score.mean()}')


# my_pipeline = Pipeline(steps =[
# #('preprocessor', KNNImputer(n_neighbors=1)),
# ('preprocessor', SimpleImputer()),
# ('scaler', StandardScaler()),
# ('model', XGBClassifier() )
# ])

# n=10
# step_size = 50
# n_estimators = []
# for i in range(1,n+1):
#     n_estimators.append(int(i*step_size))
#     #print(i)
# #neighbors= list(range(1,5))
# #neighbors = [1]
# rates = list(np.linspace(0.0001,.25,10))

# #print(rates)
# print("Starting GridSearch")
# param_grid = {
#     #'preprocessor__n_neighbors': neighbors,
#      'model__n_estimators': n_estimators,
#      'model__learning_rate': rates  
#      }


# grid_search = GridSearchCV( estimator=my_pipeline, param_grid = param_grid, cv = 3, scoring = 'accuracy', verbose = 1)
# grid_search.fit(X_train, y_train)
# print(grid_search.best_params_)
"""" Grid Search yeilded most accurate model to be the one with following parameters:
    {'model__learning_rate': 0.08340000000000002, 'model__n_estimators': 50}"""

nameless_pipeline = Pipeline(steps =[
#('preprocessor', KNNImputer(n_neighbors=1)),
('preprocessor', SimpleImputer()),
('scaler', StandardScaler()),
('model', XGBClassifier(learning_rate =0.08340000000000002, n_estimators = 50) )
])

cv_score = cross_val_score(nameless_pipeline,X_train,y_train,scoring='accuracy', cv = 5)
print(f'Expected accuracy for pipeline model is {cv_score.mean()}')

""" Fitting Full data"""
nameless_pipeline.fit(X_train,y_train)
nameless_fight_model.fit(X_train,y_train)


""" Saving for future use"""
joblib.dump(nameless_pipeline, r'C:\Users\jorda\OneDrive\Documents\GitHub\JDMmessingaround\UfcProject\SavedModels\nameless_pipeline.joblib')
joblib.dump(nameless_fight_model, r'C:\Users\jorda\OneDrive\Documents\GitHub\JDMmessingaround\UfcProject\SavedModels\nameless_model.joblib')
print('Models have been saved.')

print(f'Total run time is {time.time()- start} seconds')