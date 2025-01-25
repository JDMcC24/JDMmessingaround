import pandas as pd
import numpy as np
import sklearn as sk
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import root_mean_squared_log_error
import os, requests, zipfile
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from matplotlib import pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
import xgboost as xgb

import time 



# def unzip_file(file_path):
#     """
#     Unzips a file and saves the contents in the same directory as the ZIP file.

#     Args:
#         file_path (str): Path to the ZIP file.
#     """
#     try:
#         # Check if the file is a ZIP file
#         if not zipfile.is_zipfile(file_path):
#             print("The provided file is not a ZIP file.")
#             return

#         # Get the directory of the ZIP file
#         directory = os.path.dirname(file_path)

#         # Extract the contents
#         with zipfile.ZipFile(file_path, 'r') as zip_ref:
#             zip_ref.extractall(directory)
#             print(f"Contents extracted to: {directory}")
#     except Exception as e:
#         print(f"An error occurred: {e}")

# # Replace 'your_file.zip' with the path to your ZIP file
# zip_file_path = r"C:\Users\jorda\OneDrive\Documents\GitHub\JDMmessingaround\datasets\InsuranceData\playground-series-s4e12.zip"
# unzip_file(zip_file_path)


X = pd.read_csv(r'datasets\InsuranceData\train.csv')
X.dropna(axis = 0, subset=['Premium Amount'], inplace = True)
y = X['Premium Amount']
print(y.describe())
X.drop(['Premium Amount'], axis = 1, inplace = True)
#print(X['Education Level'].value_counts())
#print(X.columns)

# Changing categorical data to integers
label_encoder = LabelEncoder()
col = X.columns[1:]
for c in col:
    if X[c].dtype != 'int' and X[c].dtype != 'float':
         X[c] = label_encoder.fit_transform(X[c])


X_train, X_val, y_train, y_val = train_test_split(X, y,random_state=42)

print(y_train)

#Checking models
# def get_score(n_estimators):
#     my_pipeline = Pipeline(steps=[
#         ('preprocessor', SimpleImputer()),
#         ('model', RandomForestRegressor(n_estimators, random_state=42))
#     ])
#     scores = -1 * cross_val_score(my_pipeline, X_train, y_train,
#                                   cv=3,
#                                   scoring= 'neg_root_mean_squared_log_error')
#     return scores.mean()

# n=10
# step = 50
# n_estimators = []
# for i in range(1,n+1):
#     n_estimators.append(int(i*step))
# scores = []
# timescores = []
# times = time.time()
# for i in n_estimators:
#     s = get_score(i)
#     scores.append(s)
#     t = time.time()- times
#     times = time.time()
#     print(f'Time taken {t} seconds. Mean Absolute Error is {s}')
#     timescores.append(t)
    
# print(sum(timescores))

# plt.plot(n_estimators,scores)
# plt.show()

