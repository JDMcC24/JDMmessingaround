import pandas as pd
import numpy as np
import sklearn as sk
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import root_mean_squared_log_error
from sklearn.metrics import accuracy_score
import os, requests, zipfile
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from matplotlib import pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
from xgboost import XGBClassifier

import xgboost as xgb

from catboost import CatBoostClassifier

from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC




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
# zip_file_path = r"C:\Users\jorda\OneDrive\Documents\GitHub\JDMmessingaround\datasets\titanic.zip"
# unzip_file(zip_file_path)

file_path = r'C:\Users\jorda\OneDrive\Documents\GitHub\JDMmessingaround\datasets\Titanic\train.csv'
X = pd.read_csv(file_path)

label_encoder = LabelEncoder()
#print(X.Name)


#print(X.columns)
y = X.pop('Survived')
y = y.astype('int')
#print(y)
X = X.drop(columns=['Name'])
col = X.columns[1:]
print(col)
for c in col:
    if X[c].dtype != 'int' and  X[c].dtype != 'float':
        X[c] = label_encoder.fit_transform(X[c])
#print(X.describe())

X_train, X_val, y_train, y_val = train_test_split(X,y)


##Checking models
from sklearn.impute import KNNImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

my_pipeline = Pipeline(steps=[
        ('preprocessor', KNNImputer()),
        #('preprocessor', SimpleImputer()),
        ('scaler', StandardScaler()),
        ('model', XGBClassifier())
    ])


from sklearn.model_selection import GridSearchCV

# n=20
# step_size = 50
# n_estimators = []
# for i in range(1,n+1):
#     n_estimators.append(int(i*step_size))
# neighbors= list(range(1,11))
# rates = list(np.linspace(0.0001,.15,10))

# #print(rates)
# print("Starting GridSearch")
# param_grid = {'preprocessor__n_neighbors' :  neighbors,
#      'model__n_estimators': n_estimators,
#      'model__learning_rate': rates  
#      }

# grid_search = GridSearchCV( estimator=my_pipeline, param_grid = param_grid, cv = 4, scoring = 'neg_mean_absolute_error', verbose = 1)
# grid_search.fit(X_train, y_train)
# print(grid_search.best_params_)


# plt.plot(n_estimators,scores,'s')
# plt.show()

## n_estimators = 250 seems best


## Trying VotingClassifer for comparison


# Define individual models with pipelines
# model1 = Pipeline([
#     ('imputer', SimpleImputer(strategy='mean')),
#     ('classifier', XGBClassifier(random_state=42))
# ])
# model2 = Pipeline([
#     ('imputer', SimpleImputer(strategy='mean')),
#     ('classifier', CatBoostClassifier(random_state=42))
# ])
# model3 = Pipeline([
#     ('imputer', SimpleImputer(strategy='mean')),
#     ('classifier', SVC(probability=True, random_state=42))
# ])

# # Combine models into a VotingClassifier
# voting_clf = VotingClassifier(estimators=[
#     ('xbg', model1), 
#     ('cb', model2), 
#     ('svc', model3)
# ], voting='soft')

# Perform cross-validation


#cv_scores = cross_val_score(voting_clf, X_train, y_train, cv=5, scoring='accuracy')

# def get_score3(n_estimators):
#     model1 = Pipeline([
#         ('imputer', SimpleImputer(strategy='mean')),
#         ('classifier', XGBClassifier(n_estimators = n_estimators,random_state=42, learning_rate = 0.05))
#         ])
#     model2 = Pipeline([
#         ('imputer', SimpleImputer(strategy='mean')),
#         ('classifier', CatBoostClassifier(random_state=42))
#         ])
#     model3 = Pipeline([
#         ('imputer', SimpleImputer(strategy='mean')),
#         ('classifier', SVC(probability=True, random_state=42))
#         ])

# # Combine models into a VotingClassifier
#     voting_clf = VotingClassifier(estimators=[
#         ('xbg', model1), 
#         ('cb', model2), 
#         ('svc', model3)
#         ], voting='soft')
#     scores = cross_val_score(voting_clf, X_train, y_train, cv=5, scoring='accuracy')
#     return scores.mean()


# n=20
# step = 50
# n_estimators = []
# for i in range(1,n+1):
#     n_estimators.append(int(i*step))
# scores = []
# timescores = []
# times = time.time()
# for i in n_estimators:
#     s = get_score3(i)
#     scores.append(s)
#     t = time.time()- times
#     times = time.time()
#     print(f'Time taken {t} seconds. Accuracy is {s}')
#     timescores.append(t)
    
# print(sum(timescores))

# plt.plot(n_estimators,scores,'-')
# plt.show()


# Print the cross-validation scores
# print(f"Cross-Validation Accuracy Scores: {cv_scores}")
# print(f"Mean Cross-Validation Accuracy: {cv_scores.mean()}")


# Creating Full model With XGBClassifer
# titanic_model = XGBClassifier(n_estimators = 100, learning_rate = 0.05)
# titanic_model.fit(X,y)

#Creating Full model with VotingClassifer
# model1 = Pipeline([
#         ('imputer', SimpleImputer(strategy='mean')),
#         ('classifier', XGBClassifier(n_estimators = 150, learning_rate = 0.05))
#         ])
# model2 = Pipeline([
#         ('imputer', SimpleImputer(strategy='mean')),
#         ('classifier', CatBoostClassifier())
#         ])
# model3 = Pipeline([
#         ('imputer', SimpleImputer(strategy='mean')),
#         ('classifier', SVC(probability=True, ))
#         ])

# # Combine models into a VotingClassifier
# titanic_model = VotingClassifier(estimators=[
#         ('xbg', model1), 
#         ('cb', model2), 
#         ('svc', model3)
#         ], voting='soft')
# titanic_model.fit(X,y)




# """Pipline Model"""
titanic_model = Pipeline([
    ('imputer', KNNImputer(n_neighbors=10)),
    ('scaler', StandardScaler()),
    ('classifier', XGBClassifier(n_estimators = 50, learning_rate = 0.016755555555555555))])
titanic_model.fit(X,y)

""" Making and Saving Predictions"""
test = pd.read_csv(r'C:\Users\jorda\OneDrive\Documents\GitHub\JDMmessingaround\datasets\Titanic\test.csv')
test = test.drop(columns=['Name'])
#passids = test.pop('PassengerID')
col = test.columns[1:]
#print(col)
for c in col:
    if test[c].dtype != 'int' and  test[c].dtype != 'float':
        test[c] = label_encoder.fit_transform(test[c])

test_preds = titanic_model.predict(test)
scores = cross_val_score(titanic_model, X, y, cv=5, scoring='accuracy')
print(f'Expected accuracy is {scores.mean()}')


folder_path = r'C:\Users\jorda\OneDrive\Documents\GitHub\JDMmessingaround\datasets\Titanic'
file_name = 'submission.csv'
full_path = os.path.join(folder_path, file_name)

output = pd.DataFrame({'PassengerId': test.PassengerId,
                       'Survived': test_preds})

output.to_csv(full_path, index=False,)
print(output.describe())

#Most recent percentile
print(f'Most recent percentile is {(1- (2697/12992))*100 }')