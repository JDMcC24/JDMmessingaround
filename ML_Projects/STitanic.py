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
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

import xgboost as xgb





import time 
start = time.time()

"""Importing data"""
file_path = r'C:\Users\jorda\OneDrive\Documents\GitHub\JDMmessingaround\datasets\Spaceship Titanic\spaceship-titanic\train.csv'
X = pd.read_csv(file_path)
y = X.pop('Transported')

file_path = r'C:\Users\jorda\OneDrive\Documents\GitHub\JDMmessingaround\datasets\Spaceship Titanic\spaceship-titanic\test.csv'
test_X = pd.read_csv(file_path)
labels = test_X.pop('PassengerId')



#label_encoder = LabelEncoder()
# print(X.head())
#print(X.Cabin.value_counts() )


##Feature Selection$$
def split_Cabin(df):
    df['Deck'] = df['Cabin'].str.split("/", n=2, expand=True)[0]
    df['Number'] = df['Cabin'].str.split("/", n=2, expand=True)[1]
    df['Side'] = df['Cabin'].str.split("/", n=2, expand=True)[2]
    df.pop('Cabin')
    return df

X = split_Cabin(X)
X['Spent'] = X['RoomService']+ X['FoodCourt']+ X['ShoppingMall'] + X['Spa']+ X['VRDeck']
#X = X.drop(columns=['Name', 'PassengerId', 'RoomService','FoodCourt', 'ShoppingMall','Spa', 'VRDeck' ])
X = X.drop(columns=['Name', 'PassengerId'])
X['Number'] = X['Number'].astype('float')


test_X = split_Cabin(test_X)
test_X['Spent'] = test_X['RoomService']+ test_X['FoodCourt']+ test_X['ShoppingMall'] + test_X['Spa']+ test_X['VRDeck']
#test_X = test_X.drop(columns=['Name', 'RoomService','FoodCourt', 'ShoppingMall','Spa', 'VRDeck' ])
test_X = test_X.drop(columns=['Name'])
test_X['Number'] = test_X['Number'].astype('float')

print(X.columns)

label_encoder = LabelEncoder()
#cat_feature = []
for c in X.columns:
    if X[c].dtype == 'object':
        #cat_feature.append(c)
        X[c] = label_encoder.fit_transform(X[c])
        test_X[c] = label_encoder.fit_transform(test_X[c])

#X = pd.get_dummies(X)

#test_X =pd.get_dummies(test_X)

print(test_X.head())

X_train, X_val, y_train, y_val = train_test_split(X,y)

""" Searching for parameters"""
my_pipeline = Pipeline(steps =[
#('preprocessor', KNNImputer(n_neighbors=1)),
('preprocessor', SimpleImputer()),
('scaler', StandardScaler()),
('model', XGBClassifier() )
])

n=20
step_size = 20
n_estimators = []
for i in range(5,n+1):
    n_estimators.append(int(i*step_size))
    #print(i)
#neighbors= list(range(1,5))
#neighbors = [1]
rates = list(np.linspace(0.0001,.25,25))

#print(rates)
print("Starting GridSearch")
param_grid = {
    #'preprocessor__n_neighbors': neighbors,
     'model__n_estimators': n_estimators,
     'model__learning_rate': rates  
     }


# grid_search = GridSearchCV( estimator=my_pipeline, param_grid = param_grid, cv = 5, scoring = 'accuracy', verbose = 1)
# grid_search.fit(X_train, y_train)
# print(grid_search.best_params_)
# {'model__learning_rate': 0.0105125, 'model__n_estimators': 380, 'preprocessor__n_neighbors': 1}

# """Creating model"""
my_pipeline = Pipeline(steps =[
#('preprocessor', KNNImputer(n_neighbors=1)),
('preprocessor', SimpleImputer()),
('scaler', StandardScaler()),
('model', XGBClassifier(learning_rate =0.0834, n_estimators= 100 ) )
])


"""Fitting Model and making Predicitons"""
my_pipeline.fit(X,y)
test_preds = my_pipeline.predict(test_X)
test_preds = test_preds.astype(bool)


folder_path = r'C:\Users\jorda\OneDrive\Documents\GitHub\JDMmessingaround\datasets\Spaceship Titanic\spaceship-titanic'
file_name = 'submission.csv'
full_path = os.path.join(folder_path, file_name)

output = pd.DataFrame({'PassengerId': labels,
                       'Transported': test_preds})

output.to_csv(full_path, index=False,)
print(output.describe())

print(f'Expected accuracy = {cross_val_score(my_pipeline,X,y, cv = 5, scoring='accuracy').mean()}')




""" Version 2: Playing with Tensorflow and Keras"""
#import tensorflow as tf

# label_encoder = LabelEncoder()
# input_shape = (X_train.shape[1],)

# tf_model = tf.keras.Sequential([
#     tf.keras.layers.Input(shape=input_shape),
#     tf.keras.layers.Dense(10, activation='relu'),
#     tf.keras.layers.Dense(10, activation='relu'),
#     tf.keras.layers.Dense(len(np.unique(y_train)), activation='softmax')  
# ])



# import pandas as pd
# import numpy as np
# import tensorflow as tf
# from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
# from sklearn.pipeline import Pipeline, FunctionTransformer
# from sklearn.compose import ColumnTransformer
# from sklearn.base import BaseEstimator, ClassifierMixin


# Define Custom Estimator for TensorFlow Model
# class TensorFlowClassifier(BaseEstimator, ClassifierMixin):
#     def __init__(self, input_dim, num_classes):
#         self.input_dim = input_dim
#         self.num_classes = num_classes
#         self.model = self.build_model()
#         self.classes_ = None

#     def build_model(self):
#         model = tf.keras.Sequential([
#             tf.keras.layers.Input(shape=(self.input_dim,)),
#             tf.keras.layers.Dense(10, activation='relu'),
#             tf.keras.layers.Dense(10, activation='relu'),
#             tf.keras.layers.Dense(self.num_classes, activation='softmax')
#         ])
#         model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#         return model

#     def fit(self, X, y, **fit_params):
#         self.classes_ = np.unique(y)  # Set the classes_ attribute
#         self.model.fit(X, y, epochs=100, batch_size=32, validation_split=0.2, verbose=0)
#         return self

#     def predict(self, X):
#         return np.argmax(self.model.predict(X), axis=-1)

#     def predict_proba(self, X):
#         return 

# input_dim = X.shape[1]
# num_classes = len(np.unique(y))
# tf_classifier = TensorFlowClassifier(input_dim=input_dim, num_classes=num_classes)

# pipeline = Pipeline(steps =[
# ('preprocessor', KNNImputer(n_neighbors=1)),
# ('scaler', StandardScaler()),
# ('tf_model', tf_classifier) 
# ])




# # Evaluate the pipeline using cross-validation
# def get_score(n):
#     pipeline = Pipeline(steps =[
#     ('preprocessor', KNNImputer(n_neighbors=n)),    
#     ('scaler', StandardScaler()),
#     ('tf_model', tf_classifier) 
# ])
#     scores = cross_val_score(pipeline, X_train, y_train,
#                                   cv=5,scoring='accuracy')
    
#     return scores.mean()
# #c_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
# c_scores =[]
# for i in range(1,11):
#     cscore = get_score(i)
#     c_scores.append(cscore)
#     print(cscore)
 
# plt.plot(list(range(1,11)), c_scores, "--")
# plt.show()



# Print the cross-validation scores
# print(f"Cross-validation scores: {scores}")
# print(f"Mean accuracy: {scores.mean()}")

# pipeline.fit(X,y)

# test_preds = pipeline.predict(test_X)
# test_preds = test_preds.astype(bool)


# folder_path = r'C:\Users\jorda\OneDrive\Documents\GitHub\JDMmessingaround\datasets\Spaceship Titanic\spaceship-titanic'
# file_name = 'submission.csv'
# full_path = os.path.join(folder_path, file_name)

# output = pd.DataFrame({'PassengerId': labels,
#                        'Transported': test_preds})

# output.to_csv(full_path, index=False,)
# print(output.describe())

#print(f'Expected accuracy = {cross_val_score(pipeline,X,y, scoring='accuracy').mean()}')


#print(f'{(1- 310/2103)*100} Percentile so far')

print(f'Total run time {time.time()-start}')