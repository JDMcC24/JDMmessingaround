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
from datetime import datetime
#from Ufcdataclearnup import *
import joblib



starttime = time.time()


### Entry the date of the Fight, defult is Today
Date = datetime.now()
def timestamp_to_float(timestamp):
    # Convert the timestamp to a float representing seconds since the epoch
    timestamp_float = timestamp.timestamp()
    return timestamp_float

Date = timestamp_to_float(Date)

data = pd.read_csv(r'UfcFightPredictor\CleanData.cvs')
det_data = pd.read_csv(r'UfcFightPredictor\CleanFighterData.cvs')

def sec_to_time(sec):
    hours = sec // 3600
    remaining = sec % 3600
    minutes = remaining // 60
    seconds = remaining %60
    return f' Total time is {hours} hours, {minutes} minutes, and {seconds}, seconds.'


def get_fighter_stats(fighter):
    stats =  ['Height', 'Weight', 'Reach', 'Stance', 'DOB', 'SLpM',
       'Str. Acc.', 'SApM', 'Str. Def.', 'TD Avg.', 'TD Acc.', 'TD Def.',
       'Sub. Avg.', 'Wins', 'Losses', 'Draws', 'NoContests']
    fighter_stats = []
    fighter_index = det_data.loc[det_data.Name== fighter].index[0]
    for stat in stats:
        fighter_stats.append(det_data.loc[fighter_index, stat])
    return fighter_stats

def matchup_stats(redfighter,bluefighter, Date):
    stats =  ['Height', 'Weight', 'Reach', 'Stance', 'DOB', 'SLpM',
       'Str. Acc.', 'SApM', 'Str. Def.', 'TD Avg.', 'TD Acc.', 'TD Def.',
       'Sub. Avg.', 'Wins', 'Losses', 'Draws', 'NoContests']
    redstats = get_fighter_stats(redfighter)
    bluestats = get_fighter_stats(bluefighter)
    redcolumns = []
    bluecolumns = []

    for col in stats:
        redcolumns.append('Red'+col)
        bluecolumns.append('Blue'+col)
    columns = ['RedFighter', 'BlueFighter', 'Date'] + redcolumns + bluecolumns
    #[redfighter,bluefighter, Date]+ redstats + bluestats
    stats = np.array([[redfighter,bluefighter, Date]+ redstats + bluestats])
    matchup = pd.DataFrame(stats, columns = columns)
    return matchup
        

Allfeatures = ['Date', 'RedHeight', 'RedWeight', 'RedReach', 'RedDOB', 'RedSLpM', 'RedStr. Acc.', 'RedSApM', 'RedStr. Def.', 'RedTD Avg.', 'RedTD Acc.', 'RedTD Def.', 'RedSub. Avg.', 'RedWins', 'RedLosses', 'RedDraws', 'RedNoContests', 'BlueHeight', 'BlueWeight', 'BlueReach', 'BlueDOB', 'BlueSLpM', 'BlueStr. Acc.', 'BlueSApM', 'BlueStr. Def.', 'BlueTD Avg.', 'BlueTD Acc.', 'BlueTD Def.', 'BlueSub. Avg.', 'BlueWins', 'BlueLosses', 'BlueDraws', 'BlueNoContests', 'RedAge', 'BlueAge', 'RedStance_Open Stance', 'RedStance_Orthodox', 'RedStance_Southpaw', 'RedStance_Switch', 'BlueStance_Open Stance', 'BlueStance_Orthodox', 'BlueStance_Sideways', 'BlueStance_Southpaw', 'BlueStance_Switch']
column_order = Allfeatures
def get_mathup_features(RedFighter, BlueFighter, Date):
    matchup = matchup_stats(RedFighter, BlueFighter, Date)
    matchup = matchup.drop(columns=['RedFighter', 'BlueFighter'])
    dummies = pd.get_dummies(matchup[['RedStance', 'BlueStance']])
    matchup = pd.concat([matchup,dummies], axis = 1)
    matchup = matchup.drop(['RedStance', 'BlueStance'], axis = 1)
    matchup['RedDOB'] = matchup['RedDOB'].astype(float)
    matchup['Date'] = matchup['Date'].astype(float)
    matchup['BlueDOB'] = matchup['BlueDOB'].astype(float)
    matchup['RedAge'] = matchup['Date'] - matchup['RedDOB']
    matchup['BlueAge'] = matchup['Date'] - matchup['BlueDOB']
    presentfeatures=matchup.columns.to_list()
    for feature in Allfeatures:
        if feature not in presentfeatures:
            matchup[feature] = False
    matchup = matchup[column_order]
    return matchup



y = data.pop('Result')
labelencoder = LabelEncoder()
y = labelencoder.fit_transform(y)
category_mapping = dict(zip(labelencoder.classes_, range(len(labelencoder.classes_))))
X = data.drop(columns = ['RedFighter', 'BlueFighter'])
X = pd.get_dummies(X)
Allfeatures = X.columns.to_list()
#print(X['RedStance_Open Stance'])
#print(X.head())



# my_pipeline = Pipeline(steps =[
# #('preprocessor', KNNImputer(n_neighbors=1)),
# ('preprocessor', SimpleImputer()),
# ('scaler', StandardScaler()),
# ('model', XGBClassifier() )
# ])

# n=20
# step_size = 50
# n_estimators = []
# for i in range(1,n+1):
#     n_estimators.append(int(i*step_size))
#     #print(i)
# #neighbors= list(range(1,5))
# #neighbors = [1]
# rates = list(np.linspace(0.0001,.20,10))

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
"""" Grid Search best results:
{'model__learning_rate': 0.02231111111111111, 'model__n_estimators': 200}
Expected accuracy for pipeline model is 0.7324299909665763"""

#n is number of fights 'simulated'
n = 11

RedFighter = 'Ismael Bonfim'
BlueFighter = 'Nazim Sadykhov'
X1 = get_mathup_features(RedFighter,BlueFighter,Date)

import random
Model_winlist = []
Pipeline_winlist = []
for i in range(n):
    print(i+1)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=.01)
    seed = random.randint(0,100000)
    nameless_pipeline = Pipeline(steps =[
#('preprocessor', KNNImputer(n_neighbors=1)),
    ('preprocessor', SimpleImputer()),
    ('scaler', StandardScaler()),
    ('model', XGBClassifier(learning_rate =0.02231111111111111, n_estimators = 200, random_state= seed) )
    ])
    nameless_fight_model = RandomForestClassifier(random_state=seed)
    # nameless_fight_model.fit(X,y)
    # nameless_pipeline.fit(X,y)
    nameless_fight_model.fit(X_train,y_train)
    nameless_pipeline.fit(X_train,y_train)
    if nameless_fight_model.predict(X1)[0]== 3:
        Model_winlist.append(1)
    if nameless_pipeline.predict(X1)[0] ==3:
        Pipeline_winlist.append(1)

# print(Pipeline_winlist, Model_winlist)

print(f' After {n} simulations, {RedFighter} Won {(sum(Model_winlist)/n) * 100} percent of the time according to the simple model',
      f' After {n} simulations, {RedFighter} Won {(sum(Pipeline_winlist)/n) * 100} according to the pipeline Model', sep = '\n')
print(sec_to_time(time.time()-starttime))

