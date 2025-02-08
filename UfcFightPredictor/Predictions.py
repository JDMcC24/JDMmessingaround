import pandas as pd
import numpy as np
import time
from datetime import datetime,timedelta
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


""" To Do:
    - Re-create functions from Ufcdataclearnup so that it doesn't have to run every time
    - Get Data for total (T)KO's and submissions"""

def sec_to_time(sec):
    hours = sec // 3600
    remaining = sec % 3600
    minutes = remaining // 60
    seconds = round(remaining %60,2)
    return f' Total time is {hours} hours, {minutes} minutes, and {seconds}, seconds.'



# fightersdata = pd.read_csv(r'C:\Users\jorda\OneDrive\Documents\GitHub\JDMmessingaround\UfcProject\ufc_fighters_detailed_stats.csv')


""" Key to interpretation"""

def get_prediction(RedFighter, Bluefighter, value):
    if value == 1:
        return BlueFighter+ ' to win '
    if value == 3:
        return RedFighter+ ' to win '
    if value == 0:
        return 'a draw '
    if value == 2:
        return 'a no contest (lol) '



# print(X.columns)

""" Loading Models"""
nameless_model_path =r'C:\Users\jorda\OneDrive\Documents\GitHub\JDMmessingaround\UfcProject\SavedModels\nameless_model.joblib'
nameless_pipeline_path = r'C:\Users\jorda\OneDrive\Documents\GitHub\JDMmessingaround\UfcProject\SavedModels\nameless_pipeline.joblib'
nameless_model = joblib.load(nameless_model_path)
nameless_pipeline =joblib.load(nameless_pipeline_path)


# RedFighter = 'Zhang Weili'
# BlueFighter = 'Tatiana Suarez'

""" Getting Date of Fight as Float"""
def timestamp_to_float(timestamp):
    # Convert the timestamp to a float representing seconds since the epoch
    timestamp_float = timestamp.timestamp()
    return timestamp_float



""" Getting prediction Features for fight"""

Allfeatures = ['BlueStance_Switch', 'RedDraws', 'RedLosses', 'RedWeight', 'RedTD Acc.', 'BlueStance_Southpaw', 'BlueStr. Acc.', 'BlueHeight', 'RedStr. Def.', 'BlueWeight', 'BlueReach', 'RedNoContests', 'BlueTD Def.', 'RedStance_Switch', 'RedStr. Acc.', 'BlueSApM', 'BlueWins', 'RedSLpM', 'BlueStance_Orthodox', 'RedHeight', 'BlueStr. Def.', 'RedReach', 'BlueStance_Sideways', 'BlueSLpM', 'BlueTD Acc.', 'BlueTD Avg.', 'RedSApM', 'RedSub. Avg.', 'RedWins', 'BlueNoContests', 'RedStance_Orthodox', 'RedTD Def.', 'RedTD Avg.', 'RedStance_Open Stance', 'BlueStance_Open Stance', 'RedStance_Southpaw', 'BlueSub. Avg.', 'BlueLosses', 'BlueDraws', 'BlueDOB', 'Date', 'RedDOB']
column_order = ['Date', 'RedHeight', 'RedWeight', 'RedReach', 'RedDOB', 'RedSLpM',
       'RedStr. Acc.', 'RedSApM', 'RedStr. Def.', 'RedTD Avg.', 'RedTD Acc.',
       'RedTD Def.', 'RedSub. Avg.', 'RedWins', 'RedLosses', 'RedDraws',
       'RedNoContests', 'BlueHeight', 'BlueWeight', 'BlueReach', 'BlueDOB',
       'BlueSLpM', 'BlueStr. Acc.', 'BlueSApM', 'BlueStr. Def.', 'BlueTD Avg.',
       'BlueTD Acc.', 'BlueTD Def.', 'BlueSub. Avg.', 'BlueWins', 'BlueLosses',
       'BlueDraws', 'BlueNoContests', 'RedStance_Open Stance',
       'RedStance_Orthodox', 'RedStance_Southpaw', 'RedStance_Switch',
       'BlueStance_Open Stance', 'BlueStance_Orthodox', 'BlueStance_Sideways',
       'BlueStance_Southpaw', 'BlueStance_Switch']

def get_mathup_features(RedFighter, BlueFighter, Date):
    matchup = matchup_stats(RedFighter, BlueFighter, Date)
    matchup = matchup.drop(columns=['RedFighter', 'BlueFighter'])
    dummies = pd.get_dummies(matchup[['RedStance', 'BlueStance']])
    matchup = pd.concat([matchup,dummies], axis = 1)
    matchup = matchup.drop(['RedStance', 'BlueStance'], axis = 1)
    presentfeatures=matchup.columns.to_list()
    for feature in Allfeatures:
        if feature not in presentfeatures:
            matchup[feature] = False
    matchup = matchup[column_order]
    return matchup

RedFighter = 'Zhang Weili'
BlueFighter = 'Tatiana Suarez'
Date = datetime.now()+timedelta(days=1)
Date = timestamp_to_float(Date)
#print(get_mathup_features(RedFighter, BlueFighter, Date))


"""First Fight to Predict"""
RedFighter = 'Zhang Weili'
BlueFighter = 'Tatiana Suarez'
X1 = get_mathup_features(RedFighter,BlueFighter,Date)

model_prediction = nameless_model.predict(X1)
pipeline_prediction = nameless_pipeline.predict(X1)
print(f'For the first fight, the Model predicts {get_prediction(RedFighter, BlueFighter,model_prediction)} and the Pipeline predicts {get_prediction(RedFighter, BlueFighter,pipeline_prediction)}' )

"""Second Fight to Predict"""

RedFighter = 'Dricus Du Plessis'
BlueFighter = 'Sean Strickland'
X2 = get_mathup_features(RedFighter,BlueFighter,Date)

model_prediction = nameless_model.predict(X2)
pipeline_prediction = nameless_pipeline.predict(X2)
print(f'For the second fight, the Model predicts {get_prediction(RedFighter, BlueFighter,model_prediction)} and the Pipeline predicts {get_prediction(RedFighter, BlueFighter,pipeline_prediction)}' )


"""Third Fight to Predict"""

RedFighter = 'Tom Nolan'
BlueFighter = 'Viacheslav Borshchev'
X3 = get_mathup_features(RedFighter,BlueFighter,Date)

model_prediction = nameless_model.predict(X3)
pipeline_prediction = nameless_pipeline.predict(X3)
print(f'For the third fight, the Model predicts {get_prediction(RedFighter, BlueFighter,model_prediction)} and the Pipeline predicts {get_prediction(RedFighter, BlueFighter,pipeline_prediction)}' )



"""Fourth Fight to Predict"""

RedFighter = 'Jimmy Crute'
BlueFighter = 'Rodolfo Bellato'
X4 = get_mathup_features(RedFighter,BlueFighter,Date)

model_prediction = nameless_model.predict(X4)
pipeline_prediction = nameless_pipeline.predict(X4)
print(f'For the fourth fight, the Model predicts {get_prediction(RedFighter, BlueFighter,model_prediction)} and the Pipeline predicts {get_prediction(RedFighter, BlueFighter,pipeline_prediction)}' )

"""Fifth Fight to Predict"""

RedFighter = 'Jake Matthews'
BlueFighter = 'Francisco Prado'
X5 = get_mathup_features(RedFighter,BlueFighter,Date)

model_prediction = nameless_model.predict(X5)
pipeline_prediction = nameless_pipeline.predict(X5)
print(f'For the fifth fight, the Model predicts {get_prediction(RedFighter, BlueFighter,model_prediction)} and the Pipeline predicts {get_prediction(RedFighter, BlueFighter,pipeline_prediction)}' )



# RedFighter = 'Israel Adesanya'
# BlueFighter = 'Nassourdine Imavov'
# X1 = get_mathup_features(RedFighter,BlueFighter,Date)

# model_prediction = nameless_model.predict(X1)
# pipeline_prediction = nameless_pipeline.predict(X1)
# print(f'For the first fight, the Model predicts {get_prediction(RedFighter, BlueFighter,model_prediction)} and the Pipeline predicts {get_prediction(RedFighter, BlueFighter,pipeline_prediction)}' )

# RedFighter = 'Shara Magomedov'
# BlueFighter = 'Michael Page'
# X1 = get_mathup_features(RedFighter,BlueFighter,Date)

# model_prediction = nameless_model.predict(X1)
# pipeline_prediction = nameless_pipeline.predict(X1)
# print(f'For the first fight, the Model predicts {get_prediction(RedFighter, BlueFighter,model_prediction)} and the Pipeline predicts {get_prediction(RedFighter, BlueFighter,pipeline_prediction)}' )


print(sec_to_time(time.time()- start) )
print('Done')
#print(X.columns)