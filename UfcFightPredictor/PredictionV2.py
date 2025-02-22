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

import joblib


# """ To Do:
#     - Add am 'Age' feature for each fighter
#     - Re-create functions from Ufcdataclearnup so that it doesn't have to run every time
#     - Get Data for total (T)KO's and submissions"""

start = time.time()


### Entry the date of the Fight, defult is Today
Date = datetime.now()


det_data = pd.read_csv(r'UfcFightPredictor\CleanFighterData.cvs')


""" Useful Functions"""
def sec_to_time(sec):
    hours = sec // 3600
    remaining = sec % 3600
    minutes = remaining // 60
    seconds = round(remaining %60,2)
    return f' Total time is {hours} hours, {minutes} minutes, and {seconds}, seconds.'



""" Stats I'm interested in :['Name', 'Height', 'Weight', 'Reach', 'Stance', 'DOB', 'SLpM',
       'Str. Acc.', 'SApM', 'Str. Def.', 'TD Avg.', 'TD Acc.', 'TD Def.',
       'Sub. Avg.', 'Wins', 'Losses', 'Draws', 'NoContests']"""


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
        


# fightersdata = pd.read_csv(r'C:\Users\jorda\OneDrive\Documents\GitHub\JDMmessingaround\UfcProject\ufc_fighters_detailed_stats.csv')


""" Key to interpretation"""

def get_prediction(RedFighter, Bluefighter, value):
    if value == 1:
        return BlueFighter+ ' to win'
    if value == 3:
        return RedFighter+ ' to win'
    if value == 0:
        return 'a draw'
    if value == 2:
        return 'a no contest (lol)'




""" Loading Models"""
nameless_model_path =r'UfcFightPredictor\SavedModels\nameless_model.joblib'
nameless_pipeline_path = r'UfcFightPredictor\SavedModels\nameless_pipeline.joblib'
nameless_model = joblib.load(nameless_model_path)
nameless_pipeline =joblib.load(nameless_pipeline_path)



""" Getting Date of Fight as Float"""
def timestamp_to_float(timestamp):
    # Convert the timestamp to a float representing seconds since the epoch
    timestamp_float = timestamp.timestamp()
    return timestamp_float

Date = timestamp_to_float(Date)



""" Getting prediction Features for fight"""

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

# RedFighter = 'Zhang Weili'
# BlueFighter = 'Tatiana Suarez'


# print(get_mathup_features(RedFighter, BlueFighter, Date))


"""First Fight to Predict"""
RedFighter = 'Jared Cannonier'
BlueFighter = 'Gregory Rodrigues'
X1 = get_mathup_features(RedFighter,BlueFighter,Date)

model_prediction = nameless_model.predict(X1)
pipeline_prediction = nameless_pipeline.predict(X1)
print(f'For the first fight, the Model predicts {get_prediction(RedFighter, BlueFighter,model_prediction)}.', f' and the Pipeline predicts {get_prediction(RedFighter, BlueFighter,pipeline_prediction)}.','', sep = '\n' )

"""Second Fight to Predict"""

RedFighter = 'Calvin Kattar'
BlueFighter = 'Youssef Zalal'
X2 = get_mathup_features(RedFighter,BlueFighter,Date)

model_prediction = nameless_model.predict(X2)
pipeline_prediction = nameless_pipeline.predict(X2)
print(f'For the second fight, the Model predicts {get_prediction(RedFighter, BlueFighter,model_prediction)}.', f' and the Pipeline predicts {get_prediction(RedFighter, BlueFighter,pipeline_prediction)}.','', sep = '\n' )


"""Third Fight to Predict"""

RedFighter = 'Edmen Shahbazyan'
BlueFighter = 'Dylan Budka'
X3 = get_mathup_features(RedFighter,BlueFighter,Date)

model_prediction = nameless_model.predict(X3)
pipeline_prediction = nameless_pipeline.predict(X3)
print(f'For the third fight, the Model predicts {get_prediction(RedFighter, BlueFighter,model_prediction)}.', f' and the Pipeline predicts {get_prediction(RedFighter, BlueFighter,pipeline_prediction)}.','', sep = '\n' )



"""Fourth Fight to Predict"""

RedFighter = 'Ismael Bonfim'
BlueFighter = 'Nazim Sadykhov'
X4 = get_mathup_features(RedFighter,BlueFighter,Date)

model_prediction = nameless_model.predict(X4)
pipeline_prediction = nameless_pipeline.predict(X4)
print(f'For the fourth fight, the Model predicts {get_prediction(RedFighter, BlueFighter,model_prediction)}.', f' and the Pipeline predicts {get_prediction(RedFighter, BlueFighter,pipeline_prediction)}.','', sep = '\n' )

"""Fifth Fight to Predict"""

RedFighter = 'Rodolfo Vieira'
BlueFighter = 'Andre Petroski'
X5 = get_mathup_features(RedFighter,BlueFighter,Date)

model_prediction = nameless_model.predict(X5)
pipeline_prediction = nameless_pipeline.predict(X5)
print(f'For the fifth fight, the Model predicts {get_prediction(RedFighter, BlueFighter,model_prediction)}.', f' and the Pipeline predicts {get_prediction(RedFighter, BlueFighter,pipeline_prediction)}.','', sep = '\n' )



""" Hypothetical"""
# RedFighter = 'Umar Nurmagomedov'
# BlueFighter = 'Merab Dvalishvili'
# X5 = get_mathup_features(RedFighter,BlueFighter,Date)

# model_prediction = nameless_model.predict(X5)
# pipeline_prediction = nameless_pipeline.predict(X5)
# print(f'For the hypothetical fight, the Model predicts {get_prediction(RedFighter, BlueFighter,model_prediction)}.', f' and the Pipeline predicts {get_prediction(RedFighter, BlueFighter,pipeline_prediction)}.','', sep = '\n' )


print(sec_to_time(time.time()- start) )
print('Done')
# #print(X.columns)