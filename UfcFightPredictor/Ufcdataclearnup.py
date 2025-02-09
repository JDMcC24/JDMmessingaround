import pandas as pd
import numpy as np
import time 
from datetime import datetime

start = time.time()


rbr_file_path = r'UfcFightPredictor\current_roundbyround.csv'
rbrdata = pd.read_csv(rbr_file_path)

detailedstats_path = r'UfcFightPredictor\ufc_fighters_detailed_stats.csv'
det_data = pd.read_csv(detailedstats_path)
"""Fixing missing numbers"""
det_data.replace('--', np.nan, inplace=True)


""" Seperating Draws and No Contests for full record"""
def extract_first_number(value):
    if type(value) == str:
        return value.split()[0]
    else:
        return value

"""Useful Functions"""
def sec_to_time(sec):
    hours = sec // 3600
    remaining = sec % 3600
    minutes = remaining // 60
    seconds = remaining %60
    return f' Total time is {hours} hours, {minutes} minutes, and {seconds}, seconds.'

def extract_second_character(value):
    if type(value) == str:
        return value.split()[0][1]
    else:
        return value

def height_to_inches(height_str):
    if isinstance(height_str, str):
        parts = height_str.split("'")
        feet = int(parts[0])
        inches = int(parts[1].replace('"', '').strip())

        total_inches = feet * 12 + inches
        return float(total_inches)
    else:
        return float('nan')  

def convert_weight_to_number(weight_str):
    # Remove the " lbs." part and convert to integer
    weight_number = float(weight_str.replace(" lbs.", ""))
    return weight_number

def convert_reach_to_number(reach_str):
    # Remove the " part and convert to integer
    reach_number = float(reach_str.replace('"' , ''))
    return reach_number

def convert_percent_to_number(reach_str):
    # Remove the "%" part and convert to integer
    reach_number = float(reach_str.replace("%", ""))
    return reach_number

def timestamp_to_float(timestamp):
    # Convert the timestamp to a float representing seconds since the epoch
    timestamp_float = timestamp.timestamp()
    return timestamp_float

""" Seperating Draws and No Contests for full record"""
det_data['NoContests'] = det_data.Draws
det_data['Draws'] = det_data['Draws'].apply(extract_first_number)
det_data['NoContests'] = det_data['NoContests'].str.extract(r'(\(.*?\))')
det_data.NoContests.replace(np.nan, 0, inplace=True)
det_data['NoContests'] = det_data['NoContests'].apply(extract_second_character)

#print(det_data.Weight.dtype)

"""Renaming Columns for consistency"""
rbrdata = rbrdata.rename(columns = {'fighter' : 'Name'})
rbrdata = rbrdata.rename(columns = {'fight_date' : 'Date'})


"""Removing fighters with missing DOB and fights with missing event dates"""
det_data = det_data.dropna(subset = ['DOB'])
rbrdata = rbrdata.dropna(subset = ['Date'])

#print(det_data.columns.to_list())
""" Getting Standardized Dates"""
rbrdata['Date'] = pd.to_datetime(rbrdata.Date)
det_data['DOB'] = pd.to_datetime(det_data['DOB'], format='%b %d, %Y', errors = 'coerce')
det_data['DOB'] = det_data['DOB'].apply(timestamp_to_float)
rbrdata['Date'] = rbrdata['Date'].apply(timestamp_to_float)



""" Fixing Heights to floats"""
det_data['Height'] = det_data['Height'].apply(height_to_inches)





""" Making sure only consider fighters and fights in both datasets"""

UFCFighters = set(det_data.Name.unique())
Allfighters = set(rbrdata.Name.unique())
fighterset = UFCFighters.intersection(Allfighters)
droppedfighters=  UFCFighters.union(Allfighters).difference(fighterset)

for fighter in droppedfighters:
    det_data = det_data[det_data['Name']!= fighter]
    badfights = rbrdata.loc[rbrdata.Name == fighter].id.unique()
    for fight in badfights:
        rbrdata = rbrdata[rbrdata['id']!= fight]


"""Changing Weight and Reach to floats"""
det_data['Weight'] = det_data['Weight'].astype(str)
det_data['Weight'] = det_data['Weight'].apply(convert_weight_to_number)
det_data['Reach'] = det_data['Reach'].astype(str)
det_data['Reach'] = det_data['Reach'].apply(convert_reach_to_number)


""" Percentage Stats : = ['Str. Acc.', 'SApM', 'Str. Def.',  'TD Acc.', 'TD Def.']"""

""" Converting percentages to floats"""
perstats = ['Str. Acc.', 'SApM', 'Str. Def.',  'TD Acc.', 'TD Def.']
for stat in perstats:
    det_data[stat] = det_data[stat].astype(str)
    det_data[stat] = det_data[stat].apply(convert_percent_to_number)



""" Stats I'm interested in :['Name', 'Height', 'Weight', 'Reach', 'Stance', 'DOB', 'SLpM',
       'Str. Acc.', 'SApM', 'Str. Def.', 'TD Avg.', 'TD Acc.', 'TD Def.',
       'Sub. Avg.', 'Wins', 'Losses', 'Draws', 'NoContests']"""

det_data.to_csv( r'UfcFightPredictor\CleanFighterData.cvs', index=False)



def get_fighter_stats(fighter):
    stats =  ['Height', 'Weight', 'Reach', 'Stance', 'DOB', 'SLpM',
       'Str. Acc.', 'SApM', 'Str. Def.', 'TD Avg.', 'TD Acc.', 'TD Def.',
       'Sub. Avg.', 'Wins', 'Losses', 'Draws', 'NoContests']
    fighter_stats = []
    fighter_index = det_data.loc[det_data.Name== fighter].index[0]
    for stat in stats:
        fighter_stats.append(det_data.loc[fighter_index, stat])
    return fighter_stats
# print(get_fighter_stats('Islam Makhachev'))

#print(det_data.loc[det_data.Name == 'Islam Makhachev'])

# print(det_data.loc[det_data.Name == 'Islam Makhachev'].index[0])


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
        


today = datetime.today()
testmatch = matchup_stats('Islam Makhachev','Renato Moicano',today)

# print(testmatch)
match_data = testmatch


"""" Creating condensed DataFrame"""

fights = rbrdata.id.unique()
winslist = []

for fight in fights:
    X = rbrdata.loc[rbrdata.id == fight]
    redfighter = X.iloc[0,0]
    bluefighter = X.iloc[1,0]
    Winloc = X.columns.get_loc('winner')
    # if X.iloc[0,Winloc] == 'W':
    #     winslist.append(1)
    # elif X.iloc[0,Winloc] == 'L':
    #     winslist.append(-1)
    # else:
    #     winslist.append(0)
    win = X.iloc[0,Winloc]
    winslist.append(win)
    #print(win)
    Dateloc = X.columns.get_loc('Date')
    Date = X.iloc[0,Dateloc]
    newmatchup = matchup_stats(redfighter,bluefighter,Date)
    match_data = pd.concat([match_data,newmatchup], ignore_index = True)
match_data = match_data.drop(index= [0])
match_data['Result'] = winslist
#match_data = match_data.drop(match_data.columns[0],axis = 0 )
# match_data.pop('id')
# print(match_data.head())
# match_data['Date'] = match_data['Date'].apply(timestamp_to_float)
match_data.Date = match_data.Date.astype(float)
match_data.RedDOB = match_data.RedDOB.astype(float)
match_data.BlueDOB = match_data.BlueDOB.astype(float)
match_data['RedAge'] = match_data['Date'] - match_data['RedDOB']
match_data['BlueAge'] = match_data['Date'] - match_data['BlueDOB']



path = r'C:\Users\jorda\OneDrive\Documents\GitHub\JDMmessingaround\UfcProject'
filename = 'CleanData.cvs'
match_data.to_csv( r'UfcFightPredictor\CleanData.cvs', index=False)

print(f'Clean Data has been saved as '+ filename)


print('For clear up:'+  sec_to_time(time.time() - start) + '.')





