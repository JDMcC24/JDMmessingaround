import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

#from fightdata import fighter_data
import time

data = pd.read_csv(r'C:\Users\jorda\OneDrive\Documents\GitHub\JDMmessingaround\UfcProject\CleanData.cvs')
print(data.loc[data.RedFighter == 'Rodolfo Bellato'])
print(data.loc[data.BlueFighter == 'Rodolfo Bellato'])
data2 = pd.read_csv(r'C:\Users\jorda\OneDrive\Documents\GitHub\JDMmessingaround\UfcProject\ufc_fighters_detailed_stats.csv')
print(data2.loc[data2.Name == 'Rodolfo Bellato' ])

print('done')