import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

#from fightdata import fighter_data
import time

data = pd.read_csv(r'UfcFightPredictor\CleanData.cvs')
data.RedAge = data.Date - data.RedDOB

print(data.columns.to_list())


print('done')