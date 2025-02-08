import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression


data_file_path = r'C:\Users\jorda\OneDrive\Documents\GitHub\JDMmessingaround\datasets\TimeSeries\book_sales.csv'
booksales = pd.read_csv(data_file_path, index_col= 'Date', parse_dates=['Date'])
booksales = booksales.drop(['Paperback'], axis= 1)
booksales['Time'] =  np.arange(len(booksales.index))
booksales['Lag_1'] = booksales['Hardcover'].shift(1)
booksales = booksales.reindex(columns = ['Hardcover', 'Time', 'Lag_1'])
print(booksales.head())

ar = pd.read_csv(r'C:\Users\jorda\OneDrive\Documents\GitHub\JDMmessingaround\datasets\TimeSeries\ar.csv')
dtype = dtype = {
    'store_nbr': 'category',
    'family': 'category',
    'sales': 'float32',
    'onpromotion': 'uint64',
}

fig, ax = plt.subplots()
ax.plot('Time', 'Hardcover', data = booksales, color = '0.75')
ax = sns.regplot(x = 'Time', y = 'Hardcover', data = booksales, ci = None, scatter_kws=dict(color='0.25'))
ax.set_title ('Time Plot of Hardcover Sales')
plt.show()

plt.clf()
booksales = booksales.dropna()

fig2, ax2 = plt.subplots()
ax.plot('Lag_1', 'Hardcover', data = booksales, color = '0.75')
ax = sns.regplot(x = 'Lag_1', y = 'Hardcover', data = booksales, ci = None, scatter_kws=dict(color='0.25'))
ax.set_title ('Time Plot of Hardcover Sales')
plt.show()