import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

#Basic Data Frames


df = np.array(range(1,10)).reshape(3,3)
df = pd.DataFrame(df, columns = ['A','B','C'], index = ['X','Y','Z'])
# print(df.index)  
# index = Row names


# # Basic data information
# print(df.info())


# #To see the first few rows
# print(df.head())


# #To see the last few rows
# print(df.tail)


# #To see the columns 
# print(df.columns)


# #To see row names
# print(df.index)


#To see the values
#print(df.values)

# #To see basic statistical information
# print(df.describe())

# #To find unique values in a column
# print(df['A'].unique())

# #Loading csv data from a file 
# #Remark: Use a raw strin r"____" in order to avoid complicaitions with escape characters
# file_path = r"JDMmessingaround\RandomStudyingscripts\NSFG_2022_2023_FemRespPUFData.csv"
# NSFG_data = pd.read_csv(file_path)

# #Loading csv data from a url works the same way
# website = r'https://raw.githubusercontent.com/KeithGalli/complete-pandas-tutorial/refs/heads/master/warmup-data/coffee.csv'
# coffee = pd.read_csv(website)

# #Loading Excel data or parquet works similarly
# file_path = r'JDMmessingaround\datasets\olympics-data.xlsx'
# results = pd.read_excel(file_path, sheet_name= 'results') 
# print(results.head())

fruits = pd.DataFrame([[30,21]], columns = ['Apples', "Bananas"])
ingredients = pd.Series(['4 cups', '1 cup',' 2 large', '1 can'], index = ["Flour", 'Milk', 'Eggs', 'Spam'])

print(ingredients)