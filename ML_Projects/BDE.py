import pandas as pd
import sklearn as sk
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
import os, requests, zipfile



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
# zip_file_path = r"JDMmessingaround\datasets\Melbournehousing\melb_data.csv.zip"
# unzip_file(zip_file_path)

melhousing = pd.read_csv(r'JDMmessingaround\datasets\Melbournehousing\melb_data.csv')
# #print(melhousing.describe())
# #print(melhousing.columns)
melhousing = melhousing.dropna(axis = 0)
# # print(melhousing.describe())
y = melhousing.Price
melfeatures = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X = melhousing[melfeatures]
# print(X.describe())
# print(X.head)
# mel_model = DecisionTreeRegressor(random_state=42)
# mel_model.fit(X,y)
# # print( f'Making predictions for follwing 4 houses {X.head()}', f"The predidctions are { mel_model.predict(X.head())}", sep= "\n" )
# # print(y)
# predicted_home_prices = mel_model.predict(X)
# print(mean_absolute_error(y, predicted_home_prices))

from sklearn.model_selection import train_test_split


# train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)
# mel_model = DecisionTreeRegressor()
# mel_model.fit(train_X, train_y)
# val_predictions = mel_model.predict(val_X)
# print( mean_absolute_error(val_y, val_predictions))


def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes= max_leaf_nodes, random_state= 0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y,preds_val)
    return mae
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)


# # print(get_mae(10, train_X, val_X, train_y, val_y))
# for max_leaf_nodes in [5,50,500,5000]:
#     mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
#     print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, mae))

from sklearn.ensemble import RandomForestRegressor

forest_model = RandomForestRegressor(random_state = 1)
forest_model.fit(train_X,train_y)
melb_pred = forest_model.predict(val_X)
print(mean_absolute_error(val_y, melb_pred))
