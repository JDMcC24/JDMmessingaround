import os, requests, zipfile
import pandas as pd
import sklearn as sk
import time

starttime = time.time()

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
# zip_file_path = r"JDMmessingaround\datasets\winereivews\winereviews.zip"
# unzip_file(zip_file_path)

file_path = r'JDMmessingaround\datasets\winereivews\winemag-data_first150k.csv'
reviews = pd.read_csv(file_path)
Oregon_wines = reviews.loc[reviews.province == "Oregon"]
# Oregon_PN = Oregon_wines.loc[Oregon_wines.variety == "Pinot Noir"]
# print(Oregon_PN.points.max())
# winner = Oregon_PN.loc[Oregon_PN.points == Oregon_PN.points.max()]
# print(winner.price)
#print(Oregon_wines.taster_name.value_counts())
#print(Oregon_wines.winery.value_counts())

def remean_points(row):
    row.points = row.points - reviews.points.mean()
    return row

review_points_mean = reviews.points.mean()
#print((reviews.points - review_points_mean).describe())
reviews_ratios = reviews.points / reviews.price
print(reviews_ratios.loc[reviews_ratios == reviews_ratios.max()].index)
#print(reviews.loc[reviews.index == reviews_ratios.max().index])



print(time.time() - starttime)
reviews.apply()