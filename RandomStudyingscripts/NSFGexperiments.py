import os
import requests
import zipfile
import numpy as np
import pandas as pd
import time
from matplotlib import pyplot as plt
starttime = time.time()


# url = "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/NSFG/NSFG-2022-2023-FemRespPUFData.zip"
# filename = os.path.basename(url)
# current_directory = os.path.dirname(os.path.abspath(__file__))
# file_path = os.path.join(current_directory, filename)
# try:
#     # Send a GET request to the URL
#     response = requests.get(url)
#     response.raise_for_status()  # Check if the request was successful

#     # Write the content to a file
#     with open(file_path, 'wb') as file:
#         file.write(response.content)

#     print(f"File downloaded and saved as: {file_path}")
# except requests.exceptions.RequestException as e:
#     print(f"An error occurred: {e}")


# def unzip_file(file_path):
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
# zip_file_path = os.path.join(current_directory, filename)
# unzip_file(zip_file_path)


NSFG_data = pd.read_csv("ProbabilityandSatisticsforProgramers/Chapter1/NSFG_2022_2023_FemRespPUFData.csv")
#print(NSFG_data.head())
# NSFG_data.hist(column = 'RSCRAGE', bins = 30)
# plt.show()

print((NSFG_data['RSCRAGE']==15).sum()/ len(NSFG_data['RSCRAGE']))
print(time.time() - starttime)