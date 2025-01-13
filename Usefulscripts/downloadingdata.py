import os
import requests

# URL of the CSV file
url = "https://www.cdc.gov/nchs/data/nsfg/2022-2023_FemRespData.csv"

# Extract the filename from the URL
filename = os.path.basename(url)

# Get the current directory of the script
current_directory = os.path.dirname(os.path.abspath(__file__))

# Full path to save the file
file_path = os.path.join(current_directory, filename)

try:
    # Send a GET request to the URL
    response = requests.get(url)
    response.raise_for_status()  # Check if the request was successful

    # Write the content to a file
    with open(file_path, 'wb') as file:
        file.write(response.content)

    print(f"File downloaded and saved as: {file_path}")
except requests.exceptions.RequestException as e:
    print(f"An error occurred: {e}")


"""Unzipping files"""


def unzip_file(file_path):
    """
    Unzips a file and saves the contents in the same directory as the ZIP file.

    Args:
        file_path (str): Path to the ZIP file.
    """
    try:
        # Check if the file is a ZIP file
        if not zipfile.is_zipfile(file_path):
            print("The provided file is not a ZIP file.")
            return

        # Get the directory of the ZIP file
        directory = os.path.dirname(file_path)

        # Extract the contents
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(directory)
            print(f"Contents extracted to: {directory}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Replace 'your_file.zip' with the path to your ZIP file
zip_file_path = "your_file.zip"
unzip_file(zip_file_path)
